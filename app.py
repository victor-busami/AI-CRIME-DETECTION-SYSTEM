import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import os
import json
import hashlib
import secrets
from datetime import datetime, timedelta
from ultralytics import YOLO
import numpy as np
from transformers import pipeline
import re
import torch
import clip
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import base64
from io import BytesIO
import uuid
import logging
from typing import Dict, List, Tuple, Optional
import sqlite3
from contextlib import contextmanager
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F

# ------------------ ENHANCED CONFIG ------------------
st.set_page_config(
    page_title="AI Crime Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1E3A8A;
        font-size: 2.5em;
        margin-bottom: 20px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .status-urgent { color: #DC2626; font-weight: bold; }
    .status-normal { color: #D97706; font-weight: bold; }
    .status-low { color: #059669; font-weight: bold; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .alert-box {
        background-color: #FEF2F2;
        border: 2px solid #DC2626;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🚨 AI Crime Detection System</h1>', unsafe_allow_html=True)

# ------------------ ENHANCED SECURITY ------------------
class SecurityManager:
    def __init__(self):
        self.users_db = "users.db"
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for users"""
        with sqlite3.connect(self.users_db) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    active INTEGER DEFAULT 1
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    username TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    FOREIGN KEY (username) REFERENCES users (username)
                )
            ''')
            # Create default admin user
            self.create_default_admin()
    
    def create_default_admin(self):
        """Create default admin user if it doesn't exist"""
        admin_password = "CrimeAdmin2024!"
        password_hash = hashlib.sha256(admin_password.encode()).hexdigest()
        
        with sqlite3.connect(self.users_db) as conn:
            conn.execute('''
                INSERT OR IGNORE INTO users (username, password_hash, role)
                VALUES (?, ?, 'admin')
            ''', ("admin", password_hash))
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{password_hash}"
    
    def verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash"""
        try:
            salt, password_hash = stored_hash.split(":")
            return hashlib.sha256((password + salt).encode()).hexdigest() == password_hash
        except:
            # Fallback for simple hash (legacy)
            return hashlib.sha256(password.encode()).hexdigest() == stored_hash
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Authenticate user and return role"""
        with sqlite3.connect(self.users_db) as conn:
            cursor = conn.execute('''
                SELECT password_hash, role FROM users 
                WHERE username = ? AND active = 1
            ''', (username,))
            result = cursor.fetchone()
            
            if result and self.verify_password(password, result[0]):
                # Update last login
                conn.execute('''
                    UPDATE users SET last_login = CURRENT_TIMESTAMP 
                    WHERE username = ?
                ''', (username,))
                return result[1]
        return None
    
    def create_session(self, username: str) -> str:
        """Create new session"""
        session_id = str(uuid.uuid4())
        expires_at = datetime.now() + timedelta(hours=24)
        
        with sqlite3.connect(self.users_db) as conn:
            conn.execute('''
                INSERT INTO sessions (session_id, username, expires_at)
                VALUES (?, ?, ?)
            ''', (session_id, username, expires_at))
        
        return session_id
    
    def validate_session(self, session_id: str) -> Optional[str]:
        """Validate session and return username"""
        with sqlite3.connect(self.users_db) as conn:
            cursor = conn.execute('''
                SELECT username FROM sessions 
                WHERE session_id = ? AND expires_at > CURRENT_TIMESTAMP
            ''', (session_id,))
            result = cursor.fetchone()
            return result[0] if result else None

# ------------------ ENHANCED DATABASE ------------------
class DatabaseManager:
    def __init__(self):
        self.db_path = "crime_reports.db"
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    location TEXT NOT NULL,
                    latitude REAL,
                    longitude REAL,
                    image_path TEXT,
                    annotated_path TEXT,
                    description TEXT,
                    objects_detected TEXT,
                    priority TEXT,
                    sentiment TEXT,
                    keywords TEXT,
                    flagged_for_review INTEGER DEFAULT 0,
                    clip_similarity REAL,
                    severity_score REAL,
                    submitted_by TEXT,
                    reviewed_by TEXT,
                    review_timestamp TIMESTAMP,
                    status TEXT DEFAULT 'pending'
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS analytics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE,
                    location TEXT,
                    total_reports INTEGER,
                    urgent_reports INTEGER,
                    normal_reports INTEGER,
                    low_reports INTEGER,
                    avg_severity REAL
                )
            ''')
    
    def insert_report(self, report_data: Dict) -> int:
        """Insert new report and return ID"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                INSERT INTO reports (
                    location, latitude, longitude, image_path, annotated_path,
                    description, objects_detected, priority, sentiment, keywords,
                    flagged_for_review, clip_similarity, severity_score, submitted_by, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                report_data['location'], report_data['latitude'], report_data['longitude'],
                report_data['image_path'], report_data['annotated_path'], report_data['description'],
                report_data['objects_detected'], report_data['priority'], report_data['sentiment'],
                report_data['keywords'], report_data['flagged_for_review'], report_data['clip_similarity'],
                report_data['severity_score'], report_data['submitted_by'], report_data['status']
            ))
            return cursor.lastrowid
    
    def get_reports(self, limit: int = 100, filters: Dict = None) -> List[Dict]:
        """Get reports with optional filters"""
        query = "SELECT * FROM reports"
        params = []
        
        if filters:
            conditions = []
            if filters.get('priority'):
                conditions.append("priority IN ({})".format(','.join(['?' for _ in filters['priority']])))
                params.extend(filters['priority'])
            if filters.get('status'):
                conditions.append("status = ?")
                params.append(filters['status'])
            if filters.get('flagged_only'):
                conditions.append("flagged_for_review = 1")
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def update_report_status(self, report_id: int, status: str, reviewed_by: str, priority: str = None):
        """Update report status"""
        with sqlite3.connect(self.db_path) as conn:
            if priority:
                conn.execute('''
                    UPDATE reports 
                    SET status = ?, reviewed_by = ?, review_timestamp = CURRENT_TIMESTAMP,
                        priority = ?, flagged_for_review = 0
                    WHERE id = ?
                ''', (status, reviewed_by, priority, report_id))
            else:
                conn.execute('''
                    UPDATE reports 
                    SET status = ?, reviewed_by = ?, review_timestamp = CURRENT_TIMESTAMP
                    WHERE id = ?
                ''', (status, reviewed_by, report_id))

# ------------------ ENHANCED LOCATION SYSTEM ------------------
LOCATION_COORDS = {
    # Nairobi County
    "Nairobi CBD": (-1.2864, 36.8172),
    "Westlands": (-1.2696, 36.8074),
    "Kasarani": (-1.2176, 36.8907),
    "Embakasi": (-1.3218, 36.8786),
    "Kibera": (-1.3133, 36.7892),
    
    # Mombasa County
    "Mombasa CBD": (-4.0435, 39.6682),
    "Nyali": (-4.0178, 39.7064),
    "Likoni": (-4.0835, 39.6648),
    
    # Other Major Cities
    "Kisumu": (-0.0917, 34.7679),
    "Eldoret": (0.5143, 35.2698),
    "Nakuru": (-0.3031, 36.0800),
    "Thika": (-1.0395, 37.0690),
    "Machakos": (-1.5177, 37.2634),
    "Meru": (0.0467, 37.6556),
    "Nyeri": (-0.4209, 36.9483),
    "Kericho": (-0.3677, 35.2861),
    
    # Custom Location
    "Other": (0.0, 0.0)
}

# Enhanced priority weights for severity scoring
SEVERITY_WEIGHTS = {
    "fire": 0.9, "explosion": 0.95, "gun": 0.85, "knife": 0.7, "weapon": 0.75,
    "blood": 0.8, "shooting": 0.9, "murder": 0.95, "assault": 0.7, "robbery": 0.75,
    "ambulance": 0.6, "police": 0.5, "emergency": 0.8, "accident": 0.6,
    "fight": 0.65, "violence": 0.7, "theft": 0.5, "burglary": 0.6, "vandalism": 0.4
}

# ------------------ ENHANCED AI MODELS ------------------
class AIModelManager:
    def __init__(self):
        self.yolo_model = None
        self.faster_rcnn_model = None
        self.sentiment_pipeline = None
        self.clip_model = None
        self.clip_preprocess = None
        self.device = None
        
        # Optimized pre-set thresholds
        self.detection_conf = 0.3  # Lowered confidence threshold for better recall
        self.clip_threshold = 0.25  # CLIP similarity threshold
        self.auto_escalate_threshold = 0.8  # Auto-escalate severity threshold
        self.iou_threshold = 0.5  # IOU threshold for NMS
    
    @st.cache_resource
    def load_yolo_model(_self):
        """Load YOLO model with caching"""
        try:
            return YOLO("yolov8n.pt")  # Using nano version for faster inference
        except Exception as e:
            st.error(f"Failed to load YOLO model: {e}")
            return None
    
    @st.cache_resource
    def load_faster_rcnn_model(_self):
        """Load Faster R-CNN model with caching"""
        try:
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Failed to load Faster R-CNN model: {e}")
            return None
    
    @st.cache_resource
    def load_sentiment_pipeline(_self):
        """Load sentiment analysis pipeline"""
        try:
            return pipeline("sentiment-analysis", 
                          model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        except Exception as e:
            st.error(f"Failed to load sentiment model: {e}")
            return None
    
    @st.cache_resource
    def load_clip_model(_self):
        """Load CLIP model for image-text similarity"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-B/32", device=device)
            return model, preprocess, device
        except Exception as e:
            st.error(f"Failed to load CLIP model: {e}")
            return None, None, "cpu"
    
    def apply_nms(self, boxes, scores, iou_threshold=0.5):
        """Apply Non-Maximum Suppression to eliminate duplicate detections"""
        if len(boxes) == 0:
            return torch.tensor([]), torch.tensor([])
        
        boxes_tensor = torch.tensor(boxes)
        scores_tensor = torch.tensor(scores)
        
        # Convert to [x1, y1, x2, y2] format
        boxes_tensor[:, 2] = boxes_tensor[:, 0] + boxes_tensor[:, 2]
        boxes_tensor[:, 3] = boxes_tensor[:, 1] + boxes_tensor[:, 3]
        
        # Apply NMS
        keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)
        
        return boxes_tensor[keep_indices], scores_tensor[keep_indices]
    
    def detect_objects_faster_rcnn(self, image: np.ndarray) -> Tuple[List, List, List]:
        """Detect objects using Faster R-CNN"""
        if self.faster_rcnn_model is None:
            self.faster_rcnn_model = self.load_faster_rcnn_model()
        
        if self.faster_rcnn_model is None:
            return [], [], []
        
        try:
            # Convert PIL image to tensor
            img_tensor = F.to_tensor(image).unsqueeze(0)
            
            # Run inference
            with torch.no_grad():
                predictions = self.faster_rcnn_model(img_tensor)
            
            # Process results
            boxes = predictions[0]['boxes'].cpu().numpy()
            scores = predictions[0]['scores'].cpu().numpy()
            labels = predictions[0]['labels'].cpu().numpy()
            
            # Apply confidence threshold
            valid_indices = scores > self.detection_conf
            boxes = boxes[valid_indices]
            scores = scores[valid_indices]
            labels = labels[valid_indices]
            
            # Convert to [x, y, width, height] format
            boxes = [[b[0], b[1], b[2]-b[0], b[3]-b[1]] for b in boxes]
            
            # Get COCO class names
            coco_class_names = self.get_coco_class_names()
            class_names = [coco_class_names[label] for label in labels]
            
            return class_names, scores, boxes
            
        except Exception as e:
            st.error(f"Faster R-CNN detection failed: {e}")
            return [], [], []
    
    def get_coco_class_names(self):
        """Return COCO dataset class names"""
        return [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
            'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
            'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A',
            'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def detect_objects_yolo(self, image: np.ndarray) -> Tuple[List, List]:
        """Detect objects using YOLO"""
        if self.yolo_model is None:
            self.yolo_model = self.load_yolo_model()
        
        if self.yolo_model is None:
            return [], []
        
        try:
            # Use optimized pre-set confidence threshold
            results = self.yolo_model.predict(image, conf=self.detection_conf, verbose=False)
            
            # Extract detected objects and their confidence scores
            detected_objects = []
            confidence_scores = []
            boxes = []
            
            if len(results[0].boxes) > 0:
                classes = results[0].names
                for cls, conf, box in zip(results[0].boxes.cls, results[0].boxes.conf, results[0].boxes.xywh):
                    class_name = classes[int(cls)]
                    detected_objects.append(class_name)
                    confidence_scores.append(float(conf))
                    boxes.append(box.cpu().numpy().tolist())
            
            return detected_objects, confidence_scores, boxes
            
        except Exception as e:
            st.error(f"YOLO detection failed: {e}")
            return [], [], []
    
    def detect_objects_ensemble(self, image: np.ndarray) -> Tuple[np.ndarray, List[str], float]:
        """Enhanced object detection with ensemble model approach"""
        # Convert to PIL for Faster R-CNN
        pil_image = Image.fromarray(image)
        
        # Run both models
        yolo_objects, yolo_scores, yolo_boxes = self.detect_objects_yolo(image)
        frcnn_objects, frcnn_scores, frcnn_boxes = self.detect_objects_faster_rcnn(pil_image)
        
        # Combine results
        all_objects = yolo_objects + frcnn_objects
        all_scores = yolo_scores + frcnn_scores.tolist()
        all_boxes = yolo_boxes + frcnn_boxes if frcnn_objects else yolo_boxes
        
        # Apply Non-Maximum Suppression to eliminate duplicates
        if all_boxes:
            nms_boxes, nms_scores = self.apply_nms(all_boxes, all_scores, self.iou_threshold)
            # Get the indices of the kept boxes
            nms_indices = []
            for nms_box in nms_boxes.tolist():
                for i, box in enumerate(all_boxes):
                    if np.allclose(nms_box, box):
                        nms_indices.append(i)
                        break
            # Get the corresponding objects for the kept boxes
            nms_objects = [all_objects[i] for i in nms_indices]
        else:
            nms_objects = all_objects
            nms_scores = all_scores
            nms_boxes = all_boxes
        
        # Draw annotations on the image
        annotated_image = image.copy()
        draw = ImageDraw.Draw(Image.fromarray(annotated_image))
        font = ImageFont.load_default()
        
        for i, (obj, score, box) in enumerate(zip(nms_objects, nms_scores, nms_boxes)):
            x, y, w, h = box
            # Draw rectangle
            draw.rectangle([x, y, x+w, y+h], outline="red", width=2)
            # Draw label
            label = f"{obj}: {score:.2f}"
            draw.text((x, y - 10), label, fill="red", font=font)
        
        annotated_image = np.array(Image.fromarray(annotated_image))
        
        # Calculate average confidence
        # Ensure nms_scores is a list or numpy array for the check
        if isinstance(nms_scores, torch.Tensor):
            nms_scores_np = nms_scores.cpu().numpy()
        else:
            nms_scores_np = np.array(nms_scores)
        avg_confidence = np.mean(nms_scores_np) if len(nms_scores_np) > 0 else 0.0
        
        return annotated_image, nms_objects, avg_confidence
    
    def analyze_sentiment_advanced(self, text: str) -> Tuple[str, str, List[str], float]:
        """Enhanced sentiment analysis with severity scoring"""
        if self.sentiment_pipeline is None:
            self.sentiment_pipeline = self.load_sentiment_pipeline()
        
        if self.sentiment_pipeline is None:
            return "Unknown", "Neutral", [], 0.0
        
        try:
            result = self.sentiment_pipeline(text)[0]
            label_map = {"LABEL_0": "Negative", "LABEL_1": "Neutral", "LABEL_2": "Positive"}
            sentiment_label = label_map.get(result["label"], "Neutral")
            confidence = result["score"]
            
            # Enhanced crime keyword detection
            crime_keywords = [
                # Weapons & violence
                "fire", "gun", "guns", "firearm", "firearms", "knife", "knives", "explosion", "explosives", "smoke", "weapon", "weapons", "blood", "fight", "fighting", "shooting", "shoot", "shootout", "burning", "torch", "assault", "attack", "attacked", "beating", "beaten", "stab", "stabbing", "shot", "shots", "armed", "unarmed",
                # Vehicles
                "vehicle", "vehicles", "truck", "trucks", "car", "cars", "motorcycle", "motorbike", "van", "bus", "ambulance", "police", "policeman", "policewoman", "cop", "cops",
                # Crimes
                "robbery", "robber", "robbers", "robbed", "rob", "theft", "thief", "thieves", "steal", "stolen", "burglary", "burglar", "burglars", "break-in", "breakin", "break in", "vandalism", "vandal", "vandals", "arson", "arsonist", "arsonists", "kidnap", "kidnapping", "kidnapped", "kidnapper", "kidnappers", "hostage", "hostages", "fraud", "scam", "scammer", "scammers", "drugs", "drug", "trafficking", "rape", "sexual assault", "homicide", "murder", "murderer", "murderers", "manslaughter", "emergency", "accident", "accidents", "violence", "violent", "terrorist", "terrorists", "terrorism", "gang", "gangs", "gangster", "gangsters", "crime", "crimes", "illegal", "dangerous", "danger", "threat", "threaten", "threatened", "hostility", "hostile", "extortion", "extort", "blackmail", "blackmailer", "blackmailers", "shootout", "shootouts", "hostage situation", "carjacking", "carjack", "carjacked", "looting", "looters", "looter", "riot", "riots", "rioter", "rioters", "smuggling", "smuggler", "smugglers", "pickpocket", "pickpockets", "pickpocketing", "shoplifting", "shoplifter", "shoplifters", "abduction", "abducted", "abductor", "abductors", "molestation", "molester", "molesters", "cybercrime", "cyber attack", "cyberattack", "scamming", "scammed", "bribery", "bribe", "bribed", "briber", "bribers"
            ]

            found_keywords = []
            severity_score = 0.0

            lowered_text = text.lower()
            for keyword in crime_keywords:
                if keyword in lowered_text:
                    found_keywords.append(keyword)
                    severity_score += SEVERITY_WEIGHTS.get(keyword.rstrip('s'), 0.3)
            
            # Normalize severity score
            severity_score = min(severity_score, 1.0)
            
            # Determine priority: always urgent if severity is high
            if severity_score > self.auto_escalate_threshold:
                priority = "Urgent ❗"
            elif sentiment_label == "Negative" and (found_keywords or severity_score > 0.5):
                priority = "Urgent ❗"
            elif found_keywords or sentiment_label == "Negative" or severity_score > 0.3:
                priority = "Normal ⚠️"
            else:
                priority = "Low Priority ✅"
            
            return priority, sentiment_label, found_keywords, severity_score
            
        except Exception as e:
            st.error(f"Sentiment analysis failed: {e}")
            return "Unknown", "Neutral", [], 0.0
    
    def compute_clip_similarity(self, image: np.ndarray, text: str) -> float:
        """Compute CLIP similarity between image and text"""
        if self.clip_model is None:
            self.clip_model, self.clip_preprocess, self.device = self.load_clip_model()
        
        if self.clip_model is None:
            return 0.0
        
        try:
            # Preprocess image
            image_pil = Image.fromarray(image)
            image_input = self.clip_preprocess(image_pil).unsqueeze(0).to(self.device)
            
            # Tokenize text
            text_input = clip.tokenize([text]).to(self.device)
            
            # Compute similarity
            with torch.no_grad():
                image_features = self.clip_model.encode_image(image_input)
                text_features = self.clip_model.encode_text(text_input)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Compute cosine similarity
                similarity = (image_features @ text_features.T).item()
                
            return max(0.0, min(1.0, similarity))  # Clamp between 0 and 1
            
        except Exception as e:
            st.error(f"CLIP similarity computation failed: {e}")
            return 0.0

# ------------------ ENHANCED VISUALIZATION ------------------
class VisualizationManager:
    @staticmethod
    def create_interactive_map(reports: List[Dict]) -> folium.Map:
        """Create interactive map with crime reports"""
        # Center map on Nairobi
        m = folium.Map(location=[-1.2921, 36.8219], zoom_start=7)
        
        # Color mapping for priorities
        color_map = {
            "Urgent ❗": "red",
            "Normal ⚠️": "orange", 
            "Low Priority ✅": "green"
        }
        
        for report in reports:
            if report['latitude'] and report['longitude']:
                # Create popup content
                popup_html = f"""
                <div style="width: 200px;">
                    <h4>{report['location']}</h4>
                    <p><strong>Priority:</strong> {report['priority']}</p>
                    <p><strong>Time:</strong> {report['timestamp']}</p>
                    <p><strong>Objects:</strong> {report['objects_detected']}</p>
                    <p><strong>Description:</strong> {report['description'][:100]}...</p>
                </div>
                """
                
                folium.Marker(
                    location=[report['latitude'], report['longitude']],
                    popup=folium.Popup(popup_html, max_width=250),
                    icon=folium.Icon(
                        color=color_map.get(report['priority'], 'blue'),
                        icon='exclamation-triangle'
                    )
                ).add_to(m)
        
        return m
    
    @staticmethod
    def create_priority_distribution_chart(reports: List[Dict]) -> go.Figure:
        """Create priority distribution pie chart"""
        priority_counts = {}
        for report in reports:
            priority = report['priority']
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        fig = go.Figure(data=[
            go.Pie(
                labels=list(priority_counts.keys()),
                values=list(priority_counts.values()),
                hole=.3,
                marker_colors=['#DC2626', '#D97706', '#059669']
            )
        ])
        
        fig.update_layout(
            title="Priority Distribution",
            font=dict(size=14),
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_timeline_chart(reports: List[Dict]) -> go.Figure:
        """Create timeline chart of reports"""
        df = pd.DataFrame(reports)
        if df.empty:
            return go.Figure()
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        daily_counts = df.groupby(['date', 'priority']).size().reset_index(name='count')
        
        fig = px.line(
            daily_counts, 
            x='date', 
            y='count',
            color='priority',
            title='Crime Reports Over Time',
            color_discrete_map={
                'Urgent ❗': '#DC2626',
                'Normal ⚠️': '#D97706',
                'Low Priority ✅': '#059669'
            }
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Reports",
            hovermode='x unified'
        )
        
        return fig

# ------------------ INITIALIZE MANAGERS ------------------
security_manager = SecurityManager()
db_manager = DatabaseManager()
ai_manager = AIModelManager()
viz_manager = VisualizationManager()

# ------------------ SESSION STATE ------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_role" not in st.session_state:
    st.session_state.user_role = None
if "username" not in st.session_state:
    st.session_state.username = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# ------------------ ENHANCED LOGIN SYSTEM ------------------
def handle_login():
    """Handle user login with enhanced security (admin only)"""
    with st.sidebar:
        st.header("🔐 Authentication")
        
        if not st.session_state.logged_in:
            # Only show login form, no registration
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit_login = st.form_submit_button("Login")
                
                if submit_login and username and password:
                    user_role = security_manager.authenticate(username, password)
                    if user_role == 'admin':
                        session_id = security_manager.create_session(username)
                        st.session_state.logged_in = True
                        st.session_state.user_role = user_role
                        st.session_state.username = username
                        st.session_state.session_id = session_id
                        st.success(f"✅ Welcome back, {username}!")
                        st.rerun()
                    else:
                        st.error("❌ Only the admin account is allowed to log in.")
                elif submit_login:
                    st.error("❌ Invalid credentials")
        else:
            st.success(f"👋 Welcome, {st.session_state.username}")
            st.write(f"**Role:** {st.session_state.user_role}")
            
            if st.button("🚪 Logout"):
                # Clear session
                st.session_state.logged_in = False
                st.session_state.user_role = None
                st.session_state.username = None
                st.session_state.session_id = None
                st.rerun()

# ------------------ ENHANCED ADMIN DASHBOARD ------------------
def render_admin_dashboard():
    """Render enhanced admin dashboard"""
    st.header("📊 Admin Dashboard")
    
    # Get reports from database
    reports = db_manager.get_reports(limit=1000)
    
    if not reports:
        st.info("No reports submitted yet.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Reports", len(reports))
    
    with col2:
        urgent_count = sum(1 for r in reports if r['priority'] == 'Urgent ❗')
        st.metric("Urgent Reports", urgent_count)
    
    with col3:
        flagged_count = sum(1 for r in reports if r['flagged_for_review'])
        st.metric("Flagged for Review", flagged_count)
    
    with col4:
        pending_count = sum(1 for r in reports if r['status'] == 'pending')
        st.metric("Pending Review", pending_count)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 Review Queue", "🗺️ Crime Map", "📈 Analytics"])
    
    with tab1:
        render_review_queue(reports)
    
    with tab2:
        render_crime_map(reports)
    
    with tab3:
        render_analytics_dashboard(reports)

def render_review_queue(reports: List[Dict]):
    """Render admin review queue"""
    st.subheader("📋 Reports Review Queue")
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        status_filter = st.selectbox(
            "Filter by Status",
            ["all", "pending", "approved", "rejected"],
            index=0
        )
    
    with col2:
        priority_filter = st.multiselect(
            "Filter by Priority",
            ["Urgent ❗", "Normal ⚠️", "Low Priority ✅"],
            default=[]
        )
    
    with col3:
        flagged_only = st.checkbox("Show only flagged reports")
    
    # Apply filters
    filtered_reports = reports
    
    if status_filter != "all":
        filtered_reports = [r for r in filtered_reports if r['status'] == status_filter]
    
    if priority_filter:
        filtered_reports = [r for r in filtered_reports if r['priority'] in priority_filter]
    
    if flagged_only:
        filtered_reports = [r for r in filtered_reports if r['flagged_for_review']]
    
    # Display reports
    if not filtered_reports:
        st.info("No reports match the selected filters.")
        return
    
    for report in filtered_reports[:20]:  # Show first 20 reports
        with st.expander(f"Report #{report['id']} - {report['timestamp']} - {report['priority']}", 
                         expanded=False):
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write(f"**Location:** {report['location']}")
                st.write(f"**Description:** {report['description']}")
                st.write(f"**Objects Detected:** {report['objects_detected']}")
                st.write(f"**Keywords:** {report['keywords']}")
                st.write(f"**CLIP Similarity:** {report['clip_similarity']:.2f}")
                st.write(f"**Severity Score:** {report['severity_score']:.2f}")
                
                if report['flagged_for_review']:
                    st.warning("⚠️ This report has been flagged for review")
            
            with col2:
                if report['annotated_path'] and os.path.exists(report['annotated_path']):
                    st.image(report['annotated_path'], caption="AI Detection", width=200)
                
                # Admin actions
                new_priority = st.selectbox(
                    "Set Priority",
                    ["Urgent ❗", "Normal ⚠️", "Low Priority ✅"],
                    index=["Urgent ❗", "Normal ⚠️", "Low Priority ✅"].index(report['priority']),
                    key=f"priority_{report['id']}"
                )
                
                col_approve, col_reject = st.columns(2)
                
                with col_approve:
                    if st.button("✅ Approve", key=f"approve_{report['id']}"):
                        db_manager.update_report_status(
                            report['id'], 
                            'approved', 
                            st.session_state.username,
                            new_priority
                        )
                        st.success("Report approved!")
                        st.rerun()
                
                with col_reject:
                    if st.button("❌ Reject", key=f"reject_{report['id']}"):
                        db_manager.update_report_status(
                            report['id'], 
                            'rejected', 
                            st.session_state.username
                        )
                        st.success("Report rejected!")
                        st.rerun()

def render_crime_map(reports: List[Dict]):
    """Render interactive crime map"""
    st.subheader("🗺️ Interactive Crime Map")
    
    # Filter for reports with valid coordinates
    map_reports = [r for r in reports if r['latitude'] and r['longitude'] and r['latitude'] != 0.0]
    
    if not map_reports:
        st.info("No reports with valid coordinates to display on map.")
        return
    
    # Create and display interactive map
    crime_map = viz_manager.create_interactive_map(map_reports)
    st_folium(crime_map, width=700, height=500)
    
    # Location statistics
    st.subheader("📍 Location Statistics")
    location_stats = {}
    for report in reports:
        loc = report['location']
        if loc not in location_stats:
            location_stats[loc] = {'total': 0, 'urgent': 0, 'normal': 0, 'low': 0}
        location_stats[loc]['total'] += 1
        if report['priority'] == 'Urgent ❗':
            location_stats[loc]['urgent'] += 1
        elif report['priority'] == 'Normal ⚠️':
            location_stats[loc]['normal'] += 1
        else:
            location_stats[loc]['low'] += 1
    
    # Display location stats as table
    stats_df = pd.DataFrame.from_dict(location_stats, orient='index')
    stats_df = stats_df.sort_values('total', ascending=False)
    st.dataframe(stats_df, use_container_width=True)

def render_analytics_dashboard(reports: List[Dict]):
    """Render analytics dashboard"""
    st.subheader("📈 Analytics Dashboard")
    
    if not reports:
        st.info("No data available for analytics.")
        return
    
    # Charts in columns
    col1, col2 = st.columns(2)
    
    with col1:
        # Priority distribution
        priority_chart = viz_manager.create_priority_distribution_chart(reports)
        st.plotly_chart(priority_chart, use_container_width=True)
    
    with col2:
        # Timeline chart
        timeline_chart = viz_manager.create_timeline_chart(reports)
        st.plotly_chart(timeline_chart, use_container_width=True)
    
    # Detailed metrics
    st.subheader("📊 Detailed Metrics")
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(reports)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Time-based analysis
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Severity Score", f"{df['severity_score'].mean():.2f}")
    
    with col2:
        st.metric("Average CLIP Similarity", f"{df['clip_similarity'].mean():.2f}")
    
    with col3:
        flagged_percentage = (df['flagged_for_review'].sum() / len(df)) * 100
        st.metric("Flagged Reports %", f"{flagged_percentage:.1f}%")
    
    # Most common objects detected
    st.subheader("🔍 Most Common Objects Detected")
    all_objects = []
    for report in reports:
        if report['objects_detected']:
            objects = report['objects_detected'].split(', ')
            all_objects.extend(objects)
    
    if all_objects:
        object_counts = pd.Series(all_objects).value_counts().head(10)
        st.bar_chart(object_counts)
    
    # Export functionality
    st.subheader("📁 Export Data")
    if st.button("📥 Download Full Report Data"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"crime_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ------------------ ENHANCED USER SUBMISSION ------------------
def render_user_submission():
    """Render enhanced user submission interface"""
    st.header("📝 Submit Crime Report")
    
    # Enhanced form with better validation
    with st.form("crime_report_form", clear_on_submit=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Image upload with preview
            uploaded_file = st.file_uploader(
                "📷 Upload Evidence Image",
                type=["jpg", "jpeg", "png"],
                help="Upload a clear image of the incident"
            )
            
            if uploaded_file:
                # Display image preview
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width=300)
                
                # Basic image validation
                if image.size[0] < 100 or image.size[1] < 100:
                    st.warning("⚠️ Image resolution is very low. Please upload a clearer image.")
        
        with col2:
            # Location selection with search
            location = st.selectbox(
                "📍 Incident Location",
                options=list(LOCATION_COORDS.keys()),
                help="Select the location where the incident occurred"
            )
        
        # Incident description with guidance
        st.write("### 📝 Incident Description")
        description = st.text_area(
            "Describe what happened",
            height=150,
            placeholder="Please describe the incident in detail:\n- What happened?\n- When did it occur?\n- Who was involved?\n- Any immediate dangers?",
            help="Provide as much detail as possible. This helps our AI better understand the situation."
        )
        
        # Submit button
        submitted = st.form_submit_button("🚀 Submit Report", use_container_width=True)
        
        if submitted:
            if uploaded_file and description:
                process_crime_report(
                    uploaded_file=uploaded_file,
                    location=location,
                    description=description
                )
            else:
                st.error("❌ Please upload an image and provide a description.")

def extract_crime_keywords_from_objects(objects_list):
    """Extract crime-related keywords from detected objects."""
    # Use the same keywords as in analyze_sentiment_advanced
    crime_keywords = [
        "fire", "gun", "guns", "firearm", "firearms", "knife", "knives", "explosion", "explosives", "smoke", "weapon", "weapons", "blood", "fight", "fighting", "shooting", "shoot", "shootout", "burning", "torch", "assault", "attack", "attacked", "beating", "beaten", "stab", "stabbing", "shot", "shots", "armed", "unarmed",
        "vehicle", "vehicles", "truck", "trucks", "car", "cars", "motorcycle", "motorbike", "van", "bus", "ambulance", "police", "policeman", "policewoman", "cop", "cops",
        "robbery", "robber", "robbers", "robbed", "rob", "theft", "thief", "thieves", "steal", "stolen", "burglary", "burglar", "burglars", "break-in", "breakin", "break in", "vandalism", "vandal", "vandals", "arson", "arsonist", "arsonists", "kidnap", "kidnapping", "kidnapped", "kidnapper", "kidnappers", "hostage", "hostages", "fraud", "scam", "scammer", "scammers", "drugs", "drug", "trafficking", "rape", "sexual assault", "homicide", "murder", "murderer", "murderers", "manslaughter", "emergency", "accident", "accidents", "violence", "violent", "terrorist", "terrorists", "terrorism", "gang", "gangs", "gangster", "gangsters", "crime", "crimes", "illegal", "dangerous", "danger", "threat", "threaten", "threatened", "hostility", "hostile", "extortion", "extort", "blackmail", "blackmailer", "blackmailers", "shootout", "shootouts", "hostage situation", "carjacking", "carjack", "carjacked", "looting", "looters", "looter", "riot", "riots", "rioter", "rioters", "smuggling", "smuggler", "smugglers", "pickpocket", "pickpockets", "pickpocketing", "shoplifting", "shoplifter", "shoplifters", "abduction", "abducted", "abductor", "abductors", "molestation", "molester", "molesters", "cybercrime", "cyber attack", "cyberattack", "scamming", "scammed", "bribery", "bribe", "bribed", "briber", "bribers"
    ]
    found = []
    for obj in objects_list:
        for kw in crime_keywords:
            if kw in obj.lower() and kw not in found:
                found.append(kw)
    return found

def process_crime_report(uploaded_file, location, description):
    """Process and analyze submitted crime report with improved logic for priority and sentiment."""
    with st.spinner("🔍 Analyzing report with AI..."):
        image = Image.open(uploaded_file)
        img_np = np.array(image)

        # AI Analysis
        annotated_img, detected_objects, detection_confidence = ai_manager.detect_objects_ensemble(img_np)
        desc_priority, sentiment_label, found_keywords, severity_score = ai_manager.analyze_sentiment_advanced(description)
        clip_similarity = ai_manager.compute_clip_similarity(img_np, description)
        detected_crime_keywords = extract_crime_keywords_from_objects(detected_objects)

        # Enhanced validation logic with optimized thresholds
        flagged_for_review = False
        validation_issues = []
        flag_count = 0

        # Check for mismatches between description and detected objects
        if found_keywords:
            detected_str = ", ".join(detected_objects).lower()
            keyword_matches = sum(1 for kw in found_keywords if kw in detected_str)
            if keyword_matches == 0 and len(found_keywords) > 0:
                validation_issues.append("Keywords in description don't match detected objects")
                flag_count += 1

        # Check CLIP similarity using optimized threshold
        if clip_similarity < ai_manager.clip_threshold:
            validation_issues.append(f"Low image-text similarity ({clip_similarity:.2f} < {ai_manager.clip_threshold})")
            flag_count += 1

        # Check detection confidence using optimized threshold
        if detection_confidence < ai_manager.detection_conf:
            validation_issues.append(f"Low object detection confidence ({detection_confidence:.2f} < {ai_manager.detection_conf})")
            flag_count += 1

        # --- NEW LOGIC: Combine image and description for priority ---
        # If both description and image have high-severity crime keywords, or image alone is highly suspicious
        high_severity_keywords = [kw for kw in found_keywords if SEVERITY_WEIGHTS.get(kw.rstrip('s'), 0) >= 0.7]
        high_severity_detected = [kw for kw in detected_crime_keywords if SEVERITY_WEIGHTS.get(kw.rstrip('s'), 0) >= 0.7]
        # Use the higher of the two severity scores
        combined_severity = max(severity_score, sum(SEVERITY_WEIGHTS.get(kw.rstrip('s'), 0.3) for kw in detected_crime_keywords))
        combined_severity = min(combined_severity, 1.0)

        # Priority logic
        if high_severity_keywords and high_severity_detected:
            priority = "Urgent ❗"
        elif high_severity_detected and not found_keywords:
            priority = "Normal ⚠️"  # suspicious image, neutral/benign text
        elif high_severity_keywords and not high_severity_detected:
            priority = "Normal ⚠️"  # suspicious text, benign image
        elif found_keywords or detected_crime_keywords or sentiment_label == "Negative" or combined_severity > 0.3:
            priority = "Normal ⚠️"
        else:
            priority = "Low Priority ✅"

        # If the image and description are both highly severe, or the image alone is highly severe, escalate
        if (high_severity_keywords and high_severity_detected) or (combined_severity > ai_manager.auto_escalate_threshold and high_severity_detected):
            priority = "Urgent ❗"
            flagged_for_review = False
        # If there is a mismatch and at least two issues, flag for review
        if flag_count >= 2:
            flagged_for_review = True

        # Save files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = f"uploads/report_{timestamp}.png"
        annotated_path = f"uploads/annotated_{timestamp}.png"
        os.makedirs("uploads", exist_ok=True)
        image.save(img_path)
        Image.fromarray(annotated_img).save(annotated_path)
        lat, lon = LOCATION_COORDS.get(location, (0.0, 0.0))
        report_data = {
            'location': location,
            'latitude': lat,
            'longitude': lon,
            'image_path': img_path,
            'annotated_path': annotated_path,
            'description': description,
            'objects_detected': ", ".join(detected_objects),
            'priority': priority,
            'sentiment': sentiment_label,
            'keywords': ", ".join(sorted(set(found_keywords + detected_crime_keywords))),
            'flagged_for_review': flagged_for_review,
            'clip_similarity': clip_similarity,
            'severity_score': combined_severity,
            'submitted_by': st.session_state.username if hasattr(st.session_state, 'username') and st.session_state.username else 'anonymous',
            'status': 'pending'
        }
        report_id = db_manager.insert_report(report_data)
    # Display results
    st.success("✅ Report submitted successfully!")
    st.info(f"📋 Report ID: {report_id}")
    col1, col2 = st.columns(2)
    with col1:
        st.image(annotated_img, caption="🔍 AI Analysis Results", use_container_width=True)
    with col2:
        st.write("### 📊 Analysis Results")
        priority_color = {"Urgent ❗": "red", "Normal ⚠️": "orange", "Low Priority ✅": "green"}
        st.markdown(f"**Priority:** <span style='color: {priority_color.get(priority, 'black')}'>{priority}</span>", unsafe_allow_html=True)
        st.write(f"**Sentiment:** {sentiment_label}")
        st.write(f"**Severity Score:** {combined_severity:.2f}/1.0")
        st.write(f"**Detection Confidence:** {detection_confidence:.2f}")
        st.write(f"**Image-Text Similarity:** {clip_similarity:.2f}")
        if detected_objects:
            st.write(f"**Objects Detected:** {', '.join(detected_objects)}")
        if found_keywords or detected_crime_keywords:
            st.write(f"**Keywords Found:** {', '.join(sorted(set(found_keywords + detected_crime_keywords)))}")
        if flagged_for_review:
            st.warning("⚠️ This report has been flagged for admin review due to:")
            for issue in validation_issues:
                st.write(f"• {issue}")
        if combined_severity > ai_manager.auto_escalate_threshold and high_severity_detected:
            st.error("🚨 HIGH SEVERITY INCIDENT - Authorities will be notified immediately!")

# ------------------ MAIN APPLICATION LOGIC ------------------
def main():
    """Main application logic"""
    # Sidebar: Admin access button
    with st.sidebar:
        st.header("System Access")
        if "admin_mode" not in st.session_state:
            st.session_state.admin_mode = False
        if not st.session_state.admin_mode:
            if st.button("🔑 Admin Login", help="Click to access admin features (login required)"):
                st.session_state.admin_mode = True
        else:
            st.info("Admin login mode enabled. Please log in below.")
    if st.session_state.admin_mode:
        # Handle authentication for admin
        handle_login()
        if not st.session_state.logged_in:
            st.info("👋 Please log in as admin to access the admin dashboard.")
            st.markdown("""
            ### 🚨 AI Crime Detection System
            
            This system uses advanced AI to analyze crime reports and prioritize emergency responses.
            
            **Features:**
            - 🔍 Automated object detection in images
            - 🧠 Sentiment analysis of incident descriptions
            - 🗺️ Interactive crime mapping
            - 📊 Real-time analytics dashboard
            - ⚡ Intelligent priority classification
            
            **Admin Credentials:**
            - Username: `admin`
            - Password: `CrimeAdmin2024!`
            """)
            return
        # Validate session
        if st.session_state.session_id:
            valid_user = security_manager.validate_session(st.session_state.session_id)
            if not valid_user:
                st.error("Session expired. Please log in again.")
                st.session_state.logged_in = False
                st.rerun()
        # Main application interface for admin
        if st.session_state.user_role == 'admin':
            render_admin_dashboard()
        else:
            st.error("You do not have admin privileges.")
    else:
        # General user interface (no login required)
        render_user_submission()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        🚨 AI Crime Detection System | Powered by Advanced Machine Learning
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()