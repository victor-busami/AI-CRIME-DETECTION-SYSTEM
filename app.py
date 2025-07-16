import streamlit as st
from PIL import Image
import pandas as pd
import os
from datetime import datetime
from ultralytics import YOLO
import numpy as np

# Set Streamlit page config
st.set_page_config(page_title="AI Crime Detector", layout="centered")

st.title("🚨 AI Crime Detection System")
st.subheader("Upload evidence and describe the incident.")

# Load YOLO model
@st.cache_resource
def load_model():
    model = YOLO("yolov8n.pt")  # Nano version - fast and small
    return model

# Run object detection on image
def detect_objects(image):
    model = load_model()
    results = model.predict(image, conf=0.4)
    annotated_frame = results[0].plot()
    classes = results[0].names
    detected = [classes[int(cls)] for cls in results[0].boxes.cls]
    return annotated_frame, detected

# Image and description input
uploaded_file = st.file_uploader("📷 Upload Image", type=["jpg", "jpeg", "png"])
description = st.text_area("📝 Incident Description", placeholder="Describe what happened...")

# Handle submission
if st.button("Submit Report"):
    if uploaded_file and description:
        # Load and process image
        img = Image.open(uploaded_file)
        img_np = np.array(img)

        # Run YOLOv8 detection
        annotated_img, detected_objects = detect_objects(img_np)

        # Create timestamp and paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = f"report_{timestamp}.png"
        annotated_path = f"annotated_{timestamp}.png"

        # Save original and annotated image
        Image.fromarray(img_np).save(img_path)
        Image.fromarray(annotated_img).save(annotated_path)

        # Save report data
        report_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "image_path": img_path,
            "annotated_path": annotated_path,
            "description": description,
            "objects_detected": ", ".join(detected_objects)
        }

        # Save to CSV
        if not os.path.exists("reports.csv"):
            df = pd.DataFrame([report_data])
        else:
            try:
                df = pd.read_csv("reports.csv")
                df = pd.concat([df, pd.DataFrame([report_data])], ignore_index=True)
            except pd.errors.EmptyDataError:
                df = pd.DataFrame([report_data])

        df.to_csv("reports.csv", index=False)

        # Show results
        st.image(annotated_img, caption="🔍 AI Detection Result", use_column_width=True)
        st.success(f"✅ Report submitted! Objects Detected: {', '.join(detected_objects) if detected_objects else 'None'}")
    else:
        st.warning("Please upload an image and enter a description.")

# View past reports
with st.expander("📁 View Previous Reports"):
    if os.path.exists("reports.csv"):
        try:
            df = pd.read_csv("reports.csv")
            if df.empty:
                st.info("⚠️ The reports file is empty. Submit a new report to begin.")
            else:
                for _, row in df.iterrows():
                    st.write(f"🕒 {row['timestamp']}")
                    if os.path.exists(row['annotated_path']):
                        st.image(row['annotated_path'], caption=f"Detected: {row['objects_detected']}", width=400)
                    st.write(f"📄 Description: {row['description']}")
                    st.markdown("---")
        except pd.errors.EmptyDataError:
            st.info("⚠️ The reports file is corrupted or unreadable. Try submitting a new report.")
    else:
        st.info("No reports submitted yet.")
