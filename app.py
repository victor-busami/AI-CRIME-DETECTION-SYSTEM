import streamlit as st
from PIL import Image
import pandas as pd
import os
from datetime import datetime
from ultralytics import YOLO
import numpy as np
from transformers import pipeline

# ------------------ APP CONFIG ------------------
st.set_page_config(page_title="AI Crime Detector", layout="centered")
st.title("🚨 AI Crime Detection System")

# ------------------ CREDENTIALS ------------------
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "crime123"

# ------------------ SESSION STATE ------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "admin_page" not in st.session_state:
    st.session_state.admin_page = False

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# ------------------ FUNCTIONALITY ------------------
def detect_objects(image):
    model = load_model()
    results = model.predict(image, conf=0.4)
    annotated_frame = results[0].plot()
    classes = results[0].names
    detected = [classes[int(cls)] for cls in results[0].boxes.cls]
    return annotated_frame, detected

def analyze_sentiment(text):
    sentiment_pipeline = load_sentiment_pipeline()
    result = sentiment_pipeline(text)[0]
    label = result["label"].upper()
    if "NEGATIVE" in label:
        return "Urgent ❗", label
    elif "NEUTRAL" in label:
        return "Normal ⚠️", label
    else:
        return "Low Priority ✅", label

# ------------------ LOGIN PANEL ------------------
with st.sidebar:
    st.header("🔐 Admin Login")
    if not st.session_state.logged_in:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
                st.success("✅ Logged in successfully.")
                st.session_state.logged_in = True
                st.session_state.admin_page = True
            else:
                st.error("❌ Incorrect credentials.")
    else:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.admin_page = False
            st.experimental_rerun()

# ------------------ ADMIN DASHBOARD ------------------
if st.session_state.admin_page:
    st.subheader("📊 Admin Dashboard")

    if os.path.exists("reports.csv"):
        try:
            df = pd.read_csv("reports.csv")
            if df.empty:
                st.info("No reports yet.")
            else:
                # Filters
                with st.expander("🔍 Filter Reports"):
                    priority_filter = st.multiselect("Filter by Priority", options=df["priority"].unique())
                    sentiment_filter = st.multiselect("Filter by Sentiment", options=df["sentiment"].unique())
                    object_filter = st.text_input("Search Object Detected")

                    filtered_df = df.copy()
                    if priority_filter:
                        filtered_df = filtered_df[filtered_df["priority"].isin(priority_filter)]
                    if sentiment_filter:
                        filtered_df = filtered_df[filtered_df["sentiment"].isin(sentiment_filter)]
                    if object_filter:
                        filtered_df = filtered_df[filtered_df["objects_detected"].str.contains(object_filter, case=False, na=False)]

                # Show filtered results
                for _, row in filtered_df.iterrows():
                    st.write(f"🕒 {row['timestamp']}")
                    if os.path.exists(row['annotated_path']):
                        st.image(row['annotated_path'], caption=f"Detected: {row['objects_detected']}", width=400)
                    st.write(f"📄 Description: {row['description']}")
                    st.write(f"🧠 Sentiment: {row['sentiment']} | 🚦 Priority: {row['priority']}")
                    st.markdown("---")
        except Exception as e:
            st.error(f"Error reading reports: {e}")
    else:
        st.info("No reports submitted yet.")

# ------------------ USER REPORT SUBMISSION ------------------
elif not st.session_state.admin_page:
    st.subheader("📝 Submit a Crime Report")

    uploaded_file = st.file_uploader("📷 Upload Image", type=["jpg", "jpeg", "png"])
    description = st.text_area("Describe the Incident")

    if st.button("Submit Report"):
        if uploaded_file and description:
            img = Image.open(uploaded_file)
            img_np = np.array(img)
            annotated_img, detected_objects = detect_objects(img_np)
            priority, sentiment_label = analyze_sentiment(description)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = f"report_{timestamp}.png"
            annotated_path = f"annotated_{timestamp}.png"
            Image.fromarray(img_np).save(img_path)
            Image.fromarray(annotated_img).save(annotated_path)

            report_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "image_path": img_path,
                "annotated_path": annotated_path,
                "description": description,
                "objects_detected": ", ".join(detected_objects),
                "priority": priority,
                "sentiment": sentiment_label
            }

            if not os.path.exists("reports.csv"):
                df = pd.DataFrame([report_data])
            else:
                try:
                    df = pd.read_csv("reports.csv")
                    df = pd.concat([df, pd.DataFrame([report_data])], ignore_index=True)
                except pd.errors.EmptyDataError:
                    df = pd.DataFrame([report_data])

            df.to_csv("reports.csv", index=False)
            st.image(annotated_img, caption="🔍 AI Detection Result", use_column_width=True)
            st.success(f"✅ Report submitted! Objects Detected: {', '.join(detected_objects) or 'None'}")
            st.info(f"🧠 Sentiment: {sentiment_label} | 🚦 Priority: {priority}")
        else:
            st.warning("Please upload an image and enter a description.")

# ------------------ END ------------------
