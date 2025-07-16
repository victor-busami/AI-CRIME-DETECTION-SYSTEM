# 🚨 AI Crime Detection System

A Streamlit-based web application that leverages AI for real-time crime detection and reporting. The system uses an ensemble of YOLOv8 and Faster R-CNN for object detection in images, CLIP for image-text similarity, and a transformer-based model for sentiment analysis of crime descriptions. Admins can view, filter, and analyze reports on an interactive dashboard.

## Features

- **User Crime Reporting:**
  - Upload images and describe incidents.
  - AI detects objects in images (YOLOv8 + Faster R-CNN ensemble) and analyzes the sentiment of the description.
  - CLIP is used to check image-text similarity.
  - Reports are geotagged by location and stored in a SQLite database for review.

- **Admin Dashboard:**
  - Secure login for admins only (no user registration).
  - View all submitted reports with images, detected objects, sentiment, and priority.
  - Filter reports by priority, sentiment, or detected objects.
  - Visualize reports on a map and with analytics charts (trends, object frequency, sentiment distribution, location counts).
  - Export all reports as CSV.

## Tech Stack

- **Frontend & Backend:** [Streamlit](https://streamlit.io/)
- **Object Detection:** [YOLOv8](https://github.com/ultralytics/ultralytics) + [Faster R-CNN](https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html)
- **Sentiment Analysis:** [Hugging Face Transformers](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
- **Image-Text Similarity:** [CLIP](https://github.com/openai/CLIP)
- **Data Storage:** SQLite databases (`crime_reports.db`, `users.db`)
- **Image Processing:** [Pillow (PIL)](https://python-pillow.org/)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd AI-CRIME-DETECTION-SYSTEM
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
Ensure you have a `requirements.txt` file (provided) and run:
```bash
pip install -r requirements.txt
```

#### Example requirements.txt
```
torch
transformers
ftfy
tqdm
Pillow
# For CLIP
git+https://github.com/openai/CLIP.git

# Additional requirements for your app
streamlit
pandas
ultralytics
numpy
plotly
folium
streamlit-folium
torchvision
```

### 4. Download YOLOv8 Weights
Place the `yolov8n.pt` file in the project root. You can download it from the [Ultralytics YOLOv8 release page](https://github.com/ultralytics/ultralytics/releases).

### 5. Run the Application
```bash
streamlit run app.py
```

The app will be available at [http://localhost:8501](http://localhost:8501).

## Usage

- **User:**
  - Go to the app homepage.
  - Upload an image, select a location, and describe the incident.
  - Submit the report. The AI will analyze the image and description, then display the results.
  - Reports are only flagged for admin review if at least two of the following are true: (1) keywords in the description do not match detected objects, (2) image-text similarity is low, (3) object detection confidence is low. High-severity incidents are never flagged for review.

- **Admin:**
  - Use the sidebar to log in with the default credentials:
    - Username: `admin`
    - Password: `CrimeAdmin2024!`
  - Access the dashboard to view, filter, and analyze reports.

## File Structure

- `app.py` — Main Streamlit application.
- `yolov8n.pt` — YOLOv8 model weights.
- `crime_reports.db` — SQLite database for reports (auto-created).
- `users.db` — SQLite database for user authentication (auto-created).
- `uploads/` — Uploaded and annotated images.
- `requirements.txt` — Python dependencies.
- `venv/` — Python virtual environment (not included in version control).

## Troubleshooting

- **Model Download Issues:**
  - The first run may take time as models (YOLOv8, Faster R-CNN, CLIP, transformers) are downloaded automatically. Ensure you have a stable internet connection.

- **Database Files Not Found:**
  - The app will auto-create `crime_reports.db` and `users.db` if they do not exist.

- **Login Sidebar Not Updating:**
  - The sidebar updates immediately after login/logout. If you experience issues, ensure you are using a recent version of Streamlit.

- **YOLOv8 or Transformers Not Found:**
  - Ensure all dependencies are installed in your virtual environment.

## Customization

- **Change Admin Credentials:**
  - The default admin credentials are set in the database on first run. To change, edit the logic in `SecurityManager.create_default_admin()` in `app.py` and delete `users.db` to reset.

- **Add More Locations:**
  - Update the `LOCATION_COORDS` dictionary in `app.py`.

## License

This project is for educational and demonstration purposes. Please check the licenses of YOLOv8, CLIP, and Hugging Face models for production/commercial use.

---

Feel free to open issues or contribute improvements! 