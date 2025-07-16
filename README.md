# 🚨 AI Crime Detection System

A Streamlit-based web application that leverages AI for real-time crime detection and reporting. The system uses YOLOv8 for object detection in images and a transformer-based model for sentiment analysis of crime descriptions. Admins can view, filter, and analyze reports on an interactive dashboard.

## Features

- **User Crime Reporting:**
  - Upload images and describe incidents.
  - AI detects objects in images and analyzes the sentiment of the description.
  - Reports are geotagged by location and stored for review.

- **Admin Dashboard:**
  - Secure login for admins.
  - View all submitted reports with images, detected objects, sentiment, and priority.
  - Filter reports by priority, sentiment, or detected objects.
  - Visualize reports on a map and with analytics charts (trends, object frequency, sentiment distribution, location counts).
  - Export all reports as CSV.

## Tech Stack

- **Frontend & Backend:** [Streamlit](https://streamlit.io/)
- **Object Detection:** [YOLOv8](https://github.com/ultralytics/ultralytics)
- **Sentiment Analysis:** [Hugging Face Transformers](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
- **Data Storage:** CSV files
- **Image Processing:** [Pillow (PIL)](https://python-pillow.org/)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd crime-detector
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
Create a `requirements.txt` file with the following content (if not already present):

```
streamlit
pillow
pandas
ultralytics
numpy
transformers
```

Then install:
```bash
pip install -r requirements.txt
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

- **Admin:**
  - Use the sidebar to log in with the default credentials:
    - Username: `admin`
    - Password: `crime123`
  - Access the dashboard to view, filter, and analyze reports.

## File Structure

- `app.py` — Main Streamlit application.
- `yolov8n.pt` — YOLOv8 model weights.
- `reports.csv` — Generated after the first report; stores all report data.
- `venv/` — Python virtual environment (not included in version control).

## Troubleshooting

- **Missing Columns/Error Reading Reports:**
  - If you see errors about missing columns in `reports.csv`, delete the file and submit a new report to regenerate it with the correct structure.

- **Login Sidebar Not Updating:**
  - The sidebar now updates immediately after login/logout. If you experience issues, ensure you are using a recent version of Streamlit.

- **YOLOv8 or Transformers Not Found:**
  - Ensure all dependencies are installed in your virtual environment.

## Customization

- **Change Admin Credentials:**
  - Edit the `ADMIN_USERNAME` and `ADMIN_PASSWORD` variables in `app.py`.

- **Add More Locations:**
  - Update the `LOCATION_COORDS` dictionary in `app.py`.

## License

This project is for educational and demonstration purposes. Please check the licenses of YOLOv8 and Hugging Face models for production/commercial use.

---

Feel free to open issues or contribute improvements! 