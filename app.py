from flask import Flask, request, send_file, render_template, jsonify
from flask_socketio import SocketIO
import cv2
import numpy as np
from ultralytics import YOLO
import os
import logging
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'  # Required for SocketIO
socketio = SocketIO(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load YOLOv8 model
try:
    model = YOLO("best.pt")  # Path to your best.pt file
    logger.info(f"Model loaded with classes: {model.names}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Ensure upload and output directories exist
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Google Drive API setup
SCOPES = ['https://www.googleapis.com/auth/drive.file']
SERVICE_ACCOUNT_FILE = 'service_account.json'  # Path to your service account JSON
FOLDER_ID = '1NlEhNb2H7Orcl-KcKLVW2cA2bndpnowM'  # Replace with your Google Drive folder ID

def get_drive_service():
    try:
        credentials = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        service = build('drive', 'v3', credentials=credentials)
        logger.info("Google Drive service initialized")
        return service
    except Exception as e:
        logger.error(f"Failed to initialize Google Drive service: {str(e)}")
        raise

def upload_to_drive(file_path, file_name):
    try:
        service = get_drive_service()
        file_metadata = {
            'name': file_name,
            'parents': [FOLDER_ID]
        }
        media = MediaFileUpload(file_path, mimetype='video/mp4')
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        file_id = file.get('id')
        logger.info(f"File uploaded to Google Drive with ID: {file_id}")
        return file_id
    except Exception as e:
        logger.error(f"Failed to upload file to Google Drive: {str(e)}")
        raise

def format_timestamp(seconds):
    """Convert seconds to mm:ss format"""
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze_video():
    if "video" not in request.files:
        logger.error("No video file provided")
        return {"error": "No video file provided"}, 400

    video_file = request.files["video"]
    video_path = os.path.join(UPLOAD_FOLDER, video_file.filename)
    video_file.save(video_path)
    logger.info(f"Video saved to {video_path}")

    # Upload to Google Drive
    try:
        file_id = upload_to_drive(video_path, video_file.filename)
        socketio.emit('drive_upload', {'message': f'Video uploaded to Google Drive (ID: {file_id})'}, namespace='/')
    except Exception as e:
        logger.error(f"Google Drive upload failed: {str(e)}")
        return {"error": f"Google Drive upload failed: {str(e)}"}, 500

    # Process video and get violence timestamps
    output_path = os.path.join(OUTPUT_FOLDER, f"annotated_{video_file.filename}")
    try:
        violence_timestamps = process_video(video_path, output_path)
        logger.info(f"Annotated video saved to {output_path}")
        logger.info(f"Violence detected at timestamps: {violence_timestamps}")
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return {"error": f"Error processing video: {str(e)}"}, 500

    # Verify output file exists and is not empty
    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        logger.error("Output video file not created or empty")
        return {"error": "Output video file not created or empty"}, 500

    # Send timestamps via SocketIO
    socketio.emit('violence_timestamps', {'timestamps': violence_timestamps}, namespace='/')

    # Return the annotated video
    return send_file(output_path, mimetype="video/mp4")

def process_video(input_path, output_path):
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Processing video: {width}x{height}, {fps} fps, {total_frames} frames")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264 codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        raise Exception("Error initializing video writer")

    current_frame = 0
    violence_timestamps = []
    violence_detected_frames = set()  # Track frames with violence to avoid duplicates
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate current timestamp in seconds
        current_time_seconds = current_frame / fps

        # Perform YOLOv8 inference
        results = model(frame, conf=0.5)  # Adjust confidence threshold as needed

        frame_has_violence = False
        
        # Draw bounding boxes
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cls = int(box.cls[0])
                class_name = model.names[cls]
                label = f"{class_name} {confidence:.2f}"
                
                # Check if violence is detected
                if class_name.lower() == "violence":
                    frame_has_violence = True
                
                # Color: red for violence, green for normal
                color = (0, 0, 255) if class_name.lower() == "violence" else (0, 255, 0)
                
                # Draw rectangle and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                logger.debug(f"Detected: {label} at ({x1}, {y1}, {x2}, {y2})")

        # Record timestamp if violence is detected in this frame
        if frame_has_violence and current_frame not in violence_detected_frames:
            timestamp_formatted = format_timestamp(current_time_seconds)
            violence_timestamps.append(timestamp_formatted)
            violence_detected_frames.add(current_frame)
            logger.info(f"Violence detected at {timestamp_formatted} (frame {current_frame})")

        # Write the frame to output video
        out.write(frame)

        # Update progress
        current_frame += 1
        progress = (current_frame / total_frames) * 100
        socketio.emit('progress', {'percentage': progress}, namespace='/')
        logger.debug(f"Progress: {progress:.2f}% ({current_frame}/{total_frames} frames)")

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    logger.info("Video processing completed")
    
    # Remove duplicate timestamps and sort them
    violence_timestamps = sorted(list(set(violence_timestamps)))
    
    return violence_timestamps

if __name__ == "__main__":
    socketio.run(app, debug=True)
