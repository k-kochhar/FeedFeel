import cv2
import threading
import time
from flask import Flask, Response, render_template
from flask_cors import CORS
import os
import numpy as np
from ultralytics import YOLO
import torch
import sys

# Implement functions directly instead of importing from detect.py
# Check if MPS is available (Apple Silicon GPU acceleration)
def check_mps():
    if torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) is available! Using GPU acceleration.")
        return torch.device("mps")
    elif torch.backends.mps.is_built():
        print("MPS is built but not available. Check your MacOS version (needs 12.3+)")
        return torch.device("cpu")
    else:
        print("MPS not available. Using CPU instead.")
        return torch.device("cpu")

# Initialize webcam or virtual camera
def initialize_camera(camera_id=0):
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        print("Available cameras:")
        # List available cameras (may vary by system)
        for i in range(8):
            temp_cap = cv2.VideoCapture(i)
            if temp_cap.isOpened():
                print(f"  Camera ID {i} is available")
                temp_cap.release()
        return None
    return cap

class YoloStream:
    def __init__(self, camera_id=0, model_size="yolov8n.pt", conf_threshold=0.25):
        # Get the device
        self.device = check_mps()
        
        # Initialize YOLO model
        print(f"Loading YOLOv8 model: {model_size}")
        self.model = YOLO(model_size)
        
        # Move model to MPS device if available
        if self.device.type == "mps":
            self.model.to(self.device)
        
        # Initialize camera
        self.camera_id = camera_id
        self.cap = initialize_camera(camera_id)
        if self.cap is None:
            raise Exception(f"Failed to initialize camera {camera_id}")
            
        # Get camera properties
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {self.frame_width}x{self.frame_height}")
        
        # Set confidence threshold
        self.conf_threshold = conf_threshold
        
        # Variables for threading
        self.thread = None
        self.is_running = False
        self.current_frame = None
        self.lock = threading.Lock()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        # Start processing thread
        self.start()
    
    def start(self):
        if self.thread is None:
            self.is_running = True
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()
            print("YOLO detection thread started")
    
    def update(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to receive frame from camera")
                break
            
            # Rotate frame if needed (this may need adjusting based on your camera)
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            # Run YOLOv8 inference
            results = self.model(frame, conf=self.conf_threshold)
            
            # Draw detections and labels
            annotated_frame = results[0].plot()
            
            # Calculate and display FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 1:
                self.fps = self.frame_count / elapsed_time
                cv2.putText(annotated_frame, f"FPS: {self.fps:.2f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.frame_count = 0
                self.start_time = time.time()
            
            # Store the processed frame
            with self.lock:
                self.current_frame = annotated_frame.copy()
    
    def get_frame(self):
        with self.lock:
            if self.current_frame is not None:
                # Encode as JPEG
                ret, buffer = cv2.imencode('.jpg', self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                return buffer.tobytes()
            return None
    
    def stop(self):
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        print("YOLO detection thread stopped")

# Global stream object
stream = None

# Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create HTML template string as we don't have a templates folder
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>YOLO Detection Stream</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #0a192f;
            color: #e6f1ff;
            margin: 0;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        h1 {
            margin-bottom: 20px;
        }
        .stream-container {
            background-color: #172a45;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            max-width: 95%;
            width: auto;
            display: flex;
            flex-direction: column;
            align-items: center;
        }
        img {
            max-width: 100%;
            border-radius: 5px;
        }
        .stream-info {
            margin-top: 15px;
            font-size: 14px;
            color: #8892b0;
        }
    </style>
</head>
<body>
    <h1>YOLO Detection Stream</h1>
    <div class="stream-container">
        <img src="/video_feed" alt="YOLO Detection Stream">
        <div class="stream-info">
            Live YOLO object detection stream
        </div>
    </div>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML_TEMPLATE

def generate_frames():
    global stream
    while True:
        frame_data = stream.get_frame()
        if frame_data:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        else:
            # If no frame, send a blank image
            blank = np.zeros((480, 640, 3), np.uint8)
            ret, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        # Control the MJPEG stream rate
        time.sleep(0.03)  # ~30fps

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def start_server(camera_id=1, port=5001):
    global stream
    
    try:
        # Initialize YoloStream
        stream = YoloStream(camera_id=camera_id)
        
        # Start Flask server
        print(f"Starting MJPEG server on port {port}")
        app.run(host='0.0.0.0', port=port, threaded=True)
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if stream:
            stream.stop()

if __name__ == "__main__":
    # Parse command line arguments
    camera_id = 0  # Default to camera ID 1 (usually virtual camera)
    port = 5001    # Default port
    
    # Allow command line overrides
    if len(sys.argv) > 1:
        try:
            camera_id = int(sys.argv[1])
        except ValueError:
            print(f"Invalid camera ID: {sys.argv[1]}. Using default: 1")
    
    if len(sys.argv) > 2:
        try:
            port = int(sys.argv[2])
        except ValueError:
            print(f"Invalid port: {sys.argv[2]}. Using default: 5001")
    
    # Start MJPEG server
    print(f"Starting MJPEG server with camera ID {camera_id} on port {port}")
    start_server(camera_id=camera_id, port=port) 