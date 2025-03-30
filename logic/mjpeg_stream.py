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
import json
from flask_socketio import SocketIO
from dotenv import load_dotenv
import random

# Import functions from get_vib.py for embedding visualization
from get_vib import get_sentence_embedding, average_pool_embedding, sonify_embeddings

# Load environment variables from .env file
load_dotenv()

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

# Class for generating and processing embeddings for visualization
class EmbeddingVisualizer:
    def __init__(self):
        # Sample objects that might be detected
        self.sample_objects = [
            "person", "car", "bicycle", "chair", "bottle", 
            "dog", "cat", "laptop", "cup", "book",
            "cell phone", "backpack", "umbrella", "handbag"
        ]
        
        # Store last processed embedding data
        self.current_visualization = {
            "object": "",
            "raw_embedding": [],
            "pooled_embedding": [],
            "audio_signal": [],
            "stepper_pattern": [],
            "processing_time": 0
        }
    
    def generate_visualization_data(self, object_name=None):
        """Generate visualization data for a given object or random sample"""
        # If no object is provided, choose a random one
        if not object_name:
            object_name = random.choice(self.sample_objects)
        
        start_time = time.time()
        
        try:
            # Get the embedding
            embedding = get_sentence_embedding(object_name)
            
            # Sample just a portion of the embedding for visualization (it's very large)
            embedding_sample = embedding[:100]
            
            # Apply average pooling
            pooled_embedding = average_pool_embedding(embedding, pool_size=6)
            
            # Sample pooled embedding for visualization
            pooled_sample = pooled_embedding[:50]
            
            # Generate audio signal from the pooled embedding
            audio_signal = sonify_embeddings(pooled_embedding, sample_rate=44100, duration=3.0)
            
            # Sample a portion of the audio signal
            audio_sample = audio_signal[:400].tolist()
            
            # Convert audio signal to stepper motor pattern (same logic as in get_vib.py)
            # Just sample 20hz pattern for visualization
            duration = len(audio_signal) / 44100
            samples_20hz = int(duration * 20)
            indices = np.linspace(0, len(audio_signal) - 1, samples_20hz, dtype=int)
            
            # Convert audio amplitude [-1,1] to stepper speed [-250, 250]
            min_speed, max_speed = -250, 250
            pattern_20hz = audio_signal[indices]
            pattern_20hz = min_speed + (pattern_20hz + 1) * (max_speed - min_speed) / 2
            stepper_pattern = np.round(pattern_20hz).astype(int).tolist()
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Create visualization data
            self.current_visualization = {
                "object": object_name,
                "raw_embedding": embedding_sample,
                "pooled_embedding": pooled_sample.tolist(),
                "audio_signal": audio_sample,
                "stepper_pattern": stepper_pattern,
                "processing_time": round(processing_time, 2)
            }
            
            return self.current_visualization
        
        except Exception as e:
            print(f"Error generating visualization data: {str(e)}")
            return {
                "object": object_name,
                "error": str(e)
            }

class YoloStream:
    def __init__(self, camera_id=0, model_size="yolov8n.pt", conf_threshold=0.25, socketio=None):
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
        
        # Store SocketIO instance for emitting events
        self.socketio = socketio
        
        # Variables for threading
        self.thread = None
        self.is_running = False
        self.current_frame = None
        self.lock = threading.Lock()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        # Detection statistics
        self.detection_stats = {
            "total_detections": 0,
            "class_counts": {},
            "recent_detections": [],
            "fps": 0
        }
        
        # Embedding visualization
        self.embedding_visualizer = EmbeddingVisualizer()
        self.last_viz_update = time.time()
        self.viz_interval = 10.0  # Update embedding visualization every 10 seconds
        
        # Class color mapping (for consistent colors)
        self.class_colors = {}
        
        # Start processing thread
        self.start()
    
    def start(self):
        if self.thread is None:
            self.is_running = True
            self.thread = threading.Thread(target=self.update, daemon=True)
            self.thread.start()
            print("YOLO detection thread started")
    
    def get_class_color(self, class_name):
        """Generate a consistent color for each class name"""
        if class_name not in self.class_colors:
            # Generate a color based on the hash of the class name
            hash_val = hash(class_name) % 0xFFFFFF
            # Convert to hex and ensure it's bright enough
            color = f"#{hash_val:06x}"
            self.class_colors[class_name] = color
        return self.class_colors[class_name]
    
    def update(self):
        last_emit_time = time.time()
        emit_interval = 0.5  # Send data to frontend every 0.5 seconds
        
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to receive frame from camera")
                break
            
            # Rotate frame if needed (this may need adjusting based on your camera)
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            
            # Run YOLOv8 inference
            results = self.model(frame, conf=self.conf_threshold)
            
            # Process detection results for stats
            current_detections = []
            latest_object_name = None  # Track the most recent detected object
            
            if len(results[0].boxes) > 0:
                for i in range(len(results[0].boxes)):
                    box = results[0].boxes[i]
                    class_id = int(box.cls.item())
                    class_name = self.model.names[class_id]
                    confidence = float(box.conf.item())
                    
                    # Keep track of the last detected object with good confidence
                    if confidence > 0.6:
                        latest_object_name = class_name
                    
                    # Get bounding box
                    x1, y1, x2, y2 = [float(x) for x in box.xyxy[0]]
                    
                    # Calculate object size (normalized by frame dimensions)
                    width = (x2 - x1) / self.frame_width
                    height = (y2 - y1) / self.frame_height
                    size = width * height
                    
                    # Calculate position (center point)
                    center_x = (x1 + x2) / 2 / self.frame_width
                    center_y = (y1 + y2) / 2 / self.frame_height
                    
                    # Add to detection stats
                    self.detection_stats["total_detections"] += 1
                    
                    if class_name in self.detection_stats["class_counts"]:
                        self.detection_stats["class_counts"][class_name] += 1
                    else:
                        self.detection_stats["class_counts"][class_name] = 1
                    
                    # Add to current detections
                    detection = {
                        "class_name": class_name,
                        "confidence": confidence,
                        "position": {
                            "x": center_x,
                            "y": center_y
                        },
                        "size": size,
                        "timestamp": time.time(),
                        "color": self.get_class_color(class_name)
                    }
                    current_detections.append(detection)
            
            # Update recent detections list (keep last 20)
            self.detection_stats["recent_detections"] = current_detections + self.detection_stats["recent_detections"]
            if len(self.detection_stats["recent_detections"]) > 20:
                self.detection_stats["recent_detections"] = self.detection_stats["recent_detections"][:20]
            
            # Draw detections and labels
            annotated_frame = results[0].plot()
            
            # Calculate and display FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 1:
                self.fps = self.frame_count / elapsed_time
                self.detection_stats["fps"] = round(self.fps, 2)
                cv2.putText(annotated_frame, f"FPS: {self.fps:.2f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.frame_count = 0
                self.start_time = time.time()
            
            # Store the processed frame
            with self.lock:
                self.current_frame = annotated_frame.copy()
            
            # Emit detection stats via WebSocket
            current_time = time.time()
            if self.socketio and current_time - last_emit_time >= emit_interval:
                self.socketio.emit('detection_stats', self.detection_stats)
                last_emit_time = current_time
            
            # Update embedding visualization periodically
            # When the interval is reached or if a new object is detected with good confidence
            if current_time - self.last_viz_update >= self.viz_interval or latest_object_name:
                object_to_process = latest_object_name if latest_object_name else None
                viz_data = self.embedding_visualizer.generate_visualization_data(object_to_process)
                if self.socketio:
                    self.socketio.emit('embedding_visualization', viz_data)
                self.last_viz_update = current_time
    
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
socketio = SocketIO(app, cors_allowed_origins="*")  # Initialize SocketIO with CORS

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

@app.route('/stats')
def stats():
    """Return current detection stats as JSON"""
    global stream
    if stream:
        return json.dumps(stream.detection_stats)
    return json.dumps({"error": "Stream not initialized"})

@socketio.on('connect')
def handle_connect():
    print('Client connected to WebSocket')
    # Send initial embedding visualization on connect
    global stream
    if stream:
        viz_data = stream.embedding_visualizer.generate_visualization_data()
        socketio.emit('embedding_visualization', viz_data)

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected from WebSocket')

@socketio.on('request_embedding_viz')
def handle_request_embedding_viz(data):
    """Handle frontend requests for embedding visualization"""
    global stream
    if stream:
        object_name = data.get('object') if data and 'object' in data else None
        viz_data = stream.embedding_visualizer.generate_visualization_data(object_name)
        socketio.emit('embedding_visualization', viz_data)

def start_server(camera_id=1, port=5001):
    global stream
    
    try:
        # Initialize YoloStream with SocketIO
        stream = YoloStream(camera_id=camera_id, socketio=socketio)
        
        # Start SocketIO server (which will also run the Flask app)
        print(f"Starting MJPEG server with SocketIO on port {port}")
        socketio.run(app, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("Server stopped by user")
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if stream:
            stream.stop()

if __name__ == "__main__":
    # Parse command line arguments
    camera_id = 0  # Default to camera ID 0
    port = 5001    # Default port
    
    # Allow command line overrides
    if len(sys.argv) > 1:
        try:
            camera_id = int(sys.argv[1])
        except ValueError:
            print(f"Invalid camera ID: {sys.argv[1]}. Using default: 0")
    
    if len(sys.argv) > 2:
        try:
            port = int(sys.argv[2])
        except ValueError:
            print(f"Invalid port: {sys.argv[2]}. Using default: 5001")
    
    # Start MJPEG server
    print(f"Starting MJPEG server with camera ID {camera_id} on port {port}")
    start_server(camera_id=camera_id, port=port) 