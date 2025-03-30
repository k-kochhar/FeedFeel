# SixthSense!

## System Overview

The application consists of two main components:

1. **YOLO Detection Stream**: A Python-based MJPEG streaming server with real-time WebSocket data that processes camera input using YOLOv8 for object detection.
2. **Next.js Dashboard**: A web interface that displays the YOLO detection stream and provides real-time visualization of detection data.

## Features

- **Live Object Detection**: Processes camera feed with YOLOv8 for real-time object detection
- **MJPEG Streaming**: Sends the processed video as an MJPEG stream to the frontend
- **Real-time Stats**: Uses WebSockets to provide live detection statistics to the dashboard
- **Visualization Dashboard**: Displays detection metrics, object distribution, and recent detections

## Manual Setup

### YOLO MJPEG Streaming Server

```bash
cd logic
pip install -r requirements.txt  # Only needed for first-time setup
python mjpeg_stream.py [camera_id] [port]
```

- `camera_id`: Optional camera ID (default: 0, which is typically the built-in camera)
- `port`: Optional port to run the server on (default: 5001)

The stream will be available at: http://localhost:5001/video_feed

### Next.js Dashboard

```bash
cd dashboard
npm install  # Only needed for first-time setup
npm run dev
```

The dashboard will be available at: http://localhost:3000

## Camera IDs

Common camera IDs include:
- `0`: Usually the built-in webcam
- `1`: Often a virtual camera (like OBS Virtual Camera)
- `2+`: Additional cameras or virtual inputs

You can run `python logic/mjpeg_stream.py` without arguments to see a list of available cameras on your system.

## Requirements

- Python 3.x with required packages:
  - flask
  - flask-socketio
  - flask-cors
  - opencv-python
  - torch
  - ultralytics
- Node.js and npm for the dashboard
- A webcam or virtual camera (like OBS Virtual Camera)

## Dashboard Visualization

The dashboard provides real-time visualization of detection data:

1. **Detection Stats Card**: Shows total detections, FPS, and connection status
2. **Object Classes Card**: Displays distribution of detected object classes with progress bars
3. **Recent Detections Card**: Lists recently detected objects with confidence, position, and timing information

## Troubleshooting

If the stream doesn't appear:
1. Check that the MJPEG server is running (`http://localhost:5001/` should show a test page)
2. Verify that the correct camera ID is being used
3. Try a different camera ID with `python logic/mjpeg_stream.py X` (where X is 0, 1, 2, etc.)
4. Check the terminal output for any error messages

If the WebSocket connection fails:
1. Make sure the server is running with the correct port
2. Check browser console for any connection errors
3. Restart both the server and the frontend