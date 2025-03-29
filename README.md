# FeedFeel

This application combines YOLO object detection with a Next.js dashboard for visualization.

## System Overview

The application consists of two main components:

1. **YOLO Detection Stream**: A Python-based MJPEG streaming server that processes camera input using YOLOv8 for object detection and streams the processed video.
2. **Next.js Dashboard**: A web interface that displays the YOLO detection stream and provides additional controls.

## Quick Start

The easiest way to run the application is using the provided start script:

```bash
# Run with default settings (camera ID 1, port 5001)
./start_app.sh

# Run with a specific camera ID
./start_app.sh -c 0

# Run with a custom port
./start_app.sh -p 8080

# Run with both custom camera ID and port
./start_app.sh -c 0 -p 8080
```

This script will:
1. Check for required dependencies and install them if needed
2. Start the MJPEG streaming server from the logic folder
3. Launch the Next.js frontend
4. Provide URLs to access both services

## Camera IDs

Common camera IDs include:
- `0`: Usually the built-in webcam
- `1`: Often a virtual camera (like OBS Virtual Camera)
- `2+`: Additional cameras or virtual inputs

You can run `python logic/mjpeg_stream.py` without arguments to see a list of available cameras on your system.

## Manual Setup

If you prefer to start components individually:

### YOLO MJPEG Streaming Server

```bash
cd logic
python mjpeg_stream.py [camera_id] [port]
```

- `camera_id`: Optional camera ID (default: 1, which is typically a virtual camera)
- `port`: Optional port to run the server on (default: 5001)

The stream will be available at: http://localhost:[port]/video_feed

### Next.js Dashboard

```bash
cd dashboard
npm install  # Only needed for first-time setup
npm run dev
```

The dashboard will be available at: http://localhost:3000

## Configuration

- The MJPEG server uses camera ID 1 by default (typically a virtual camera)
- You can change the camera ID by passing it as an argument to the script or using the `-c` option
- The Next.js dashboard connects to the MJPEG server at http://localhost:5001/video_feed by default

## Requirements

- Python 3.x with required packages:
  - flask
  - flask-cors
  - opencv-python
  - torch
  - ultralytics
- Node.js and npm for the dashboard
- A webcam or virtual camera (like OBS Virtual Camera)

## Troubleshooting

If the stream doesn't appear:
1. Check that the MJPEG server is running (`http://localhost:5001/` should show a test page)
2. Verify that the correct camera ID is being used
3. Try a different camera ID with `./start_app.sh -c X` (where X is 0, 1, 2, etc.)
4. Check the terminal output for any error messages