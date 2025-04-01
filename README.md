# SixthSense - [First Place at HackPrinceton](https://devpost.com/software/sixthsense-xuw41r)
<p align="center">
<img width="720" alt="Landing" src="https://github.com/user-attachments/assets/afbbfd91-ff50-4660-b3c5-5fa49827e41c" />
</p>

## Overview

Over 253 million people worldwide live with visual impairments. SixthSense redefines assistive technology by introducing a groundbreaking dual-feedback system that allows users to "feel" their environment through touch without overloading their hearing.

By combining **Meta glasses**, **YOLOv8 object detection**, and **custom haptic feedback** through a wearable glove and gauntlet, SixthSense lets users perceive both **what** an object is and **where** it is, providing a tactile language that becomes intuitive over time.

<p align="center">
  <img width="400" alt="Screenshot 2025-03-29 at 9 10 00 PM" src="https://github.com/user-attachments/assets/b616b54d-4ada-4f9f-b986-8cb1945075f0" />
  <img width="400" alt="PHOTO-2025-03-30-08-49-17 4" src="https://github.com/user-attachments/assets/f87148e4-debf-474e-bb8a-57618bdace7e" />
</p>

<p align="center">
  <img width="400" alt="PHOTO-2025-03-30-08-49-17 2" src="https://github.com/user-attachments/assets/a3a6b518-fc53-4fde-a6eb-bde62786e38d" />
  <img width="400" alt="PHOTO-2025-03-30-08-49-17 3" src="https://github.com/user-attachments/assets/5e29a134-ca37-4874-8cb0-59da1410fd47" />
</p>

---

## What It Does

- **Visual Recognition**: Captures real-world scenes using Meta glasses
- **Object Detection**: Identifies objects using YOLOv8
- **Tactile Signatures**: Converts object embeddings into unique vibration patterns using inverse Fourier transforms
- **Directional Guidance**: Uses servo motors to physically guide users' hands toward object locations
- **Simultaneous Sensory Composition**: Merges multiple objects’ tactile signals into a harmonic, non-overwhelming experience

---

## Key Innovations

- **Brain-Inspired Architecture**: Separates object identity ("what") from location ("where"), mirroring human visual processing
- **Embedding-to-Vibration Mapping**: Translates semantic object vectors into unique, recognizable tactile patterns
- **Harmonic Touch Language**: Similar objects produce related sensations, allowing users to intuitively understand complex scenes
- **Minimal Auditory Interference**: Preserves hearing by relying solely on touch for feedback

---

## System Components

1. **YOLO Detection Stream**  
   Python-based MJPEG streaming server with WebSocket support to detect and serve object data in real-time.

2. **Next.js Dashboard**  
   Web interface for developers and testers to visualize detections and track statistics.

3. **Haptic Feedback System**  
   Arduino-controlled glove and gauntlet delivering dual feedback: vibration patterns (identity) and servo-based guidance (location).

---

## Built With

- `Python`, `YOLOv8`, `OpenCV`, `Arduino`, `Next.js`, `Flask`, `WebSockets`, `Fourier Transforms`

---

## Check us out

- Devpost Submission: [https://devpost.com/software/sixthsense-xuw41r]
- Demo Video: [https://www.youtube.com/watch?v=vNlNHXVv8kw]
