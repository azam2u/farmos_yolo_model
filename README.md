# FarmOS Plant Detector

This repository contains a custom trained YOLO11n object detection model for identifying 5 specific plant species in real-time camera feeds. The model was trained on a dataset of images cleanly segmented using a dual-pipeline of U-2-Net (rembg) and SAM3.

## Supported Classes:
- JapaneseFraxinus
- Kuroganemochi
- Shirakashi
- camellia
- ichigonoki

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/azam2u/farmos_yolo_model.git
   cd farmos_yolo_model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Real-Time Inference

To run the live webcam inference script:
```bash
python realtime_yolo.py
```

The script will attempt to open your default webcam (`/dev/video0`) and draw bounding boxes around any of the 5 plant species it detects in real-time. Press `q` while the video window is focused to exit.

## Model Details
- Architecture: YOLO11n
- Input Size: 224x224
- Training Epochs: 10
- Validation mAP50: 0.995
