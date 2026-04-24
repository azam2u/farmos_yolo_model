# FarmOS Plant Classifier

This repository contains a YOLOv8 Nano classification model trained to recognize 14 different plant species. The pipeline includes scripts for segmenting raw data, balancing the dataset, training the model, and running live inference.

## Features
- **14-Class Plant Recognition**: Accurately classifies species such as `JapaneseFraxinus`, `sakura`, `camellia_yabutsubaki`, and `Kuroganemochi`.
- **Live Webcam Inference**: Built-in support for real-time predictions via webcam.
- **Data Pipeline**: Automated tools to extract instances using SAM3 PCS, pool them by hierarchy (Whole Plant > Buds > Leaf), and perfectly balance classes.

## The Model
The model (`weights/yolov8n_plant_classifier_22Apr.pt`) was trained for 100 epochs on a dynamically balanced dataset of 500 images per class (80% train / 20% val). 
- **Architecture**: YOLOv8 Nano (`yolov8n-cls.pt`)
- **Input Size**: 224x224
- **Performance**: Achieved 100% confidence across validation subsets.

## Setup & Installation
It is highly recommended to run this inside a Conda environment with `ultralytics` installed.

```bash
conda create -n imagerecog python=3.10
conda activate imagerecog
pip install ultralytics opencv-python
```

## How to Run Inference

We provide a simple script to run inference on either a single static image or via a live camera feed.

### 1. Live Webcam Inference
To open your camera and see real-time predictions rendered on screen:
```bash
python inference_yolo_22ndapr.py --live
```
*(Press 'q' in the camera window to stop).*

### 2. Single Image Inference
To test a specific image from the terminal:
```bash
python inference_yolo_22ndapr.py --image /path/to/your/test_image.jpg
```
This will pop up a window showing the image with the label overlay and print the exact confidence percentage in the terminal.

## Pipeline Scripts Included

If you wish to reproduce or expand the dataset, the following utility scripts are included:

- **`extract_fruits_buds.py`**: Uses the SAM3 Promptable Concept Segmentation (PCS) model to automatically segment and crop bounding boxes of plants from raw video frames.
- **`build_yolo_dataset.py`**: A dataset generator that pulls from multiple source directories (`Segmented_PCS`, `Segmented_Fruits_Buds`, `Segmented_Single_Leaf`). It uses a strict fallback priority to prioritize "Whole Plant" images, filling any deficits with Buds and then Leaves, guaranteeing perfectly balanced classes.
- **`train_yolo_22ndapr.py`**: The training configuration script used to kick off the YOLOv8 classification training.
