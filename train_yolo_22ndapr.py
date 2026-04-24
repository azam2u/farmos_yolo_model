from ultralytics import YOLO

def main():
    print("Initializing YOLOv8 Nano classification model...")
    model = YOLO('yolov8n-cls.pt')

    print("Beginning training on the YOLO_Plant_Dataset_22ndapr dataset...")
    # Train the model with recommended augmentations
    results = model.train(
        data='/home/cvl/farmos_env/YOLO_Plant_Dataset_22ndapr',
        epochs=100,
        imgsz=224,
        batch=16,
        workers=4,
        degrees=15.0,  # Rotations to help with plant variations
        scale=0.2,     # Zoom in/out
        fliplr=0.5,    # Horizontal flips
        project='/home/cvl/farmos_env/runs/classify',
        name='plant_classifier_22ndapr'
    )
    print("Training finished! Results saved to /home/cvl/farmos_env/runs/classify/plant_classifier_22ndapr")

if __name__ == "__main__":
    main()
