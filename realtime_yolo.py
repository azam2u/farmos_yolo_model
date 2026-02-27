import cv2
from ultralytics import YOLO
import sys

def main():
    # Load the best trained model weights
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "weights", "best.pt")
    print(f"Loading YOLO model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # Open the default camera (index 0)
    # Try index 0, if it fails, maybe try 1
    camera_idx = 0
    cap = cv2.VideoCapture(camera_idx)

    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_idx}. Searching for other cameras...")
        # fallback to index 1 just in case
        camera_idx = 1
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            print("Error: Could not open any camera. Please check your USB connection or permissions.")
            sys.exit(1)

    print(f"Camera initialized on index {camera_idx}! Press 'q' in the video window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from camera. Exiting...")
            break

        # Run YOLO inference on the frame
        # Setting a confidence threshold of 0.5 to minimize noise
        results = model.predict(source=frame, conf=0.5, verbose=False)

        # Plot predictions (bounding boxes and class labels) on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame in a window
        cv2.imshow("FarmOS - Real-Time Plant Detection", annotated_frame)

        # Check if the user pressed 'q' to quit (wait 1 ms between frames)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

    # Clean up and release hardware resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
