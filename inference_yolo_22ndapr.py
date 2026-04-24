import argparse
from ultralytics import YOLO
import cv2
import sys

def main():
    parser = argparse.ArgumentParser(description="YOLO Plant Classification Inference")
    parser.add_argument('--image', type=str, help="Path to an image for single inference")
    parser.add_argument('--live', action='store_true', help="Run live inference using webcam (source 0)")
    parser.add_argument('--model', type=str, default='weights/yolov8n_plant_classifier_22Apr.pt', help="Path to the trained model")
    
    args = parser.parse_args()
    
    if not args.image and not args.live:
        print("Please provide either --image <path> or use the --live flag.")
        parser.print_help()
        sys.exit(1)
        
    print(f"Loading YOLO model from {args.model}...")
    model = YOLO(args.model)
    
    if args.image:
        print(f"Running inference on image: {args.image}")
        # Run prediction and show image window
        results = model.predict(source=args.image, show=True)
        
        # Also print the top prediction to the terminal
        result = results[0]
        top_idx = result.probs.top1
        pred_label = result.names[top_idx]
        conf = result.probs.top1conf.item()
        print(f"\n--- Result ---")
        print(f"Prediction: {pred_label} (Confidence: {conf:.2%})")
        
        print("\nPress any key in the image window to close it.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    elif args.live:
        print("\nStarting live inference on webcam...")
        print("A window will open with the live feed and real-time predictions.")
        print("Press 'q' in the live window or CTRL+C in the terminal to stop.")
        
        # YOLO's built-in prediction automatically handles capturing and rendering the webcam!
        # stream=True acts as a generator for memory efficiency.
        results = model.predict(source=0, show=True, stream=True)
        
        try:
            for r in results:
                # We iterate through the generator to keep the stream alive
                pass
        except KeyboardInterrupt:
            print("\nLive inference stopped.")
            
if __name__ == "__main__":
    main()
