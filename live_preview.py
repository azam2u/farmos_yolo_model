#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO


WINDOW_TITLE = "FarmOS Leaf YOLO Preview"


def parse_args():
    parser = argparse.ArgumentParser(description="Preview and confirm YOLO plant classification.")
    parser.add_argument("--model", required=True, help="Path to the YOLO classification model.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Minimum confidence displayed as a confirmed classification.",
    )
    parser.add_argument("--save-dir", required=True, help="Directory for the confirmed frame.")
    return parser.parse_args()


def open_camera(camera_index):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera index {camera_index}.")
    return cap


def classify_frame(model, frame):
    results = model.predict(source=frame, verbose=False)
    if not results or results[0].probs is None:
        return frame.copy(), "", 0.0

    result = results[0]
    class_id = int(result.probs.top1)
    label = str(result.names[class_id])
    confidence = float(result.probs.top1conf.item() * 100.0)
    return result.plot(), label, confidence


def draw_preview(frame, label, confidence, threshold_percent):
    preview = np.ascontiguousarray(frame).copy()
    shown_label = label if confidence >= threshold_percent else "Low confidence"
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 94), (0, 0, 0), -1)
    cv2.putText(
        preview,
        f"{shown_label} ({confidence:.2f}%)",
        (14, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (80, 230, 80),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        "ENTER/SPACE: confirm  O: overwrite  N: create new  Q/ESC: cancel",
        (14, 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return preview


def save_frame(frame, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.abspath(os.path.join(save_dir, f"capture_live_{timestamp}.jpg"))
    if not cv2.imwrite(path, frame):
        raise RuntimeError(f"Could not save confirmed frame to {path}.")
    return path


def main():
    args = parse_args()
    cap = None
    try:
        model = YOLO(args.model)
        if model.task != "classify":
            raise RuntimeError(f"Expected a classification model, received task '{model.task}'.")

        cap = open_camera(args.camera)
        cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
        threshold_percent = max(0.0, args.conf * 100.0 if args.conf <= 1.0 else args.conf)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Camera opened but failed to provide a frame.")

            annotated, label, confidence = classify_frame(model, frame)
            preview = draw_preview(annotated, label, confidence, threshold_percent)
            cv2.imshow(WINDOW_TITLE, preview)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                print(json.dumps({"status": "cancelled"}))
                return 0

            mode = None
            if key in (10, 13, 32):
                mode = "ask"
            elif key == ord("o"):
                mode = "overwrite"
            elif key == ord("n"):
                mode = "create_new"

            if mode:
                image_path = save_frame(frame, args.save_dir)
                print(
                    json.dumps(
                        {
                            "status": "confirmed",
                            "mode": mode,
                            "image_path": image_path,
                            "label": label,
                            "confidence": confidence,
                        }
                    )
                )
                return 0
    except Exception as exc:
        print(f"Live YOLO preview failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
