#!/usr/bin/env python3
import argparse
import json
import sys

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO plant classification on one image.")
    parser.add_argument("--image", required=True, help="Path to the input image.")
    parser.add_argument("--model", required=True, help="Path to the YOLO classification model.")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Accepted for compatibility with farmOS Leaf MCP.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        model = YOLO(args.model)
        results = model.predict(source=args.image, verbose=False)
        if not results or results[0].probs is None:
            raise RuntimeError("The model returned no classification probabilities.")

        result = results[0]
        class_id = int(result.probs.top1)
        label = str(result.names[class_id])
        confidence = float(result.probs.top1conf.item() * 100.0)
        print(json.dumps({"label": label, "confidence": confidence}))
    except Exception as exc:
        print(f"YOLO inference failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
