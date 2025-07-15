#!/usr/bin/env python
"""
Converts a PyTorch (.pt) YOLO model to ONNX format.

This script takes a .pt model file as input and exports it to the ONNX format,
allowing for various export options such as image size, opset version,
model simplification, and dynamic axes.

Usage:
  python scripts/convert_to_onnx.py --pt_model_path yolov8n.pt

With optional arguments:
  python scripts/convert_to_onnx.py --pt_model_path runs/train/exp/weights/best.pt --imgsz 640 --opset 12 --simplify --dynamic
"""
import argparse
import os
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Convert a YOLO .pt model to ONNX format.")
    parser.add_argument(
        "--pt_model_path",
        type=str,
        required=True,
        help="Path to the input .pt model file (e.g., yolov8n.pt or path/to/your/best.pt)."
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Image size for export (e.g., 640). Defaults to the model's default."
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="ONNX opset version (e.g., 12). Defaults to the latest compatible."
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Whether to simplify the ONNX model. Default is True."
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Whether to use dynamic axes. Default is False."
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Whether to include NMS (Non-Maximum Suppression) in the exported model. Default is False."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Full path to save the exported ONNX model (e.g., data/models/yolov8n.onnx). "
             "If not provided, it defaults to the model name with .onnx extension in the "
             "directory of the input .pt model or the current working directory if the input is just a name."
    )

    args = parser.parse_args()

    # Load the .pt model
    model = YOLO(args.pt_model_path)

    # Prepare export arguments
    export_args = {"format": "onnx"}
    if args.imgsz is not None:
        export_args["imgsz"] = args.imgsz
    if args.opset is not None:
        export_args["opset"] = args.opset
    if args.simplify:
        export_args["simplify"] = True
    if args.dynamic:
        export_args["dynamic"] = True
    if args.nms:
        export_args["nms"] = True
        
    # If output_path is specified, pass it to the export function.
    # Ultralytics' export function has a 'path' argument.
    if args.output_path:
        export_args["path"] = args.output_path
        # Ensure the directory for the output path exists
        output_dir = os.path.dirname(args.output_path)
        if output_dir: # Create directory if output_path includes a directory
            os.makedirs(output_dir, exist_ok=True)


    # Export the model
    try:
        exported_model_path = model.export(**export_args)
        print(f"Successfully exported model to: {exported_model_path}")
    except Exception as e:
        print(f"An error occurred during export: {e}")
        if not args.output_path:
            # Attempt to construct the expected ONNX path for a more informative error if export fails before returning path
            base_name = os.path.splitext(os.path.basename(args.pt_model_path))[0]
            expected_onnx_path = f"{base_name}.onnx"
            print(f"Default ONNX model path would have been: {expected_onnx_path} (in the same directory as the script or model if path is relative)")


if __name__ == "__main__":
    main()