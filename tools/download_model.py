#!/usr/bin/env python
"""
Downloads a YOLO model from Ultralytics.

This script takes a model name as a command-line argument,
instantiates the YOLO object from the ultralytics library,
which automatically downloads the model if it's not present locally.
It then prints a confirmation message with the model name.

Example usage:
    python scripts/download_model.py --model_name yolov8n.pt
    ./scripts/download_model.py --model_name yolov9c.pt
"""
import argparse
import os
from ultralytics import YOLO
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

def main():
    """
    Parses command-line arguments, downloads/loads the specified YOLO model,
    saves it to the specified output directory, and prints a confirmation message.
    """
    parser = argparse.ArgumentParser(
        description="Download a YOLO model from Ultralytics and save it to a specified directory."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="The name of the YOLO model to download (e.g., yolov8n.pt, yolov9c.pt)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/models",
        help="Directory to save the downloaded model. Defaults to 'data/models'."
    )
    args = parser.parse_args()

    print(f"Attempting to load/download model: {args.model_name}...")
    try:
        # Load the model (this will download it if not present in cache)
        model = YOLO(args.model_name)
        print(f"Successfully loaded/downloaded model: {args.model_name} into cache.")

        # Ensure the output directory exists
        os.makedirs(args.output_dir, exist_ok=True)

        # Determine the filename (it should be the same as model_name if it's a .pt file)
        # For official models, model_name is like 'yolov8n.pt'
        # If model_name is a path, os.path.basename will extract the filename.
        file_name = os.path.basename(args.model_name)
        if not file_name.endswith(".pt"):
            # This case should ideally not happen if users provide 'yolovX.pt'
            # but as a fallback, append .pt if it's a stem like 'yolov8n'
            if file_name in GITHUB_ASSETS_STEMS:
                 file_name += ".pt"
            else:
                # If it's a custom name without .pt, this might be an issue
                # For now, we assume model_name includes .pt for pre-trained models
                print(f"Warning: Model name '{args.model_name}' does not end with .pt. Proceeding with the name as is for the output file.")


        desired_save_path = os.path.join(args.output_dir, file_name)

        # Export the model to the desired path.
        # For a .pt model, exporting to 'pt' format essentially saves it.
        # The export() method returns the path to the saved file.
        saved_path = model.export(format="pt", path=desired_save_path) # Ultralytics handles overwrite by default
        
        print(f"Model '{args.model_name}' successfully saved to: {saved_path}")

    except Exception as e:
        print(f"Error during model processing for {args.model_name}: {e}")

if __name__ == "__main__":
    main()