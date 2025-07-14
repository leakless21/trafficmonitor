#!/usr/bin/env python3
"""
Script to run YOLO plate detection on all images in a folder.
Crops detected license plates, converts them to grayscale, and saves with the same filename.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from ultralytics import YOLO
from loguru import logger

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    logger.remove()
    log_level = "DEBUG" if verbose else "INFO"
    logger.add(sys.stderr, level=log_level, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", colorize=True)

def load_yolo_model(model_path: str) -> YOLO:
    """Load YOLO model for plate detection."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)
    return model

def get_image_files(folder_path: str) -> List[Path]:
    """Get all image files from the specified folder."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    # Common image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    
    image_files = []
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    logger.info(f"Found {len(image_files)} image files in {folder_path}")
    return image_files

def detect_plates(model: YOLO, image: np.ndarray, conf_threshold: float = 0.5) -> List[Tuple[List[int], float]]:
    """Detect license plates in the image and return bounding boxes with confidence scores."""
    results = model.predict(image, conf=conf_threshold, verbose=False)
    
    detections = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            confidence = box.conf.item()
            detections.append((bbox, confidence))
    
    return detections

def crop_and_convert_plates(image: np.ndarray, detections: List[Tuple[List[int], float]]) -> List[np.ndarray]:
    """Crop detected plates and convert them to grayscale."""

    cropped_plates = []
    height, width = image.shape[:2]  # Compute once

    append_crop = cropped_plates.append  # Local var lookup is faster

    for bbox, confidence in detections:
        # Fast/explicit coords unpack, int conversion (tuple unpack is fastest here)
        try:
            x1, y1, x2, y2 = bbox
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
        except Exception as e:
            logger.warning(f"Invalid bbox entry: {bbox} ({e})")
            continue

        # Clip to image bounds in single statements
        x1 = 0 if x1 < 0 else x1 if x1 <= width else width
        x2 = 0 if x2 < 0 else x2 if x2 <= width else width
        y1 = 0 if y1 < 0 else y1 if y1 <= height else height
        y2 = 0 if y2 < 0 else y2 if y2 <= height else height

        # Skip invalid bbox fast
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"Invalid bounding box: [{x1}, {y1}, {x2}, {y2}]")
            continue

        plate_crop = image[y1:y2, x1:x2]

        if plate_crop.size == 0:
            logger.warning(f"Empty crop for bounding box: [{x1}, {y1}, {x2}, {y2}]")
            continue

        # Fastest check for color/grayscale with shape[2] if present
        if plate_crop.ndim == 3 and plate_crop.shape[2] == 3:
            plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        else:
            plate_gray = plate_crop

        append_crop(plate_gray)
        # Logging conditionally (string interpolation delayed until needed)
        logger.debug(
            "Cropped plate with confidence %.3f, size: %s", confidence, plate_gray.shape
        )

    return cropped_plates

def process_image(model: YOLO, image_path: Path, output_folder: Path, conf_threshold: float = 0.5) -> int:
    """Process a single image file."""
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Failed to load image: {image_path}. Skipping.")
            return 0
        
        # Detect plates
        detections = detect_plates(model, image, conf_threshold)
        
        if not detections:
            logger.info(f"No plates detected in: {image_path.name}. Skipping.")
            return 0
        
        logger.info(f"Detected {len(detections)} plate(s) in: {image_path.name}")
        
        # Crop and convert plates
        cropped_plates = crop_and_convert_plates(image, detections)
        
        if not cropped_plates:
            logger.warning(f"No valid plates cropped from: {image_path.name}. This might indicate issues with bounding box coordinates.")
            return 0
        
        # Save cropped plates
        base_name = image_path.stem
        extension = image_path.suffix
        
        saved_count = 0
        for i, plate in enumerate(cropped_plates):
            if len(cropped_plates) == 1:
                # Single plate: keep original name
                output_filename = f"{base_name}{extension}"
            else:
                # Multiple plates: add index
                output_filename = f"{base_name}_plate_{i+1}{extension}"
            
            output_path = output_folder / output_filename
            
            # Save grayscale plate
            success = cv2.imwrite(str(output_path), plate)
            if success:
                saved_count += 1
                logger.debug(f"Saved: {output_path}")
            else:
                logger.error(f"Failed to save: {output_path}")
        
        logger.info(f"Saved {saved_count} plate(s) from: {image_path.name}")
        return saved_count
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return 0

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Batch YOLO plate detection and cropping")
    parser.add_argument("input_folder", help="Input folder containing images")
    parser.add_argument("output_folder", help="Output folder for cropped plates")
    parser.add_argument("--model", "-m", default="data/models/lp.pt", 
                       help="Path to YOLO model file (default: data/models/lp.pt)")
    parser.add_argument("--confidence", "-c", type=float, default=0.6,
                       help="Confidence threshold for detection (default: 0.6)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Create output folder
        output_folder = Path(args.output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output folder: {output_folder}")
        
        # Load model
        model = load_yolo_model(args.model)
        
        # Get image files
        image_files = get_image_files(args.input_folder)
        
        if not image_files:
            logger.warning("No image files found in input folder")
            return
        
        # Process images
        total_processed_images = 0
        total_plates_saved = 0
        unprocessed_image_files = []
        
        for image_path in image_files:
            logger.info(f"Processing: {image_path.name}")
            plates_saved = process_image(model, image_path, output_folder, args.confidence)
            
            if plates_saved > 0:
                total_processed_images += 1
                total_plates_saved += plates_saved
            else:
                unprocessed_image_files.append(image_path)
        
        # Summary
        logger.info(f"Processing complete!")
        logger.info(f"Images processed: {total_processed_images}/{len(image_files)}")
        logger.info(f"Total plates saved: {total_plates_saved}")
        
        if unprocessed_image_files:
            logger.warning("The following images were not processed:")
            for file_path in unprocessed_image_files:
                logger.warning(f"  - {file_path.name}")
        else:
            logger.info("All images were processed successfully.")
        
    except Exception as e:
        logger.error(f"Script failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 