#!/usr/bin/env python3
"""
PaddleOCR to CSV Script

This script processes images using PaddleOCR and outputs the results 
to CSV format similar to train_anotaciones.csv.

Usage:
    python scripts/paddleocr_to_csv.py --input_dir path/to/images --output_csv output.csv
"""

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
from paddleocr import PaddleOCR
from tqdm import tqdm


def setup_logging() -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('paddleocr_processing.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def initialize_ocr(use_gpu: bool = False, lang: str = 'en') -> PaddleOCR:
    """
    Initialize PaddleOCR instance.
    
    Args:
        use_gpu: Whether to use GPU acceleration
        lang: Language for OCR recognition
        
    Returns:
        Initialized PaddleOCR instance
    """
    # Initialize PaddleOCR with v3.x compatible parameters
    return PaddleOCR(
        use_textline_orientation=True,  # Enable text line orientation classification
        lang=lang,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec"
    )


def process_image(image_path: Path, ocr: PaddleOCR) -> Optional[str]:
    """
    Process a single image with PaddleOCR.
    
    Args:
        image_path: Path to the image file
        ocr: PaddleOCR instance
        
    Returns:
        Extracted text or None if no text found
    """
    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            logging.warning(f"Could not read image: {image_path}")
            return None
        
        # Run OCR using the preferred predict method
        result = ocr.predict(image)
        
        if not result or len(result) == 0:
            return None
        
        # Extract text from the OCRResult object (PaddleOCR v3.x format)
        ocr_result = result[0]
        
        # Access data as dictionary keys
        if 'rec_texts' not in ocr_result or 'rec_scores' not in ocr_result:
            return None
            
        rec_texts = ocr_result['rec_texts']
        rec_scores = ocr_result['rec_scores']
        
        if not rec_texts or not rec_scores:
            return None
        
        # Extract text with confidence filtering
        texts = []
        for text, confidence in zip(rec_texts, rec_scores):
            # Filter by confidence threshold
            if confidence > 0.5:
                # Clean text (remove spaces, special characters for plate numbers)
                cleaned_text = ''.join(c for c in text if c.isalnum())
                if cleaned_text:
                    texts.append(cleaned_text)
        
        # Combine all texts or return the longest one
        if texts:
            # For license plates, usually return the longest valid text
            return max(texts, key=len) if len(texts) > 1 else texts[0]
        
        return None
        
    except Exception as e:
        logging.error(f"Error processing {image_path}: {e}")
        return None


def get_image_files(input_dir: Path, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) -> List[Path]:
    """
    Get all image files from input directory.
    
    Args:
        input_dir: Directory containing images
        extensions: Supported image file extensions
        
    Returns:
        List of image file paths
    """
    image_files = []
    for ext in extensions:
        image_files.extend(input_dir.glob(f"**/*{ext}"))
        image_files.extend(input_dir.glob(f"**/*{ext.upper()}"))
    
    return sorted(image_files)


def process_images_to_csv(
    input_dir: Path, 
    output_csv: Path, 
    use_gpu: bool = False,
    lang: str = 'en',
    relative_paths: bool = True
) -> None:
    """
    Process all images in directory and save results to CSV.
    
    Args:
        input_dir: Directory containing images
        output_csv: Output CSV file path
        use_gpu: Whether to use GPU acceleration
        lang: Language for OCR
        relative_paths: Whether to use relative paths in CSV
    """
    logger = setup_logging()
    logger.info(f"Starting OCR processing for directory: {input_dir}")
    
    # Initialize OCR
    logger.info("Initializing PaddleOCR...")
    ocr = initialize_ocr(use_gpu=use_gpu, lang=lang)
    
    # Get all image files
    image_files = get_image_files(input_dir)
    logger.info(f"Found {len(image_files)} image files")
    
    if not image_files:
        logger.warning("No image files found in the specified directory")
        return
    
    # Process images and write to CSV
    processed_count = 0
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header
        writer.writerow(['image_path', 'plate_text'])
        
        # Process each image
        for image_path in tqdm(image_files, desc="Processing images"):
            plate_text = process_image(image_path, ocr)
            
            if plate_text:
                # Determine path to write
                if relative_paths:
                    # Make path relative to input directory
                    relative_path = image_path.relative_to(input_dir)
                    path_to_write = str(relative_path).replace('\\', '/')
                else:
                    path_to_write = str(image_path)
                
                writer.writerow([path_to_write, plate_text])
                processed_count += 1
                logger.debug(f"Processed: {path_to_write} -> {plate_text}")
            else:
                logger.debug(f"No text detected in: {image_path}")
    
    logger.info(f"Processing complete! Processed {processed_count} images with detected text")
    logger.info(f"Results saved to: {output_csv}")


def main():
    """Main function to handle command line arguments and execute processing."""
    parser = argparse.ArgumentParser(
        description="Process images with PaddleOCR and output to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/paddleocr_to_csv.py --input_dir lp_data/test_images --output_csv results.csv
  python scripts/paddleocr_to_csv.py --input_dir data/plates --output_csv annotations.csv --use_gpu --lang en
        """
    )
    
    parser.add_argument(
        '--input_dir', 
        type=str, 
        required=True,
        help='Directory containing images to process'
    )
    
    parser.add_argument(
        '--output_csv', 
        type=str, 
        required=True,
        help='Output CSV file path'
    )
    
    parser.add_argument(
        '--use_gpu', 
        action='store_true',
        help='Use GPU acceleration (requires CUDA)'
    )
    
    parser.add_argument(
        '--lang', 
        type=str, 
        default='en',
        choices=['en', 'ch', 'french', 'german', 'korean', 'japan'],
        help='Language for OCR recognition (default: en)'
    )
    
    parser.add_argument(
        '--absolute_paths', 
        action='store_true',
        help='Use absolute paths in CSV instead of relative paths'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return 1
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        return 1
    
    # Create output directory if needed
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Process images
    try:
        process_images_to_csv(
            input_dir=input_dir,
            output_csv=output_csv,
            use_gpu=args.use_gpu,
            lang=args.lang,
            relative_paths=not args.absolute_paths
        )
        return 0
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 