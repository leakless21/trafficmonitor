#!/usr/bin/env python3
"""
OCR Dataset Processor

This script processes image datasets using multiple OCR engines and outputs 
results to CSV format. Supports both fast_plate_ocr and PaddleOCR engines.

Usage:
    python scripts/ocr_dataset_processor.py --input_dir path/to/images --output_csv results.csv --engine fast_plate_ocr
"""

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class OCRResult:
    """Data class for OCR results."""
    text: str
    confidence: float
    processing_time: float
    engine: str


class BaseOCREngine:
    """Base class for OCR engines."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = "base"
    
    def process_image(self, image: np.ndarray) -> Optional[OCRResult]:
        """Process image and return OCR result."""
        raise NotImplementedError


class FastPlateOCREngine(BaseOCREngine):
    """Fast Plate OCR engine using ONNX models."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "fast_plate_ocr"
        
        try:
            from fast_plate_ocr import LicensePlateRecognizer
            
            hub_model_name = config.get("hub_model_name", "cct-s-v1-global-model")
            device = config.get("device", "auto")
            self.conf_threshold = config.get("conf_threshold", 0.5)
            
            self.reader = LicensePlateRecognizer(hub_ocr_model=hub_model_name, device=device)
            logging.info(f"FastPlateOCR initialized with model: {hub_model_name} on device: {device}")
            
        except ImportError:
            raise ImportError("fast_plate_ocr is not installed. Install with: pip install fast-plate-ocr")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize FastPlateOCR: {e}")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for OCR (no preprocessing needed for new FastPlateOCR)."""
        return image
    
    def process_image(self, image: np.ndarray) -> Optional[OCRResult]:
        """Process image with FastPlateOCR."""
        start_time = time.time()
        
        try:
            raw_results = self.reader.run(image, return_confidence=True)
            
            if not raw_results or not isinstance(raw_results, tuple) or len(raw_results) != 2:
                return None
            
            plate_texts, confidence = raw_results
            
            if not plate_texts or confidence.size == 0:
                return None
            
            plate_text = plate_texts[0]
            char_confidence = confidence[0]
            overall_confidence = np.mean(char_confidence) if char_confidence.size > 0 else 0.0
            
            if len(plate_text) < 3 or overall_confidence < self.conf_threshold:
                return None
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                text=plate_text,
                confidence=float(overall_confidence),
                processing_time=processing_time,
                engine=self.name
            )
            
        except Exception as e:
            logging.error(f"Error processing image with FastPlateOCR: {e}")
            return None


class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR engine."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "paddleocr"
        
        try:
            from paddleocr import PaddleOCR
            
            lang = config.get("lang", "en")
            self.conf_threshold = config.get("conf_threshold", 0.5)
            
            # Initialize with updated parameters for PaddleOCR v3.x
            self.reader = PaddleOCR(
                use_textline_orientation=True,  # Enable for better multi-line text detection
                use_doc_orientation_classify=False,  # Keep disabled for license plates
                use_doc_unwarping=False,  # Keep disabled for license plates
                lang=lang,
                text_detection_model_name="PP-OCRv5_mobile_det",
                text_recognition_model_name="PP-OCRv5_mobile_rec",
                # Additional parameters for better license plate detection (using correct v3 parameter names)
                text_det_thresh=0.3,  # Lower threshold for text detection (default: 0.3)
                text_det_box_thresh=0.5,  # Box threshold for text detection (default: 0.6)
                text_det_unclip_ratio=1.6  # Unclip ratio for text boxes (default: 1.5)
            )
            logging.info(f"PaddleOCR initialized with lang: {lang}")
            
        except ImportError:
            raise ImportError("PaddleOCR is not installed. Install with: pip install paddlepaddle paddleocr")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PaddleOCR: {e}")
    
    def process_image(self, image: np.ndarray) -> Optional[OCRResult]:
        """Process image with PaddleOCR."""
        start_time = time.time()

        try:
            # Use predict() method with NumPy array input
            results = self.reader.predict(image)

            if not results or not results[0]:
                return None

            # Extract text and confidence from results based on PaddleOCR v3.x format
            ocr_result = results[0]
            if not isinstance(ocr_result, dict) or 'rec_texts' not in ocr_result or 'rec_scores' not in ocr_result:
                logging.warning("PaddleOCR result format not as expected. Expected dict with 'rec_texts' and 'rec_scores'.")
                return None

            rec_texts = ocr_result['rec_texts']
            rec_scores = ocr_result['rec_scores']

            if not rec_texts or not rec_scores:
                return None

            texts = []
            confidences = []

            for text, confidence in zip(rec_texts, rec_scores):
                if confidence > self.conf_threshold:
                    # Clean text for license plates
                    cleaned_text = ''.join(c for c in text if c.isalnum())
                    if len(cleaned_text) >= 3:
                        texts.append(cleaned_text)
                        confidences.append(confidence)

            if not texts:
                return None

            # For multi-line license plates, combine all detected text lines.
            # Prioritize concatenation for license plates as they are typically a single string.
            if len(texts) == 1:
                plate_text = texts[0]
                overall_confidence = confidences[0]
            else:
                plate_text = ''.join(texts)
                overall_confidence = np.mean(confidences) if confidences else 0.0

            processing_time = time.time() - start_time

            return OCRResult(
                text=plate_text,
                confidence=float(overall_confidence),
                processing_time=processing_time,
                engine=self.name
            )

        except Exception as e:
            logging.error(f"Error processing image with PaddleOCR: {e}")
            return None


def setup_logging(log_file: str = "ocr_dataset_processing.log") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def get_ocr_engine(engine_name: str, config: Dict[str, Any]) -> BaseOCREngine:
    """Get OCR engine instance."""
    engines = {
        "fast_plate_ocr": FastPlateOCREngine,
        "paddleocr": PaddleOCREngine
    }
    
    if engine_name not in engines:
        raise ValueError(f"Unknown engine: {engine_name}. Available: {list(engines.keys())}")
    
    return engines[engine_name](config)


def get_image_files(input_dir: Path, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')) -> List[Path]:
    """Get all image files from input directory."""
    print(f"[DEBUG] get_image_files called with input_dir: {input_dir}")
    image_files = []
    for ext in extensions:
        found_files = list(input_dir.glob(f"**/*{ext}"))
        found_files.extend(list(input_dir.glob(f"**/*{ext.upper()}")))
        print(f"[DEBUG] Glob pattern **/*{ext} found {len(found_files)} files.")
        image_files.extend(found_files)
    
    # Remove duplicates that can occur on case-insensitive filesystems
    unique_image_files = sorted(list(set(image_files)))
    print(f"[DEBUG] Total unique image files found: {len(unique_image_files)}")
    if unique_image_files:
        print(f"[DEBUG] Sample files: {unique_image_files[:5]}")
    return unique_image_files


def process_dataset(
    input_dir: Path,
    output_csv: Path,
    engine_name: str,
    engine_config: Dict[str, Any],
    relative_paths: bool = True,
    batch_size: int = 100,
    save_intermediate: bool = True
) -> Dict[str, Any]:
    """
    Process dataset with OCR and save results to CSV.
    
    Args:
        input_dir: Directory containing images
        output_csv: Output CSV file path
        engine_name: OCR engine to use
        engine_config: Configuration for OCR engine
        relative_paths: Whether to use relative paths in CSV
        batch_size: Number of images to process before saving intermediate results
        save_intermediate: Whether to save intermediate results
        
    Returns:
        Dictionary with processing statistics
    """
    logger = setup_logging()
    logger.info(f"Starting OCR processing for dataset: {input_dir}")
    logger.info(f"Using OCR engine: {engine_name}")
    
    # Initialize OCR engine
    try:
        ocr_engine = get_ocr_engine(engine_name, engine_config)
    except Exception as e:
        logger.error(f"Failed to initialize OCR engine: {e}")
        raise
    
    # Get all image files
    image_files = get_image_files(input_dir)
    logger.info(f"Found {len(image_files)} image files")
    
    if not image_files:
        logger.warning("No image files found in the specified directory")
        return {"total_images": 0, "processed": 0, "successful": 0}
    
    # Process images and write to CSV
    processed_count = 0
    successful_count = 0
    failed_count = 0
    total_processing_time = 0.0
    
    # Create output directory if needed
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare intermediate file
    temp_csv = output_csv.with_suffix('.tmp.csv') if save_intermediate else None
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write header with additional fields
        writer.writerow(['image_path', 'plate_text', 'confidence', 'processing_time', 'engine'])
        
        # Process each image
        for i, image_path in enumerate(tqdm(image_files, desc=f"Processing with {engine_name}")):
            try:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    logger.warning(f"Could not read image: {image_path}")
                    failed_count += 1
                    continue
                
                # Process with OCR
                result = ocr_engine.process_image(image)
                processed_count += 1
                
                if result:
                    # Determine path to write
                    if relative_paths:
                        try:
                            relative_path = image_path.relative_to(input_dir)
                            path_to_write = str(relative_path).replace('\\', '/')
                        except ValueError:
                            path_to_write = str(image_path)
                    else:
                        path_to_write = str(image_path)
                    
                    # Write result to CSV
                    writer.writerow([
                        path_to_write,
                        result.text,
                        f"{result.confidence:.4f}",
                        f"{result.processing_time:.4f}",
                        result.engine
                    ])
                    
                    successful_count += 1
                    total_processing_time += result.processing_time
                    
                    logger.debug(f"Processed: {path_to_write} -> {result.text} (conf: {result.confidence:.3f})")
                else:
                    failed_count += 1
                    logger.debug(f"No text detected in: {image_path}")
                
                # Save intermediate results
                if save_intermediate and temp_csv and (i + 1) % batch_size == 0:
                    csvfile.flush()
                    logger.info(f"Processed {i + 1}/{len(image_files)} images. "
                              f"Successful: {successful_count}, Failed: {failed_count}")
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                failed_count += 1
                continue
    
    # Clean up temporary file
    if temp_csv and temp_csv.exists():
        temp_csv.unlink()
    
    # Calculate statistics
    avg_processing_time = total_processing_time / successful_count if successful_count > 0 else 0
    
    stats = {
        "total_images": len(image_files),
        "processed": processed_count,
        "successful": successful_count,
        "failed": failed_count,
        "success_rate": successful_count / len(image_files) if len(image_files) > 0 else 0,
        "avg_processing_time": avg_processing_time,
        "total_processing_time": total_processing_time,
        "engine": engine_name
    }
    
    logger.info("Processing complete!")
    logger.info(f"Total images: {stats['total_images']}")
    logger.info(f"Successful: {stats['successful']} ({stats['success_rate']:.1%})")
    logger.info(f"Failed: {stats['failed']}")
    logger.info(f"Average processing time: {stats['avg_processing_time']:.4f}s")
    logger.info(f"Results saved to: {output_csv}")
    
    return stats


def main():
    """Main function to handle command line arguments and execute processing."""
    parser = argparse.ArgumentParser(
        description="Process image dataset with OCR and output to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use FastPlateOCR (default)
  python scripts/ocr_dataset_processor.py --input_dir lp_data/CarTGMTCrop --output_csv results.csv
  
  # Use PaddleOCR with specific language
  python scripts/ocr_dataset_processor.py --input_dir data/test_images --output_csv results.csv --engine paddleocr --lang en
  
  # Process with custom confidence threshold
  python scripts/ocr_dataset_processor.py --input_dir images --output_csv annotations.csv --conf_threshold 0.7
  
  # Use absolute paths and custom batch size
  python scripts/ocr_dataset_processor.py --input_dir data/plates --output_csv results.csv --absolute_paths --batch_size 50
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
        '--engine', 
        type=str, 
        choices=['fast_plate_ocr', 'paddleocr'],
        default='fast_plate_ocr',
        help='OCR engine to use (default: fast_plate_ocr)'
    )
    
    parser.add_argument(
        '--conf_threshold', 
        type=float, 
        default=0.5,
        help='Confidence threshold for OCR results (default: 0.5)'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        help='Device to use for FastPlateOCR (auto, cpu, cuda) (default: auto)'
    )
    
    parser.add_argument(
        '--lang', 
        type=str, 
        default='en',
        choices=['en', 'ch', 'french', 'german', 'korean', 'japan'],
        help='Language for PaddleOCR (default: en)'
    )
    
    parser.add_argument(
        '--use_gpu', 
        action='store_true',
        help='Use GPU acceleration for PaddleOCR'
    )
    
    parser.add_argument(
        '--absolute_paths', 
        action='store_true',
        help='Use absolute paths in CSV instead of relative paths'
    )
    
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=100,
        help='Batch size for intermediate saves (default: 100)'
    )
    
    parser.add_argument(
        '--no_intermediate', 
        action='store_true',
        help='Disable intermediate result saving'
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
    
    # Prepare output path
    output_csv = Path(args.output_csv)
    
    # Prepare engine configuration
    engine_config = {
        "conf_threshold": args.conf_threshold,
        "device": args.device,
        "lang": args.lang,
        "use_gpu": args.use_gpu
    }
    
    # Add engine-specific configurations
    if args.engine == "fast_plate_ocr":
        engine_config["hub_model_name"] = "cct-s-v1-global-model"
    
    # Process dataset
    try:
        stats = process_dataset(
            input_dir=input_dir,
            output_csv=output_csv,
            engine_name=args.engine,
            engine_config=engine_config,
            relative_paths=not args.absolute_paths,
            batch_size=args.batch_size,
            save_intermediate=not args.no_intermediate
        )
        
        print("\n📊 Processing Summary:")
        print(f"  Engine: {stats['engine']}")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Successful: {stats['successful']} ({stats['success_rate']:.1%})")
        print(f"  Failed: {stats['failed']}")
        print(f"  Average processing time: {stats['avg_processing_time']:.4f}s")
        print(f"  Output saved to: {output_csv}")
        
        return 0
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 