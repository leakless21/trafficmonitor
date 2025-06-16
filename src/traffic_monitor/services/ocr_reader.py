import multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
from queue import Empty, Full
from typing import Dict, Any, Tuple

import cv2
import numpy as np
from loguru import logger

from fast_plate_ocr import ONNXPlateRecognizer
from ..utils.custom_types import PlateDetectionMessage, OCRResultMessage

class OCRReader:
    def __init__(self, config: Dict[str, Any]):
        hub_model_name = config.get("hub_model_name", "global-plates-mobile-vit-v2-model")
        device = config.get("device", "auto")
        self.conf_threshold = config.get("conf_threshold", 0.5)

        try:
            self.reader = ONNXPlateRecognizer(hub_model_name, device=device)
            logger.info(f"OCR reader initialized with model: {hub_model_name} on device: {device}")
        except Exception as e:
            logger.error(f"Failed to initialize OCR reader: {e}")
            raise
    def _preprocess_plate(self, plate_image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        
    def read_plate(self, plate_image: np.ndarray) -> Tuple[str, float] | None:
        gray_plate = self._preprocess_plate(plate_image)
        try:
            raw_results = self.reader.run(gray_plate, return_confidence=True)
        except Exception as e:
            logger.error(f"Failed to read plate: {e}")
            return None
        
        if not raw_results:
            logger.warning("No plate detected in the frame")
            return None
        
        if not isinstance(raw_results, tuple) or len(raw_results) != 2:
            logger.error(f"Invalid results format from OCR reader: type={type(raw_results)}, length={len(raw_results)}")
            return None
        
        plate_texts, confidence = raw_results
        
        # Check if the plate text is valid
        if not plate_texts or confidence.size == 0:
            logger.warning("Plate text is too short to be valid")
            return None
        
        plate_text = plate_texts[0]
        char_confidence = confidence[0]
        overall_confidence = np.mean(char_confidence) if char_confidence.size > 0 else 0.0
        
        if len(plate_text) < 3:
            logger.warning("Plate text is too short to be valid")
            return None
        # Check if the confidence is above the threshold
        if overall_confidence < self.conf_threshold:
            logger.debug(f"OCR result '{plate_text}' with confidence {confidence} below threshold {self.conf_threshold}")
            return None
        
        # Return the OCR result
        return (plate_text, float(confidence))
