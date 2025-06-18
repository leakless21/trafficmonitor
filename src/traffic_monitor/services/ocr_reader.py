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
            self.reader = ONNXPlateRecognizer(hub_ocr_model=hub_model_name, device=device)
            logger.info(f"[OCRReader] OCR reader initialized with model: {hub_model_name} on device: {device}")
        except Exception as e:
            logger.error(f"[OCRReader] Failed to initialize OCR reader: {e}")
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
            logger.debug(f"OCR result '{plate_text}' with confidence {overall_confidence} below threshold {self.conf_threshold}")
            return None
        
        # Return the OCR result
        return (plate_text, float(overall_confidence))
    
def ocr_reader_process(config: Dict[str, Any], lp_detector_output_queue: Queue, ocr_reader_output_queue: Queue, shutdown_event: Event):
    from ..utils.logging_config import setup_logging
    setup_logging()  # Setup logging for this process
    
    process_name = mp.current_process().name
    logger.info(f"[OCRReader] Process {process_name} started")
    try:
        ocr_reader = OCRReader(config)
        while not shutdown_event.is_set():
            try:
                lp_message: PlateDetectionMessage = lp_detector_output_queue.get(timeout=1)
            except Empty:
                continue

            if lp_message is None:
                logger.warning(f"[OCRReader] Received None message, shutting down.")
                lp_detector_output_queue.put(None)
                break

            jpeg_data = lp_message['frame_data_jpeg']
            frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
            x1, y1, x2, y2 = lp_message['plate_bbox_original']
            if x1 >= x2 or y1 >= y2:
                continue
            plate_image = frame[y1:y2, x1:x2]
            ocr_results = ocr_reader.read_plate(plate_image)
            if ocr_results:
                lp_text, ocr_confidence = ocr_results
                ocr_result_message: OCRResultMessage = {
                    "frame_id": lp_message['frame_id'],
                    "camera_id": lp_message['camera_id'],
                    "timestamp": lp_message['timestamp'],
                    "vehicle_id": int(lp_message['vehicle_id']),
                    "lp_text": lp_text,
                    "ocr_confidence": ocr_confidence,
                }
                ocr_reader_output_queue.put(ocr_result_message, timeout=1)
                vehicle_class = lp_message.get('vehicle_class', 'unknown')
                logger.info(f"[OCRReader] Detected plate '{lp_text}' for {vehicle_class} (ID: {lp_message['vehicle_id']}) with confidence {ocr_confidence:.3f}")
            else:
                vehicle_class = lp_message.get('vehicle_class', 'unknown')
                logger.debug(f"[OCRReader] No plate text extracted from {vehicle_class} (ID: {lp_message['vehicle_id']})")
    except Exception as e:
        logger.exception(f"[OCRReader] Process {process_name} crashed: {e}")
        if 'ocr_reader_output_queue' in locals() and ocr_reader_output_queue:
            ocr_reader_output_queue.put(None, timeout=1)
    finally:
        logger.info(f"[OCRReader] Process {process_name} shutting down")