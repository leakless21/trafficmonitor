import multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
from queue import Empty, Full
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np
from loguru import logger

from fast_plate_ocr import ONNXPlateRecognizer

# ADD IMPORT INSIDE TRY to avoid if not installed
try:
    from paddleocr import PaddleOCR  # type: ignore
except ImportError:  # pragma: no cover
    PaddleOCR = None  # type: ignore

from ..utils.custom_types import PlateDetectionMessage, OCRResultMessage

class OCRReader:
    def __init__(self, config: Dict[str, Any]):
        self.backend: str = config.get("backend", "fast_plate_ocr").lower()
        self.conf_threshold = config.get("conf_threshold", 0.5)

        if self.backend == "fast_plate_ocr":
            hub_model_name = config.get("hub_model_name", "global-plates-mobile-vit-v2-model")
            device = config.get("device", "auto")

            try:
                self.reader = ONNXPlateRecognizer(hub_ocr_model=hub_model_name, device=device)
                logger.info(f"[OCRReader] FastPlateOCR initialized with model: {hub_model_name} on device: {device}")
            except Exception as e:
                logger.error(f"[OCRReader] Failed to initialize FastPlateOCR reader: {e}")
                raise
        elif self.backend == "paddleocr":
            if PaddleOCR is None:
                raise ImportError("paddleocr package is not installed. Please install paddleocr to use this backend.")
            # Map config keys for PaddleOCR v5 API
            use_gpu: bool = bool(config.get("use_gpu", False))
            device = "cuda" if use_gpu else "cpu"
            # Allow explicit device override from config (map common values)
            device_override = config.get("device")
            if device_override:
                if device_override.startswith("gpu"):
                    device = "cuda"
                elif device_override == "cpu":
                    device = "cpu"
                elif device_override == "auto":
                    device = "auto"
            lang: str = str(config.get("lang", "en"))
            try:
                # Initialize PaddleOCR with v5 API (device parameter instead of use_gpu)
                self.reader = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    lang=lang,
                    device=device,
                    text_detection_model_name="PP-OCRv4_mobile_det",
                    text_recognition_model_name="en_PP-OCRv4_mobile_rec",
                )
                logger.info(f"[OCRReader] PaddleOCR initialized with language: {lang} | Device: {device}")
            except Exception as e:
                logger.error(f"[OCRReader] Failed to initialize PaddleOCR reader: {e}")
                raise
        else:
            raise ValueError(f"Unsupported OCR backend: {self.backend}")

    def _preprocess_plate(self, plate_image: np.ndarray) -> np.ndarray:
        if self.backend == "fast_plate_ocr":
            return cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        # No preprocessing required for paddleocr (expects BGR/RGB image)
        return plate_image

    def _read_plate_fast(self, plate_image: np.ndarray) -> Optional[Tuple[str, float]]:
        gray_plate = self._preprocess_plate(plate_image)
        try:
            raw_results = self.reader.run(gray_plate, return_confidence=True)  # type: ignore[attr-defined]
        except Exception as e:
            logger.error(f"Failed to read plate using FastPlateOCR: {e}")
            return None

        if not raw_results or not isinstance(raw_results, tuple) or len(raw_results) != 2:
            logger.debug("FastPlateOCR returned no valid results")
            return None

        plate_texts, confidence = raw_results

        if not plate_texts or confidence.size == 0:
            return None

        plate_text = plate_texts[0]
        char_confidence = confidence[0]
        overall_confidence = np.mean(char_confidence) if char_confidence.size > 0 else 0.0

        if len(plate_text) < 3 or overall_confidence < self.conf_threshold:
            return None

        return plate_text, float(overall_confidence)

    def _read_plate_paddle(self, plate_image: np.ndarray) -> Optional[Tuple[str, float]]:
        try:
            results = self.reader.predict(plate_image)  # type: ignore[attr-defined]
        except Exception as e:
            logger.error(f"Failed to read plate using PaddleOCR: {e}")
            return None

        if not results:
            return None

        # PaddleOCR v3 returns list with dict entries
        ocr_result = results[0]
        rec_texts = ocr_result.get("rec_texts", [])
        rec_scores = ocr_result.get("rec_scores", [])

        if not rec_texts or not rec_scores:
            return None

        cleaned_results: list[Tuple[str, float]] = []
        for text, conf in zip(rec_texts, rec_scores):
            # ensure numeric confidence (could be list for each char)
            if isinstance(conf, (list, tuple, np.ndarray)):
                conf_value = float(np.mean(conf)) if len(conf) else 0.0
            else:
                conf_value = float(conf)

            if conf_value >= self.conf_threshold:
                cleaned_text = "".join(c for c in text if c.isalnum())
                if cleaned_text:
                    cleaned_results.append((cleaned_text, conf_value))

        if not cleaned_results:
            return None

        # Select best candidate: longest text then highest confidence
        best_text, best_conf = max(cleaned_results, key=lambda x: (len(x[0]), x[1]))
        return best_text, best_conf

    def read_plate(self, plate_image: np.ndarray) -> Tuple[str, float] | None:
        if self.backend == "fast_plate_ocr":
            return self._read_plate_fast(plate_image)
        elif self.backend == "paddleocr":
            return self._read_plate_paddle(plate_image)
        else:
            logger.error(f"Unsupported backend during inference: {self.backend}")
            return None
    
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
                    "vehicle_id": lp_message['vehicle_id'],
                    "lp_text": lp_text,
                    "ocr_confidence": ocr_confidence,
                }
                
                # Real-time behavior: drop old OCR result if queue is full
                try:
                    try:
                        ocr_reader_output_queue.get_nowait()  # Remove old OCR result if queue is full
                    except Empty:
                        pass  # Queue was empty, which is fine
                    
                    ocr_reader_output_queue.put_nowait(ocr_result_message)  # Put new OCR result without blocking
                except Full:
                    # This should never happen with get_nowait() + put_nowait() pattern, but keep for safety
                    logger.warning(f"[OCRReader] Output queue is full, dropping OCR result for vehicle {lp_message['vehicle_id']}")
                except Exception as e:
                    logger.exception(f"[OCRReader] Error putting OCR result on output queue: {e}")
                
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