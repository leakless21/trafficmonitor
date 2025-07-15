import multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
from queue import Empty
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np
from loguru import logger

from fast_plate_ocr import LicensePlateRecognizer

# ADD IMPORT INSIDE TRY to avoid if not installed
try:
    from paddleocr import PaddleOCR  # type: ignore
except ImportError:  # pragma: no cover
    PaddleOCR = None  # type: ignore

from ..utils.custom_types import PlateDetectionMessage, OCRResultMessage
from ..utils.minidb import configure_database, write_plate_result
from ..utils.queue_utils import safe_put

class TextRecognitionService:
    def __init__(self, config: Dict[str, Any]):
        self.backend: str = config.get("backend", "fast_plate_ocr").lower()
        self.conf_threshold = config.get("conf_threshold", 0.5)

        if self.backend == "fast_plate_ocr":
            hub_model_name = config.get("hub_model_name", "cct-s-v1-global-model")
            device = config.get("device", "auto")

            try:
                self.reader = LicensePlateRecognizer(hub_ocr_model=hub_model_name, device=device)
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
            lang: str = str(config.get("lang", "en"))
            try:
                # Initialize PaddleOCR with v5 API (device parameter instead of use_gpu)
                self.reader = PaddleOCR(
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                    lang=lang,
                    device=device,
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_mobile_rec",
                    text_det_thresh=0.3,  # Lower threshold for text detection (default: 0.3)
                    text_det_box_thresh=0.5,  # Box threshold for text detection (default: 0.6)
                    text_det_unclip_ratio=1.6  # Unclip ratio for text boxes (default: 1.5)
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
        try:
            raw_results = self.reader.run(plate_image, return_confidence=True)  # type: ignore[attr-defined]
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
            # Ensure numeric confidence (could be list for each char in some versions)
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

        # If only one high-confidence text line, return it directly
        if len(cleaned_results) == 1:
            return cleaned_results[0]

        # Handle multi-line license plates (e.g., two-line plates):
        # Concatenate all detected text segments preserving OCR reading order.
        combined_text: str = "".join(text for text, _ in cleaned_results)
        combined_conf: float = float(np.mean([conf for _, conf in cleaned_results]))

        # If the combined text meets basic length and confidence requirements, use it.
        if len(combined_text) >= 3 and combined_conf >= self.conf_threshold:
            return combined_text, combined_conf

        # Fallback to previous behaviour: choose the longest single line.
        best_text, best_conf = max(cleaned_results, key=lambda x: len(x[0]))
        return best_text, best_conf

    def read_plate(self, plate_image: np.ndarray) -> Tuple[str, float] | None:
        if self.backend == "fast_plate_ocr":
            return self._read_plate_fast(plate_image)
        elif self.backend == "paddleocr":
            return self._read_plate_paddle(plate_image)
        else:
            logger.error(f"Unsupported backend during inference: {self.backend}")
            return None
    
def text_recognition_process(config: Dict[str, Any], lp_detector_output_queue: Queue, ocr_reader_output_queue: Queue, shutdown_event: Event):
    from ..utils.logging_config import setup_logging
    setup_logging()  # Setup logging for this process
    
    # Configure database for this subprocess (inherits path from parent config)
    configure_database(config)
    
    # Cache to avoid redundant OCR for the same vehicle within a short window
    cache_duration_sec: float = float(config.get("cache_duration_sec", 2.0))
    ocr_cache: dict[int, float] = {}  # vehicle_id -> last_ocr_timestamp

    process_name = mp.current_process().name
    offline_mode = config.get("offline_mode", False)
    service_name = config.get("service_name", "TextRecognitionService")
    min_plate_width: int = int(config.get("min_plate_width", 20))
    min_plate_height: int = int(config.get("min_plate_height", 10))
    logger.info(f"[TextRecognitionService] Process {process_name} started")
    try:
        ocr_reader = TextRecognitionService(config)
        while not shutdown_event.is_set():
            try:
                lp_message: PlateDetectionMessage = lp_detector_output_queue.get(timeout=1)
            except Empty:
                continue

            if lp_message is None:
                logger.info("[OCRReader] Received shutdown sentinel – propagating downstream and exiting")
                try:
                    ocr_reader_output_queue.put(None, timeout=1)
                except Exception as e:
                    logger.warning(f"[TextRecognitionService] Failed to propagate shutdown sentinel: {e}")
                break

            jpeg_data = lp_message['frame_data_jpeg']
            frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
            x1, y1, x2, y2 = lp_message['plate_bbox_original']
            if x1 >= x2 or y1 >= y2:
                continue

            # Identify vehicle early for logging
            vehicle_id = lp_message['vehicle_id']

            # Skip OCR for plates that are below minimum size thresholds
            plate_width = x2 - x1
            plate_height = y2 - y1
            if plate_width < min_plate_width or plate_height < min_plate_height:
                logger.trace(f"[OCRReader] Skipping OCR for vehicle {vehicle_id} due to small plate size ({plate_width}x{plate_height})")
                continue

            # Skip OCR if we recently processed this vehicle
            msg_ts = lp_message['timestamp']
            last_ts = ocr_cache.get(vehicle_id)
            if last_ts is not None and (msg_ts - last_ts) < cache_duration_sec:
                logger.trace(f"[OCRReader] Skipping OCR for vehicle {vehicle_id} within cache window ({msg_ts - last_ts:.2f}s)")
                continue

            plate_image = frame[y1:y2, x1:x2]
            ocr_results = ocr_reader.read_plate(plate_image)
            if ocr_results:
                lp_text, ocr_confidence = ocr_results

                # Update cache on successful read
                ocr_cache[vehicle_id] = msg_ts
                ocr_result_message: OCRResultMessage = {
                    "frame_id": lp_message['frame_id'],
                    "camera_id": lp_message['camera_id'],
                    "timestamp": lp_message['timestamp'],
                    "vehicle_id": lp_message['vehicle_id'],
                    "lp_text": lp_text,
                    "ocr_confidence": ocr_confidence,
                }
                
                # Use mode-aware queue operation
                success = safe_put(ocr_reader_output_queue, ocr_result_message, offline_mode, service_name)
                if not success:
                    logger.warning(f"[{service_name}] Failed to put OCR result for vehicle {lp_message['vehicle_id']}")
                
                vehicle_class = lp_message.get('vehicle_class', 'unknown')
                logger.info(f"[OCRReader] Detected plate '{lp_text}' for {vehicle_class} (ID: {lp_message['vehicle_id']}) with confidence {ocr_confidence:.3f}")
                # Persist to database
                try:
                    write_plate_result(
                        camera_id=lp_message['camera_id'],
                        vehicle_id=lp_message['vehicle_id'],
                        vehicle_class=vehicle_class,
                        lp_text=lp_text,
                        ocr_conf=ocr_confidence,
                        ts=int(lp_message['timestamp'] * 1000)
                    )
                except Exception as db_exc:
                    logger.exception(f"[OCRReader] Failed to write plate result to database: {db_exc}")
            else:
                vehicle_class = lp_message.get('vehicle_class', 'unknown')
                logger.debug(f"[OCRReader] No plate text extracted from {vehicle_class} (ID: {lp_message['vehicle_id']})")
    except Exception as e:
        logger.exception(f"[OCRReader] Process {process_name} crashed: {e}")
        if 'ocr_reader_output_queue' in locals() and ocr_reader_output_queue:
            ocr_reader_output_queue.put(None, timeout=1)
    finally:
        logger.info(f"[OCRReader] Process {process_name} shutting down")