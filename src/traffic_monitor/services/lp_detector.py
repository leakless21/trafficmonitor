import multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
from queue import Empty, Full
from typing import Dict, Any
import os

import ultralytics
import cv2
import numpy as np
from loguru import logger

from ..utils.custom_types import FrameMessage, PlateDetectionMessage, VehicleTrackingMessage

class LPDetector:
    """
    Encapsulates the license plate detection model and its configuration.
    Handles loading the model, setting confidence thresholds, and processing detection results.
    """
    def __init__(self, model_path: str, conf_threshold: float):
        """
        Initializes the LPDetector with the specified model and confidence threshold.
        """
        logger.info(f"[LPDetector] Attempting to load model from {model_path}...")
        try:
            self.model = ultralytics.YOLO(model_path)
            self.conf_threshold = conf_threshold
            logger.info(f"[LPDetector] Model loaded successfully from {model_path}")
        except Exception as e:
            logger.exception(f"[LPDetector] Failed to load model from {model_path}: {e}")
            raise # Re-raise the exception to propagate the error
    
    def find_plates(self, frame: np.ndarray) -> tuple[list[int], float] | None:
        """
        Finds license plates in the given frame.
        """
        logger.info("[LPDetector] Running inference...")
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        if not results or not results[0].boxes:
            logger.info("[LPDetector] No plates found.")
            return None
        best_plate = results[0].boxes[0]
        bbox = best_plate.xyxy[0].tolist()
        confidence = best_plate.conf.item()
        return (bbox, confidence)

def lp_detector_process(
        config: Dict[str, Any],
        input_queue: Queue,
        output_queue: Queue,
        shutdown_event: Event
):
    """
    Processes license plate detection in a separate process.
    """
    from ..utils.logging_config import setup_logging
    setup_logging()  # Setup logging for this process
    
    process_name = mp.current_process().name
    logger.info(f"[LPDetectorProcess] Starting process {process_name}")

    lp_detector: LPDetector | None = None
    try:
        model_path = config.get("lp_detector", {}).get("model_path", "data/models/plate_v8n.pt")
        conf_threshold = config.get("lp_detector", {}).get("conf_threshold", 0.6)

        if not model_path or not conf_threshold:
            logger.error("[LPDetectorProcess] Missing required configuration parameters for LPDetector.")
            output_queue.put(None)
            return # Exit process

        if not os.path.exists(model_path):
            logger.error(f"[LPDetectorProcess] Model file not found at: {model_path}")
            output_queue.put(None)
            return # Exit process

        logger.info(f"[LPDetectorProcess] Initializing LPDetector with model_path: {model_path}, conf_threshold: {conf_threshold}")
        lp_detector = LPDetector(model_path, conf_threshold)
        logger.info(f"[LPDetectorProcess] LPDetector initialized successfully.")

    except Exception as e:
        logger.exception(f"[LPDetectorProcess] Failed to initialize LPDetector: {e}")
        if output_queue:
            output_queue.put(None) # Signal shutdown to other processes
        return # Exit process

    # Main processing loop
    try:
        while not shutdown_event.is_set():
            try:
                message: VehicleTrackingMessage = input_queue.get(timeout=1)
            except Empty:
                continue
            if message is None:
                logger.info(f"[LPDetectorProcess] Received None message, shutting down process")
                output_queue.put(None)
                break
        
            jpeg_bytes = message["frame_data_jpeg"]
            original_frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)

            lp_detections = []
            for vehicle in message["tracked_objects"]:
                bbox = vehicle["bbox_xyxy"]
                vehicle_crop = original_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                
                if vehicle_crop.size == 0:
                    logger.debug(f"[{process_name}] Empty vehicle crop for {vehicle['class_name']} (ID: {vehicle['track_id']})")
                    continue
                
                detections = lp_detector.find_plates(vehicle_crop)
                
                if detections is not None:
                    logger.debug(f"[{process_name}] Detected license plate on {vehicle['class_name']} (ID: {vehicle['track_id']})")
                else:
                    logger.debug(f"[{process_name}] No license plate detected on {vehicle['class_name']} (ID: {vehicle['track_id']})")

                if detections is not None:
                    lp_bbox_on_crop, lp_confidence = detections
                    lp_x1_crop, lp_y1_crop, lp_x2_crop, lp_y2_crop = lp_bbox_on_crop

                    final_lp_x1 = bbox[0] + lp_x1_crop
                    final_lp_y1 = bbox[1] + lp_y1_crop
                    final_lp_x2 = bbox[0] + lp_x2_crop
                    final_lp_y2 = bbox[1] + lp_y2_crop

                    final_lp_bbox = [int(c) for c in [final_lp_x1, final_lp_y1, final_lp_x2, final_lp_y2]]

                    plate_message: PlateDetectionMessage = {
                        "frame_id": message["frame_id"],
                        "camera_id": message["camera_id"],
                        "timestamp": message["timestamp"],
                        "frame_data_jpeg": message["frame_data_jpeg"],
                        "frame_height": message["frame_height"],
                        "frame_width": message["frame_width"],
                        "original_frame_height": message["original_frame_height"],
                        "original_frame_width": message["original_frame_width"],
                        "vehicle_id": vehicle["track_id"],
                        "vehicle_class": vehicle["class_name"],
                        "plate_bbox_original": final_lp_bbox,
                        "plate_confidence": lp_confidence
                    }

                    logger.debug(f"[LPDetectorProcess] Found plate for vehicle {vehicle['track_id']} with confidence {lp_confidence}")
                    try:
                        output_queue.put(plate_message)
                    except Full:
                        logger.warning(f"[LPDetectorProcess] Output queue is full, dropping plate message for vehicle {vehicle['track_id']}")
                    except Exception as e:
                        logger.exception(f"[LPDetectorProcess] Error putting plate message on output queue: {e}")
                else:
                    logger.debug(f"[LPDetectorProcess] No plate found for vehicle {vehicle['track_id']}")

    except Exception as e:
        logger.exception(f"[LPDetectorProcess] Error in main processing loop: {e}")
        if output_queue:
            output_queue.put(None)
    finally:
        logger.info(f"[LPDetectorProcess] Shutting down process {process_name}")
            
