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

from ..utils.custom_types import FrameMessage, PlateDetectionMessage, TrackedVehicleMessage

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
        confidence = float(best_plate.conf)
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
                message: TrackedVehicleMessage = input_queue.get(timeout=1)
            except Empty:
                continue
            if message is None:
                logger.info(f"[LPDetectorProcess] Received None message, shutting down process")
                output_queue.put(None)
                break
        
            jpeg_bytes = message["frame_data_jpeg"]
            original_frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)

            for vehicle in message["tracked_objects"]:
                vx1, vy1, vx2, vy2 = vehicle["bbox_xyxy"]
                if vx1 >= vx2 or vy1 >= vy2:
                    continue
                vehicle_crop = original_frame[vy1:vy2, vx1:vx2]

                plate_results = lp_detector.find_plates(vehicle_crop)

                if plate_results:
                    lp_bbox_on_crop, lp_confidence = plate_results
                    lp_x1_crop, lp_y1_crop, lp_x2_crop, lp_y2_crop = lp_bbox_on_crop

                    final_lp_x1 = vx1 + lp_x1_crop
                    final_lp_y1 = vy1 + lp_y1_crop
                    final_lp_x2 = vx1 + lp_x2_crop
                    final_lp_y2 = vy1 + lp_y2_crop

                    final_lp_bbox = [final_lp_x1, final_lp_y1, final_lp_x2, final_lp_y2]

                    plate_message: PlateDetectionMessage = {
                        "frame_id": message["frame_id"],
                        "camera_id": message["camera_id"],
                        "timestamp": message["timestamp"],
                        "frame_data_jpeg": message["frame_data_jpeg"],
                        "frame_height": message["frame_height"],
                        "frame_width": message["frame_width"],
                        "vehicle_id": str(vehicle["track_id"]),
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
            
