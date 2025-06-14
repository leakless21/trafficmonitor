import multiprocessing as mp
from multiprocessing.synchronize import Event 
from multiprocessing.queues import Queue 
from queue import Empty, Full
from typing import Dict, Any

import ultralytics
import cv2
import numpy as np
import base64
from loguru import logger

from ..utils.logging_config import setup_logging
from ..utils.config_loader import load_config
from ..utils.custom_types import FrameMessage, VehicleDetectionMessage, Detection

class VehicleDetector:
    # Encapsulates the vehicle detection model and its configuration
    def __init__(self, model_path: str, conf_threshold: float, class_mapping: dict[int, str]):
        self.model = ultralytics.YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.class_mapping = class_mapping
        logger.info(f"VehicleDetector initialized with model: {model_path}, conf_threshold: {conf_threshold}, class_mapping: {class_mapping}")
        
    def process_results(self, results) -> list[Detection]:
        # Process YOLO results into a list of Detection objects
        detections: list[Detection] = []
        # Results can be a list of Detection objects or a single Detection object
        if not results or not results[0]:
            return detections
        for box in results[0].boxes:
            class_id = int(box.cls)
            if class_id in self.class_mapping: # Check if the class ID is in the class mapping
                bbox = box.xyxy[0].tolist() # Get the bounding box coordinates
                confidence = float(box.conf) # Get the confidence score
                detections_dict: Detection = {
                    "bbox_xyxy": bbox,
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": self.class_mapping[class_id]
                }
                detections.append(detections_dict)
        return detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        # Detect vehicles in the frame
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        processed_results = self.process_results(results)
        return processed_results
    
def vehicle_detector_process(
        config: Dict[str, Any],
        input_queue: Queue,
        output_queue: Queue,
        shutdown_event: Event
):
    """
    Main functions for the vehicle detector process.
    - Get frames
    - Detect vehicles
    - Put the results in the output queue
    """
    process_name = mp.current_process().name
    logger.info(f"[{process_name}] Vehicle Detector process started.")

    # 1. Extract configuration
    try:
        model_path = config.get("model_path")
        conf_threshold = config.get("conf_threshold", 0.5)
        class_mapping = {int(k): v for k, v in config.get("class_mapping", {}).items()}
        if not model_path or not conf_threshold or not class_mapping:
            logger.error(f"[{process_name}] Invalid configuration. model_path: {model_path}, conf_threshold: {conf_threshold}, class_mapping: {class_mapping}")
            return
    
        # 2. Initialize the vehicle detector
        vehicle_detector = VehicleDetector(model_path, conf_threshold, class_mapping)
        
        while not shutdown_event.is_set():
            # 3. Get a frame from the input queue
            try:
                frame_message: FrameMessage = input_queue.get(timeout=1)
                # 3.1 Check if the frame is None
            except Empty:
                continue

            # 4. Shutdown the process if the frame is None
            if frame_message is None:
                logger.warning("Received None frame message. Shutting down.")
                output_queue.put(None)
                break

            # 5. Get the frame data and decode it
            jpeg_as_base64 = frame_message["frame_data_jpeg"]
            jpeg_binary = base64.b64decode(jpeg_as_base64)
            img_array = np.frombuffer(jpeg_binary, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # 6. Detect vehicles
            detections = vehicle_detector.detect(frame)
            logger.debug(f"[{process_name}] Detected {len(detections)} vehicles in frame {frame_message['frame_id']}")

            # 7. Augment the message with the detections
            output_message: VehicleDetectionMessage = {
                "frame_id": frame_message["frame_id"],
                "frame_width": frame_message["frame_width"],
                "frame_height": frame_message["frame_height"],
                "camera_id": frame_message["camera_id"],
                "timestamp": frame_message["timestamp"],
                "frame_data_jpeg": frame_message["frame_data_jpeg"],
                "detections": detections
            }
            # 8. Put the message in the output queue
            try:
                output_queue.put(output_message)
            except Full:
                logger.warning(f"[{process_name}] Output queue is full. Dropping message.")
                continue
    
    except Exception as e:
        logger.exception(f"[{process_name}] Vehicle Detector process crashed: {e}")
    finally:
        logger.info(f"[{process_name}] Vehicle Detector process finished.")