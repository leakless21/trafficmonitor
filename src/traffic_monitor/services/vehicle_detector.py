import multiprocessing as mp
from multiprocessing.synchronize import Event 
from multiprocessing.queues import Queue 
from queue import Empty, Full
from typing import Dict, Any

import ultralytics
import cv2
import numpy as np
from loguru import logger

from ..utils.custom_types import FrameMessage, VehicleDetectionMessage, Detection
from ..utils.logging_config import setup_logging

class VehicleDetector:
    """
    Encapsulates the vehicle detection model and its configuration.
    Handles loading the model, setting confidence thresholds, and processing detection results.
    """
    def __init__(self, model_path: str, conf_threshold: float, class_mapping: dict[int, str]):
        """
        Initializes the VehicleDetector with the specified model, confidence threshold, and class mapping.

        Args:
            model_path (str): Path to the YOLO model weights.
            conf_threshold (float): Confidence threshold for detections.
            class_mapping (dict[int, str]): A dictionary mapping class IDs to class names.
        """
        try:
            self.model = ultralytics.YOLO(model_path)
            logger.info(f"[VehicleDetector] YOLO model loaded successfully from: {model_path}")
        except Exception as e:
            logger.exception(f"[VehicleDetector] Failed to load YOLO model from {model_path}: {e}")
            raise # Re-raise the exception to propagate the error

        self.conf_threshold = conf_threshold
        self.class_mapping = class_mapping
        logger.info(f"[VehicleDetector] Initialized with conf_threshold: {conf_threshold}, class_mapping: {class_mapping}")
        
    def process_results(self, results) -> list[Detection]:
        """
        Processes the raw output from the YOLO model into a standardized list of Detection objects.

        Args:
            results: The raw detection results from the YOLO model.

        Returns:
            list[Detection]: A list of dictionaries, each representing a detected object.
        """
        detections: list[Detection] = []
        # Ensure results are not empty and contain detectable objects
        if not results or not results[0]:
            return detections
        
        # Iterate through detected bounding boxes
        for box in results[0].boxes:
            class_id = int(box.cls)
            # Only consider detections for classes specified in the mapping
            if class_id in self.class_mapping: 
                bbox = box.xyxy[0].tolist() # Get bounding box coordinates [x1, y1, x2, y2]
                confidence = float(box.conf) # Get detection confidence score
                
                detections_dict: Detection = {
                    "bbox_xyxy": bbox,
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": self.class_mapping[class_id]
                }
                detections.append(detections_dict)
        return detections

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Performs vehicle detection on a given frame.

        Args:
            frame (np.ndarray): The input image frame as a NumPy array.

        Returns:
            list[Detection]: A list of detected objects.
        """
        # Run YOLO prediction on the frame with the specified confidence threshold
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        # Process the raw results into a structured list of detections
        processed_results = self.process_results(results)
        return processed_results
    
def vehicle_detector_process(
        config: Dict[str, Any],
        input_queue: Queue,
        output_queue: Queue,
        shutdown_event: Event
):
    print(f"[VehicleDetectorProcess] Process starting...") # Very early print for debugging
    """
    Main process function for the vehicle detection service.

    This function continuously reads frames from the input queue, performs vehicle detection
    using the VehicleDetector, and puts the detection results into the output queue.
    It gracefully handles shutdown signals and manages queue operations.

    Args:
        config (Dict[str, Any]): Configuration dictionary for the detector.
        input_queue (Queue): Queue to receive FrameMessage objects.
        output_queue (Queue): Queue to send VehicleDetectionMessage objects.
        shutdown_event (Event): An event to signal the process to shut down.
    """
    setup_logging(config.get("loguru")) # Initialize logging for this process
    process_name = mp.current_process().name
    logger.info(f"[{process_name}] Vehicle Detector process started.")

    try:
        # Load configuration parameters for the detector
        model_path = config.get("model_path")
        conf_threshold = config.get("conf_threshold", 0.5)
        class_mapping = {int(k): v for k, v in config.get("class_mapping", {}).items()}
        
        # Validate essential configuration parameters
        if not model_path or not conf_threshold or not class_mapping:
            logger.error(f"[{process_name}] Invalid configuration. model_path: {model_path}, conf_threshold: {conf_threshold}, class_mapping: {class_mapping}")
            return # Exit if configuration is invalid
    
        # Initialize the vehicle detector instance
        try:
            logger.info(f"[{process_name}] Initializing vehicle detector with model: {model_path}, conf_threshold: {conf_threshold}, class_mapping: {class_mapping}")
            vehicle_detector = VehicleDetector(model_path, conf_threshold, class_mapping)
            logger.info(f"[{process_name}] Vehicle detector initialized.")
        except Exception as e:
            logger.exception(f"[{process_name}] Failed to initialize VehicleDetector: {e}")
            return # Exit if initialization fails
        
        while not shutdown_event.is_set():
            logger.debug(f"[{process_name}] Attempting to get frame from input queue...")
            try:
                # Attempt to get a frame message from the input queue with a timeout
                frame_message: FrameMessage = input_queue.get(timeout=1)
                logger.debug(f"[{process_name}] Received frame {frame_message.get('frame_id')} from input queue.")
            except Empty:
                # If the queue is empty, continue to the next iteration
                logger.trace(f"[{process_name}] Input queue is empty. Waiting for frames.")
                continue

            # Check for a None message, which indicates a shutdown signal from the upstream process
            if frame_message is None:
                logger.warning(f"[{process_name}] Received None frame message. Shutting down.")
                output_queue.put(None) # Propagate shutdown signal to downstream processes
                break

            # Decode the JPEG binary frame data into an OpenCV image array
            jpeg_binary = frame_message["frame_data_jpeg"]
            img_array = np.frombuffer(jpeg_binary, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            logger.debug(f"[{process_name}] Decoded frame {frame_message.get('frame_id')}. Performing detection...")

            # Perform vehicle detection on the current frame
            detections = vehicle_detector.detect(frame)
            logger.debug(f"[{process_name}] Detected {len(detections)} vehicles in frame {frame_message['frame_id']}")

            # Construct the output message with detection results
            output_message: VehicleDetectionMessage = {
                "frame_id": frame_message["frame_id"],
                "frame_width": frame_message["frame_width"],
                "frame_height": frame_message["frame_height"],
                "camera_id": frame_message["camera_id"],
                "timestamp": frame_message["timestamp"],
                "frame_data_jpeg": frame_message["frame_data_jpeg"],
                "detections": detections
            }
            
            # Attempt to put the processed message into the output queue
            try:
                output_queue.put(output_message)
            except Full:
                # Log a warning if the output queue is full and drop the message
                logger.warning(f"[{process_name}] Output queue is full. Dropping message.")
                continue
    
    except KeyboardInterrupt:
        logger.info(f"[{process_name}] KeyboardInterrupt received. Shutting down.")
        shutdown_event.set()
        if not output_queue.full():
            output_queue.put(None) # Propagate shutdown signal
    except Exception as e:
        # Catch and log any unexpected exceptions that occur during the process
        logger.exception(f"[{process_name}] Vehicle Detector process crashed: {e}")
        # Re-raise the exception to ensure the process truly terminates if needed
        raise 
    finally:
        # Log process completion upon normal shutdown or exception
        logger.info(f"[{process_name}] Vehicle Detector process finished.")