import multiprocessing as mp
from multiprocessing.synchronize import Event 
from multiprocessing.queues import Queue 
from queue import Empty, Full
from typing import Dict, Any

import ultralytics
import cv2
import numpy as np
from loguru import logger
import time

from ..utils.custom_types import FrameMessage, VehicleDetectionMessage, Detection
from ..utils.logging_config import setup_logging
from ..utils.queue_utils import safe_put, log_queue_stats

class VehicleDetectionService:
    """
    Encapsulates the vehicle detection model and its configuration.
    Handles loading the model, setting confidence thresholds, and processing detection results.
    """
    def __init__(self, model_path: str, conf_threshold: float, class_mapping: dict[int, str]):
        """
        Initializes the VehicleDetectionService with the specified model, confidence threshold, and class mapping.

        Args:
            model_path (str): Path to the YOLO model weights.
            conf_threshold (float): Confidence threshold for detections.
            class_mapping (dict[int, str]): A dictionary mapping class IDs to class names.
        """
        try:
            self.model = ultralytics.YOLO(model_path)
            logger.info(f"[VehicleDetectionService] YOLO model loaded successfully from: {model_path}")
        except Exception as e:
            logger.exception(f"[VehicleDetectionService] Failed to load YOLO model from {model_path}: {e}")
            raise # Re-raise the exception to propagate the error

        self.conf_threshold = conf_threshold
        self.class_mapping = class_mapping
        logger.info(f"[VehicleDetectionService] Initialized with conf_threshold: {conf_threshold}, class_mapping: {class_mapping}")
        
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
            class_id = box.cls.item()  # Use .item() to extract scalar from array
            # Only consider detections for classes specified in the mapping
            if class_id in self.class_mapping: 
                bbox = [int(c) for c in box.xyxy[0].tolist()] # Get bounding box coordinates [x1, y1, x2, y2]
                confidence = box.conf.item()  # Use .item() to extract scalar from array
                
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
    
def vehicle_detection_process(
        config: Dict[str, Any],
        input_queue: Queue,
        output_queue: Queue,
        shutdown_event: Event
):
    print(f"[VehicleDetectorProcess] Process starting...") # Very early print for debugging
    """
    Main process function for the vehicle detection service.

    This function continuously reads frames from the input queue, performs vehicle detection
            using the VehicleDetectionService, and puts the detection results into the output queue.
    It gracefully handles shutdown signals and manages queue operations.

    Args:
        config (Dict[str, Any]): Configuration dictionary for the detector.
        input_queue (Queue): Queue to receive FrameMessage objects.
        output_queue (Queue): Queue to send VehicleDetectionMessage objects.
        shutdown_event (Event): An event to signal the process to shut down.
    """
    setup_logging(config.get("loguru")) # Initialize logging for this process
    process_name = mp.current_process().name
    offline_mode = config.get("offline_mode", False)
    service_name = config.get("service_name", "VehicleDetectionService")
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
            vehicle_detector = VehicleDetectionService(model_path, conf_threshold, class_mapping)
            logger.info(f"[{process_name}] Vehicle detector initialized.")
        except Exception as e:
            logger.exception(f"[{process_name}] Failed to initialize VehicleDetectionService: {e}")
            return # Exit if initialization fails
        
        while not shutdown_event.is_set():
            try:
                # Attempt to get a frame message from the input queue with a timeout
                frame_message: FrameMessage = input_queue.get(timeout=1)
                
                # Check for shutdown signal
                if frame_message is None:
                    logger.info(f"[{process_name}] Received shutdown signal.")
                    break
                
                # Log processing start for debugging
                logger.debug(f"[{process_name}] Processing frame {frame_message['frame_id']}")
                
                # Decode the JPEG frame data back to an OpenCV image
                jpeg_bytes = frame_message["frame_data_jpeg"]
                frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.warning(f"[{process_name}] Failed to decode frame {frame_message['frame_id']}. Skipping.")
                    continue
                
                # Perform vehicle detection on the frame
                detections = vehicle_detector.detect(frame)
                
                # Enhanced logging with class-specific information
                if detections:
                    class_counts = {}
                    for det in detections:
                        class_name = det["class_name"]
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    
                    class_summary = ", ".join([f"{count} {class_name}{'s' if count > 1 else ''}" for class_name, count in class_counts.items()])
                    logger.debug(f"[{process_name}] Detected {len(detections)} objects in frame {frame_message['frame_id']}: {class_summary}")
                else:
                    logger.debug(f"[{process_name}] No objects detected in frame {frame_message['frame_id']}")

                # Construct the output message with detection results
                output_message: VehicleDetectionMessage = {
                    "frame_id": frame_message["frame_id"],
                    "frame_width": frame_message["frame_width"],
                    "frame_height": frame_message["frame_height"],
                    "og_frame_width": frame_message["og_frame_width"],
                    "og_frame_height": frame_message["og_frame_height"],
                    "og_fps": frame_message["og_fps"],
                    "camera_id": frame_message["camera_id"],
                    "timestamp": frame_message["timestamp"],
                    "frame_data_jpeg": frame_message["frame_data_jpeg"],
                    "detections": detections
                }
                
                # Use mode-aware queue operation
                success = safe_put(output_queue, output_message, offline_mode, service_name)
                if not success:
                    logger.warning(f"[{service_name}] Failed to put detection for frame {frame_message['frame_id']}")
                    continue
                
            except Empty:
                # If no message is available, continue the loop to check the shutdown event
                continue
            except Exception as e:
                # Log any unexpected errors but continue processing to prevent process crash
                logger.error(f"[{process_name}] Error processing frame: {e}", exc_info=True)
                # Add small delay to prevent tight error loops
                time.sleep(0.1)
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