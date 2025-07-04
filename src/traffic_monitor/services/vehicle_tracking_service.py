import multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
from queue import Empty, Full
import cv2
import numpy as np
from loguru import logger
from boxmot import create_tracker
from ..utils.custom_types import FrameMessage, VehicleDetectionMessage, Detection, TrackedObject, TrackedVehicleMessage
from pathlib import Path
from typing import Any, Dict, List
from ..utils.logging_config import setup_logging
from ..utils.queue_utils import safe_put, log_queue_stats


class VehicleTracker:
    """
    Manages the vehicle tracking logic using the BoxMOT library.
    Initializes the tracker and processes raw detections into tracked objects.
    """

    def __init__(
        self,
        tracker_type: str,
        reid_model_path: Path,
        device: str,
        half: bool,
        per_class: bool,
        tracker_config_path: Path
    ):
        try:
            self.tracker = create_tracker(
                tracker_type=tracker_type,
                tracker_config=tracker_config_path,
                reid_weights=reid_model_path,
                device=device,
                half=half,
                per_class=per_class
            )
            logger.info(f"VehicleTracker initialized: {tracker_type} on {device}")
        except Exception as e:
            logger.exception(f"Failed to create tracker {tracker_type}: {e}")
            raise # Re-raise the exception to propagate the error

    def _detections_to_numpy(self, detections: List[Detection]) -> np.ndarray:
        """
        Converts a list of detection dictionaries into a NumPy array format expected by the tracker.

        Args:
            detections (List[Detection]): A list of detection dictionaries.

        Returns:
            np.ndarray: A NumPy array where each row represents a detection
                        [x1, y1, x2, y2, confidence, class_id].
        """
        if not detections:
            return np.array([])
        # Extract bounding box, confidence, and class_id for each detection
        return np.array([[detection["bbox_xyxy"][0], detection["bbox_xyxy"][1], detection["bbox_xyxy"][2], detection["bbox_xyxy"][3], detection["confidence"], detection["class_id"]] for detection in detections])
    
    def _tracks_to_dict(self, tracks: np.ndarray, class_mapping: Dict[int, str]) -> List[TrackedObject]:
        """
        Converts a NumPy array of tracked objects from the tracker into a list of TrackedObject dictionaries.

        Args:
            tracks (np.ndarray): A NumPy array representing tracked objects.
                                 Expected format: [x1, y1, x2, y2, track_id, confidence, class_id, detection_index].
            class_mapping (Dict[int, str]): A dictionary mapping class IDs to class names.

        Returns:
            List[TrackedObject]: A list of dictionaries, each representing a tracked object.
        """
        if tracks.size == 0:
            return []
        return [
            {
                "bbox_xyxy": [int(c) for c in track[:4]],  # Bounding box coordinates
                "track_id": int(track[4]),        # Track ID
                "confidence": float(track[5]),     # Confidence score
                "class_id": int(track[6]),         # Class ID
                "class_name": class_mapping.get(int(track[6]), "unknown") # Map class ID to name, default to "unknown"
            }
            for track in tracks
        ]
    
    def update(self, detections: list[Detection], class_mapping: dict[int, str], frame: np.ndarray) -> list[TrackedObject]:
        """
        Updates the tracker with new detections and retrieves the current list of tracked objects.

        Args:
            detections (list[Detection]): A list of new detections.
            class_mapping (dict[int, str]): A dictionary mapping class IDs to class names.

        Returns:
            list[TrackedObject]: A list of tracked objects with their attributes.
        """
        if not detections:
            track_numpy = self.tracker.update(np.array([]), frame)
        else:
            detection_numpy = self._detections_to_numpy(detections)
            track_numpy = self.tracker.update(detection_numpy, frame)

        # Convert the tracked objects (numpy array) to a list of dictionaries
        return self._tracks_to_dict(track_numpy, class_mapping)
    
def vehicle_tracker_process(config: Dict[str, Any], input_queue: Queue, output_queue: Queue, shutdown_event: Event):
    """
    The main process function for the vehicle tracker.

    This function continuously retrieves vehicle detection messages from the input queue,
    updates the vehicle tracker, and puts the tracked vehicle messages into the output queue.
    It gracefully shuts down when the shutdown event is set or a None message is received.

    Args:
        config (Dict[str, Any]): Configuration dictionary for the tracker.
        input_queue (Queue): Queue to receive VehicleDetectionMessage objects.
        output_queue (Queue): Queue to send TrackedVehicleMessage objects.
        shutdown_event (Event): An event to signal the process to shut down.
    """
    setup_logging(config.get("loguru")) # Initialize logging for this process
    process_name = mp.current_process().name
    offline_mode = config.get("offline_mode", False)
    service_name = config.get("service_name", "VehicleTracker")
    logger.info(f"Vehicle Tracker process {process_name} started")

    try:
        # Convert class mapping keys to integers
        class_mapping = {int(k): v for k, v in config.get("class_mapping", {}).items()}
        tracker_type = config.get("tracker_type", "bytetrack")
        reid_model_path = config.get("reid_model_path", Path("data/models/reid.pt"))
        device = config.get("device", "cpu")
        half = config.get("half", False)
        per_class = config.get("per_class", None)

        # Construct the absolute path for tracker_config_path
        tracker_config_path = Path(__file__).resolve().parent.parent / "config" / "trackers" / f"{tracker_type}.yaml"

        try:
            tracker = VehicleTracker(
                tracker_type=tracker_type,
                reid_model_path=reid_model_path,
                device=device,
                half=half,
                per_class=per_class,
                tracker_config_path=tracker_config_path
            )
            logger.info(f"Vehicle tracker initialized with {tracker_type}")
        except Exception as e:
            logger.exception(f"Failed to initialize VehicleTracker: {e}")
            return # Exit if initialization fails

        while not shutdown_event.is_set():
            try:
                # Get vehicle detection message from the input queue with a timeout
                vehicle_detection_message: VehicleDetectionMessage = input_queue.get(timeout=1)
                logger.trace(f"Received detection message for frame {vehicle_detection_message.get('frame_id')}")
            except Empty:
                # Continue if the queue is empty after the timeout
                logger.trace("Input queue empty, waiting for detection messages")
                continue
            
            # Handle shutdown signal received as None message
            if vehicle_detection_message is None:
                logger.info("Received shutdown signal. Terminating.")
                output_queue.put(None) # Signal downstream processes to shut down as well
                break

            # Convert JPEG binary data back to an OpenCV frame
            jpeg_binary = vehicle_detection_message["frame_data_jpeg"]
            img_array = np.frombuffer(jpeg_binary, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # Perform tracking on the detected objects
            detections = vehicle_detection_message["detections"]
            tracked_objects = tracker.update(detections, class_mapping, frame)
            
            # Log tracking results with sampling to reduce noise
            frame_id = vehicle_detection_message['frame_id']
            if tracked_objects and hash(frame_id) % 30 == 0:  # Sample logging every ~30 frames
                class_tracks = {}
                for obj in tracked_objects:
                    class_name = obj["class_name"]
                    class_tracks[class_name] = class_tracks.get(class_name, 0) + 1
                
                class_summary = ", ".join([f"{count} {class_name}{'s' if count > 1 else ''}" 
                                         for class_name, count in class_tracks.items()])
                logger.debug(f"Tracking {len(tracked_objects)} objects in frame {frame_id}: {class_summary}")

            # Create the output message
            output_message = TrackedVehicleMessage(
                frame_id=vehicle_detection_message["frame_id"],
                camera_id=vehicle_detection_message["camera_id"],
                timestamp=vehicle_detection_message["timestamp"],
                frame_data_jpeg=jpeg_binary,
                frame_height=vehicle_detection_message["frame_height"],
                frame_width=vehicle_detection_message["frame_width"],
                og_frame_height=vehicle_detection_message["og_frame_height"],
                og_frame_width=vehicle_detection_message["og_frame_width"],
                og_fps=vehicle_detection_message["og_fps"],
                tracked_objects=tracked_objects
            )
            
            # Use mode-aware queue operation
            success = safe_put(output_queue, output_message, offline_mode, service_name)
            if success:
                logger.trace(f"Queued tracking results for frame {frame_id}")
            else:
                logger.warning(f"[{service_name}] Failed to put tracking for frame {frame_id}")
                continue
            
    except Exception as e:
        # Log any exceptions that occur during the process and propagate the shutdown signal
        logger.error(f"Error in Vehicle Tracker process: {e}")
        output_queue.put(None)
        raise
    finally:
        logger.info(f"Vehicle Tracker process {process_name} finished")

