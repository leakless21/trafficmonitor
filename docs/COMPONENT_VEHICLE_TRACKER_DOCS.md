## Component: Vehicle Tracker

### Domain: Vehicle Tracking

This document outlines the `VehicleTracker` component, responsible for handling multi-object tracking of vehicles within the traffic monitoring system. It leverages the BoxMOT library to process raw detections into persistent, tracked objects.

### Related Classes and Files:

- **`VehicleTracker` Class**

  - **Location:** `src/traffic_monitor/services/vehicle_tracker.py`
  - **Purpose:** Encapsulates the BoxMOT tracker initialization and update logic. It acts as the interface between raw detections and the tracking algorithm.
  - **Key Methods:**
    - `__init__(self, tracker_type: str, reid_model_path: Path, device: str, half: bool, per_class: bool, tracker_config: Path)`: Initializes the BoxMOT tracker with the specified configuration.
    - `_detections_to_numpy(self, detections: List[Detection]) -> np.ndarray`: Converts a list of detection dictionaries to a NumPy array format required by BoxMOT.
    - `_tracks_to_dict(self, tracks: np.ndarray, class_mapping: Dict[int, str]) -> List[TrackedObject>`: Converts the tracker's output NumPy array of tracked objects into a list of standardized `TrackedObject` dictionaries.
    - `update(self, detections: list[Detection], class_mapping: dict[int, str], frame: np.ndarray) -> list[TrackedObject]`: The main method to update the tracker with new detections and retrieve the current set of tracked objects.

- **`vehicle_tracker_process` Function**
  - **Location:** `src/traffic_monitor/services/vehicle_tracker.py`
  - **Purpose:** This is the multiprocessing entry point for the vehicle tracking service. It continuously reads detection messages from an input queue, performs tracking using the `VehicleTracker` class, and publishes tracked vehicle messages to an output queue.
  - **Key Parameters:**
    - `config (Dict[str, Any])`: Configuration dictionary containing parameters for the tracker (e.g., `tracker_type`, `reid_model_path`, `tracker_config`).
    - `input_queue (Queue)`: Multiprocessing queue to receive `VehicleDetectionMessage` objects from upstream processes (e.g., `VehicleDetector`).
    - `output_queue (Queue)`: Multiprocessing queue to send `TrackedVehicleMessage` objects to downstream processes.
    - `shutdown_event (Event)`: An event flag used to signal the process to gracefully shut down.

## Vehicle Counter Component

### Overview

The Vehicle Counter component is responsible for counting vehicles that cross a predefined virtual line within the video feed. It processes tracked vehicle data and updates counts based on vehicle movements.

### Related Classes and Files

- **`Counter` Class**: Located in `src/traffic_monitor/services/vehicle_counter.py`
  - Manages the counting logic, including tracking vehicle positions relative to the counting line and maintaining counts.
- **`vehicle_counter_process` Function**: Located in `src/traffic_monitor/services/vehicle_counter.py`
  - The main entry point for the vehicle counter as a multiprocessing service. It initializes the `Counter` and handles message queues for input and output.
- **Configuration**: Counting line coordinates are defined in `src/traffic_monitor/config/settings.yaml` under the `vehicle_counter` section. The format is a list of line definitions, where each line is represented by two points: `[[x1, y1], [x2, y2]]`. For example: `counting_lines: - [[0, 750], [1920, 750]] - [[500, 0], [500, 1080]]`.
