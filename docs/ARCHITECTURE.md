## Architecture Design and Deployment Considerations

### VehicleTracker Component

**Purpose:** The `VehicleTracker` component is responsible for managing vehicle tracking logic using the BoxMOT library. It initializes the tracker and processes raw detections from the `VehicleDetector` into tracked objects.

**Area of Responsibility:**

- Initializing the BoxMOT tracker with specified configuration.
- Converting raw detection data into a format suitable for the tracker.
- Updating the tracker with new detections and retrieving tracked objects.
- Converting tracked objects from the tracker's internal format to a standardized output format (`TrackedObject` dictionaries).

**Compute Requirements:**

- Primarily CPU-bound, but can leverage GPU if `device` is set to `cuda` and a compatible ReID model is provided.
- Memory usage depends on the number of tracked objects and frame resolution.

**Storage Requirements:**

- Requires access to `reid_model_path` (e.g., `data/models/reid.pt`) for ReID models.
- Requires access to `tracker_config` (e.g., `src/traffic_monitor/config/bytetrack.yaml`) for tracker-specific configurations.

**Interfaces:**

- **Input:** Receives `VehicleDetectionMessage` objects from the `VehicleDetector` process via a multiprocessing queue. Each message contains frame data and a list of `Detection` objects.
- **Output:** Sends `TrackedVehicleMessage` objects to downstream processes (e.g., for visualization or data logging) via a multiprocessing queue. Each message includes tracked objects, frame metadata, and JPEG-encoded frame data.

**Dependencies:**

- **Internal:** `multiprocessing`, `cv2`, `numpy`, `loguru`, `pathlib`, `src.traffic_monitor.utils.custom_types`.
- **External:** `boxmot` library for tracking functionalities.

### Inter-process Communication

The system utilizes `multiprocessing.Queue` for inter-process communication between the `FrameGrabber`, `VehicleDetector`, and `VehicleTracker` components. The following queues are configured with a `maxsize` of 100 to accommodate processing loads and prevent frame drops:

- **`frame_grabber_output_queue`**: Transfers `FrameMessage` objects from `FrameGrabber` to `VehicleDetector`.
- **`vehicle_detector_output_queue`**: Transfers `VehicleDetectionMessage` objects from `VehicleDetector` to `VehicleTracker`.
- **`vehicle_tracker_output_queue`**: Transfers `TrackedVehicleMessage` objects from `VehicleTracker` to downstream processes (e.g., for visualization or data logging).

**Configuration:**

- `tracker_type` (str): Type of tracker to use (e.g., "bytetrack").
- `reid_model_path` (Path): Path to the ReID model weights.
- `device` (str): Device to run the tracker on (e.g., "cpu", "cuda").
- `half` (bool): Whether to use half-precision (FP16) for inference (typically for GPU).
- `per_class` (bool | None): Whether to track objects per class.
- `tracker_config` (Path): Path to the tracker-specific configuration file (e.g., `src/traffic_monitor/config/bytetrack.yaml`).

### Logging Configuration

**Purpose:** The logging system is configured to provide clear and actionable insights into the application's runtime behavior, facilitating debugging and monitoring.

**Area of Responsibility:**

- Centralized logging setup via `src/traffic_monitor/utils/logging_config.py`.
- Customizable logging levels and formats.
- Output to both console and file for comprehensive record-keeping.
- **Multiprocessing Support:** Each child process (VehicleDetector, VehicleTracker, LPDetector, OCRReader) independently sets up logging to ensure proper log output from all processes.

**Configuration:**

- The default terminal output level is set to `INFO` to minimize verbose debug messages in the console.
- Logging parameters such as level, format, file path, rotation, retention, and compression can be configured via `loguru` section in `src/traffic_monitor/config/settings.yaml`.
- **Process-Specific Logging:** Each process function calls `setup_logging()` at startup to ensure consistent logging configuration across all processes.

**Dependencies:**

- Each multiprocessing service (`lp_detector_process`, `ocr_reader_process`, etc.) must import and call `setup_logging()` to initialize logging properly.

### Vehicle Counter Service

- **Technical Requirements**: Handles vehicle counting based on predefined counting lines. Processes `TrackedVehicleMessage` and outputs `VehicleCountMessage`.
- **Area of Responsibility**: Detecting when a tracked vehicle crosses a designated line and maintaining counts by class and total.
- **Compute**: Primarily CPU-bound, performing geometric calculations and dictionary operations.
- **Storage**: Stores current vehicle positions and counted track IDs in memory.
- **Interface toward other components**:
  - **Input**: Receives `TrackedVehicleMessage` from `Vehicle Tracker` via an input queue.
  - **Output**: Sends `VehicleCountMessage` to `Main Supervisor` via an output queue.
- **Dependency to other components**: Depends on `shapely` for geometric operations and `loguru` for logging. Receives data from `Vehicle Tracker`.
