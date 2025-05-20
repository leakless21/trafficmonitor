**Traffic Monitoring System: Prototype Build Guide (src Layout & Iterative Approach)**

**1. Introduction & Project Goal**

*   **Goal:** To develop a prototype system that processes video input to detect, track, and count vehicles, and perform license plate recognition.
*   **Approach:** We'll build it as a series of interconnected Python processes (simulating microservices) within a `src` layout. Initially, these processes will communicate using Python's built-in `multiprocessing.Queue` for simplicity. This modular design allows for easier testing and understanding of each component.
*   **Core Functionality (The "What"):**
    1.  **Video Ingestion:** Read video from a file or live webcam.
    2.  **Vehicle Detection:** Identify vehicles (e.g., cars, trucks, buses) in each video frame.
    3.  **Vehicle Tracking:** Follow each detected vehicle from one frame to the next, assigning it a consistent ID.
    4.  **License Plate Detection:** Locate the license plate area on each tracked vehicle.
    5.  **License Plate Recognition (OCR):** Read the characters from the detected license plate.
    6.  **Vehicle Counting:** Count vehicles as they cross a predefined virtual line in the video.
    7.  **Visualization:** Display the video feed with all detections, tracks, plate numbers, and counts overlaid.

**2. Understanding the `src` Layout**

*   **Purpose:** The `src` (source) layout is a standard way to organize Python projects. Your main application code lives inside a directory named `src`. This clearly separates your code from other project files like configuration (`pyproject.toml`), tests, documentation, etc.
*   **Structure:**
    ```
    traffic_monitoring_prototype/  <-- Project Root
    ├── src/
    │   └── traffic_monitor/       <-- Your main Python package
    │       ├── __init__.py          # Makes 'traffic_monitor' a package
    │       ├── main_supervisor.py   # Orchestrates all processes
    │       ├── services/            # Sub-package for individual processing modules
    │       │   ├── __init__.py      # Makes 'services' a package
    │       │   ├── frame_grabber.py
    │       │   ├── vehicle_detector.py
    │       │   └── ... (other service .py files)
    │       └── shared_utils/        # Sub-package for common helper functions
    │           ├── __init__.py      # Makes 'shared_utils' a package
    │           ├── logging_config.py
    │           └── constants.py       # For storing shared, code-intrinsic constant values (e.g., message keys)
    │           └── custom_types.py    # For defining shared message structures (TypedDicts/Pydantic models)
    │       └── config/              # For storing configuration files
    │           └── settings.yaml    # External configuration (model paths, thresholds, etc.)
    ├── tests/                     # For PyTest unit and integration tests
    │   ├── __init__.py
    │   └── ... (test files)
    ├── weights/                   # Stores your pre-trained model files (.pt)
    ├── output/                    # For saving processed videos or logs (optional)
    ├── pyproject.toml             # Pixi project configuration
    └── .envrc                     # Direnv environment configuration
    ```
*   **`__init__.py` files:** These empty files are crucial. They tell Python that the directories (`traffic_monitor`, `services`, `shared_utils`) are "packages," allowing you to import modules from them using dot notation (e.g., `from traffic_monitor.services import frame_grabber`).
*   **Documentation Files (Create these as you go):**
    *   `REQUIREMENTS.md`: Detail functional and non-functional requirements.
    *   `ARCHITECTURE.MD`: Describe the overall system architecture, data flow, and component responsibilities.
    *   `COMPONENT_{SERVICE_NAME}_DOCS.md`: For each service, document its specific role, inputs, outputs, and configurations.
    *   `GAP_ANALYSIS.md`: Track bugs and missing features during testing.

**3. System Architecture (Initial Prototype using `multiprocessing.Queue`)**

*   **Concept:** A central "Supervisor" script launches each core piece of functionality as an independent Python process. These processes communicate by passing data through shared "Queues." Think of it like a factory assembly line where items (video frames and their processed data) move from one station (process) to the next via conveyor belts (queues).
*   **Components:**
    *   **`src/traffic_monitor/main_supervisor.py`:** The "factory manager." It creates the queues, loads configurations, creates a shared shutdown event, and starts all the worker processes.
    *   **`multiprocessing.Queue` Instances:** The "conveyor belts." These are special Python objects that allow safe communication of data between processes. Crucially, set a `maxsize` to prevent memory issues due to backpressure.
    *   **`multiprocessing.Event` Instance:** A shared event (`shutdown_event`) created by the supervisor and passed to all services to signal a graceful shutdown.
    *   **Service Processes (e.g., `FrameGrabberProcess`, `VehicleDetectorProcess`):** Each worker "station" that performs a specific task. Each runs its logic defined in a corresponding `.py` file in `src/traffic_monitor/services/`.

**4. Core Technologies**

*   **Python 3.8+:** The programming language.
*   **`multiprocessing.Queue`:** For communication between your Python processes in this initial prototype.
*   **Ultralytics YOLOv8/11:** For deep learning-based object detection (vehicles and license plates). You'll use your `.pt` model weight files.
*   **BoxMOT:** A library for object tracking (to follow vehicles).
*   **PaddleOCR:** An OCR library for reading license plate text.
*   **OpenCV (`cv2`):** For all video and image processing tasks (reading frames, drawing, displaying).
*   **NumPy:** For efficient numerical operations, especially with image data.
*   **Loguru:** For easy and powerful logging, helping you see what each process is doing. Configure it to include process names for clarity.
*   **Pixi:** To manage your project's Python environment and dependencies.
*   **Direnv:** To automatically load environment configurations when you enter the project directory.
*   **Ruff:** For code formatting and linting (to be integrated).
*   **Pytest:** For writing and running unit and integration tests (to be integrated).

**5. Detailed Project Setup Steps**

1.  **Install Tools:** Ensure you have Python (Pixi will manage its version), Pixi, Direnv, and Git.
2.  **Create Directory Structure:** Follow the structure shown in Section 2 ("Understanding the `src` Layout"). Create placeholder `REQUIREMENTS.md`, `ARCHITECTURE.MD`, etc.
3.  **`pyproject.toml` (in project root):**
    *   Define project metadata (name, version).
    *   Specify dependencies (Python, opencv-python, ultralytics, numpy, loguru initially). Add others like `boxmot`, `paddleocr`, `pyyaml` (for config) later using `pixi add <package>`. Add `ruff`, `pytest` as dev dependencies (`pixi add --dev <package>`).
    *   Create a `[tasks.start]` entry: `start = "python -m traffic_monitor.main_supervisor"`.
    *   Add Ruff tasks: `lint = "ruff check ."` and `format = "ruff format ."`.
4.  **`.envrc` (in project root):**
    *   `layout pixi`
    *   Optionally, `export PYTHONPATH="${PWD}/src:${PYTHONPATH}"` if you face import issues, though `python -m` should generally handle it.
    *   `direnv allow` to activate.
5.  **Install Dependencies:** Run `pixi install` in the project root.
6.  **Place Model Weights:** Copy your `vehicle_detector.pt` and `plate_detector.pt` files into the `weights/` directory.
7. **Configuration File (Example: `src/traffic_monitor/config/settings.yaml`):**
    *   Create a YAML file to store configurations for all services (model paths, thresholds, video source, line coordinates, etc.). The supervisor will load this and pass relevant sections to each service.
    *   **This is for parameters you might want to change without altering Python code.**
8.  **Shared Utilities (Example: `src/traffic_monitor/shared_utils/logging_config.py`):**
    *   Create a function to set up Loguru with a consistent format, including process names/IDs (e.g., using `{process.name}`).
    *   Example: `logger.add(sys.stderr, format="{time} {level} [{process.name}] {message}")`
9.  **Shared Constants (Example: `src/traffic_monitor/shared_utils/constants.py`):**
    *   Define any shared constant values that are intrinsic to the code's logic and not typically configured externally. Examples: standard dictionary keys for messages (e.g., `FRAME_ID_KEY = "frame_id"`), default internal parameters if not exposed in `settings.yaml`.
    *   **Purpose:** Avoids "magic strings/numbers" scattered in your code, improving readability and making refactoring safer (e.g., change a key in one place).
    *   **For a beginner:** While `settings.yaml` handles external configuration, this file helps keep the Python code itself cleaner. If you have very few such constants, you might define them in the module where they are used, but a central `constants.py` is better as the project grows.
10. **Shared Types (Example: `src/traffic_monitor/shared_utils/custom_types.py`):**
    *    Define `TypedDict` or Pydantic models for the structure of messages passed between services. **These are Python code and cannot be defined in `settings.yaml`.**
    *    **Why this is crucial (even for a beginner aiming for industry practice):**
        *   **Clarity & Explicit Contracts:** Clearly defines the "shape" of data your services expect and produce (e.g., `class FrameMessage(TypedDict): frame_id: str; timestamp: float; ...`). This is vital for understanding data flow.
        *   **Error Prevention:** Helps catch errors early. Static type checkers (like MyPy, integrated with Ruff) can find issues with `TypedDict` usage before runtime. Pydantic models offer runtime validation.
        *   **Maintainability:** Makes it easier to update message structures and understand the impact of changes.
    *   **Recommendation for prototype:** Start with `TypedDict` from Python's `typing` module. It's simple and provides significant benefits with static analysis.

**6. Service Modules: Detailed Descriptions**

Each service is a Python module (e.g., `frame_grabber.py`) within `src/traffic_monitor/services/`. It will contain a primary function (e.g., `frame_grabber_process`) that is run by the `main_supervisor.py` in a separate process.

**General Principles for All Services:**
*   **Function Signature:** Will typically include `config: dict`, relevant `input_q` (or queues), `output_q` (or None), and the shared `shutdown_event: multiprocessing.Event`.
*   **Error Handling:** Implement robust `try...except Exception as e:` blocks within the main processing loop. Log exceptions with context (e.g., `frame_id`) using `logger.exception()`. Decide whether to skip a problematic message or terminate.
*   **Graceful Shutdown:** Periodically check `if shutdown_event.is_set(): break` in the main loop. Timeouts on `input_q.get()` allow for this check.
*   **Resource Management:** Use `try...finally` blocks to ensure resources (models, file handles, camera captures) are released cleanly, especially when the service exits.
*   **Logging:** Log key events: start, stop, message received (e.g., `frame_id`), processing summary, message sent, errors.
*   **Type Hinting & Docstrings:** Consistently use Python type hints for function arguments and return values. Write clear docstrings for all modules, classes, and functions.
*   **Serialization:** Base64 encoded JPEGs for image data is good. For other complex data in messages, consider serializing to JSON strings if moving to Redis later.

---

**Service 1: Frame Grabber (`src/traffic_monitor/services/frame_grabber.py`)**

*   **Purpose:** To read video frames from a source (file or webcam) and make them available to other services.
*   **Function Signature (example):** `def frame_grabber_process(config: dict, output_q: multiprocessing.Queue, shutdown_event: multiprocessing.Event):`
*   **Input:**
    *   `config`: A Python dictionary containing settings like `video_source` (e.g., `"path/to/video.mp4"` or `0` for webcam), `camera_id`, and potentially desired FPS.
*   **Output (Message put onto `output_q`):**
    *   A Python dictionary per frame:
        ```
        {
            "frame_id": (string) # Unique ID for this frame
            "camera_id": (string) # Identifier for the video source
            "timestamp": (float) # Time of capture
            "frame_data_jpeg": (string) # Frame image: OpenCV BGR -> JPEG -> Base64 encoded string
            "frame_height": (int)
            "frame_width": (int)
        }
        ```
    *   If end of video file or `shutdown_event` is set: Puts `None` onto `output_q` to signal downstream services to shut down.
*   **Key Logic:**
    1.  Initialize `cv2.VideoCapture` using `config["video_source"]`.
    2.  Use a `try...finally` block to ensure `video_capture.release()` is called.
    3.  Loop (check `not shutdown_event.is_set()`):
        *   Read a frame. If `ret` is false (no frame), send `None` to `output_q` and exit the loop.
        *   Generate `frame_id` (e.g., using `uuid.uuid4()`) and `timestamp`.
        *   Convert the OpenCV frame (NumPy array) to a JPEG byte string (`cv2.imencode`), then encode that to a Base64 string. This makes it easy to pass through queues.
        *   Create the message dictionary with all data.
        *   Use `output_q.put(message_dict, timeout=1)` to send the message. Handle `queue.Full` exceptions if necessary, or rely on timeout to re-check shutdown event.
    3.  Release `cv2.VideoCapture` when done (in `finally` block).
*   **Imports needed:** `cv2`, `base64`, `time`, `uuid`, `multiprocessing`, `loguru`, `queue` (for `queue.Full` exception).

---

**Service 2: Vehicle Detector (`src/traffic_monitor/services/vehicle_detector.py`)**

*   **Purpose:** To take raw frames and identify vehicles within them.
*   **Function Signature:** `def vehicle_detector_process(config: dict, input_q: multiprocessing.Queue, output_q: multiprocessing.Queue, shutdown_event: multiprocessing.Event):`
*   **Input (Message received from `input_q`):** The dictionary produced by the Frame Grabber.
*   **Output (Message put onto `output_q`):**
    *   A Python dictionary per frame:
        ```
        {
            # --- Passthrough fields from input message ---
            "frame_id": (string), "camera_id": (string), "timestamp": (float),
            "frame_data_jpeg": (string), "frame_height": (int), "frame_width": (int),
            # --- New fields ---
            "detections": (list) # List of dictionaries, one for each detected vehicle
                [
                    {
                        "bbox_xyxy": [int, int, int, int], # Pixel coordinates [x1, y1, x2, y2]
                        "confidence": (float), # Detection confidence score
                        "class_id": (int),     # Numeric ID for the detected class
                        "class_name": (string) # Human-readable class name (e.g., "car")
                    },
                    ...
                ]
        }
        ```
    *   If input message is `None` or `shutdown_event` is set: Puts `None` onto `output_q`.
*   **Key Logic:**
    1.  **Initialization (once at process start):**
        *   Load the Ultralytics YOLOv8/11 vehicle detection model: `model = YOLO(config["model_path"])`.
        *   The `config` dictionary will provide `model_path` (e.g., `"weights/vehicle_detector.pt"`), `conf_threshold`, `iou_threshold`, and a `class_mapping` dictionary (e.g., `{0: "person", 2: "car", ...}`).
    2.  Loop (check `not shutdown_event.is_set()`):
        *   Get a message from `input_q` with timeout (e.g., `message = input_q.get(timeout=0.1)`). Handle `queue.Empty` by continuing to allow shutdown check.
        *   If `message is None`, send `None` to `output_q` and exit.
        *   Implement `try...except` for message processing.
        *   Decode `message["frame_data_jpeg"]` to get the OpenCV frame.
        *   Perform inference: `results = model.predict(frame, conf=..., iou=...)`.
        *   Iterate through `results.boxes.data` (or the appropriate attribute for your Ultralytics version) to get bounding boxes (`xyxy`), confidence scores, and class IDs.
        *   Use the `class_mapping` from `config` to get the `class_name`.
        *   Build the `detections` list.
        *   Create the output message dictionary (including passthrough fields).
        *   Use `output_q.put(output_message, timeout=1)`. Handle `queue.Full`.
*   **Imports needed:** `ultralytics.YOLO`, `cv2`, `base64`, `numpy`, `multiprocessing`, `loguru`, `queue`.

---

**Service 3: Vehicle Tracker (`src/traffic_monitor/services/vehicle_tracker.py`)**

*   **Purpose:** To take the detected vehicles from each frame and assign a consistent ID to each vehicle as it moves across frames.
*   **Function Signature:** `def vehicle_tracker_process(config: dict, input_q: multiprocessing.Queue, output_q: multiprocessing.Queue, shutdown_event: multiprocessing.Event):`
*   **Input (Message received from `input_q`):** The dictionary produced by the Vehicle Detector.
*   **Output (Message put onto `output_q`):**
    *   A Python dictionary per frame:
        ```
        {
            # --- Passthrough fields ---
            "frame_id": (string), ..., "frame_width": (int),
            # --- New fields ---
            "tracked_objects": (list) # List of dictionaries, one for each tracked vehicle
                [
                    {
                        "bbox_xyxy": [int, int, int, int],
                        "track_id": (int),     # Unique ID assigned by the tracker
                        "class_id": (int),
                        "class_name": (string),
                        "confidence": (float)  # Detection confidence (passthrough)
                    },
                    ...
                ]
        }
        ```
    *   If input message is `None` or `shutdown_event` is set: Puts `None` onto `output_q`.
*   **Key Logic:**
    1.  **Initialization (once at process start):**
        *   Initialize a BoxMOT tracker object (e.g., `BoTSORT`, `ByteTrack`). Example: `tracker = BoTSORT(model_weights=Path(config["osnet_model_path"]), device='cpu', per_class=False)`.
        *   Ensure `config` provides paths for any necessary tracker-specific weights (like OSNet for BoTSORT if using ReID).
        *   The `config` will pass any necessary tracker parameters and the `class_mapping`.
    2.  Loop (check `not shutdown_event.is_set()`):
        *   Get `detection_message` from `input_q` (with timeout). Handle `queue.Empty`.
        *   If `detection_message is None`, send `None` to `output_q` and exit.
        *   Implement `try...except` for message processing.
        *   Extract the `detections` list from `detection_message`.
        *   Convert these detections into the NumPy array format required by BoxMOT (usually `[x1, y1, x2, y2, score, class_id]`).
        *   Decode `detection_message["frame_data_jpeg"]` to get the OpenCV frame (some trackers use appearance features from the frame).
        *   Update the tracker: `tracks = tracker.update(detections_numpy_array, frame)`.
        *   The `tracks` output from BoxMOT is usually a NumPy array like `[x1, y1, x2, y2, track_id, score, class_id]`.
        *   Iterate through `tracks` to build the `tracked_objects` list, using `class_mapping` for `class_name`.
        *   Create the output message dictionary.
        *   Use `output_q.put(output_message, timeout=1)`. Handle `queue.Full`.
*   **Dependencies to add:** `boxmot`. `cv2`, `base64`, `numpy`, `multiprocessing`, `loguru`, `queue`, `pathlib`.

---

**Service 4: License Plate Detector (`src/traffic_monitor/services/lp_detector.py`)**

*   **Purpose:** For each tracked vehicle, to specifically find the bounding box of its license plate.
*   **Function Signature:** `def lp_detector_process(config: dict, input_q: multiprocessing.Queue, output_q: multiprocessing.Queue, shutdown_event: multiprocessing.Event):`
*   **Input (Message received from `input_q`):** The dictionary produced by the Vehicle Tracker.
*   **Output (Message put onto `output_q`):** *Multiple messages can be produced from one input message if multiple vehicles have plates.* Each output message represents *one* potential license plate found.
    *   A Python dictionary per detected plate:
        ```
        {
            # --- Passthrough fields (from the original frame context) ---
            "frame_id": (string), "camera_id": (string), "timestamp": (float),
            "frame_data_jpeg": (string), # Original full frame, crucial for OCR later
            # --- Vehicle and Plate Info ---
            "vehicle_track_id": (int),    # The track_id of the vehicle this plate belongs to
            "vehicle_class_name": (string), # e.g., "car"
            "plate_bbox_orig": [int, int, int, int], # LP bbox in *original full frame* coordinates
            "plate_confidence": (float) # Confidence of the LP detection
        }
        ```
    *   If input message is `None` or `shutdown_event` is set: Puts `None` onto `output_q`.
*   **Key Logic:**
    1.  **Initialization:** Load the YOLOv8/11 license plate detection model: `lp_model = YOLO(config["model_path"])`. `config` provides `model_path` (e.g., `"weights/plate_detector.pt"`), `conf_threshold`.
    2.  Loop (check `not shutdown_event.is_set()`):
        *   Get `tracked_vehicle_message` from `input_q` (with timeout). Handle `queue.Empty`.
        *   If `tracked_vehicle_message is None`, send `None` to `output_q` and exit.
        *   Implement `try...except` for message processing.
        *   Decode `tracked_vehicle_message["frame_data_jpeg"]` to get the `original_frame`.
        *   Iterate through each `vehicle` in `tracked_vehicle_message["tracked_objects"]`:
            *   Get the vehicle's bounding box (`vehicle_bbox_xyxy`).
            *   Crop this vehicle region from the `original_frame`: `vehicle_crop = original_frame[y1:y2, x1:x2]`.
            *   Perform license plate detection *on this `vehicle_crop`*: `lp_results = lp_model.predict(vehicle_crop, conf=...)`.
            *   If a license plate is found in `lp_results`:
                *   Get its bounding box (`lp_bbox_on_crop`) which is relative to `vehicle_crop`.
                *   **Crucial:** Convert `lp_bbox_on_crop` coordinates to be relative to the `original_frame`. (e.g., `lp_x1_orig = vehicle_bbox_xyxy[0] + lp_bbox_on_crop[0]`). Double-check this math.
                *   Create the output message dictionary for this specific plate.
                *   Use `output_q.put(output_message, timeout=1)`. Handle `queue.Full`.
*   **Imports needed:** `ultralytics.YOLO`, `cv2`, `base64`, `numpy`, `multiprocessing`, `loguru`, `queue`.

---

**Service 5: OCR Reader (`src/traffic_monitor/services/ocr_reader.py`)**

*   **Purpose:** To take the cropped image of a license plate and read the text characters on it.
*   **Function Signature:** `def ocr_reader_process(config: dict, input_q: multiprocessing.Queue, output_q: multiprocessing.Queue, shutdown_event: multiprocessing.Event):`
*   **Input (Message received from `input_q`):** The dictionary produced by the License Plate Detector (one per plate).
*   **Output (Message put onto `output_q`):**
    *   A Python dictionary per OCR attempt:
        ```
        {
            # --- Passthrough fields ---
            "frame_id": (string), "camera_id": (string), "timestamp": (float),
            "vehicle_track_id": (int),
            # --- New fields ---
            "license_plate_text": (string), # The recognized text
            "ocr_confidence": (float)    # Confidence of the OCR result
        }
        ```
    *   If input message is `None` or `shutdown_event` is set: Puts `None` onto `output_q`.
*   **Key Logic:**
    1.  **Initialization:** Initialize PaddleOCR: `ocr_engine = PaddleOCR(use_angle_cls=True, lang=config.get("lang", "en"), show_log=False)`. `config` provides `lang`.
    2.  Loop (check `not shutdown_event.is_set()`):
        *   Get `lp_candidate_message` from `input_q` (with timeout). Handle `queue.Empty`.
        *   If `lp_candidate_message is None`, send `None` to `output_q` and exit.
        *   Implement `try...except` for message processing.
        *   Decode `lp_candidate_message["frame_data_jpeg"]` to get `original_frame`.
        *   Get `lp_candidate_message["plate_bbox_orig"]`.
        *   Crop the license plate image from the `original_frame` using these coordinates: `plate_image_crop = original_frame[y1:y2, x1:x2]`.
        *   Perform OCR on `plate_image_crop`: `ocr_result = ocr_engine.ocr(plate_image_crop, cls=True)`.
        *   Parse the `ocr_result` (PaddleOCR's output format) to extract the most likely text string and its confidence score. Handle cases where no text is found.
        *   Create the output message dictionary.
        *   Use `output_q.put(output_message, timeout=1)`. Handle `queue.Full`.
*   **Dependencies to add:** `paddleocr`, `paddlepaddle`. `cv2`, `base64`, `numpy`, `multiprocessing`, `loguru`, `queue`.

---

**Service 6: Vehicle Counter (`src/traffic_monitor/services/vehicle_counter.py`)**

*   **Purpose:** To count vehicles that cross a predefined virtual line in the scene.
*   **Function Signature:** `def vehicle_counter_process(config: dict, input_q: multiprocessing.Queue, output_q: multiprocessing.Queue, shutdown_event: multiprocessing.Event):`
*   **Input (Message received from `input_q`):** The dictionary from the Vehicle Tracker (`tracked_vehicles_q`).
*   **Output (Message put onto `output_q`):** *Published periodically or when counts change significantly.*
    *   A Python dictionary:
        ```
        {
            "camera_id": (string),
            "timestamp": (float), # Timestamp of this count update
            "counts_by_class": (dict), # e.g., {"car": 10, "truck": 5}
            "total_count": (int)
        }
        ```
*   **Key Logic:**
    1.  **Initialization:**
        *   Define the counting line: `line_points = config["counting_line"]` (e.g., `[(x1,y1), (x2,y2)]`).
        *   Initialize state variables: `vehicle_last_pos = {}` (to store the last known position of each track ID's center point), `counted_ids_this_session = set()` (to prevent double-counting a single crossing), `current_counts = {"total": 0}`.
    2.  Loop (check `not shutdown_event.is_set()`):
        *   Get `tracked_vehicle_message` from `input_q` (with timeout). Handle `queue.Empty`.
        *   If `tracked_vehicle_message is None`, consider sending a final count update if desired, then exit loop (no need to propagate `None` unless Visualizer specifically needs a final count update signal).
        *   Implement `try...except` for message processing.
        *   For each `vehicle` in `tracked_vehicle_message["tracked_objects"]`:
            *   Get `track_id`, `class_name`, and `bbox_xyxy`.
            *   Calculate a representative point for the vehicle (e.g., bottom-center of its bounding box).
            *   If `track_id` is in `vehicle_last_pos`:
                *   You have the vehicle's `previous_point` and `current_point`.
                *   Implement a **line intersection check**: Does the line segment from `previous_point` to `current_point` intersect the predefined `counting_line`? (Search for "line segment intersection algorithm"; libraries like Shapely can help, or implement with basic geometry and NumPy).
                *   If it intersects AND `track_id` is not in `counted_ids_this_session` (or hasn't been counted recently):
                    *   Increment `current_counts["total"]` and `current_counts.setdefault(class_name, 0) += 1`.
                    *   Add `track_id` to `counted_ids_this_session`.
                    *   Set a flag `counts_have_changed = True`.
            *   Update `vehicle_last_pos[track_id]` with the `current_point`.
        *   If `counts_have_changed` (or after a certain time interval):
            *   Create the count update message.
            *   Use `output_q.put(count_update_message, timeout=1)`. Handle `queue.Full`.
            *   Reset `counts_have_changed` flag.
*   **Imports needed:** `numpy` (potentially for line math), `multiprocessing`, `loguru`, `queue`.

---

**Service 7: Visualizer (`src/traffic_monitor/services/visualizer.py`)**

*   **Purpose:** To gather all processed information and display it as an annotated video feed. This is often the most complex in terms of managing and synchronizing data from multiple sources.
*   **Function Signature:** `def visualizer_process(config: dict, tracked_vehicles_q_in: multiprocessing.Queue, ocr_results_q_in: multiprocessing.Queue, counting_updates_q_in: multiprocessing.Queue, shutdown_event: multiprocessing.Event):`
    *   *(Note: It gets `frame_data_jpeg` via the `tracked_vehicles_q_in` message).*
*   **Input (Messages from multiple queues):**
    *   From `tracked_vehicles_q_in`: Contains frame data, bounding boxes, track IDs.
    *   From `ocr_results_q_in`: Contains license plate text for specific track IDs.
    *   From `counting_updates_q_in`: Contains the latest vehicle counts.
*   **Output:** Displays an annotated video window using `cv2.imshow()`.
*   **Key Logic:**
    1.  **Initialization:**
        *   Data buffers to hold the latest information:
            *   `ocr_data_for_tracks = {}` (stores `{track_id: {"text": "ABC123", "confidence": 0.85, "timestamp": ...}}` to age out old OCR data).
            *   `latest_vehicle_counts = None`.
            *   `fps_calculator = FPSCounter()` (a helper class to calculate FPS).
    2.  Loop (check `not shutdown_event.is_set()`):
        *   **Non-blockingly get messages from all input queues:**
            *   Use `tracked_vehicles_q_in.get(block=True, timeout=0.01)` (or `get_nowait()` with try-except `queue.Empty`). If a message (let's call it `track_msg`) is received:
                *   This `track_msg` is the "driver" for rendering a frame because it contains `frame_data_jpeg` and `tracked_objects`.
                *   If `track_msg is None`, set `shutdown_event.set()` to signal all other services, then break the loop and clean up.
                *   Decode `track_msg["frame_data_jpeg"]` to get the `current_display_frame`.
                *   Start with this `current_display_frame`.
                *   Draw all `tracked_objects` from `track_msg` onto it (bboxes, track IDs, class names).
                *   For each `track_id` in `track_msg["tracked_objects"]`, check `ocr_data_for_tracks` to see if recent OCR text is available. If yes, draw it near the vehicle.
                *   If `latest_vehicle_counts` is available, draw the count statistics on the frame.
                *   Draw calculated FPS.
                *   `cv2.imshow("Traffic Monitor", current_display_frame)`.
                *   Handle `cv2.waitKey(1)` for display updates and 'q' to quit. If 'q' is pressed, set `shutdown_event.set()`, break the loop, and allow cleanup.
            *   Use `ocr_results_q_in.get_nowait()`: If an OCR message arrives (and is not `None`), update `ocr_data_for_tracks` with the new text and timestamp for that `vehicle_track_id`.
            *   Use `counting_updates_q_in.get_nowait()`: If a count message arrives (and is not `None`), update `latest_vehicle_counts`.
        *   Implement logic to periodically clear very old entries from `ocr_data_for_tracks` based on their timestamp to prevent memory growth.
        *   If the `track_msg` from `tracked_vehicles_q_in` is `None` (propagated from upstream services that already handled shutdown_event), break the loop and clean up.
    3.  `cv2.destroyAllWindows()` when done (in `finally` block).
*   **Imports needed:** `cv2`, `base64`, `numpy`, `multiprocessing`, `loguru`, `collections.deque` (optional for managing frame IDs), `queue`.

---

**7. Supervisor Script (`src/traffic_monitor/main_supervisor.py`)**

*   **Purpose:** The main script you run. It initializes all the queues and starts each service function as a separate `multiprocessing.Process`.
*   **Key Logic:**
    1.  Import all service functions (e.g., `from .services.frame_grabber import frame_grabber_process`).
    2.  Import `load_config` function (e.g., from `shared_utils.config_loader`).
    3.  `multiprocessing.set_start_method('spawn', force=True)`: Call this early, especially for cross-platform compatibility.
    4.  Create a `shutdown_event = multiprocessing.Event()`.
    5.  Load main configuration (e.g., from `src/traffic_monitor/config/settings.yaml`).
    6.  Create all the `multiprocessing.Queue(maxsize=SOME_REASONABLE_NUMBER)` instances. `maxsize` is crucial to prevent a fast producer from overwhelming a slow consumer and using up all memory.
    7.  Prepare `config` dictionaries for each service by extracting relevant parts from the loaded main configuration. Include a `service_name` in each for better logging.
    8.  For each service:
        *   Create a `multiprocessing.Process(target=service_function, name="ServiceName", args=(config_dict, input_q_or_tuple_of_qs, output_q_or_None, shutdown_event))`.
        *   Store the process object in a list.
    9.  Start all processes: `for p in process_list: p.start()`.
    10. Implement a loop to keep the main supervisor process alive. It can wait for the `shutdown_event` or handle `KeyboardInterrupt`.
        ```python
        # Example supervisor main loop
        try:
            while not shutdown_event.is_set():
                # Check if all vital processes are alive; if not, trigger shutdown
                # all_alive = all(p.is_alive() for p in process_list_subset_considered_vital)
                # if not all_alive:
                #    logger.error("A vital process died. Initiating shutdown.")
                #    shutdown_event.set()
                #    break
                time.sleep(0.5) # Keep main thread alive, periodically check event
        except KeyboardInterrupt:
            logger.info("Ctrl+C received. Initiating shutdown.")
            shutdown_event.set()
        finally:
            logger.info("Supervisor initiating cleanup...")
            for p in process_list:
                if p.is_alive():
                    logger.info(f"Waiting for {p.name} to join...")
                    p.join(timeout=5) # Give process time to exit gracefully
                if p.is_alive():
                    logger.warning(f"{p.name} did not join in time. Terminating.")
                    p.terminate() # Force terminate if stuck
                    p.join() # Wait for termination
            
            # Clean up queues
            # for q_name, q_instance in all_queues_dict.items():
            #    logger.info(f"Closing queue: {q_name}")
            #    q_instance.close()
            #    q_instance.join_thread()
            logger.info("All processes joined. Supervisor exiting.")
        ```
    11. Implement graceful shutdown: On `KeyboardInterrupt` (Ctrl+C) or if `shutdown_event` is set by a service (like Visualizer), set `shutdown_event.set()`. Then iterate through the process list, call `p.join(timeout=...)` to wait for them to exit. If they don't exit cleanly, `p.terminate()` and then `p.join()` again.
    12. After all processes are joined, close all queues (`q.close()`, `q.join_thread()`).

---

**8. Running the Prototype**

1.  **Navigate to Project Root:** Open your terminal in the `traffic_monitoring_prototype/` directory.
2.  **Activate Pixi Environment:** Direnv should do this automatically. If not, `pixi shell`.
3.  **Run Supervisor:** `pixi run start` (which should execute `python -m traffic_monitor.main_supervisor`).
    *   The `python -m traffic_monitor.main_supervisor` command tells Python to run `main_supervisor.py` as a module that's part of the `traffic_monitor` package (which Python finds in your `src` directory).

---

**9. Iterative Development and Debugging Tips**

*   **One Service at a Time:**
    1.  Start by implementing only the `FrameGrabberProcess` and a very simple "consumer" in your supervisor that just gets messages from `raw_frames_q` and logs them. Run this to make sure frame grabbing and basic queue communication work.
    2.  Then implement `VehicleDetectorProcess`. Have the supervisor consume from `vehicle_detections_q` and log results.
    3.  Continue this pattern, adding one service at a time and verifying its output before moving to the next.
*   **Logging, Logging, Logging:** Use `loguru` extensively. In each service, log:
    *   When it starts/stops.
    *   Key configuration values it's using (be careful not to log sensitive data if any).
    *   When it receives a message (maybe log the `frame_id`).
    *   A summary of the processing it did (e.g., "Detected 3 vehicles").
    *   When it sends a message.
    *   Any errors encountered (use `logger.exception()` for stack traces).
    *   Including `process.name` or a configured `service_name` in your log format helps distinguish logs from different services.
*   **Small Test Videos:** Use short, simple video clips for initial testing.
*   **Queue `maxsize`:** Setting a `maxsize` on your queues is vital. If one service is much slower than its producer, an unbounded queue will consume all your RAM. A `maxsize` creates "backpressure." Start with a small `maxsize` (e.g., 10-20) and tune if necessary.
*   **Error Handling:** Wrap potentially problematic code (e.g., `queue.get()`, model inference, file operations) in `try...except` blocks within each service. Log errors with context.
*   **Debugging `multiprocessing`:** Debugging multiple processes can be tricky. Rely heavily on logging. Some IDEs offer limited support for debugging child processes.
*   **Use `pytest` Early:** Start writing simple unit tests for helper functions or specific logic within services as you develop them. This can be done in the `tests/` directory.

---

**10. Next Steps: Towards a More Robust System (Beyond this Prototype)**

Once the core logic of all services is working with `multiprocessing.Queue`:

1.  **Transition to Redis Streams:**
    *   **Why?** `multiprocessing.Queue` only works for processes on the *same machine* and managed by the same parent Python script. Redis is an external message broker that allows true decoupling – services can run anywhere and don't need to know about each other directly, only about Redis. It also offers persistence and better scalability features.
    *   **How?**
        *   Run a Redis server (e.g., using Docker: `docker run -d -p 6379:6379 redis`).
        *   In each service, replace `multiprocessing.Queue` `put()` calls with `redis_client.xadd("your_stream_name", message_dict)`.
        *   Replace `get()` calls with `redis_client.xreadgroup(...)` (which is more complex as it involves consumer groups and message acknowledgments (`xack`)).
        *   Message data sent to Redis should ideally be simple key-value pairs where values are strings/bytes. Complex Python objects within your message dict might need to be serialized to JSON strings (e.g., `json.dumps(detections_list)`).
2.  **Containerize with Docker:**
    *   **Why?** Docker packages each service and its dependencies into a portable "container." This ensures it runs the same way anywhere and simplifies deployment.
    *   **How?**
        *   Write a `Dockerfile` for each service. This file specifies how to build an image for that service (e.g., copy the Python code from `src/traffic_monitor/services/`, install Python dependencies like OpenCV and Ultralytics).
        *   Write a `docker-compose.yml` file in your project root. This file defines all your services (including your Redis service) and how they connect and run together. It replaces your `main_supervisor.py`. Each service in `docker-compose.yml` will run its respective Python script (e.g., `python -m traffic_monitor.services.frame_grabber`).
        *   Configuration (like Redis hostname, model paths inside the container) will be passed via environment variables in the `docker-compose.yml` file.
3.  **Formalize Documentation:** Keep `REQUIREMENTS.md`, `ARCHITECTURE.MD`, and `COMPONENT_{NAME}_DOCS.md` updated as the system evolves.
4.  **Comprehensive Testing:** Expand unit tests and add integration tests using `pytest`.
5.  **CI/CD:** Set up GitHub Actions or GitLab CI for automated testing, linting, and building.

This detailed guide should provide a solid foundation. Remember to build and test incrementally! Good luck!