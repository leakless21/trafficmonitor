Okay, here's the full updated guide, incorporating the `src` layout and keeping the detailed explanations for a beginner. I will focus on the structure, purpose, inputs, outputs, and key logic points for each component, rather than full code blocks.

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
    │           └── logging_config.py
    ├── weights/                   # Stores your pre-trained model files (.pt)
    ├── output/                    # For saving processed videos or logs (optional)
    ├── pyproject.toml             # Pixi project configuration
    └── .envrc                     # Direnv environment configuration
    ```
*   **`__init__.py` files:** These empty files are crucial. They tell Python that the directories (`traffic_monitor`, `services`, `shared_utils`) are "packages," allowing you to import modules from them using dot notation (e.g., `from traffic_monitor.services import frame_grabber`).

**3. System Architecture (Initial Prototype using `multiprocessing.Queue`)**

*   **Concept:** A central "Supervisor" script launches each core piece of functionality as an independent Python process. These processes communicate by passing data through shared "Queues." Think of it like a factory assembly line where items (video frames and their processed data) move from one station (process) to the next via conveyor belts (queues).
*   **Components:**
    *   **`src/traffic_monitor/main_supervisor.py`:** The "factory manager." It creates the queues and starts all the worker processes.
    *   **`multiprocessing.Queue` Instances:** The "conveyor belts." These are special Python objects that allow safe communication of data between processes.
    *   **Service Processes (e.g., `FrameGrabberProcess`, `VehicleDetectorProcess`):** Each worker "station" that performs a specific task. Each runs its logic defined in a corresponding `.py` file in `src/traffic_monitor/services/`.

**4. Core Technologies**

*   **Python 3.8+:** The programming language.
*   **`multiprocessing.Queue`:** For communication between your Python processes in this initial prototype.
*   **Ultralytics YOLOv8:** For deep learning-based object detection (vehicles and license plates). You'll use your `.pt` model weight files.
*   **BoxMOT:** A library for object tracking (to follow vehicles).
*   **PaddleOCR:** An OCR library for reading license plate text.
*   **OpenCV (`cv2`):** For all video and image processing tasks (reading frames, drawing, displaying).
*   **NumPy:** For efficient numerical operations, especially with image data.
*   **Loguru:** For easy and powerful logging, helping you see what each process is doing.
*   **Pixi:** To manage your project's Python environment and dependencies.
*   **Direnv:** To automatically load environment configurations when you enter the project directory.

**5. Detailed Project Setup Steps**

1.  **Install Tools:** Ensure you have Python (Pixi will manage its version), Pixi, Direnv, and Git.
2.  **Create Directory Structure:** Follow the structure shown in Section 2 ("Understanding the `src` Layout").
3.  **`pyproject.toml` (in project root):**
    *   Define project metadata (name, version).
    *   Specify dependencies (Python, opencv-python, ultralytics, numpy, loguru initially). Add others like `boxmot`, `paddleocr` later using `pixi add <package>`.
    *   Create a `[tasks.start]` entry: `start = "python -m traffic_monitor.main_supervisor"` for easy execution.
4.  **`.envrc` (in project root):**
    *   `layout pixi`
    *   Optionally, `export PYTHONPATH="${PWD}/src:${PYTHONPATH}"` if you face import issues, though `python -m` should generally handle it.
    *   `direnv allow` to activate.
5.  **Install Dependencies:** Run `pixi install` in the project root.
6.  **Place Model Weights:** Copy your `vehicle_detector.pt` and `plate_detector.pt` files into the `weights/` directory.
7.  **Shared Utilities (Example: `src/traffic_monitor/shared_utils/logging_config.py`):**
    *   Create a function to set up Loguru with a consistent format, potentially including process names/IDs for clarity when multiple processes are logging.

**6. Service Modules: Detailed Descriptions**

Each service is a Python module (e.g., `frame_grabber.py`) within `src/traffic_monitor/services/`. It will contain a primary function (e.g., `frame_grabber_process`) that is run by the `main_supervisor.py` in a separate process.

---

**Service 1: Frame Grabber (`src/traffic_monitor/services/frame_grabber.py`)**

*   **Purpose:** To read video frames from a source (file or webcam) and make them available to other services.
*   **Function Signature (example):** `def frame_grabber_process(config: dict, output_q: multiprocessing.Queue):`
*   **Input:**
    *   `config`: A Python dictionary containing settings like `video_source` (e.g., `"path/to/video.mp4"` or `0` for webcam) and `camera_id`.
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
    *   If end of video file: Puts `None` onto `output_q` to signal downstream services to shut down.
*   **Key Logic:**
    1.  Initialize `cv2.VideoCapture` using `config["video_source"]`.
    2.  Loop continuously:
        *   Read a frame. If `ret` is false (no frame), send `None` to `output_q` and exit the loop.
        *   Generate `frame_id` (e.g., using `uuid.uuid4()`) and `timestamp`.
        *   Convert the OpenCV frame (NumPy array) to a JPEG byte string (`cv2.imencode`), then encode that to a Base64 string. This makes it easy to pass through queues.
        *   Create the message dictionary with all data.
        *   Use `output_q.put(message_dict, timeout=1)` to send the message. The timeout prevents the process from hanging if the queue is full (meaning a downstream service is slow).
    3.  Release `cv2.VideoCapture` when done.
*   **Imports needed:** `cv2`, `base64`, `time`, `uuid`, `multiprocessing`, `loguru`.

---

**Service 2: Vehicle Detector (`src/traffic_monitor/services/vehicle_detector.py`)**

*   **Purpose:** To take raw frames and identify vehicles within them.
*   **Function Signature:** `def vehicle_detector_process(config: dict, input_q: multiprocessing.Queue, output_q: multiprocessing.Queue):`
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
    *   If input message is `None`: Puts `None` onto `output_q`.
*   **Key Logic:**
    1.  **Initialization (once at process start):**
        *   Load the Ultralytics YOLOv8 vehicle detection model: `model = YOLO(config["model_path"])`.
        *   The `config` dictionary will provide `model_path` (e.g., `"weights/vehicle_detector.pt"`), `conf_threshold`, `iou_threshold`, and a `class_mapping` dictionary (e.g., `{0: "person", 2: "car", ...}`).
    2.  Loop continuously:
        *   Get a message from `input_q`: `message = input_q.get()`. If `None`, send `None` to `output_q` and exit.
        *   Decode `message["frame_data_jpeg"]`: Base64 decode, then `cv2.imdecode` to get the OpenCV frame.
        *   Perform inference: `results = model.predict(frame, conf=..., iou=...)`.
        *   Iterate through `results.boxes.data` (or the appropriate attribute for your Ultralytics version) to get bounding boxes (`xyxy`), confidence scores, and class IDs.
        *   Use the `class_mapping` from `config` to get the `class_name`.
        *   Build the `detections` list.
        *   Create the output message dictionary (including passthrough fields).
        *   Use `output_q.put(output_message, timeout=1)`.
*   **Imports needed:** `ultralytics.YOLO`, `cv2`, `base64`, `numpy`, `multiprocessing`, `loguru`.

---

**Service 3: Vehicle Tracker (`src/traffic_monitor/services/vehicle_tracker.py`)**

*   **Purpose:** To take the detected vehicles from each frame and assign a consistent ID to each vehicle as it moves across frames.
*   **Function Signature:** `def vehicle_tracker_process(config: dict, input_q: multiprocessing.Queue, output_q: multiprocessing.Queue):`
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
    *   If input message is `None`: Puts `None` onto `output_q`.
*   **Key Logic:**
    1.  **Initialization (once at process start):**
        *   Initialize a BoxMOT tracker object (e.g., `BoTSORT`, `ByteTrack`). Example: `tracker = BoTSORT(model_weights=Path('path/to/osnet_model.pt'), device='cpu', per_class=False)`.
        *   The `config` will pass any necessary tracker parameters and the `class_mapping`.
    2.  Loop continuously:
        *   Get `detection_message` from `input_q`. If `None`, send `None` to `output_q` and exit.
        *   Extract the `detections` list from `detection_message`.
        *   Convert these detections into the NumPy array format required by BoxMOT (usually `[x1, y1, x2, y2, score, class_id]`).
        *   Decode `detection_message["frame_data_jpeg"]` to get the OpenCV frame (some trackers use appearance features from the frame).
        *   Update the tracker: `tracks = tracker.update(detections_numpy_array, frame)`.
        *   The `tracks` output from BoxMOT is usually a NumPy array like `[x1, y1, x2, y2, track_id, score, class_id]`.
        *   Iterate through `tracks` to build the `tracked_objects` list, using `class_mapping` for `class_name`.
        *   Create the output message dictionary.
        *   Use `output_q.put(output_message, timeout=1)`.
*   **Dependencies to add:** `boxmot`. `cv2`, `base64`, `numpy`, `multiprocessing`, `loguru`.

---

**Service 4: License Plate Detector (`src/traffic_monitor/services/lp_detector.py`)**

*   **Purpose:** For each tracked vehicle, to specifically find the bounding box of its license plate.
*   **Function Signature:** `def lp_detector_process(config: dict, input_q: multiprocessing.Queue, output_q: multiprocessing.Queue):`
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
    *   If input message is `None`: Puts `None` onto `output_q`.
*   **Key Logic:**
    1.  **Initialization:** Load the YOLOv8 license plate detection model: `lp_model = YOLO(config["model_path"])`. `config` provides `model_path` (e.g., `"weights/plate_detector.pt"`), `conf_threshold`.
    2.  Loop continuously:
        *   Get `tracked_vehicle_message` from `input_q`. If `None`, send `None` to `output_q` and exit.
        *   Decode `tracked_vehicle_message["frame_data_jpeg"]` to get the `original_frame`.
        *   Iterate through each `vehicle` in `tracked_vehicle_message["tracked_objects"]`:
            *   Get the vehicle's bounding box (`vehicle_bbox_xyxy`).
            *   Crop this vehicle region from the `original_frame`: `vehicle_crop = original_frame[y1:y2, x1:x2]`.
            *   Perform license plate detection *on this `vehicle_crop`*: `lp_results = lp_model.predict(vehicle_crop, conf=...)`.
            *   If a license plate is found in `lp_results`:
                *   Get its bounding box (`lp_bbox_on_crop`) which is relative to `vehicle_crop`.
                *   **Crucial:** Convert `lp_bbox_on_crop` coordinates to be relative to the `original_frame`. (e.g., `lp_x1_orig = vehicle_bbox_xyxy + lp_bbox_on_crop`).
                *   Create the output message dictionary for this specific plate.
                *   Use `output_q.put(output_message, timeout=1)`.
*   **Imports needed:** `ultralytics.YOLO`, `cv2`, `base64`, `numpy`, `multiprocessing`, `loguru`.

---

**Service 5: OCR Reader (`src/traffic_monitor/services/ocr_reader.py`)**

*   **Purpose:** To take the cropped image of a license plate and read the text characters on it.
*   **Function Signature:** `def ocr_reader_process(config: dict, input_q: multiprocessing.Queue, output_q: multiprocessing.Queue):`
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
    *   If input message is `None`: Puts `None` onto `output_q`.
*   **Key Logic:**
    1.  **Initialization:** Initialize PaddleOCR: `ocr_engine = PaddleOCR(use_angle_cls=True, lang=config.get("lang", "en"), show_log=False)`. `config` provides `lang`.
    2.  Loop continuously:
        *   Get `lp_candidate_message` from `input_q`. If `None`, send `None` to `output_q` and exit.
        *   Decode `lp_candidate_message["frame_data_jpeg"]` to get `original_frame`.
        *   Get `lp_candidate_message["plate_bbox_orig"]`.
        *   Crop the license plate image from the `original_frame` using these coordinates: `plate_image_crop = original_frame[y1:y2, x1:x2]`.
        *   Perform OCR on `plate_image_crop`: `ocr_result = ocr_engine.ocr(plate_image_crop, cls=True)`.
        *   Parse the `ocr_result` (PaddleOCR's output format) to extract the most likely text string and its confidence score. Handle cases where no text is found.
        *   Create the output message dictionary.
        *   Use `output_q.put(output_message, timeout=1)`.
*   **Dependencies to add:** `paddleocr`, `paddlepaddle`. `cv2`, `base64`, `numpy`, `multiprocessing`, `loguru`.

---

**Service 6: Vehicle Counter (`src/traffic_monitor/services/vehicle_counter.py`)**

*   **Purpose:** To count vehicles that cross a predefined virtual line in the scene.
*   **Function Signature:** `def vehicle_counter_process(config: dict, input_q: multiprocessing.Queue, output_q: multiprocessing.Queue):`
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
    2.  Loop continuously:
        *   Get `tracked_vehicle_message` from `input_q`. If `None`, exit loop (no need to propagate `None` from this service unless Visualizer specifically needs a final count update signal).
        *   For each `vehicle` in `tracked_vehicle_message["tracked_objects"]`:
            *   Get `track_id`, `class_name`, and `bbox_xyxy`.
            *   Calculate a representative point for the vehicle (e.g., bottom-center of its bounding box).
            *   If `track_id` is in `vehicle_last_pos`:
                *   You have the vehicle's `previous_point` and `current_point`.
                *   Implement a **line intersection check**: Does the line segment from `previous_point` to `current_point` intersect the predefined `counting_line`? (Search for "line segment intersection algorithm").
                *   If it intersects AND `track_id` is not in `counted_ids_this_session` (or hasn't been counted recently):
                    *   Increment `current_counts["total"]` and `current_counts.setdefault(class_name, 0) += 1`.
                    *   Add `track_id` to `counted_ids_this_session`.
                    *   Set a flag `counts_have_changed = True`.
            *   Update `vehicle_last_pos[track_id]` with the `current_point`.
        *   If `counts_have_changed` (or after a certain time interval):
            *   Create the count update message.
            *   Use `output_q.put(count_update_message, timeout=1)`.
            *   Reset `counts_have_changed` flag.
*   **Imports needed:** `numpy` (potentially for line math), `multiprocessing`, `loguru`.

---

**Service 7: Visualizer (`src/traffic_monitor/services/visualizer.py`)**

*   **Purpose:** To gather all processed information and display it as an annotated video feed. This is often the most complex in terms of managing and synchronizing data from multiple sources.
*   **Function Signature:** `def visualizer_process(config: dict, tracked_vehicles_q_in: multiprocessing.Queue, ocr_results_q_in: multiprocessing.Queue, counting_updates_q_in: multiprocessing.Queue):`
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
    2.  Loop continuously:
        *   **Non-blockingly get messages from all input queues:**
            *   Use `tracked_vehicles_q_in.get(block=True, timeout=0.01)` (or `get_nowait()` with try-except `queue.Empty`). If a message (let's call it `track_msg`) is received:
                *   This `track_msg` is the "driver" for rendering a frame because it contains `frame_data_jpeg` and `tracked_objects`.
                *   Decode `track_msg["frame_data_jpeg"]` to get the `current_display_frame`.
                *   Start with this `current_display_frame`.
                *   Draw all `tracked_objects` from `track_msg` onto it (bboxes, track IDs, class names).
                *   For each `track_id` in `track_msg["tracked_objects"]`, check `ocr_data_for_tracks` to see if recent OCR text is available. If yes, draw it near the vehicle.
                *   If `latest_vehicle_counts` is available, draw the count statistics on the frame.
                *   Draw calculated FPS.
                *   `cv2.imshow("Traffic Monitor", current_display_frame)`.
                *   Handle `cv2.waitKey(1)` for display updates and 'q' to quit. If 'q' is pressed, signal other processes to terminate (e.g., by closing their input queues or a shared event).
            *   Use `ocr_results_q_in.get_nowait()`: If an OCR message arrives, update `ocr_data_for_tracks` with the new text and timestamp for that `vehicle_track_id`.
            *   Use `counting_updates_q_in.get_nowait()`: If a count message arrives, update `latest_vehicle_counts`.
        *   Implement logic to periodically clear very old entries from `ocr_data_for_tracks` based on their timestamp to prevent memory growth.
        *   If the `track_msg` from `tracked_vehicles_q_in` is `None`, break the loop and clean up.
    3.  `cv2.destroyAllWindows()` when done.
*   **Imports needed:** `cv2`, `base64`, `numpy`, `multiprocessing`, `loguru`, `collections.deque` (optional for managing frame IDs).

---

**7. Supervisor Script (`src/traffic_monitor/main_supervisor.py`)**

*   **Purpose:** The main script you run. It initializes all the queues and starts each service function as a separate `multiprocessing.Process`.
*   **Key Logic:**
    1.  Import all service functions (e.g., `from .services.frame_grabber import frame_grabber_process`).
    2.  `multiprocessing.set_start_method('spawn', force=True)`: Call this early, especially for cross-platform compatibility.
    3.  Create all the `multiprocessing.Queue(maxsize=SOME_REASONABLE_NUMBER)` instances. `maxsize` is crucial to prevent a fast producer from overwhelming a slow consumer and using up all memory.
    4.  Define `config` dictionaries for each service (hardcode for now, or load from a file later). These configs will include model paths, thresholds, queue names (though queues are passed directly), etc.
    5.  For each service:
        *   Create a `multiprocessing.Process(target=service_function, name="ServiceName", args=(config_dict, input_q_or_tuple_of_qs, output_q_or_None))`.
        *   Store the process object in a list.
    6.  Start all processes: `for p in process_list: p.start()`.
    7.  Implement a loop to keep the main supervisor process alive while child processes are running.
    8.  Implement graceful shutdown: On `KeyboardInterrupt` (Ctrl+C), iterate through the process list, call `p.terminate()` on each, and then `p.join(timeout=...)` to wait for them to exit. If they don't exit, `p.kill()`.

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
    *   Key configuration values it's using.
    *   When it receives a message (maybe log the `frame_id`).
    *   A summary of the processing it did (e.g., "Detected 3 vehicles").
    *   When it sends a message.
    *   Any errors encountered.
    *   Including `process.name` or `process.id` in your log format helps distinguish logs from different services.
*   **Small Test Videos:** Use short, simple video clips for initial testing.
*   **Queue `maxsize`:** Setting a `maxsize` on your queues is vital. If one service is much slower than its producer, an unbounded queue will consume all your RAM. A `maxsize` creates "backpressure."
*   **Error Handling:** Wrap potentially problematic code (e.g., `queue.get()`, model inference, file operations) in `try...except` blocks within each service.

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

This detailed guide should provide a solid foundation. Remember to build and test incrementally! Good luck!