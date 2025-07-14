# CHAPTER 3. SYSTEM METHODOLOGY

## 1.  System Architecture

The traffic monitoring system is designed with a modular, multiprocessing architecture for real-time, high-throughput video analysis. It comprises a series of decoupled services, each executing a distinct stage of the processing pipeline: frame capture, vehicle detection, multi-object tracking, vehicle counting, license plate detection, and optical character recognition (OCR). A central supervisor orchestrates these services, enabling concurrent operation across multiple processor cores. This design yields three principal advantages:

*   **Parallelism**: Each service operates in its own process. For instance, the Vehicle Detection service can perform model inference on frame N while the Vehicle Tracking service processes frame N-1. This creates a deep pipeline that substantially increases system throughput (Frames Per Second) compared to a sequential execution model.
*   **Modularity and Extensibility**: Services are self-contained and communicate via standardized message-passing queues. This architectural pattern facilitates the modification, replacement, or addition of new functionalities with minimal impact on the overall system.
*   **Resilience**: The isolation of services prevents an error in a non-critical component (e.g., the LPR stream) from halting the entire system.

The figure below provides a high-level overview of the system's data processing pipeline. Each box represents a distinct service, and the arrows depict the flow of data messages between them.

![System Overview](https://i.imgur.com/your-diagram-url.png)
*Figure 3.1: System overview*

The pipeline is structured as follows:

*   **The Core Pipeline**: An initial sequential pipeline identifies and tracks objects.
    *   **Frame Capture**: The `FrameCaptureService` ingests video frames from a source.
    *   **Vehicle Detection**: The `VehicleDetectionService` performs inference to generate bounding boxes for objects.
    *   **Vehicle Tracking**: The `VehicleTrackingService` assigns a persistent `track_id` to each object, maintaining its state across frames. This is the final stage of the sequential process.
*   **The Parallel Processing Fork**: After tracking, the pipeline deliberately forks into multiple parallel processing streams to maximize efficiency. An `EventDistributionService` duplicates the output from the tracker for each downstream consumer.
    *   **License Plate Recognition (LPR) Stream**: A two-stage process for reading license plates.
        *   **Plate Detection**: The `LicensePlateDetectionService` finds the location of the license plate on the vehicle image.
        *   **Plate Recognition**: The `TextRecognitionService` performs OCR on the plate region to extract the text.
    *   **Vehicle Counting Stream**: In parallel, the `VehicleCountingService` applies a line-crossing algorithm to determine if vehicles have crossed predefined virtual lines.
*   **Data Aggregation for Visualization**: All processing streams converge at the final stage.
    *   **Visualization**: The `VisualizationService` subscribes to the outputs of the tracking, counting, and LPR streams. It aggregates this information—bounding boxes, track IDs, counts, and plate text—and renders it onto the original video frame for display or storage.

## 2.  Dataflow and Inter-Process Communication

The system's backbone is its inter-process communication (IPC) protocol, which ensures that isolated services operate as a cohesive pipeline. To maintain strict process isolation and prevent the race conditions inherent in shared-memory models, the system exclusively uses `multiprocessing.Queue` for all inter-service communication. This standard Python library provides a process-safe, thread-safe FIFO (First-In, First-Out) message queue.

This choice offers several advantages:
*   **Decoupling**: Services operate without direct knowledge of each other. A producer service places a message on its output queue, and a consumer retrieves it from an input queue.
*   **Data Safety**: The operating system manages the underlying data transfer, guaranteeing that messages are exchanged without corruption or conflict.
*   **Flow Control**: Queues naturally buffer data, smoothing out variations in processing time between services. If a computationally intensive service like vehicle detection slows down, upstream services can continue to queue frames (up to a fixed limit) without being blocked.

Data messages adhere to a defined structure using Python's `TypedDict`, as specified in `src/traffic_monitor/utils/custom_types.py`. This serves as an API contract between services and allows static analysis tools to detect potential data errors. As a data packet moves through the pipeline, it is progressively enriched with new information, ensuring that downstream services have access to a complete context from all preceding stages.

![Data Enrichment](https://i.imgur.com/your-diagram-url.png)
*Figure 3.2: Example of progressive data enrichment in message payloads*

## 3.  Core Modules and Algorithms

### 3.1.  Frame Capture

The `FrameCaptureService` is the entry point of the video processing pipeline. Its responsibility is to ingest video data and prepare it for downstream processing.

*   **Algorithm**: The service utilizes the OpenCV library (`cv2.VideoCapture`) to connect to video sources. To manage computational load, it resizes frames and processes only every Nth frame, as configured. To optimize IPC throughput, the raw frame (a large NumPy array) is encoded into the compressed JPEG format using `cv2.imencode`. While this introduces minor computational overhead and lossy compression, the significantly reduced data payload is critical for preventing the IPC queue from becoming a bottleneck.
*   **Input**: A video source identifier (e.g., file path, camera index).
*   **Output**: For each processed frame, it emits a `FrameMessage` containing the JPEG-encoded frame and associated metadata.

![Sample Frame](https://i.imgur.com/your-diagram-url.png)
*Figure 3.3: A sample frame*

### 3.2.  Vehicle Detection

The `VehicleDetectionService` identifies and localizes all objects of interest within a given video frame.

*   **Algorithm**: This service leverages a pre-trained You Only Look Once (YOLO) object detection model, loaded via the `ultralytics` library. The service decodes the incoming JPEG frame, performs a forward pass with the YOLO model, filters the results to remove detections below a specified confidence threshold, and maps the model's numeric `class_id` to a human-readable name (e.g., "car").
*   **Input**: `FrameMessage`.
*   **Output**: `VehicleDetectionMessage`, an augmented version of the input containing a list of all detected objects.

![Vehicle Detection](https://i.imgur.com/your-diagram-url.png)
*Figure 3.4: Visualization of vehicle detection results*

### 3.3.  Vehicle Tracking

The `VehicleTrackingService` assigns a persistent identity to each detected object as it moves through the scene.

*   **Algorithm**: The service implements the tracking-by-detection paradigm using the BoxMOT library, which supports algorithms like ByteTrack and BoTSORT. For each frame, the tracker's `update` method uses a combination of motion prediction (via a Kalman filter) and optional appearance similarity to associate new detections with existing tracks.
*   **Input**: `VehicleDetectionMessage`.
*   **Output**: `TrackedVehicleMessage`, where the `detections` list is replaced by a `tracked_objects` list, with each object now including a unique `track_id`.

![Vehicle Tracking](https://i.imgur.com/your-diagram-url.png)
*Figure 3.5: Visualization of tracked vehicles with persistent IDs*

### 3.4.  Vehicle Counting

The `VehicleCountingService` tallies vehicles that cross user-defined virtual lines.

*   **Algorithm**: The logic is based on a geometric line-crossing algorithm using the `shapely` library. A crossing event is registered if a vehicle's movement vector between frames intersects with a virtual line. To prevent a single vehicle from being counted multiple times, the service maintains a set of `counted_track_ids`. A vehicle is only counted if its ID is not in this set. After being counted, its ID is added. The ID is removed from the set only when the tracking service reports that the vehicle is no longer being tracked (i.e., the track ID is lost). This ensures a vehicle is counted exactly once per continuous track.
*   **Input**: `TrackedVehicleMessage`.
*   **Output**: A `VehicleCountMessage` is emitted only when a crossing event occurs.

![Before Counting](https://i.imgur.com/your-diagram-url.png)
*Figure 3.6: Vehicle approaching the counting line*

![After Counting](https://i.imgur.com/your-diagram-url.png)
*Figure 3.7: Vehicle state after crossing the counting line*

### 3.5.  License Plate Detection

The `LicensePlateDetectionService` is the first stage of the LPR pipeline, responsible for locating the plate within a vehicle image.

*   **Algorithm**: This service employs a second, specialized YOLO model fine-tuned for license plate detection. For each incoming tracked vehicle, the service first crops the vehicle's bounding box from the main frame and runs the plate detector on this smaller, higher-resolution image.
*   **Input**: `TrackedVehicleMessage`.
*   **Output**: `PlateDetectionMessage`, containing the bounding box of the license plate.

![License Plate Detection](https://i.imgur.com/your-diagram-url.png)
*Figure 3.8: A detected license plate on a cropped vehicle image*

### 3.6.  License Plate Recognition

The `TextRecognitionService` completes the LPR pipeline by performing Optical Character Recognition (OCR).

*   **Algorithm**: This service is designed for modularity and supports multiple OCR backends, including the specialized FastPlateOCR and the general-purpose PaddleOCR engine. The service passes the cropped plate image to the selected engine and processes the output to extract the plate text and a confidence score.
*   **Input**: `PlateDetectionMessage`.
*   **Output**: `OCRResultMessage`, containing the recognized `lp_text` and its `ocr_confidence`.

![LPR Result](https://i.imgur.com/your-diagram-url.png)
*Figure 3.9: Final LPR result overlaid on the vehicle*

## 4.  Supporting Modules

### 4.1.  Configuration Management

System adaptability is achieved through a multi-layered configuration strategy managed by the `cli.py` and `config_loader.py` modules. The primary method is a central `settings.yaml` file, which uses a hierarchical structure that mirrors the application's service-oriented architecture. For modularity, algorithm-specific hyperparameters (e.g., for different trackers) are stored in separate configuration files.

![settings.yaml](https://i.imgur.com/your-diagram-url.png)
*Figure 3.10: Snippet of the hierarchical settings.yaml file*

### 4.2.  Database Architecture

The system employs SQLite for its data persistence layer, chosen for its deployment simplicity and self-contained nature. The database architecture follows a "distributed write" model, where individual services are responsible for persisting their own results. For example, the `VehicleCountingService` writes a new row to the database each time a counting event occurs.

This design is simple and keeps the data persistence logic co-located with the service that generates the data. However, it means multiple processes may attempt to write to the same SQLite database file concurrently. While infrequent writes from different services may not cause issues, this approach carries a risk of `database is locked` errors under heavy, simultaneous load from multiple services.

The schema is designed to accommodate the multi-stage processing pipeline while maintaining referential integrity, with each service writing its results to dedicated tables.

### 4.3.  Main Supervisor

The `main_supervisor.py` script is the central orchestrator of the application. It is responsible for parsing the final configuration, initializing the `multiprocessing.Queue` instances for communication, instantiating each service as a separate `multiprocessing.Process`, and managing the graceful startup and shutdown of the entire system.

### 4.4.  Visualization

The `VisualizationService` acts as a primary data sink, consuming the outputs from other services to provide real-time visual feedback. The service uses an asynchronous, state-based mechanism for data synchronization. It maintains the most recent state for vehicle counts and OCR results in memory. When a new video frame message arrives from the tracking stream, it is rendered immediately using the tracking data from that message, overlaid with the latest available count and OCR data. This approach prioritizes a fluid frame rate over perfect synchronization, which is suitable for real-time monitoring. Its features include:

*   Drawing bounding boxes, track IDs, and class labels.
*   Displaying recognized license plate text.
*   Overlaying virtual counting lines.
*   Displaying a statistics panel with FPS and vehicle counts.

The service can either display the annotated video in a live window or write it to a video file.

![Visualization](https://i.imgur.com/your-diagram-url.png)
*Figure 3.11: A visualized frame with integrated data overlays*

## 5. System Limitations

While the proposed architecture is robust, it is important to acknowledge its limitations. The performance of the computer vision models is intrinsically tied to the quality of the input video. Factors such as poor lighting, adverse weather conditions (e.g., rain, fog), and extreme camera angles can significantly degrade the accuracy of both vehicle detection and license plate recognition.

Furthermore, the vehicle tracking algorithm can be challenged by heavy occlusion, where vehicles are hidden behind other objects for extended periods, potentially leading to identity switches. Finally, the overall system throughput is ultimately constrained by the performance of the slowest component in the sequential portion of the pipeline, which is typically the vehicle detection model running on the available hardware.