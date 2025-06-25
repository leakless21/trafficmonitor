# Capstone Project Report: Real-time Traffic Monitoring System

## 1. Introduction

### 1.1 Problem Statement

Modern urban environments face increasing challenges in managing traffic flow efficiently, ensuring road safety, and gathering accurate data for urban planning. Traditional methods of traffic monitoring often involve manual observation or expensive, inflexible hardware installations. There is a critical need for an automated, scalable, and cost-effective solution that can provide real-time insights into traffic dynamics, including vehicle detection, tracking, counting, and license plate recognition.

### 1.2 Objectives

This project aims to develop a robust and efficient real-time traffic monitoring system capable of processing video streams to identify, track, and count vehicles, and optionally, to perform license plate recognition. The primary objectives are derived from the functional and non-functional requirements:

**Functional Requirements:**

- **Video Stream Processing (FR1):** Ingest and decode video streams from various sources (local files, IP cameras).
- **Vehicle Detection (FR2):** Detect vehicles within each frame, providing bounding boxes, confidence scores, and class IDs using a pre-trained YOLO model.
- **Vehicle Tracking (FR3):** Track vehicles across frames, assigning unique IDs and updating their information, supporting configurable tracker types (e.g., BoxMOT with ByteTrack) and Re-ID models.
- **Multiprocessing Architecture (FR4):** Implement a concurrent architecture using multiprocessing and inter-process communication (queues) for efficient data flow.
- **Configuration Management (FR5):** Load all system parameters from a centralized YAML configuration file.
- **Logging (FR6):** Provide structured, detailed, and class-specific logging for debugging, informational, and error messages across all processes.
- **Vehicle Counting (FR7):** Count vehicles crossing predefined counting lines, preventing double-counting, and maintaining class-specific tallies.
- **Visualization and Display (FR8):** Display real-time video feeds with overlays for bounding boxes, track IDs, class labels, counting lines, counts, FPS, and OCR results.

**Non-Functional Requirements:**

- **Performance (NFR1):** Achieve near real-time processing with minimal latency.
- **Scalability (NFR2):** Design a multiprocessing architecture that allows for potential scaling of individual components.
- **Maintainability (NFR3):** Ensure a clear, modular codebase with type hints and docstrings.
- **Reliability (NFR4):** Implement graceful handling of empty queues, process shutdowns, and robust error logging.

### 1.3 Scope

The project focuses on the development of the core traffic monitoring pipeline, encompassing video ingestion, object detection, multi-object tracking, vehicle counting, and real-time visualization. It also includes the integration of license plate detection and optical character recognition (OCR) capabilities. The system is designed to be configurable for various video sources and adaptable to different detection and tracking models. It does not include advanced features like traffic prediction, anomaly detection, or a web-based user interface.

## 2. Methodology

### 2.1 Architecture Design

The Traffic Monitoring System is designed with a modular and multiprocessing-based architecture to achieve concurrency and improve performance. Data flows through a series of specialized services, each running as a separate process and communicating via `multiprocessing.Queue` instances. This design ensures that computationally intensive tasks, such as video decoding, object detection, and tracking, can run in parallel without blocking the main application thread.

```text
+-------------------+      +---------------------+      +--------------------+
|   Frame Grabber   |      |   Vehicle Detector  |      |   Vehicle Tracker  |
| (frame_grabber.py)|----->|(vehicle_detector.py)|----->|(vehicle_tracker.py)|
|     (Process 1)   |      |      (Process 2)    |      |      (Process 3)   |
+-------------------+      +---------------------+      +--------------------+
                                       |
                                       | (vehicle_tracker_output_queue)
                                       V
                               +------------------+
                               |    Distributor   |
                               | (distributor.py) |
                               |   (Process 8)    |
                               +------------------+
                                     /  |  \
                                    /   |   \
                     (lp_detector_input_queue) (vehicle_counter_input_queue) (visualizer_input_queue)
                                   V    V    V
+-------------------+      +-------------------+      +-------------------+
|    LP Detector    |      |   Vehicle Counter |      |     Visualizer    |
|  (lp_detector.py) |      | (vehicle_counter.py)|    |  (visualizer.py)  |
|     (Process 4)   |      |     (Process 6)   |    |    (Process 7)    |
+-------------------+      +-------------------+      +--------^----------+
         |                                |                       |
         | (lp_detector_output_queue)     | (vehicle_counter_output_queue)
         V                                V                       |
+-------------------+                     +-----------------------+
|    OCR Reader     |
|  (ocr_reader.py)  |
|     (Process 5)   |
+-------------------+
         |
         | (ocr_reader_output_queue)
         V
+--------------------------------------------------------------------------+
|                               Visualizer                                 |
|                            (visualizer.py)                               |
|                               (Process 7)                                |
+--------------------------------------------------------------------------+
         ^
         |
+-------------------+
|   Main Supervisor |
| (main_supervisor.py)|
|     (Orchestrator)  |
+-------------------+
```

#### Key Components:

1.  **Frame Grabber (`frame_grabber.py`):**

    - **Purpose:** Ingests video streams (from files or IP cameras) and decodes frames.
    - **Responsibility:** Reads frames from the video source and places `FrameMessage` objects (containing frame data and metadata) into `frame_grabber_output_queue`.
    - **Compute:** I/O-bound, handles video decoding.
    - **Technology:** Leverages `opencv-python` for video capture and decoding.

2.  **Vehicle Detector (`vehicle_detector.py`):**

    - **Purpose:** Detects vehicles within incoming video frames.
    - **Responsibility:** Receives `FrameMessage` objects, runs a YOLO model for object detection, and outputs `VehicleDetectionMessage` objects (with detection bounding boxes, confidence, class IDs) to `vehicle_detector_output_queue`.
    - **Compute:** GPU/CPU-bound, performs inference with a YOLO model.
    - **Dependency:** Utilizes `ultralytics` library for YOLO models. Often integrates with `onnx` and `onnxruntime` for optimized inference with ONNX-converted models.

3.  **Vehicle Tracker (`vehicle_tracker.py`):**

    - **Purpose:** Tracks detected vehicles across multiple frames, assigning unique IDs.
    - **Responsibility:** Receives `VehicleDetectionMessage` objects, updates the BoxMOT tracker with new detections, and generates `TrackedVehicleMessage` objects (containing tracked object information, frame metadata, and JPEG-encoded frame data) to `vehicle_tracker_output_queue`.
    - **Compute:** Primarily CPU-bound, but can leverage GPU if a compatible Re-ID model is used.
    - **Dependency:** Utilizes `boxmot` library, supporting various trackers like ByteTrack, DeepOCSORT, and OCSORT.

4.  **LP Detector (`lp_detector.py`):**

    - **Purpose:** Detects license plates within detected vehicle bounding boxes.
    - **Responsibility:** Processes vehicle detections to identify regions likely containing license plates.
    - **Compute:** CPU/GPU-bound, performs inference with a specialized license plate detection model.
    - **Technology:** Integrates with underlying models for license plate detection, often leveraging capabilities similar to `fast-plate-ocr`'s detection part.

5.  \*\*OCR Reader (`ocr_reader.py`):

    - **Purpose:** Performs Optical Character Recognition on detected license plates.
    - **Responsibility:** Receives license plate regions from the `LP Detector` and extracts alphanumeric characters using the `fast-plate-ocr` library.
    - **Compute:** CPU-bound, performs OCR.
    - **Technology:** Employs the `fast-plate-ocr` library for character recognition.

6.  **Vehicle Counter (`vehicle_counter.py`):**

    - **Purpose:** Counts vehicles crossing predefined virtual counting lines.
    - **Responsibility:** Receives `TrackedVehicleMessage` objects, uses geometric calculations (`shapely`) to determine line crossings, maintains class-specific counts, and sends `VehicleCountMessage` objects to the `Main Supervisor`.
    - **Compute:** CPU-bound, performs geometric calculations and updates counts.
    - **Technology:** Relies on the `shapely` library for precise geometric computations (e.g., line intersection detection).

7.  **Visualizer (`visualizer.py`):**

    - **Purpose:** Renders real-time video output with overlays.
    - **Responsibility:** Receives `TrackedVehicleMessage` objects, draws bounding boxes, track IDs, class labels, counting lines, and displays real-time counts and FPS using OpenCV.
    - **Compute:** CPU-bound, handles rendering and display.
    - **Technology:** Extensively uses `opencv-python` for all visualization tasks, including drawing shapes, text, and displaying video frames.

8.  **Main Supervisor (`main_supervisor.py`):**
    - **Purpose:** Orchestrates the entire system.
    - **Responsibility:** Initializes all services, sets up inter-process queues, manages configuration loading, and handles graceful shutdown of all child processes.
    - **Technology:** Utilizes Python's `multiprocessing` module for process management and inter-process communication (`multiprocessing.Queue`). Employs `pyyaml` for configuration loading and `loguru` for centralized logging.

#### Inter-process Communication:

The system uses `multiprocessing.Queue` for robust and efficient data transfer between components. Key queues include:

- `frame_grabber_output_queue`: `FrameGrabber` -> `VehicleDetector`
- `vehicle_detector_output_queue`: `VehicleDetector` -> `VehicleTracker`
- `vehicle_tracker_output_queue`: `VehicleTracker` -> `Distributor`
- `lp_detector_input_queue`: `Distributor` -> `LPDetector`
- `vehicle_counter_input_queue`: `Distributor` -> `VehicleCounter`
- `visualizer_input_queue`: `Distributor` -> `Visualizer` (for tracked objects and initial frame)
- `lp_detector_output_queue`: `LPDetector` -> `OCRReader`
- `ocr_reader_output_queue`: `OCRReader` -> `Visualizer`
- `vehicle_counter_output_queue`: `VehicleCounter` -> `Visualizer`

These queues have a configurable `maxsize` (e.g., 60 in the implementation) to buffer messages and prevent bottlenecks. Data serialization for complex objects is handled using `dill` to facilitate transfer across processes.

#### Configuration Management:

All configurable parameters are loaded from `src/traffic_monitor/config/settings.yaml`. This includes video sources, model paths, confidence thresholds, tracker parameters, logging settings, and counting line coordinates, ensuring flexibility and ease of modification.

### 2.2 Key Algorithms and Techniques

- **Object Detection:** Utilizes YOLO models (e.g., YOLOv8, YOLOv11) from the `ultralytics` library for high-performance and accurate vehicle detection. Models are often converted to ONNX format for optimized inference using `onnxruntime`.
- **Multi-Object Tracking (MOT):** Employs the `boxmot` library, which integrates various state-of-the-art trackers like ByteTrack, DeepOCSORT, and StrongSORT. This allows for persistent tracking of vehicles across frames using unique track IDs.
- **License Plate Recognition:** Integrates `fast-plate-ocr` for robust OCR capabilities on detected license plates, leveraging pre-trained models.
- **Geometric Operations:** Uses the `shapely` library for precise geometric calculations, specifically for detecting when vehicle bounding boxes intersect or cross predefined counting lines.
- **Concurrency:** Leverages Python's `multiprocessing` module to run different stages of the pipeline in parallel, maximizing CPU/GPU utilization and maintaining real-time performance.
- **Structured Logging:** Implements `loguru` for comprehensive and easily configurable logging across all processes, crucial for debugging and monitoring.

## 3. Technologies Used and Tech Stack

This project is built primarily with Python, leveraging a comprehensive set of libraries and tools for computer vision, machine learning inference, and system management.

### 3.1 Programming Language

- **Python 3.11:** The core development language, chosen for its extensive libraries, readability, and strong community support in machine learning and computer vision.

### 3.2 Core Libraries and Frameworks

- **`opencv-python` (>=4.11.0.86):** Essential for video stream ingestion, frame manipulation, and real-time visualization (drawing bounding boxes, text, lines).
- **`ultralytics` (>=8.3.135):** Provides an easy-to-use interface for YOLO (You Only Look Once) models, which are used for efficient and accurate vehicle detection.
- **`onnx` (>=1.12.0), `onnxslim` (>=0.1.53), `onnxruntime` (>=1.22.0):** Used for optimizing and running ONNX (Open Neural Network Exchange) format models, enabling faster and more portable model inference, especially for deployment.
- **`boxmot` (>=13.0.9):** The primary library for multi-object tracking, offering an abstraction over various tracking algorithms (e.g., ByteTrack, DeepOCSORT).
- **`fast-plate-ocr` (>=0.3.0):** A specialized library for license plate detection and optical character recognition, facilitating the extraction of license plate numbers.
- **`pytest` (>=8.4.0):** The testing framework used for writing and running unit and integration tests, ensuring code reliability and preventing regressions.
- **`shapely` (>=2.1.1):** A powerful library for computational geometry, used specifically for handling polygons and lines to accurately detect vehicle crossings over predefined counting lines.
- **`loguru` (>=0.7.3):** A flexible and powerful logging library, used to implement structured, colored, and file-based logging across all multiprocessing components.
- **`pyyaml` (>=6.0.2):** Used for parsing and loading configuration parameters from YAML files, providing a clear and human-readable way to manage system settings.
- **`dill` (>=0.4.0):** Utilized for serialization, particularly useful in multiprocessing contexts where complex Python objects need to be passed between processes.

### 3.3 Models

- **YOLO Detection Models:** Pre-trained YOLO models (e.g., `yolo11n.pt`, `yolo11s.onnx`) are used for vehicle detection. These models identify vehicle classes (car, truck, bus, motorcycle, bicycle, person) and their bounding boxes.
- **Re-Identification (Re-ID) Models:** While optional, the architecture supports integration of Re-ID models (e.g., `reid.pt`) with `boxmot` for improved tracking robustness, especially in cases of occlusion.
- **License Plate Detection and OCR Models:** Specific models are utilized by `fast-plate-ocr` for accurate license plate detection and character recognition.

### 3.4 Tools

- **Pixi:** A modern package and environment manager, used for dependency management and ensuring consistent development and deployment environments.
- **Git:** Version control system for collaborative development and tracking code changes.

## 4. Detailed Workflow

The traffic monitoring system operates as a pipeline, with each stage handled by a dedicated multiprocessing service. The `main_supervisor.py` orchestrates the entire workflow.

1.  **System Initialization (`main_supervisor.py`):**

    - Loads all configuration parameters from `settings.yaml`.
    - Initializes `multiprocessing.Queue` objects for inter-process communication.
    - Creates and starts separate processes for `FrameGrabber`, `VehicleDetector`, `VehicleTracker`, `Distributor`, `LPDetector`, `OCRReader`, `VehicleCounter`, and `Visualizer`.
    - Passes necessary configurations and queue references to each process.
    - Sets up logging for the main supervisor and ensures child processes can configure their own logging.

2.  **Video Ingestion and Frame Grabbing (`frame_grabber.py`):**

    - The `FrameGrabber` process continuously reads frames from the configured video source (e.g., `data/videos/input/platetest.mp4`).
    - Each captured frame, along with its metadata (e.g., frame number, timestamp), is encapsulated in a `FrameMessage` object.
    - `FrameMessage` objects are then placed into the `frame_grabber_output_queue` for consumption by the `VehicleDetector`.

3.  **Vehicle Detection (`vehicle_detector.py`):**

    - The `VehicleDetector` process retrieves `FrameMessage` objects from `frame_grabber_output_queue`.
    - It performs object detection on each frame using the loaded YOLO model.
    - Detected vehicles are represented as `Detection` objects, including bounding box coordinates, confidence scores, and class IDs.
    - These detections are then packaged into `VehicleDetectionMessage` objects and put into the `vehicle_detector_output_queue`.

4.  **Vehicle Tracking (`vehicle_tracker.py`):**

    - The `VehicleTracker` process consumes `VehicleDetectionMessage` objects from `vehicle_detector_output_queue`.
    - It updates the `boxmot` tracker with the new detections, allowing the system to maintain persistent track IDs for vehicles across frames.
    - The tracker outputs `TrackedObject` dictionaries for each identified vehicle, containing its current bounding box, unique track ID, class, and other relevant information.
    - `TrackedVehicleMessage` objects, containing the tracked objects and original frame data, are sent to the `vehicle_tracker_output_queue`.

5.  **Data Distribution (`distributor.py`):**

    - The `Distributor` process receives `TrackedVehicleMessage` objects from the `vehicle_tracker_output_queue`.
    - Its primary role is to efficiently distribute these messages to multiple downstream consumers (LP Detector, Vehicle Counter, and Visualizer) by placing them into their respective input queues.
    - This ensures that all relevant services receive the necessary tracked vehicle data for parallel processing without redundant tracking.

6.  **License Plate Detection and OCR (`lp_detector.py` and `ocr_reader.py`):**

    - The `LP Detector` process receives `TrackedVehicleMessage` objects from the `Distributor`.
    - It identifies regions likely containing license plates within the tracked vehicle bounding boxes and outputs `LicensePlateDetectionMessage` objects.
    - The `OCR Reader` process then consumes `LicensePlateDetectionMessage` objects from the `LP Detector`.
    - It performs Optical Character Recognition to extract alphanumeric characters from the detected license plate regions, producing `OCRResultMessage` objects.

7.  **Vehicle Counting (`vehicle_counter.py`):**

    - The `Vehicle Counter` process receives `TrackedVehicleMessage` objects from the `Distributor`.
    - It uses geometric calculations (`shapely`) to determine when tracked vehicles cross predefined virtual counting lines.
    - It maintains and updates class-specific and total vehicle counts, preventing double-counting.
    - `VehicleCountMessage` objects, containing the latest counts, are sent to the `Visualizer`.

8.  **Visualization and Display (`visualizer.py`):**

    - The `Visualizer` process is the final stage, responsible for rendering the real-time output.
    - It receives `TrackedVehicleMessage` objects (original frame data and tracked objects) from the `Distributor`.
    - It also receives `OCRResultMessage` objects from the `OCR Reader` and `VehicleCountMessage` objects from the `Vehicle Counter`.
    - It draws vehicle bounding boxes, track IDs, class labels, license plate text, counting lines, and displays real-time counts and FPS information using OpenCV.
    - This component presents the comprehensive real-time traffic monitoring data to the user.

9.  **Logging and Monitoring:**

    - Throughout the pipeline, `loguru` is used to log events, detections, tracking updates, counting statistics, and errors.
    - Logging is configured for both console output (INFO level) and file output (`traffic_monitor.log`), with rotation and retention policies.
    - Each multiprocessing child process independently sets up its logging to ensure comprehensive log capture from all parts of the system.

10. **System Shutdown:**

    - The `main_supervisor` manages a `shutdown_event` which, when set, signals all child processes to terminate gracefully, ensuring proper resource release and clean exit.

## 5. Challenges and Solutions

During the development of the Traffic Monitoring System, several technical challenges were encountered and subsequently resolved to ensure the system's robustness and correctness.

### 5.1 Performance Optimization

- **Challenge:** Maintaining real-time performance (e.g., 20-30 FPS) was a significant challenge, especially with multiple compute-intensive tasks like object detection, tracking, and visualization running concurrently. Bottlenecks could occur at any stage, leading to dropped frames or increased latency.
- **Solution:**
  1.  **Multiprocessing:** The primary solution was the adoption of a `multiprocessing` architecture. By offloading distinct tasks (frame grabbing, detection, tracking, LP detection, OCR, counting, visualization) to separate processes, CPU cores could be fully utilized, and I/O operations (video decoding) could run in parallel with computation (model inference).
  2.  **Optimized Model Inference:**
      - **Half-precision (FP16):** Where supported by hardware (e.g., NVIDIA GPUs), enabling half-precision inference further reduced computation time and memory bandwidth requirements.
  3.  **Efficient Data Transfer:** `multiprocessing.Queue` with a predefined `maxsize` (e.g., 100) was used to buffer data between processes. This prevented immediate blocking of producers if consumers were temporarily slow and minimized inter-process communication overhead.
  4.  **Hardware Acceleration:** Configuration options for `device` (e.g., "cuda" for GPU) allowed the system to leverage GPU acceleration for detection and tracking models, drastically speeding up inference times for these critical components.

### 5.2 Efficient Data Handling with Custom Types

- **Challenge:** Managing complex data structures (e.g., detection results, tracked objects, frame metadata) across multiple processes and ensuring type safety and clarity was challenging. Using raw dictionaries or tuples could lead to hard-to-debug errors and reduce code readability.
- **Solution:** Implemented `custom_types.py` to define clear, immutable `dataclasses` (e.g., `FrameMessage`, `Detection`, `TrackedObject`, `VehicleDetectionMessage`, `TrackedVehicleMessage`, `VehicleCountMessage`).
  - **Clarity and Readability:** Dataclasses provide a structured way to define data objects with named fields and type hints, making the code more readable and easier to understand for developers.
  - **Type Safety:** Type hints (e.g., `List[Detection]`, `cv2.Mat`) improved type checking during development, catching potential errors early.
  - **Serialization with `dill`:** For passing these custom dataclass objects between multiprocessing queues, `dill` was used. Unlike Python's standard `pickle`, `dill` can serialize a wider range of Python objects, including complex custom objects, making inter-process communication seamless without requiring manual serialization/deserialization logic.
  - **Maintainability:** Centralizing data structure definitions in `custom_types.py` improved maintainability, as changes to data schemas could be managed in a single location.

## 6. Results and Discussion

The developed Real-time Traffic Monitoring System successfully integrates various computer vision and machine learning components into a cohesive and efficient pipeline. By leveraging a multiprocessing architecture, the system achieves near real-time performance, fulfilling the `NFR1: Performance` requirement. The modular design, supported by clear interfaces and configurable parameters, enhances `NFR3: Maintainability` and allows for future extensions.

The system accurately detects and tracks vehicles, providing crucial data points such as bounding boxes, unique track IDs, and vehicle classes. The vehicle counting functionality, empowered by `shapely`'s geometric capabilities, reliably records vehicle movements across user-defined lines, preventing double-counting and offering class-specific statistics. The integration of license plate detection and OCR adds a valuable layer of detail for specific use cases.

The implemented logging strategy (`NFR4: Reliability`) provides comprehensive insights into the system's operation, aiding in debugging and performance monitoring. All identified challenges during development were addressed, demonstrating the project's robustness and the effectiveness of the chosen solutions.

## 7. Conclusion

This capstone project successfully demonstrates the design and implementation of a real-time traffic monitoring system. The project highlights the benefits of a modular, multiprocessing architecture for handling complex computer vision pipelines. It effectively integrates state-of-the-art object detection (YOLO), multi-object tracking (BoxMOT), and optical character recognition technologies.

The system provides a solid foundation for various traffic intelligence applications, from basic vehicle counting to more advanced analytics. Future work could include integrating a more sophisticated UI, supporting a wider range of video sources (e.g., live camera feeds with network resilience), implementing traffic density analysis, and exploring advanced predictive analytics based on the gathered data. Further optimization for embedded systems or cloud deployment could also be considered to enhance scalability and deployment flexibility.
