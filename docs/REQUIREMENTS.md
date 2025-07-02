## Functional Requirements

### FR1: Video Stream Processing

- The system shall be able to ingest video streams from various sources (e.g., local video files, IP cameras).
- The system shall decode video frames for further processing.
- The system shall support configurable frame skipping to optimize performance.

### FR2: Vehicle Detection

- The system shall detect vehicles within each video frame.
- The system shall identify the bounding box, confidence score, and class ID for each detected vehicle.
- The system shall support configurable confidence thresholds for detection.
- The system shall utilize a pre-trained YOLO model for object detection.
- The system shall use correct YOLO COCO class mappings (e.g., 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck).

### FR3: Vehicle Tracking

- The system shall track detected vehicles across multiple frames, assigning a unique track ID to each persistent object.
- The system shall update tracked objects with their current bounding box, confidence, and class information.
- The system shall support configurable tracker types (e.g., ByteTrack).
- The system shall allow for the integration of Re-ID models for improved tracking accuracy.

### FR4: Multiprocessing Architecture

- The system shall utilize a multiprocessing architecture to handle frame grabbing, vehicle detection, and vehicle tracking concurrently.
- The system shall use inter-process communication (queues) for passing data between modules.
- The system shall support graceful shutdown of all processes.

### FR5: Configuration Management

- The system shall load configuration parameters from a YAML file (`settings.yaml`).
- The system shall allow configuration of video sources, model paths, confidence thresholds, and tracker parameters.

### FR6: Logging ✅ **IMPLEMENTED**

- The system shall implement structured logging for debugging, informational, and error messages. ✅
- The system shall support configurable log levels and formats with environment variable control. ✅
- The system shall provide detailed class-specific logging for vehicle detection, tracking, and counting. ✅
- The system shall log specific vehicle class names (e.g., "car", "bicycle", "truck") instead of generic "vehicle" terms. ✅
- The system shall include track IDs and class information in vehicle crossing detection logs. ✅
- The system shall provide class-specific count summaries in detection and tracking logs. ✅
- The system shall automatically redact sensitive information (license plates, API keys, passwords) from logs. ✅
- The system shall support JSON structured logging for production environments. ✅
- The system shall suppress noise from third-party libraries (matplotlib, urllib3, etc.). ✅
- The system shall provide process-aware logging for multiprocessing environments. ✅
- The system shall implement appropriate log levels (TRACE for high-frequency, DEBUG for summaries, INFO for lifecycle events). ✅

### FR7: Vehicle Counting

- The system shall count vehicles crossing predefined counting lines.
- The system shall use geometric line intersection to detect when a vehicle crosses a counting line.
- The system shall maintain separate counts by vehicle class (car, truck, bus, etc.).
- The system shall prevent double-counting of the same vehicle crossing the same line.
- The system shall support multiple counting lines simultaneously.
- The system shall provide counting line coordinates configuration via YAML settings.

### FR8: Visualization and Display

- The system shall display real-time video with tracking overlays using OpenCV.
- The system shall draw vehicle bounding boxes with track IDs and class labels.
- The system shall display counting lines on the video feed from the start of the application.
- The system shall provide configurable appearance for counting lines (color, thickness).
- The system shall label each counting line with a unique identifier.
- The system shall display real-time vehicle counts and FPS information.
- The system shall support display of OCR results for license plates when available.
- The system shall provide visual feedback for all counting zones from the first frame.

### FR9: OCR Processing

- The system shall support processing images with multiple OCR engines (e.g., FastPlateOCR, PaddleOCR).
- The system shall be able to extract text and confidence scores from processed images.
- The system shall handle different OCR engine configurations.
- The system shall log errors encountered during OCR processing.

### FR10: Data Export

- The system shall support exporting data to various formats (e.g., CSV, JSON, XML).
- The system shall allow for selective export of data based on vehicle class or time range.
- The system shall provide detailed export logs for tracking and counting activities.

### FR11: Data Persistence ✅ **IMPLEMENTED**

- The system shall persist plate recognition results and vehicle count data in SQLite for later analytics. ✅
- The system shall store timestamp, camera ID, vehicle ID, vehicle class, license plate text, and OCR confidence for each plate detection. ✅
- The system shall store timestamp, camera ID, total count, and class-specific counts for vehicle counting events. ✅
- The system shall use WAL mode for better concurrency and automatic retry on database locks. ✅
- The system shall support configurable database path and SQLite optimization settings via settings.yaml. ✅
- The system shall automatically create database directories and apply PRAGMA settings from configuration. ✅
- The system shall provide an option to reset (delete and re-create) the database file on every startup. ✅
- The system shall index vehicle class data for efficient querying and filtering by vehicle type. ✅
- The system shall eliminate duplicate plate readings per vehicle using confidence-based selection. ✅
- The system shall maintain a complete audit trail of all OCR attempts while providing authoritative "latest" results. ✅
- The system shall implement "best confidence wins" logic to ensure only the highest quality plate reading is used for each vehicle. ✅

## Non-Functional Requirements

### NFR1: Performance

- The system shall process video frames with minimal latency to support near real-time traffic monitoring.
- The tracking component should be optimized for efficient processing of detections.

### NFR2: Scalability

- The multiprocessing architecture should allow for potential scaling of individual components (e.g., running multiple detectors or trackers).

### NFR3: Maintainability

- The codebase shall adhere to a clear project structure and modular design.
- The code shall include type hints and docstrings for improved readability and maintainability.

### NFR4: Reliability

- The system shall handle empty input queues and gracefully manage process shutdowns.
- The system shall log errors and exceptions for debugging.
