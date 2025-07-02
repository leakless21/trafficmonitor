# Traffic Monitor System

## Project Overview

The Traffic Monitor System is a robust, modular, and real-time video analytics application designed for comprehensive traffic monitoring. It leverages advanced computer vision techniques, including YOLO for vehicle detection, BoxMOT for multi-object tracking, and various OCR engines for license plate recognition. The system is built with a multiprocessing architecture to ensure efficient and concurrent processing of video streams, providing functionalities like vehicle counting, speed estimation, and detailed data logging.

## Features

- **Real-time Video Processing**: Ingests and processes video streams from diverse sources (local files, IP cameras).
- **Advanced Vehicle Detection**: Utilizes YOLO models for accurate and efficient vehicle detection with configurable confidence thresholds.
- **Robust Multi-Object Tracking**: Integrates BoxMOT for persistent vehicle tracking across frames, supporting various tracker types and Re-ID models.
- **Multiprocessing Architecture**: Employs a distributed multiprocessing design to concurrently handle frame grabbing, vehicle detection, tracking, OCR, and data persistence, ensuring high throughput and responsiveness.
- **Vehicle Counting**: Precisely counts vehicles crossing user-defined virtual counting lines, with class-specific counts and anti-double-counting mechanisms.
- **Optical Character Recognition (OCR)**: Integrates multiple OCR engines (FastPlateOCR, PaddleOCR) for accurate license plate recognition.
- **Real-time Visualization**: Provides an intuitive graphical user interface (GUI) via OpenCV, displaying live video feeds with bounding boxes, track IDs, class labels, counting lines, real-time counts, and FPS.
- **Configurable Settings**: All operational parameters, including video sources, model paths, thresholds, tracker types, and logging settings, are managed via a centralized YAML configuration.
- **Comprehensive Logging**: Implements structured, process-aware logging with configurable levels and formats, outputting to both console and files for debugging and monitoring.
- **Data Persistence**: Stores critical data such as plate recognition results and vehicle counts into a lightweight SQLite database, enabling post-analysis and reporting.
- **Modular and Extensible Design**: Built with a clear project structure and modular components, facilitating easy maintenance, updates, and future enhancements.

## Project Structure

```
trafficmonitor/
├── src/
│   └── traffic_monitor/        # Main package (importable)
│       ├── cli.py             # Command-line interface
│       ├── main_supervisor.py # Main orchestrator
│       ├── config/            # Configuration files
│       ├── services/          # Core processing services
│       └── utils/             # Utilities and helpers
├── tools/                     # Development scripts and utilities
├── data/
│   ├── models/                # AI model weights (git-ignored)
│   └── videos/                # Sample videos
├── tests/                     # Test suite
├── docs/                      # Documentation
└── pyproject.toml            # Package configuration
```

## Architecture

The system is designed with a multiprocessing architecture, where each core functionality operates as an independent process, communicating via inter-process queues. Key components include:

- **`MainSupervisor`**: Orchestrates the entire system, launching and managing all worker processes.
- **`FrameGrabber`**: Ingests video streams, decodes frames, and prepares them for further processing. Supports frame skipping and resizing.
- **`VehicleDetector`**: Detects vehicles within frames using a YOLO model, identifying bounding boxes, confidence scores, and class IDs.
- **`VehicleTracker`**: Tracks detected vehicles across frames using the BoxMOT library, assigning unique track IDs and maintaining object states.
- **`VehicleCounter`**: Counts vehicles based on geometric intersections with predefined counting lines, preventing double-counting.
- **`LPDetector`**: (Implicit, based on OCR input) Likely responsible for detecting license plates within vehicle bounding boxes before passing them to the OCRReader.
- **`OCRReader`**: Performs Optical Character Recognition on license plate crops using configurable OCR engines (FastPlateOCR, PaddleOCR).
- **`Visualizer`**: Renders real-time video feeds with overlays, including bounding boxes, track IDs, counting lines, and dynamic statistics.
- **`Persistence (minidb)`**: Manages data storage in an SQLite database for plate recognition results and vehicle count data, ensuring robust and concurrent write operations.
- **`Logging`**: A centralized logging system provides detailed insights into the application's runtime behavior, supporting multiprocessing environments and configurable output.

Inter-process communication is managed using `multiprocessing.Queue` to ensure efficient data flow and prevent bottlenecks.

## Setup and Installation

**Prerequisites:**

- Python 3.10-3.11
- OpenCV
- PyTorch (for YOLO and Re-ID models)
- `pixi` (recommended for dependency management)

**1. Clone the repository:**

```bash
git clone https://github.com/your-username/trafficmonitor.git
cd trafficmonitor
```

**2. Install dependencies using Pixi (Recommended):**

If you have Pixi installed, you can set up the environment and install dependencies with:

```bash
pixi install
pixi shell
```

**3. Manual Installation (if not using Pixi):**

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: `venv\Scripts\activate`
pip install -e .
```

**4. Download Models:**

Pre-trained models for YOLO, Re-ID, and OCR are required. You can download them using the provided script:

```bash
python tools/download_model.py
```

Ensure that the `data/models/` directory contains the necessary `.pt` and `.onnx` files as specified in `src/traffic_monitor/config/settings.yaml`.

## Configuration

All system configurations are managed in `src/traffic_monitor/config/settings.yaml`. Key parameters include:

- `video_source`: Path to your video file or camera index.
- `resize_resolution`: Target resolution for processed frames.
- `detector`: YOLO model path, confidence threshold, and IOU threshold.
- `tracker`: Tracker type (e.g., `bytetrack`), Re-ID model path, and device (`cpu` or `cuda`).
- `ocr`: OCR backend (`fast_plate_ocr` or `paddleocr`) and model paths.
- `counting_lines`: Coordinates for virtual counting lines.
- `logging`: Logging levels, file paths, and rotation settings.
- `database`: SQLite database path and optimization settings.

**Example `settings.yaml` snippet:**

```yaml
video_source: "data/videos/your_video.mp4"
resize_resolution: [1280, 720]

detector:
  model_path: "data/models/yolo11n.pt"
  conf_threshold: 0.25
  iou_threshold: 0.45

tracker:
  tracker_type: "bytetrack"
  reid_model_path: "data/models/lp.pt" # Example, adjust as per your Re-ID model
  device: "cpu"

ocr:
  backend: "fast_plate_ocr"
  model_path: "data/models/plate_v8n.pt"

counting_lines:
  - name: "Line 1"
    coords: [[100, 200], [500, 200]]
    class_filter: ["car", "bus", "truck"]

logging:
  level: "INFO"
  file: "logs/traffic_monitor.log"

database:
  path: "traffic_monitor.db"
  optimize: true
```

## Usage

### Command Line Interface

The system now provides a clean CLI interface:

```bash
# Using the CLI directly
traffic-monitor

# With options
traffic-monitor --verbose --config path/to/custom/settings.yaml

# Using pixi
pixi run traffic-monitor

# Legacy method (still works)
python src/traffic_monitor/main_supervisor.py
```

### Available Commands

```bash
# Start the main monitoring system
traffic-monitor

# Development utilities (in tools/)
python tools/batch_plate_crop.py
python tools/ocr_evaluation.py
python tools/comparison_evaluation.py

# Or via pixi tasks
pixi run batch_crop
pixi run ocr_eval
pixi run ocr_compare
```

## Testing

Unit and integration tests are located in the `test/` directory. To run tests, use `pytest`:

```bash
pytest
```

## Future Enhancements

- **Speed Estimation**: Implement functionality to estimate vehicle speeds.
- **Anomaly Detection**: Develop modules for detecting unusual traffic patterns or events.
- **Web Interface**: Create a web-based dashboard for remote monitoring and data visualization.
- **Cloud Deployment**: Explore deployment options on cloud platforms (e.g., AWS, Azure, GCP) for scalable and resilient operations.
- **Advanced Analytics**: Integrate with data analytics tools for deeper insights into traffic flow and patterns.

## Contributing

Contributions are welcome! Please refer to `CONTRIBUTING.md` (if available) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
