# COMPONENT_OCR_DOCS.md

## OCR Component Documentation

### Overview

The OCR (Optical Character Recognition) component, primarily implemented by the `OCRReader` class and its associated engine classes (`FastPlateOCREngine`, `PaddleOCREngine`), is responsible for extracting text from image regions, specifically designed for license plate recognition within the traffic monitoring system. It supports integration with various OCR engines, offering flexibility and adaptability to different performance and accuracy requirements.

### Technical Requirements

- Support for multiple OCR engines (e.g., FastPlateOCR, PaddleOCR).
- Ability to process image crops and return recognized text with confidence scores.
- Configurable parameters for each OCR engine.
- Robust error handling and logging for OCR processing failures.

### Area of Responsibility

- Initializing the selected OCR engine based on configuration.
- Performing OCR on input image data.
- Handling pre-processing (e.g., grayscale conversion for FastPlateOCR) and post-processing (e.g., text cleaning, confidence aggregation) of OCR results.
- Providing a standardized output format for OCR results (`OCRResult` dataclass).

### Classes and Files

#### `OCRResult` (src/traffic_monitor/utils/custom_types.py)

A dataclass used to standardize the output of OCR operations.

- **Attributes:**
  - `text` (str): The recognized text.
  - `confidence` (float): The confidence score of the recognition.
  - `processing_time` (float): Time taken to process the image by the OCR engine.
  - `engine` (str): Name of the OCR engine used.

#### `BaseOCREngine` (scripts/ocr_dataset_processor.py)

An abstract base class defining the interface for all OCR engines.

- **Methods:**
  - `__init__(self, config: Dict[str, Any])`: Initializes the engine with a configuration dictionary.
  - `process_image(self, image: np.ndarray) -> Optional[OCRResult]`: Abstract method to be implemented by concrete OCR engine classes. Processes an image and returns an `OCRResult`.

#### `FastPlateOCREngine` (scripts/ocr_dataset_processor.py)

Implements the `BaseOCREngine` for FastPlateOCR, an ONNX-based OCR solution.

- **Initialization Parameters (from `config`):**
  - `hub_model_name` (str, optional): Model name for FastPlateOCR (default: "global-plates-mobile-vit-v2-model").
  - `device` (str, optional): Device to run inference on ("auto", "cpu", "cuda") (default: "auto").
  - `conf_threshold` (float, optional): Confidence threshold for filtering OCR results (default: 0.5).
- **Key Methods:**
  - `_preprocess_image(self, image: np.ndarray) -> np.ndarray`: Converts image to grayscale if it's a color image.
  - `process_image(self, image: np.ndarray) -> Optional[OCRResult]`: Runs FastPlateOCR on the image and returns `OCRResult`. Handles cases of no detection or low confidence.

#### `PaddleOCREngine` (scripts/ocr_dataset_processor.py)

Implements the `BaseOCREngine` for PaddleOCR.

- **Initialization Parameters (from `config`):**
  - `lang` (str, optional): Language for PaddleOCR (default: "en").
  - `conf_threshold` (float, optional): Confidence threshold for filtering OCR results (default: 0.5).
- **Key Methods:**
  - `process_image(self, image: np.ndarray) -> Optional[OCRResult]`: Runs PaddleOCR on the image and returns `OCRResult`. Includes logic to clean recognized text and select the best result.

#### `OCRReader` (src/traffic_monitor/services/ocr_reader.py)

(Note: This class is conceptual based on the architecture, actual implementation details will be in `src/traffic_monitor/services/ocr_reader.py`)
This service integrates with the `main_supervisor.py` and is responsible for managing the OCR process within the multiprocessing pipeline.

- **Key Responsibilities:**
  - Consuming image data (e.g., cropped license plates) from an input queue.
  - Invoking the appropriate `BaseOCREngine` (FastPlateOCR or PaddleOCR) to perform text recognition.
  - Publishing OCR results to an output queue for other components (e.g., `Visualizer`, data logging).

### Dependencies

- `cv2` (OpenCV): For image manipulation.
- `numpy`: For numerical operations on images and confidence scores.
- `loguru`: For structured logging.
- `pathlib`: For path manipulation.
- `fast_plate_ocr` (external library): Used by `FastPlateOCREngine`.
- `paddleocr` (external library): Used by `PaddleOCREngine`.
- `src.traffic_monitor.utils.custom_types`: For `OCRResult` and other custom data types.

### Configuration

The OCR component is configured via the `ocr_reader` section in `src/traffic_monitor/config/settings.yaml`.

```yaml
ocr_reader:
  # Backend can be "fast_plate_ocr" (default) or "paddleocr"
  backend: "paddleocr"

  # --- Backend-agnostic parameters ---
  conf_threshold: 0.5

  # --- PaddleOCR-specific parameters ---
  lang: "en" # Language for recognition
  use_gpu: false # Set to true if running with CUDA-enabled PaddlePaddle

  # --- FastPlateOCR-specific parameters ---
  hub_model_name: "global-plates-mobile-vit-v2-model"
  device: "auto" # "cpu" | "cuda" | "auto"
```

If `backend` is omitted the system defaults to `fast_plate_ocr` to preserve backward compatibility. When `paddleocr` is chosen the `lang` and `use_gpu` flags are passed directly to the `PaddleOCR` constructor.

### Data Flow

1. `LPDetector` (or another component) detects license plate regions and sends image crops to `OCRReader`'s input queue.
2. `OCRReader` retrieves the image crop.
3. `OCRReader` passes the image crop to the configured OCR engine's `process_image` method.
4. The OCR engine returns an `OCRResult` object.
5. `OCRReader` publishes the `OCRResult` to an output queue.
6. Downstream components (e.g., `Visualizer`, `main_supervisor.py` for logging) consume the `OCRResult` to display or record the recognized text.
