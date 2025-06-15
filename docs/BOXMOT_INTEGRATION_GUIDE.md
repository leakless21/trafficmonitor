# BoxMOT Integration Guide for Python Applications

## Table of Contents

1. [Introduction to BoxMOT](#introduction-to-boxmot)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Basic Usage](#basic-usage)
5. [Advanced Features and Configuration](#advanced-features-and-configuration)
6. [Examples and Best Practices](#examples-and-best-practices)
7. [Troubleshooting Tips](#troubleshooting-tips)
8. [API Reference](#api-reference)

## Introduction to BoxMOT

BoxMOT is a comprehensive multi-object tracking (MOT) library that provides pluggable, state-of-the-art tracking modules for segmentation, object detection, and pose estimation models. It addresses the fragmented nature of the MOT field by offering a standardized collection of tracking algorithms that can be easily integrated into computer vision applications.

### Key Features:

- **Multiple Tracking Algorithms**: Supports 6 state-of-the-art trackers including ByteTrack, BotSort, StrongSort, OcSort, DeepOcSort, and BoostTrack
- **Hardware Flexibility**: Optimized for various hardware constraints from CPU-only setups to high-end GPUs
- **ReID Integration**: Includes automatic downloading of re-identification models for appearance-based tracking
- **Easy Integration**: Seamless integration with popular detection models (YOLO, Torchvision, etc.)
- **Performance Optimized**: Designed for real-time tracking applications

### Performance Comparison:

| Tracker    | HOTA↑  | MOTA↑  | IDF1↑  | FPS  |
| ---------- | ------ | ------ | ------ | ---- |
| BoostTrack | 69.253 | 75.914 | 83.206 | 25   |
| BotSort    | 68.885 | 78.222 | 81.344 | 46   |
| StrongSort | 68.05  | 76.185 | 80.763 | 17   |
| ByteTrack  | 67.68  | 78.039 | 79.157 | 1265 |
| DeepOcSort | 67.509 | 75.83  | 79.976 | 12   |
| OcSort     | 66.441 | 74.548 | 77.899 | 1483 |

## Installation

### Prerequisites

- Python >= 3.9
- PyTorch (for deep learning models)
- OpenCV (for video processing)

### Basic Installation

Install BoxMOT using pip in a Python virtual environment:

```bash
# Create and activate virtual environment (recommended)
python -m venv boxmot_env
source boxmot_env/bin/activate  # On Windows: boxmot_env\Scripts\activate

# Install BoxMOT
pip install boxmot
```

### Development Installation

For development or to access the latest features:

```bash
git clone https://github.com/mikel-brostrom/boxmot.git
cd boxmot
pip install -e .
```

### Verify Installation

```python
import boxmot
print(f"BoxMOT version: {boxmot.__version__}")
print(f"Available trackers: {boxmot.TRACKERS}")
```

## Core Concepts

### 1. Detection Format

BoxMOT expects detections in a specific format:

- **Input**: NumPy array of shape `(N, 6)` where N is the number of detections
- **Format**: `[x1, y1, x2, y2, confidence, class_id]`
  - `(x1, y1)`: Top-left corner coordinates
  - `(x2, y2)`: Bottom-right corner coordinates
  - `confidence`: Detection confidence score (0.0 to 1.0)
  - `class_id`: Object class identifier

### 2. Tracking Output

The tracker returns results in the format:

- **Output**: NumPy array of shape `(M, 8)` where M is the number of tracked objects
- **Format**: `[x1, y1, x2, y2, track_id, confidence, class_id, detection_index]`
  - `track_id`: Unique identifier for the tracked object
  - `detection_index`: Index of the detection in the original detection array

### 3. Tracker Types

- **Motion-only trackers**: ByteTrack, OcSort (fast, no appearance features)
- **Appearance-based trackers**: StrongSort, BotSort, DeepOcSort, BoostTrack (more accurate, require ReID models)

### 4. ReID Models

For appearance-based tracking, BoxMOT supports various ReID models:

- **Lightweight**: `lmbn_n_cuhk03_d.pt`, `osnet_x0_25_market1501.pt`
- **Heavyweight**: `clip_market1501.pt`, `clip_vehicleid.pt`

## Basic Usage

### Simple Tracking Example

```python
import cv2
import numpy as np
from boxmot import create_tracker

# Initialize tracker
tracker = create_tracker(
    tracker_type='bytetrack',  # Choose from: bytetrack, botsort, strongsort, ocsort, deepocsort, boosttrack
    tracker_config=None
    per_class=None      # Use default config
    reid_weights=None,         # Not needed for ByteTrack
    device='cpu',              # or 'cuda' for GPU
    half=False                 # Use FP32
    evolve_param_dict=None
)

# Example detection data (replace with your detector output)
# Format: [x1, y1, x2, y2, confidence, class_id]
detections = np.array([
    [100, 100, 200, 200, 0.9, 0],  # Person detection
    [300, 150, 400, 250, 0.8, 0],  # Another person
])

# Dummy frame (replace with actual video frame)
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Update tracker with detections
tracks = tracker.update(detections, frame)

print(f"Number of tracks: {len(tracks)}")
for track in tracks:
    x1, y1, x2, y2, track_id, conf, cls, det_idx = track
    print(f"Track ID: {track_id}, Box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
```

### Complete Video Tracking Pipeline

```python
import cv2
import numpy as np
from pathlib import Path
from boxmot import create_tracker

def track_video(video_path, tracker_type='bytetrack'):
    """
    Complete video tracking pipeline
    """
    # Initialize tracker
    tracker = create_tracker(
        tracker_type=tracker_type,
        tracker_config=None,
        reid_weights=Path('osnet_x0_25_msmt17.pt') if tracker_type in ['strongsort', 'botsort', 'deepocsort'] else None,
        device='cpu',
        half=False
    )

    # Open video
    cap = cv2.VideoCapture(video_path)

    # Initialize your detector here (example with dummy detections)
    def get_detections(frame):
        # Replace this with your actual detection model
        # This is just a placeholder
        return np.array([[100, 100, 200, 200, 0.9, 0]])

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get detections from your detector
        detections = get_detections(frame)

        # Update tracker
        tracks = tracker.update(detections, frame)

        # Visualize results
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls, det_idx = track

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Draw track ID
            cv2.putText(frame, f'ID: {int(track_id)}',
                       (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display frame
        cv2.imshow('Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

# Usage
track_video('path/to/your/video.mp4', 'bytetrack')
```

## Advanced Features and Configuration

### 1. Tracker Configuration

Each tracker can be configured with custom parameters. You can either use default configurations or provide custom ones:

```python
from boxmot import create_tracker, get_tracker_config

# Get default config path
config_path = get_tracker_config('bytetrack')
print(f"Config path: {config_path}")

# Create tracker with custom parameters
tracker = create_tracker(
    tracker_type='strongsort',
    tracker_config=config_path,
    reid_weights=Path('osnet_x0_25_msmt17.pt'),
    device='cuda',
    half=True,  # Use FP16 for faster inference
    per_class=False  # Track all classes together
)
```

### 2. Custom Configuration Parameters

Common configuration parameters across trackers:

```python
# Example custom parameters for ByteTrack
custom_params = {
    'track_thresh': 0.5,      # Detection confidence threshold
    'track_buffer': 30,       # Number of frames to keep lost tracks
    'match_thresh': 0.8,      # Matching threshold for data association
    'frame_rate': 30          # Video frame rate
}

# For appearance-based trackers (StrongSort, BotSort, etc.)
appearance_params = {
    'max_dist': 0.2,          # Maximum distance for ReID matching
    'max_iou_distance': 0.7,  # Maximum IoU distance for matching
    'max_age': 70,            # Maximum age of tracks
    'n_init': 3,              # Number of consecutive detections before track confirmation
    'nn_budget': 100          # Maximum size of appearance descriptor gallery
}
```

### 3. ReID Model Selection

Choose appropriate ReID models based on your requirements:

```python
# Lightweight models (faster, less accurate)
lightweight_models = [
    'lmbn_n_cuhk03_d.pt',
    'osnet_x0_25_market1501.pt',
    'mobilenetv2_x1_4_msmt17.engine'
]

# Heavyweight models (slower, more accurate)
heavyweight_models = [
    'clip_market1501.pt',
    'clip_vehicleid.pt',
    'resnet50_msmt17.onnx'
]

# Usage
tracker = create_tracker(
    tracker_type='strongsort',
    reid_weights=Path('osnet_x0_25_msmt17.pt'),  # Choose based on your needs
    device='cuda',
    half=True
)
```

### 4. Class-Specific Tracking

Enable per-class tracking for better performance:

```python
tracker = create_tracker(
    tracker_type='botsort',
    reid_weights=Path('osnet_x0_25_msmt17.pt'),
    device='cpu',
    per_class=True  # Track each class separately
)
```

### 5. Multi-Class Filtering

Filter specific classes during tracking:

```python
def filter_detections_by_class(detections, allowed_classes=[0, 1, 2]):
    """
    Filter detections to only include specific classes
    allowed_classes: List of class IDs to track (e.g., [0] for person only)
    """
    if len(detections) == 0:
        return detections

    mask = np.isin(detections[:, 5], allowed_classes)
    return detections[mask]

# Usage in tracking loop
detections = get_detections(frame)
detections = filter_detections_by_class(detections, allowed_classes=[0])  # Track persons only
tracks = tracker.update(detections, frame)
```

## Examples and Best Practices

### 1. Integration with YOLO Detection

```python
import torch
from ultralytics import YOLO
from boxmot import create_tracker
import cv2
import numpy as np

class YOLOTracker:
    def __init__(self, yolo_model_path, tracker_type='bytetrack'):
        # Initialize YOLO detector
        self.detector = YOLO(yolo_model_path)

        # Initialize tracker
        self.tracker = create_tracker(
            tracker_type=tracker_type,
            reid_weights=Path('osnet_x0_25_msmt17.pt') if tracker_type in ['strongsort', 'botsort'] else None,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            half=torch.cuda.is_available()
        )

    def detect_and_track(self, frame):
        # Run YOLO detection
        results = self.detector(frame, verbose=False)

        # Convert YOLO results to BoxMOT format
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = box.cls[0].cpu().numpy()
                    detections.append([x1, y1, x2, y2, conf, cls])

        detections = np.array(detections) if detections else np.empty((0, 6))

        # Update tracker
        tracks = self.tracker.update(detections, frame)

        return tracks

# Usage
tracker = YOLOTracker('yolov8n.pt', 'bytetrack')
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    tracks = tracker.detect_and_track(frame)

    # Visualize tracks
    for track in tracks:
        x1, y1, x2, y2, track_id, conf, cls, _ = track
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('YOLO + BoxMOT', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 2. Batch Processing for Multiple Videos

```python
from pathlib import Path
import json

def process_video_batch(video_dir, output_dir, tracker_type='bytetrack'):
    """
    Process multiple videos and save tracking results
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Initialize tracker once
    tracker = create_tracker(
        tracker_type=tracker_type,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    for video_path in video_dir.glob('*.mp4'):
        print(f"Processing: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        results = []

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get detections (replace with your detector)
            detections = get_detections(frame)

            # Update tracker
            tracks = tracker.update(detections, frame)

            # Store results
            frame_results = {
                'frame': frame_idx,
                'tracks': [
                    {
                        'track_id': int(track[4]),
                        'bbox': [float(x) for x in track[:4]],
                        'confidence': float(track[5]),
                        'class': int(track[6])
                    }
                    for track in tracks
                ]
            }
            results.append(frame_results)
            frame_idx += 1

        # Save results
        output_file = output_dir / f"{video_path.stem}_tracks.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        cap.release()
        print(f"Saved results to: {output_file}")

# Usage
process_video_batch('input_videos/', 'tracking_results/', 'bytetrack')
```

### 3. Real-time Performance Optimization

```python
import time
from collections import deque

class PerformanceOptimizedTracker:
    def __init__(self, tracker_type='bytetrack', max_fps=30):
        self.tracker = create_tracker(
            tracker_type=tracker_type,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            half=True  # Use FP16 for speed
        )
        self.max_fps = max_fps
        self.frame_time = 1.0 / max_fps
        self.fps_history = deque(maxlen=30)

    def track_frame(self, frame, detections):
        start_time = time.time()

        # Update tracker
        tracks = self.tracker.update(detections, frame)

        # Calculate FPS
        process_time = time.time() - start_time
        self.fps_history.append(1.0 / max(process_time, 0.001))

        # Frame rate limiting
        if process_time < self.frame_time:
            time.sleep(self.frame_time - process_time)

        return tracks

    def get_average_fps(self):
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

# Usage
tracker = PerformanceOptimizedTracker('bytetrack', max_fps=30)

# In your main loop
tracks = tracker.track_frame(frame, detections)
print(f"Average FPS: {tracker.get_average_fps():.1f}")
```

### 4. Best Practices

#### Memory Management

```python
# For long-running applications, periodically reset tracker
frame_count = 0
reset_interval = 1000  # Reset every 1000 frames

while True:
    # ... tracking code ...

    frame_count += 1
    if frame_count % reset_interval == 0:
        # Reset tracker to prevent memory accumulation
        tracker = create_tracker(tracker_type, ...)
        print(f"Tracker reset at frame {frame_count}")
```

#### Error Handling

```python
def robust_tracking(frame, detections, tracker):
    """
    Robust tracking with error handling
    """
    try:
        # Validate input
        if detections.shape[1] != 6:
            raise ValueError(f"Expected 6 columns in detections, got {detections.shape[1]}")

        # Check for valid detections
        if len(detections) == 0:
            return np.empty((0, 8))

        # Update tracker
        tracks = tracker.update(detections, frame)

        return tracks

    except Exception as e:
        print(f"Tracking error: {e}")
        return np.empty((0, 8))  # Return empty tracks on error
```

#### Confidence Filtering

```python
def filter_low_confidence_detections(detections, min_confidence=0.5):
    """
    Filter out low-confidence detections before tracking
    """
    if len(detections) == 0:
        return detections

    high_conf_mask = detections[:, 4] >= min_confidence
    return detections[high_conf_mask]

# Usage
detections = get_detections(frame)
detections = filter_low_confidence_detections(detections, min_confidence=0.6)
tracks = tracker.update(detections, frame)
```

## Troubleshooting Tips

### Common Issues and Solutions

#### 1. Import Errors

```python
# Problem: ModuleNotFoundError: No module named 'boxmot'
# Solution: Ensure proper installation
pip install boxmot

# Problem: CUDA out of memory
# Solution: Use CPU or reduce batch size
tracker = create_tracker(tracker_type='bytetrack', device='cpu')
```

#### 2. Detection Format Issues

```python
# Problem: Tracker expects specific detection format
# Solution: Always validate detection format
def validate_detections(detections):
    if not isinstance(detections, np.ndarray):
        detections = np.array(detections)

    if len(detections.shape) != 2 or detections.shape[1] != 6:
        raise ValueError(f"Detections must be (N, 6) array, got {detections.shape}")

    return detections.astype(np.float32)

# Usage
detections = validate_detections(raw_detections)
```

#### 3. Performance Issues

```python
# Problem: Slow tracking performance
# Solutions:
# 1. Use faster tracker
tracker = create_tracker('bytetrack')  # Fastest option

# 2. Use GPU acceleration
tracker = create_tracker('bytetrack', device='cuda', half=True)

# 3. Reduce detection frequency
frame_skip = 2  # Process every 2nd frame
if frame_count % frame_skip == 0:
    tracks = tracker.update(detections, frame)
```

#### 4. Memory Leaks

```python
# Problem: Memory usage increases over time
# Solution: Implement periodic cleanup
import gc

def cleanup_tracker(tracker, frame_count, cleanup_interval=1000):
    if frame_count % cleanup_interval == 0:
        # Force garbage collection
        gc.collect()

        # Optionally recreate tracker
        if hasattr(tracker, 'reset'):
            tracker.reset()
```

#### 5. ReID Model Download Issues

```python
# Problem: ReID model download fails
# Solution: Manual download and path specification
from pathlib import Path

reid_model_path = Path('models/osnet_x0_25_msmt17.pt')
if not reid_model_path.exists():
    print("Please download ReID model manually")
    # Provide download instructions or implement manual download

tracker = create_tracker(
    tracker_type='strongsort',
    reid_weights=reid_model_path if reid_model_path.exists() else None
)
```

### Debugging Tips

#### 1. Enable Verbose Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Track detection statistics
def debug_detections(detections, frame_idx):
    print(f"Frame {frame_idx}: {len(detections)} detections")
    if len(detections) > 0:
        print(f"  Confidence range: {detections[:, 4].min():.3f} - {detections[:, 4].max():.3f}")
        print(f"  Classes: {np.unique(detections[:, 5])}")
```

#### 2. Visualize Tracking Results

```python
def visualize_tracking_debug(frame, detections, tracks):
    """
    Debug visualization showing both detections and tracks
    """
    debug_frame = frame.copy()

    # Draw detections in red
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        cv2.putText(debug_frame, f'Det: {conf:.2f}', (int(x1), int(y1) - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

    # Draw tracks in green
    for track in tracks:
        x1, y1, x2, y2, track_id, conf, cls, _ = track
        cv2.rectangle(debug_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(debug_frame, f'ID: {int(track_id)}', (int(x1), int(y1) - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return debug_frame
```

## API Reference

### Core Functions

#### `create_tracker()`

Creates and returns a tracker instance.

```python
def create_tracker(
    tracker_type: str,
    tracker_config: Optional[str] = None,
    reid_weights: Optional[Path] = None,
    device: Optional[str] = None,
    half: Optional[bool] = None,
    per_class: Optional[bool] = None,
    evolve_param_dict: Optional[dict] = None
) -> BaseTracker
```

**Parameters:**

- `tracker_type`: Tracker algorithm ('bytetrack', 'botsort', 'strongsort', 'ocsort', 'deepocsort', 'boosttrack')
- `tracker_config`: Path to configuration file (optional)
- `reid_weights`: Path to ReID model weights (required for appearance-based trackers)
- `device`: Computing device ('cpu', 'cuda')
- `half`: Use half-precision (FP16) for faster inference
- `per_class`: Enable per-class tracking
- `evolve_param_dict`: Custom parameters dictionary

#### `get_tracker_config()`

Returns the path to the default configuration file for a tracker.

```python
def get_tracker_config(tracker_type: str) -> Path
```

### Tracker Classes

All trackers inherit from `BaseTracker` and implement the following interface:

#### `update()`

Updates the tracker with new detections.

```python
def update(
    self,
    dets: np.ndarray,
    img: np.ndarray
) -> np.ndarray
```

**Parameters:**

- `dets`: Detection array of shape (N, 6) with format [x1, y1, x2, y2, conf, cls]
- `img`: Current frame as numpy array

**Returns:**

- Tracking results array of shape (M, 8) with format [x1, y1, x2, y2, track_id, conf, cls, det_idx]

#### `plot_results()`

Visualizes tracking results on the frame.

```python
def plot_results(
    self,
    img: np.ndarray,
    show_trajectories: bool = False
) -> np.ndarray
```

### Available Trackers

#### ByteTrack

- **Type**: Motion-only
- **Speed**: Very fast (1265 FPS)
- **Use case**: Real-time applications, simple scenarios

#### BotSort

- **Type**: Appearance + Motion
- **Speed**: Fast (46 FPS)
- **Use case**: Balanced performance and accuracy

#### StrongSort

- **Type**: Appearance + Motion
- **Speed**: Moderate (17 FPS)
- **Use case**: High accuracy requirements

#### OcSort

- **Type**: Motion-only
- **Speed**: Very fast (1483 FPS)
- **Use case**: Simple tracking, high-speed requirements

#### DeepOcSort

- **Type**: Appearance + Motion
- **Speed**: Slow (12 FPS)
- **Use case**: Complex scenarios, high accuracy

#### BoostTrack

- **Type**: Appearance + Motion
- **Speed**: Moderate (25 FPS)
- **Use case**: Latest state-of-the-art performance

### Constants

```python
# Available tracker types
TRACKERS = [
    "bytetrack",
    "botsort",
    "strongsort",
    "ocsort",
    "deepocsort",
    "hybridsort",
    "boosttrack"
]

# Version information
__version__ = "13.0.9"
```

### Example Integration Patterns

#### Pattern 1: Simple Integration

```python
from boxmot import create_tracker

tracker = create_tracker('bytetrack')
tracks = tracker.update(detections, frame)
```

#### Pattern 2: Advanced Configuration

```python
from boxmot import create_tracker, get_tracker_config
from pathlib import Path

config_path = get_tracker_config('strongsort')
tracker = create_tracker(
    tracker_type='strongsort',
    tracker_config=config_path,
    reid_weights=Path('osnet_x0_25_msmt17.pt'),
    device='cuda',
    half=True,
    per_class=False
)
```

#### Pattern 3: Custom Parameters

```python
custom_params = {
    'track_thresh': 0.6,
    'track_buffer': 50,
    'match_thresh': 0.9
}

tracker = create_tracker(
    tracker_type='bytetrack',
    evolve_param_dict=custom_params
)
```

This comprehensive guide should provide you with all the necessary information to successfully integrate BoxMOT into your Python applications. For the most up-to-date information and additional examples, refer to the [official BoxMOT repository](https://github.com/mikel-brostrom/boxmot).
