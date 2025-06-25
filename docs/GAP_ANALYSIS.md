## Deprecation Warnings

### NumPy Array Scalar Conversion

**Issue**: DeprecationWarnings for converting ndarray to scalar in vehicle detection and license plate detection services.

**Affected Files**:

- ~~`src/traffic_monitor/services/lp_detector.py:44`~~ ✅ **FIXED**
- ~~`src/traffic_monitor/services/vehicle_detector.py:57,61`~~ ✅ **FIXED**

**Status**: ✅ **RESOLVED** - Fixed by using `.item()` method instead of `float()`/`int()` conversion.

**Fix Applied**:

```python
# Before (deprecated):
confidence = float(best_plate.conf)
class_id = int(box.cls)

# After (correct):
confidence = best_plate.conf.item()
class_id = box.cls.item()
```

## LPDetector and OCRReader Service Issues

### Missing Logging in Child Processes

**Issue**: LPDetector and OCRReader processes were not producing logs because multiprocessing child processes need to set up their own logging configuration.

**Affected Files**:

- `src/traffic_monitor/services/lp_detector.py:49` ✅ **FIXED**
- `src/traffic_monitor/services/ocr_reader.py:58` ✅ **FIXED**

**Status**: ✅ **RESOLVED** - Added `setup_logging()` call at the beginning of each process function.

**Fix Applied**:

```python
def lp_detector_process(...):
    from ..utils.logging_config import setup_logging
    setup_logging()  # Setup logging for this process
    # ... rest of function
```

### Missing OCR Reader Configuration

**Issue**: The `settings.yaml` file was missing the `ocr_reader` configuration section, causing the OCR process to fail with missing configuration.

**Affected Files**:

- `src/traffic_monitor/config/settings.yaml` ✅ **FIXED**

**Status**: ✅ **RESOLVED** - Added OCR reader configuration section.

**Fix Applied**:

```yaml
ocr_reader:
  hub_model_name: "global-plates-mobile-vit-v2-model"
  device: "auto"
  conf_threshold: 0.5
```

## VehicleCounter Service Issues

### Counting Line Configuration Structure Mismatch

**Issue**: The VehicleCounter Counter class was expecting counting line configuration in format `[[x1,y1], [x2,y2]]` but the YAML configuration provides it as `[[[x1,y1], [x2,y2]]]` (list of lines), causing initialization to fail.

**Affected Files**:

- `src/traffic_monitor/services/vehicle_counter.py:17-30` ✅ **FIXED**

**Status**: ✅ **RESOLVED** - Updated Counter class to handle both single line and multi-line configuration formats.

**Error Message**: `[Counter] Failed to initialize line config` and `[Counter] Line config is in an unknown format.`

**Root Cause**: The Counter.**init** method tried to access `counting_lines_config[0][0]` directly, but when config is `[[[0.31, 0.22], [0.85, 0.33]]]`, this results in accessing `[0.31, 0.22]` instead of `0.31`, causing type detection to fail.

**Fix Applied**:

```python
# Before (failed with nested list structure):
def __init__(self, counting_lines_config: list):
    self.line_config_raw = counting_lines_config
    # ... code tried to access counting_lines_config[0][0] directly

# After (handles both formats):
def __init__(self, counting_lines_config: list):
    if not counting_lines_config:
        logger.error("[Counter] Empty counting lines configuration")
        self.line_config_raw = []
    elif len(counting_lines_config[0]) > 0 and isinstance(counting_lines_config[0][0], list):
        # Config is a list of lines: [[[x1,y1], [x2,y2]], [[x3,y3], [x4,y4]]]
        self.line_config_raw = counting_lines_config[0]  # Use first line
    else:
        # Config is a single line: [[x1,y1], [x2,y2]]
        self.line_config_raw = counting_lines_config
```

### Frame Queue Overflow Issues

**Issue**: FrameGrabber is dropping frames due to full output queues, indicating the processing pipeline cannot keep up with the frame rate.

**Affected Files**:

- `src/traffic_monitor/main_supervisor.py:64-69` ⚠️ **IN PROGRESS**

**Status**: ⚠️ **IN PROGRESS** - Queue sizes need optimization

**Error Message**: `[FrameGrabber] Output queue is full. Frame {frame_id} dropped.`

**Root Cause**: All queues are set to `maxsize=60` but with 30 FPS processing and multiple heavy operations (detection, tracking, OCR), the pipeline gets backed up.

**Proposed Fix**: Adjust queue sizes based on processing stage complexity and add adaptive frame dropping strategies.

## Configuration Issues

### Incorrect YOLO Class Mapping

**Issue**: The vehicle detector class mapping was using incorrect YOLO COCO class IDs, causing misclassification of detected vehicles.

**Affected Files**:

- `src/traffic_monitor/config/settings.yaml` ✅ **FIXED**

**Status**: ✅ **RESOLVED** - Updated class mapping to use correct YOLO COCO class IDs.

**Fix Applied**:

```yaml
# Before (incorrect IDs):
class_mapping:
  1: "person"
  2: "bicycle"
  3: "car"
  4: "motorcycle"
  6: "bus"      # Wrong - should be 5
  8: "truck"    # Wrong - should be 7

# After (correct COCO IDs):
class_mapping:
  0: "person"
  1: "bicycle"
  2: "car"
  3: "motorcycle"
  5: "bus"
  7: "truck"
```

## Coordinate Conversion Issues

### Data Structure Mismatch in Coordinate Conversion

**Issue**: The `relative_to_absolute_coords` function in `utils.py` was expecting individual coordinate pairs but was receiving a list of lines, causing a TypeError when the vehicle counter tried to convert relative coordinates to absolute.

**Error Message**: `int() argument must be a string, a bytes-like object or a real number, not 'list'`

**Affected Files**:

- `src/traffic_monitor/utils/utils.py` ✅ **FIXED**
- `src/traffic_monitor/services/vehicle_counter.py` ✅ **FIXED**
- `src/traffic_monitor/services/visualizer.py` ✅ **FIXED**

**Status**: ✅ **RESOLVED** - Updated function signature to handle list of lines correctly.

**Fix Applied**:

```python
# Before (incorrect - expected single line):
def relative_to_absolute_coords(relative_coords: List[List[float]], frame_width: int, frame_height: int) -> List[List[int]]:
    for coord_pair in relative_coords:  # Error: coord_pair was actually a full line
        # ...

# After (correct - handles multiple lines):
def relative_to_absolute_coords(relative_lines: List[List[List[float]]], frame_width: int, frame_height: int) -> List[List[List[int]]]:
    for line in relative_lines:
        for coord_pair in line:
            # ... correct processing
```

## Data Flow Issues

### Missing Original Frame Dimensions in VehicleDetector and VehicleTracker

**Issue**: VehicleDetector and VehicleTracker processes are not passing through the original frame dimensions (`og_frame_width` and `og_frame_height`) to downstream processes, causing VehicleCounter to crash with a KeyError.

**Affected Files**:

- `src/traffic_monitor/services/vehicle_detector.py:178-188` ✅ **FIXED**
- `src/traffic_monitor/services/vehicle_tracker.py:195-205` ✅ **FIXED**

**Status**: ✅ **RESOLVED** - Added `og_frame_width` and `og_frame_height` to output messages.

**Error Message**: `KeyError: 'og_frame_width'` in VehicleCounter process

**Root Cause**: The `TrackedVehicleMessage` type inherits from `FrameMessage` which requires these fields, but they were being omitted in the message construction.

**Fix Applied**:

```python
# In vehicle_detector.py:
output_message: VehicleDetectionMessage = {
    "frame_id": frame_message["frame_id"],
    "frame_width": frame_message["frame_width"],
    "frame_height": frame_message["frame_height"],
    "og_frame_width": frame_message["og_frame_width"],     # Added
    "og_frame_height": frame_message["og_frame_height"],   # Added
    "camera_id": frame_message["camera_id"],
    "timestamp": frame_message["timestamp"],
    "frame_data_jpeg": frame_message["frame_data_jpeg"],
    "detections": detections
}

# In vehicle_tracker.py:
output_message = TrackedVehicleMessage(
    frame_id=vehicle_detection_message["frame_id"],
    camera_id=vehicle_detection_message["camera_id"],
    timestamp=vehicle_detection_message["timestamp"],
    frame_data_jpeg=jpeg_binary,
    frame_height=vehicle_detection_message["frame_height"],
    frame_width=vehicle_detection_message["frame_width"],
    og_frame_height=vehicle_detection_message["og_frame_height"],  # Added
    og_frame_width=vehicle_detection_message["og_frame_width"],    # Added
    tracked_objects=tracked_objects
)
```

## GUI Environment Issues

### OpenCV GUI Support Missing in Windows Environment

**Issue**: OpenCV installation lacks GUI support, causing Visualizer process to fail when trying to create windows. This is common in headless environments or certain Windows configurations.

**Affected Files**:

- `src/traffic_monitor/services/visualizer.py:169-175` ✅ **FIXED**

**Status**: ✅ **RESOLVED** - Added headless mode fallback that saves frames to video file instead of displaying them.

**Error Message**:

```
OpenCV(4.11.0) D:\a\opencv-python\opencv-python\opencv\modules\highgui\src\window.cpp:1301: error: (-2:Unspecified error) The function is not implemented. Rebuild the library with Windows, GTK+ 2.x or Cocoa support.
```

**Fix Applied**:

```python
# Added GUI availability detection and headless mode fallback
gui_available = False
headless_mode = False

try:
    cv2.imshow("Traffic Monitor", test_img)
    gui_available = True
except Exception as window_error:
    logger.info(f"[Visualizer] Switching to headless mode - frames will be saved to file instead.")
    headless_mode = True

# Added video writer for headless mode
if headless_mode:
    video_writer.write(display_frame)
    logger.debug(f"[Visualizer] Saved frame to video file (headless mode)")
```

## Visualizer Service Issues

### Missing og_fps Field in TrackedVehicleMessage

**Issue**: The `VehicleTracker` was not passing through the `og_fps` field when creating `TrackedVehicleMessage`, causing the Visualizer to crash when trying to initialize the video writer.

**Affected Files**:

- `src/traffic_monitor/services/vehicle_tracker.py:204-214` ✅ **FIXED**

**Status**: ✅ **RESOLVED** - Added `og_fps` field to the TrackedVehicleMessage output.

**Error Message**: `[Visualizer] Error processing frame: 'og_fps'`

**Root Cause**: The `TrackedVehicleMessage` inherits from `FrameMessage` which includes `og_fps` field, but it was missing from the message construction in vehicle_tracker.py.

**Fix Applied**:

```python
# In vehicle_tracker.py:
# Before (missing og_fps):
output_message = TrackedVehicleMessage(
    frame_id=vehicle_detection_message["frame_id"],
    camera_id=vehicle_detection_message["camera_id"],
    timestamp=vehicle_detection_message["timestamp"],
    frame_data_jpeg=jpeg_binary,
    frame_height=vehicle_detection_message["frame_height"],
    frame_width=vehicle_detection_message["frame_width"],
    og_frame_height=vehicle_detection_message["og_frame_height"],
    og_frame_width=vehicle_detection_message["og_frame_width"],
    tracked_objects=tracked_objects
)

# After (includes og_fps):
output_message = TrackedVehicleMessage(
    frame_id=vehicle_detection_message["frame_id"],
    camera_id=vehicle_detection_message["camera_id"],
    timestamp=vehicle_detection_message["timestamp"],
    frame_data_jpeg=jpeg_binary,
    frame_height=vehicle_detection_message["frame_height"],
    frame_width=vehicle_detection_message["frame_width"],
    og_frame_height=vehicle_detection_message["og_frame_height"],
    og_frame_width=vehicle_detection_message["og_frame_width"],
    og_fps=vehicle_detection_message["og_fps"],  # Added
    tracked_objects=tracked_objects
)

# In vehicle_detector.py:
# Before (missing og_fps):
output_message: VehicleDetectionMessage = {
    "frame_id": frame_message["frame_id"],
    "frame_width": frame_message["frame_width"],
    "frame_height": frame_message["frame_height"],
    "og_frame_width": frame_message["og_frame_width"],
    "og_frame_height": frame_message["og_frame_height"],
    "camera_id": frame_message["camera_id"],
    "timestamp": frame_message["timestamp"],
    "frame_data_jpeg": frame_message["frame_data_jpeg"],
    "detections": detections
}

# After (includes og_fps):
output_message: VehicleDetectionMessage = {
    "frame_id": frame_message["frame_id"],
    "frame_width": frame_message["frame_width"],
    "frame_height": frame_message["frame_height"],
    "og_frame_width": frame_message["og_frame_width"],
    "og_frame_height": frame_message["og_frame_height"],
    "og_fps": frame_message["og_fps"],  # Added
    "camera_id": frame_message["camera_id"],
    "timestamp": frame_message["timestamp"],
    "frame_data_jpeg": frame_message["frame_data_jpeg"],
    "detections": detections
}
```

### High OCR Confidence Threshold Rejecting Valid Plates

**Issue**: The OCR confidence threshold was set too high (0.8), causing many valid license plate readings to be rejected. Detected plates with confidence 0.44 were being discarded.

**Affected Files**:

- `src/traffic_monitor/config/settings.yaml` ✅ **FIXED**

**Status**: ✅ **RESOLVED** - Lowered OCR confidence threshold from 0.8 to 0.4.

**Error Message**: `OCR result '1H33M1P__' with confidence 0.4399036169052124 below threshold 0.8`

**Root Cause**: Real-world license plate OCR often produces confidence scores in the 0.4-0.6 range due to factors like lighting, angle, and image quality. A threshold of 0.8 was too restrictive.

**Fix Applied**:

```yaml
# Before (too restrictive):
ocr_reader:
  hub_model_name: "global-plates-mobile-vit-v2-model"
  device: "auto"
  conf_threshold: 0.8

# After (more permissive):
ocr_reader:
  hub_model_name: "global-plates-mobile-vit-v2-model"
  device: "auto"
  conf_threshold: 0.4
```

### Video Writer Path Duplication Issue

**Issue**: The Visualizer's video writer was creating files with duplicated paths like `data\videos\output\data\videos\output\output_20250626_043508.mp4`, causing video writer initialization to fail.

**Affected Files**:

- `src/traffic_monitor/services/visualizer.py:84` ✅ **FIXED**

**Status**: ✅ **RESOLVED** - Fixed path construction in `_init_video_writer` method.

**Error Message**: `[Visualizer] Failed to initialize video writer to data\videos\output\data\videos\output\output_20250626_043508.mp4`

**Root Cause**: The filename was incorrectly including the output path, causing duplication when combined with `Path(self.output_path) / filename`.

**Fix Applied**:

```python
# Before (duplicated path):
def _init_video_writer(self, frame_width: int, frame_height: int, og_fps: float):
    filename = f"{self.output_path}/output_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    filepath = Path(self.output_path) / filename  # Results in duplication

# After (correct path):
def _init_video_writer(self, frame_width: int, frame_height: int, og_fps: float):
    filename = f"output_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    filepath = Path(self.output_path) / filename  # Correct path construction
    filepath.parent.mkdir(parents=True, exist_ok=True)  # Create directory if needed
    # Try multiple codecs for compatibility
```

### Video Writer Directory and Codec Issues

**Issue**: Video writer was failing due to missing output directory and potential codec compatibility issues on Windows.

**Affected Files**:

- `src/traffic_monitor/services/visualizer.py:84-96` ✅ **FIXED**

**Status**: ✅ **RESOLVED** - Added directory creation and fallback codec support.

**Error Message**: `[Visualizer] Failed to initialize video writer to data\videos\output\output_20250626_043734.mp4`

**Root Causes**:

1. Output directory `data/videos/output/` didn't exist
2. Primary codec "mp4v" might not be available on Windows systems

**Fix Applied**:

```python
# Before (no directory creation, single codec):
def _init_video_writer(self, frame_width: int, frame_height: int, og_fps: float):
    filename = f"output_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    filepath = Path(self.output_path) / filename
    fourcc = cv2.VideoWriter.fourcc(*self.output_fourcc)
    self.video_writer = cv2.VideoWriter(str(filepath), fourcc, og_fps, (frame_width, frame_height))

# After (directory creation + codec fallback):
def _init_video_writer(self, frame_width: int, frame_height: int, og_fps: float):
    filename = f"output_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    filepath = Path(self.output_path) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    codecs_to_try = [self.output_fourcc, "XVID", "MJPG", "mp4v"]
    for codec in codecs_to_try:
        # Try each codec until one works
```

## Video Output Speed-Up Issue

### Problem Description

**Issue**: Output video appears to speed up after approximately 5 seconds of processing, making vehicles move much faster than in the original video.

**Symptoms**:

- Video starts at normal speed
- After 5-10 seconds, vehicles appear to move much faster
- Output video duration is shorter than input video duration
- FPS counter shows normal values but video playback is accelerated

### Root Cause Analysis

**Primary Cause**: Frame dropping due to processing pipeline bottlenecks combined with constant FPS video writing.

**Detailed Mechanism**:

1. **Initial Phase (0-5 seconds)**: All processes work normally, frames processed at regular intervals
2. **Load Increase**: Heavy vehicle detection/tracking load causes queue backlogs
3. **Frame Dropping**: Frame grabber drops frames when queues are full (`queue.Full` exceptions)
4. **Process Crashes**: Some processes crash under heavy load (visible in logs as "One or more processes are dead")
5. **Video Output Mismatch**: VideoWriter continues writing at original FPS, but fewer frames available
6. **Result**: Video appears sped up because fewer frames span the same time duration

**Evidence from Logs**:

```
[FrameGrabber] Output queue is full. Frame <id> dropped.
[MainProcess] One or more processes are dead. Shutting down.
```

### Solution Implemented

**Approach**: Stabilize the processing pipeline to prevent frame drops rather than artificially limiting frame rate.

**Changes Made**:

1. **Conservative Queue Sizes**: Reduced queue sizes to apply backpressure earlier and prevent memory issues

   ```python
   frame_grabber_output_queue = mp.Queue(maxsize=20)     # Was 30
   vehicle_detector_output_queue = mp.Queue(maxsize=25)  # Was 45
   vehicle_tracker_output_queue = mp.Queue(maxsize=40)   # Was 90
   ```

2. **Improved Frame Timing Tracking**: Added timing validation in visualizer to detect frame processing issues

   ```python
   # Log frame writing progress periodically
   if self.frame_count % 100 == 0:
       elapsed_time = current_time - self.video_start_time
       expected_frames = elapsed_time * frame_msg["og_fps"]
       frame_ratio = self.frame_count / expected_frames
   ```

3. **Enhanced Logging**: Better frame drop detection and FPS monitoring in frame grabber

   ```python
   logger.warning(f"[{process_name}] Output queue is full. Frame {message['frame_id']} dropped.")
   ```

4. **Diagnostic Tools**: Created `test_video_timing_debug.py` to analyze frame processing patterns and identify bottlenecks

### Testing and Validation

**Use the diagnostic test**:

```bash
python test/test_video_timing_debug.py
```

This will show:

- Input video properties (FPS, duration, frame count)
- Frame processing intervals and gaps
- Output video comparison with input
- Warning indicators for frame dropping

**Expected Output**:

```
Input Video Analysis:
- FPS: 30.0
- Frame Count: 900
- Duration: 30.00 seconds

Frame Processing Analysis:
- Frames processed: 850
- Average processing interval: 0.033 seconds
- Expected interval: 0.033 seconds
- Processing speed ratio: 1.0x

Video Comparison:
Input:  900 frames @ 30.0 FPS = 30.00s
Output: 850 frames @ 30.0 FPS = 28.33s
Duration ratio: 0.94x
WARNING: Output video is significantly shorter - indicates frame dropping!
```

### Status

🔄 **IN PROGRESS** - Pipeline stability improvements implemented, testing required to validate effectiveness.

### Next Steps

1. Run diagnostic test to measure current frame drop rate
2. If frame drops persist, consider:
   - Further queue size reduction
   - Process priority adjustments
   - Hardware-specific optimizations (GPU utilization)
   - Frame skip strategies that maintain timing consistency
