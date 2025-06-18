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

### Missing Logging Setup in VehicleCounter Process

**Issue**: VehicleCounter process was not producing any logs because it was missing the logging setup call and loguru configuration.

**Affected Files**:

- `src/traffic_monitor/services/vehicle_counter.py` ✅ **FIXED**
- `src/traffic_monitor/main_supervisor.py` ✅ **FIXED**

**Status**: ✅ **RESOLVED** - Added `setup_logging()` call and loguru config passing.

**Fix Applied**:

```python
# In vehicle_counter.py:
from ..utils.logging_config import setup_logging

def vehicle_counter_process(config: dict, input_queue: Queue, output_queue: Queue, shutdown_event: Event):
    setup_logging(config.get("loguru"))  # Initialize logging for this process
    # ... rest of function

# In main_supervisor.py:
vc_config["loguru"] = loguru_config
```

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
