# Queue Management & Frame-Dropping Improvements Changelog

## 🚀 Major Feature: Dual-Mode Queue Management System

**Date:** 2025-01-27  
**Version:** Queue Management v2.0  
**Type:** Major Enhancement

### 📋 Overview

Implemented a comprehensive dual-mode queue management system that automatically switches between real-time and offline processing modes based on the `save_to_file` configuration. This addresses the fundamental trade-off between low-latency display and complete frame preservation.

### 🎯 Problem Solved

**Before:** The pipeline used fixed `maxsize=1` queues with hard-coded frame-dropping logic, causing:

- **Frame loss in offline mode:** When `save_to_file=true`, frames were still dropped if processing couldn't keep up
- **Unnecessary slow-motion risk:** Small queues could cause stuttering if any stage briefly lagged
- **Inflexible architecture:** No way to adjust behavior for different use cases

**After:** Dynamic queue sizing and mode-aware frame handling:

- **Real-time mode:** Small queues (size=3) with leaky behavior for low latency
- **Offline mode:** Large/unbounded queues with blocking behavior to preserve every frame

---

## 🔧 Technical Changes

### 1. New Queue Management Utilities (`src/traffic_monitor/utils/queue_utils.py`)

#### **Added Functions:**

- `is_offline_mode(config)` - Detects processing mode from configuration
- `get_queue_size_for_mode(offline_mode)` - Returns appropriate queue size (3 vs unbounded)
- `put_realtime(queue, message)` - Drops oldest frame if queue full (leaky behavior)
- `put_offline(queue, message)` - Blocks until frame can be enqueued (preserves all frames)
- `safe_put(queue, message, offline_mode)` - Mode-aware wrapper function
- `log_queue_stats(queue, service_name, frame_count)` - Monitoring helper

#### **Design Principles:**

```python
# Real-time mode (save_to_file=false)
Queue(maxsize=3)           # Small buffer for low latency
get_nowait() + put_nowait() # Drop old frames when full

# Offline mode (save_to_file=true)
Queue(maxsize=0)           # Unbounded queue
put(blocking=True)         # Back-pressure preserves all frames
```

### 2. Dynamic Queue Initialization (`src/traffic_monitor/main_supervisor.py`)

#### **Before:**

```python
# Hard-coded small queues
frame_grabber_output_queue = mp.Queue(maxsize=1)
vehicle_detector_output_queue = mp.Queue(maxsize=1)
# ... all queues maxsize=1
```

#### **After:**

```python
# Mode-aware queue sizing
offline_mode = is_offline_mode(vis_config)
queue_size = get_queue_size_for_mode(offline_mode)  # 0 or 3

frame_grabber_output_queue = mp.Queue(maxsize=queue_size)
vehicle_detector_output_queue = mp.Queue(maxsize=queue_size)
# ... all queues use dynamic sizing
```

#### **Runtime Behavior:**

- **Real-time mode:** `queue_size = 3` (small buffer, bounded latency)
- **Offline mode:** `queue_size = 0` (unbounded, no frame loss)

### 3. Service-Level Updates

Updated all pipeline services to use mode-aware queue operations:

#### **Frame Grabber** (`src/traffic_monitor/services/frame_grabber.py`)

- **Before:** Hard-coded `get_nowait() + put_nowait()` pattern
- **After:** `safe_put(output_queue, message, offline_mode, service_name)`
- **Benefit:** Frames preserved in offline mode, real-time behavior in live mode

#### **Vehicle Detector** (`src/traffic_monitor/services/vehicle_detector.py`)

- **Before:** Fixed drop-old logic regardless of mode
- **After:** Mode-aware queue operations with enhanced logging
- **Benefit:** Detection results preserved for complete video analysis

#### **Vehicle Tracker** (`src/traffic_monitor/services/vehicle_tracker.py`)

- **Before:** Always dropped old tracking results when queue full
- **After:** Preserves tracking continuity in offline mode
- **Benefit:** Complete vehicle path tracking for analysis/export

#### **Distributor** (`src/traffic_monitor/services/distributor.py`)

- **Before:** `distributor_process(input_queue, output_queues, shutdown_event)`
- **After:** `distributor_process(offline_mode, input_queue, output_queues, shutdown_event)`
- **Benefit:** Consistent frame distribution across all branches

#### **LP Detector** (`src/traffic_monitor/services/lp_detector.py`)

- **Before:** Hardcoded real-time queue behavior
- **After:** Mode-aware plate detection queuing
- **Benefit:** No missed license plates in offline processing

#### **OCR Reader** (`src/traffic_monitor/services/ocr_reader.py`)

- **Before:** Dropped OCR results when queue full
- **After:** Preserves all OCR results in offline mode
- **Benefit:** Complete license plate text extraction

#### **Vehicle Counter** (`src/traffic_monitor/services/vehicle_counter.py`)

- **Before:** Real-time counting with potential missed counts
- **After:** Guaranteed count preservation in offline mode
- **Benefit:** Accurate vehicle statistics for complete video

### 4. Enhanced Visualizer Logic (`src/traffic_monitor/services/visualizer.py`)

#### **Frame Age Filtering Improvement:**

```python
# Before: Always dropped frames older than 1 second
frame_age = current_time - frame_msg["timestamp"]
if frame_age > 1.0:
    return frame  # Always drop old frames

# After: Only drop in real-time mode
if not self.save_to_file:  # Only for real-time display
    frame_age = current_time - frame_msg["timestamp"]
    if frame_age > 1.0:
        return frame  # Skip processing to maintain real-time
```

#### **Benefit:**

- **Real-time mode:** Maintains low-latency display by dropping stale frames
- **Offline mode:** Processes every frame for complete video output

### 5. Configuration Documentation (`src/traffic_monitor/config/settings.yaml`)

Added clear documentation of the automatic mode detection:

```yaml
visualizer:
  save_to_file: true # Controls queue management mode


  # Queue management mode is automatically determined based on save_to_file:
  # - save_to_file: false -> real-time mode (small queues, drop old frames for low latency)
  # - save_to_file: true  -> offline mode (large/unbounded queues, preserve all frames)
```

---

## 📊 Performance Impact

### Memory Usage

- **Real-time mode:** Minimal increase (3-frame buffer vs 1-frame)
- **Offline mode:** Proportional to processing speed mismatch
  - Queue memory = (input rate - processing rate) × frame size × time
  - Self-regulating via back-pressure

### Latency

- **Real-time mode:** Slight improvement (3-frame buffer vs lock-step)
- **Offline mode:** N/A (latency not relevant for file output)

### Throughput

- **Real-time mode:** Maintained or improved (better parallelism)
- **Offline mode:** Limited by slowest component (usually encoder/disk)

---

## 🔍 Monitoring & Observability

### New Logging Features

1. **Mode detection logging:**

   ```
   Queue management mode: offline (preserve all frames), queue size: unbounded
   ```

2. **Service-level queue statistics:**

   ```
   [FrameGrabber] Frame 500, Queue size: 2
   ```

3. **Enhanced error context:**
   ```
   [VehicleTracker] Failed to put tracking for frame frame_12345
   ```

### Recommended Monitoring

- **Real-time mode:** Monitor `queue.qsize()` should stay ≤ 3
- **Offline mode:** Monitor `frames_dropped` should be 0
- **Both modes:** Track processing FPS vs input FPS

---

## 🚀 Migration Guide

### For Existing Deployments

No configuration changes required! The system automatically detects mode based on existing `save_to_file` setting.

### For Custom Integrations

If you directly instantiate services, pass the `offline_mode` parameter:

```python
# Old way
config = {"model_path": "...", "conf_threshold": 0.5}

# New way
config = {
    "model_path": "...",
    "conf_threshold": 0.5,
    "offline_mode": False  # or True for preservation mode
}
```

---

## 🧪 Testing Recommendations

### Verify Real-Time Mode

```bash
# Set real-time mode
save_to_file: false

# Expected behavior:
# - Low latency display (< 100ms behind live)
# - Some frame drops under load (acceptable)
# - Queue sizes stay small (≤ 3)
```

### Verify Offline Mode

```bash
# Set offline mode
save_to_file: true

# Expected behavior:
# - All frames preserved in output video
# - No dropped frame warnings in logs
# - Queues may grow during processing spikes
```

---

## 📚 References

- **Best Practices Document:** [`docs/QUEUE_MANAGEMENT_BEST_PRACTICES.md`](./QUEUE_MANAGEMENT_BEST_PRACTICES.md)
- **Implementation:** [`src/traffic_monitor/utils/queue_utils.py`](../src/traffic_monitor/utils/queue_utils.py)
- **Research Sources:** OBS Studio, GStreamer docs, Mux video processing guidelines

---

## 🎯 Future Enhancements

1. **Adaptive queue sizing** - Dynamically adjust queue size based on processing performance
2. **Quality-aware dropping** - In real-time mode, prefer to drop frames with fewer detections
3. **Branch-specific policies** - Different queue strategies for different output branches
4. **Prometheus metrics** - Expose queue statistics for monitoring dashboards

---

**✅ Status:** Fully implemented and tested  
**🔗 Related Issues:** Frame dropping in offline mode, slow-motion video output  
**👥 Impact:** All video processing workflows, both real-time and batch
