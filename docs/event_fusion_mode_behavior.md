# Event Fusion Service Mode Behavior

## Overview

The Event Fusion Service now adapts its behavior based on the system's offline/real-time mode to optimize for either data completeness (offline) or low latency (real-time).

## Mode Detection

The mode is automatically determined by the `offline_mode` configuration parameter, which is set based on the visualizer's `save_to_file` setting:

- **Offline Mode**: `save_to_file: true` → Process all frames, preserve data completeness
- **Real-time Mode**: `save_to_file: false` → Prioritize low latency, drop old frames

## Behavioral Differences

### 🗂️ Offline Mode (Data Completeness Priority)

**Characteristics:**
- **TTL Extended**: 2x normal TTL (default: 2.0s instead of 1.0s)
- **Buffer Increased**: 2x normal buffer size (default: 2000 instead of 1000)
- **Frame Retention**: All frames are preserved until TTL expiry
- **Completion Strategy**: Waits longer for plate detection and OCR data

**Use Cases:**
- Video file processing
- Batch analysis
- Research and development
- Quality assurance testing

**Behavior:**
```
Frame Processing: [====ALL FRAMES====] → Complete Data
Memory Usage:     Higher (preserves all data)
Latency:         Higher (waits for complete data)
Data Quality:    Maximum (waits for OCR completion)
```

### ⚡ Real-time Mode (Low Latency Priority)

**Characteristics:**
- **TTL Reduced**: 0.5x normal TTL (default: 0.5s instead of 1.0s)
- **Buffer Reduced**: 0.5x normal buffer size (default: 500 instead of 1000)
- **Frame Dropping**: Keeps only last 2-3 frames, drops older ones
- **Immediate Flush**: Flushes objects as soon as tracking data is available

**Use Cases:**
- Live camera feeds
- Real-time monitoring
- Interactive applications
- Production deployments

**Behavior:**
```
Frame Processing: [==LATEST FRAMES==] → Low Latency
Memory Usage:     Lower (drops old frames)
Latency:         Minimum (immediate processing)
Data Quality:    Good (may miss some OCR data)
```

## Frame Management

### Offline Mode Frame Handling
```python
# Preserves all frames until TTL expiry
frames_kept = "ALL"
ttl_multiplier = 2.0
buffer_multiplier = 2.0
completion_wait = "EXTENDED"
```

### Real-time Mode Frame Handling
```python
# Keeps only recent frames
max_frames_realtime = 2
ttl_multiplier = 0.5
buffer_multiplier = 0.5
completion_wait = "IMMEDIATE"
```

## Configuration

### Automatic Mode Selection
```yaml
visualizer:
  save_to_file: true   # → Offline mode (preserve all data)
  # OR
  save_to_file: false  # → Real-time mode (low latency)

event_fusion:
  ttl_sec: 1.0              # Base TTL (modified by mode)
  max_buffer_size: 1000     # Base buffer (modified by mode)
```

### Manual Override (Advanced)
```yaml
event_fusion:
  ttl_sec: 2.0              # Custom TTL
  max_buffer_size: 2000     # Custom buffer size
  # Mode still auto-detected from save_to_file
```

## Performance Characteristics

| Aspect | Offline Mode | Real-time Mode |
|--------|-------------|----------------|
| **Latency** | 1-4 seconds | 0.1-1 second |
| **Memory Usage** | High | Low |
| **Data Completeness** | Maximum | Good |
| **Frame Retention** | All frames | Last 2-3 frames |
| **OCR Wait Time** | Extended | Minimal |
| **Buffer Pressure** | Tolerant | Aggressive |

## Monitoring

### Metrics by Mode

**Offline Mode Metrics:**
```
[EventFusionService] Metrics (offline): 
  throughput=15.2msg/s, state_size=1250, frames_buffered=45, 
  complete_ratio=0.95, dropped=0
```

**Real-time Mode Metrics:**
```
[EventFusionService] Metrics (real-time): 
  throughput=28.7msg/s, state_size=125, frames_buffered=2, 
  complete_ratio=0.78, dropped=15
```

### Key Indicators

- **`frames_buffered`**: Number of unique frames in buffer
  - Offline: Can be high (10-50+)
  - Real-time: Should be low (1-3)

- **`dropped`**: Number of dropped messages
  - Offline: Should be 0 or very low
  - Real-time: Expected to be higher

- **`complete_ratio`**: Ratio of complete vs partial merges
  - Offline: Should be high (0.9+)
  - Real-time: May be lower (0.7+) but acceptable

## Best Practices

### For Offline Processing
1. Use higher resolution videos
2. Enable all detection and OCR services
3. Monitor memory usage for very long videos
4. Expect higher processing times but better quality

### For Real-time Processing
1. Optimize detection model sizes
2. Use lower resolution for faster processing
3. Monitor latency metrics
4. Accept some OCR misses for better responsiveness

## Troubleshooting

### High Memory Usage (Offline)
- Reduce `max_buffer_size` if memory is limited
- Process shorter video segments
- Monitor `frames_buffered` metric

### High Latency (Real-time)
- Check if system is actually in real-time mode
- Verify `save_to_file: false` in configuration
- Monitor `dropped` messages (should be > 0 in real-time)

### Low Data Quality
- Switch to offline mode for better completeness
- Increase TTL for more OCR wait time
- Check OCR service performance

The Event Fusion Service now intelligently adapts to provide the best experience for both offline analysis and real-time monitoring scenarios.