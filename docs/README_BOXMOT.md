# BoxMOT Integration for Traffic Monitor

This directory contains comprehensive documentation and examples for integrating BoxMOT (Box Multi-Object Tracking) into your Python traffic monitoring application.

## Quick Start

### 1. Installation

```bash
# Install BoxMOT and dependencies
pip install -r requirements_boxmot.txt

# Or install BoxMOT directly
pip install boxmot
```

### 2. Basic Usage

```python
from boxmot import create_tracker
import numpy as np

# Initialize tracker
tracker = create_tracker('bytetrack')

# Your detection data (N x 6): [x1, y1, x2, y2, confidence, class_id]
detections = np.array([[100, 100, 200, 200, 0.9, 0]])

# Dummy frame
frame = np.zeros((480, 640, 3), dtype=np.uint8)

# Update tracker
tracks = tracker.update(detections, frame)
print(f"Tracks: {tracks}")
```

### 3. Run Example

```bash
# Run the traffic tracking example
python examples/boxmot_integration_example.py
```

## Documentation

- **[Complete Integration Guide](BOXMOT_INTEGRATION_GUIDE.md)** - Comprehensive documentation covering all aspects of BoxMOT integration
- **[Example Script](../examples/boxmot_integration_example.py)** - Practical example for traffic monitoring

## Key Features

- **6 State-of-the-Art Trackers**: ByteTrack, BotSort, StrongSort, OcSort, DeepOcSort, BoostTrack
- **Hardware Flexibility**: CPU and GPU support
- **Real-time Performance**: Up to 1483 FPS with OcSort
- **Easy Integration**: Simple API for quick integration

## Tracker Comparison

| Tracker    | Type                | Speed (FPS) | Use Case                    |
| ---------- | ------------------- | ----------- | --------------------------- |
| ByteTrack  | Motion-only         | 1265        | Real-time, simple scenarios |
| OcSort     | Motion-only         | 1483        | High-speed requirements     |
| BotSort    | Appearance + Motion | 46          | Balanced performance        |
| StrongSort | Appearance + Motion | 17          | High accuracy               |
| DeepOcSort | Appearance + Motion | 12          | Complex scenarios           |
| BoostTrack | Appearance + Motion | 25          | Latest SOTA performance     |

## Configuration

### For Real-time Applications

```python
tracker = create_tracker('bytetrack', device='cuda', half=True)
```

### For High Accuracy

```python
from pathlib import Path
tracker = create_tracker(
    'strongsort',
    reid_weights=Path('osnet_x0_25_msmt17.pt'),
    device='cuda'
)
```

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure BoxMOT is installed: `pip install boxmot`
2. **CUDA Out of Memory**: Use CPU: `device='cpu'`
3. **Slow Performance**: Use ByteTrack or OcSort for speed
4. **Poor Tracking**: Try appearance-based trackers with ReID models

### Getting Help

- Check the [complete integration guide](BOXMOT_INTEGRATION_GUIDE.md) for detailed troubleshooting
- Visit the [BoxMOT GitHub repository](https://github.com/mikel-brostrom/boxmot) for issues and updates

## Integration with Your Application

Replace the dummy detector in the example with your actual detection model:

```python
def your_detector(frame):
    # Your detection logic here
    # Return detections in format: [x1, y1, x2, y2, confidence, class_id]
    return detections

# In your tracking loop
detections = your_detector(frame)
tracks = tracker.update(detections, frame)
```

## Performance Tips

1. **Use appropriate tracker for your needs**:

   - Real-time: ByteTrack or OcSort
   - Accuracy: StrongSort or BoostTrack
   - Balanced: BotSort

2. **Optimize for your hardware**:

   - GPU: Enable `half=True` for FP16
   - CPU: Use motion-only trackers

3. **Filter detections**:

   - Set confidence thresholds
   - Filter by relevant classes only

4. **Memory management**:
   - Reset tracker periodically for long-running applications
   - Use garbage collection if needed

## License

This integration follows the BoxMOT license (AGPL-3.0). See the [BoxMOT repository](https://github.com/mikel-brostrom/boxmot) for details.
