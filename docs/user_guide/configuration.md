# Configuration

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
