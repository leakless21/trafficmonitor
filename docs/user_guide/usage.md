# Usage

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
