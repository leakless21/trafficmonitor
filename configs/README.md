# Traffic Monitor Configuration

This directory contains all configuration files for the Traffic Monitor system.

## Structure

```
configs/
├── base/
│   └── default.yaml          # Base configuration with all default settings
├── environments/
│   ├── development.yaml      # Development environment overrides
│   └── production.yaml       # Production environment overrides
├── benchmarks/
│   ├── fast.yaml            # Fast benchmark configuration
│   ├── perf_fast.yaml       # Performance benchmark (fast)
│   ├── perf_standard.yaml   # Performance benchmark (standard)
│   └── prod.yaml            # Production benchmark
└── trackers/
    ├── bytetrack.yaml       # ByteTrack tracker configuration
    ├── botsort.yaml         # BotSORT tracker configuration
    └── [other trackers...]  # Additional tracker configurations
```

## Usage

### Basic Usage
```bash
# Use default configuration
traffic-monitor --config configs/base/default.yaml

# Use development environment
traffic-monitor --config configs/environments/development.yaml

# Use production environment
traffic-monitor --config configs/environments/production.yaml
```

### Configuration Inheritance

Environment configurations inherit from the base configuration and override specific settings:

1. **Base Configuration** (`configs/base/default.yaml`): Contains all default settings
2. **Environment Configuration**: Overrides specific settings for the target environment
3. **Runtime Overrides**: Command-line arguments can override any configuration value

### Environment Variables

You can override configuration values using environment variables:

```bash
# Override log level
LOG_LEVEL=DEBUG traffic-monitor --config configs/base/default.yaml

# Use JSON logging
LOG_FORMAT=json traffic-monitor --config configs/base/default.yaml
```

## Configuration Sections

- **frame_grabber**: Video input and frame processing settings
- **vehicle_detector**: YOLO model and detection parameters
- **vehicle_tracker**: Multi-object tracking configuration
- **lp_detector**: License plate detection settings
- **ocr_reader**: OCR engine configuration
- **vehicle_counter**: Counting line definitions
- **visualizer**: Display and output settings
- **loguru**: Logging configuration
- **database**: Database connection and settings
- **summary_service**: Report generation settings