# Traffic Monitor

A robust, modular, and real-time video analytics application designed for comprehensive traffic monitoring. Leverages advanced computer vision techniques including YOLO for vehicle detection, BoxMOT for multi-object tracking, and various OCR engines for license plate recognition.

## Features

- **Real-time Video Processing**: Process video streams from files or IP cameras
- **Advanced Vehicle Detection**: YOLO-based detection with configurable thresholds
- **Multi-Object Tracking**: BoxMOT integration with various tracker types
- **License Plate Recognition**: Multiple OCR engines (FastPlateOCR, PaddleOCR)
- **Vehicle Counting**: Precise counting with virtual counting lines
- **Real-time Visualization**: OpenCV-based GUI with live feeds
- **Multiprocessing Architecture**: Concurrent processing for high throughput
- **Comprehensive Logging**: Structured logging with configurable levels
- **Data Persistence**: SQLite database for results and analytics

## Quick Start

### 1. Setup Development Environment

```bash
# Install dependencies and setup development environment
make dev-setup

# Or manually:
uv sync
uv run pre-commit install  # Optional: for code quality hooks
```

### 2. Download Required Models

```bash
# Download required AI models
bash data/models/download_models.sh

# Or manually download and place models in:
# - data/models/vehicle/8n/best.engine (vehicle detection)
# - data/models/plate/5nu/best.engine (license plate detection)
```

### 3. Run Traffic Monitor

```bash
# Development mode (with GUI, verbose logging)
make run-dev

# Production mode (no GUI, optimized settings)
make run-prod

# Or run directly with custom config
traffic-monitor --config configs/environments/development.yaml --video path/to/video.mp4
```

## Installation

### Prerequisites

- **Python 3.11+**
- **CUDA-capable GPU** (recommended for optimal performance)
- **uv** package manager (install with `pip install uv`)

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3.11 python3-pip libgl1-mesa-glx libglib2.0-0

# macOS
brew install python@3.11

# Windows
# Download Python 3.11+ from python.org
```

### Python Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd traffic-monitor

# Install dependencies
uv sync

# Verify installation
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

## Configuration

### Environment Configurations

The system supports multiple environment configurations:

- **Development** (`configs/environments/development.yaml`): Verbose logging, GUI enabled, lower thresholds
- **Production** (`configs/environments/production.yaml`): Optimized for performance, no GUI, structured logging

### Key Configuration Sections

```yaml
# Video input settings
frame_grabber:
  video_source: "path/to/video.mp4"  # Video file or camera index
  resize_resolution: [1920, 1080]    # Output resolution
  process_every_n_frame: 1           # Frame sampling rate

# Vehicle detection settings
vehicle_detector:
  model_path: "data/models/vehicle/8n/best.engine"
  conf_threshold: 0.65               # Detection confidence threshold

# License plate detection
lp_detector:
  model_path: "data/models/plate/5nu/best.engine"
  conf_threshold: 0.7

# OCR settings
ocr_reader:
  backend: fast_plate_ocr            # or paddle_ocr
  hub_model_name: "cct-s-v1-global-model"

# Counting lines (normalized coordinates)
vehicle_counter:
  counting_lines:
    - [[0.31, 0.22], [0.85, 0.33]]   # Line from point A to point B
```

### Environment Variables

Override configuration with environment variables:

```bash
# Logging
export LOG_LEVEL=DEBUG
export LOG_FORMAT=json

# Model paths
export VEHICLE_MODEL_PATH="/path/to/custom/model.engine"
export PLATE_MODEL_PATH="/path/to/custom/plate/model.engine"

# Data directories
export DATA_DIR="/custom/data/path"
```

## Usage Examples

### Basic Video Processing

```bash
# Process a video file with default settings
traffic-monitor --config configs/base/default.yaml --video data/samples/videos/traffic.mp4

# Process with custom configuration
traffic-monitor --config configs/environments/development.yaml --video /path/to/video.mp4
```

### Batch Processing

```bash
# Process multiple videos
python tools/batch_processing/batch_run_traffic_monitor.py \
  --config configs/environments/production.yaml \
  --input-dir data/videos/input/ \
  --output-dir data/videos/output/
```

### Real-time Camera Processing

```bash
# Use webcam (camera index 0)
traffic-monitor --config configs/environments/development.yaml --video 0

# Use IP camera
traffic-monitor --config configs/environments/development.yaml --video "rtsp://camera-ip:port/stream"
```

## Development

### Available Commands

```bash
make help              # Show all available commands
make dev-setup         # Setup development environment
make test              # Run all tests
make test-unit         # Run unit tests only
make test-integration  # Run integration tests
make lint              # Check code quality
make format            # Format code
make clean             # Clean build artifacts
make benchmark         # Run performance benchmarks
```

### Project Structure

```
traffic-monitor/
├── src/traffic_monitor/     # Main application code
│   ├── services/           # Core services (detection, tracking, etc.)
│   └── utils/              # Utility modules
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── fixtures/          # Test data
├── configs/               # Configuration files
│   ├── base/             # Base configurations
│   ├── environments/     # Environment-specific configs
│   └── trackers/         # Tracker configurations
├── tools/                # Development and processing tools
│   ├── data_processing/  # Data processing scripts
│   ├── benchmarking/     # Performance benchmarks
│   └── visualization/    # Visualization tools
└── data/                 # Data directory
    ├── models/           # AI models (download separately)
    ├── samples/          # Sample data
    └── outputs/          # Generated outputs
```

### Adding New Features

1. **Create feature branch**: `git checkout -b feature/new-feature`
2. **Write tests first**: Add tests in `tests/unit/` or `tests/integration/`
3. **Implement feature**: Add code in appropriate `src/traffic_monitor/` module
4. **Run quality checks**: `make lint && make test`
5. **Update documentation**: Update relevant docs and this README
6. **Submit pull request**: Include description and test results

### Code Quality

The project uses modern Python tooling:

- **ruff**: Fast linting and formatting
- **mypy**: Static type checking
- **pytest**: Testing framework
- **pre-commit**: Automated quality checks

## Troubleshooting

### Common Issues

**Model not found errors**:
```bash
# Download models
bash data/models/download_models.sh

# Check model paths in config
grep -r "model_path" configs/
```

**CUDA/GPU issues**:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU-only mode
export CUDA_VISIBLE_DEVICES=""
```

**Permission errors**:
```bash
# Fix data directory permissions
chmod -R 755 data/
mkdir -p data/outputs/videos data/outputs/reports
```

**Memory issues**:
- Reduce `resize_resolution` in config
- Increase `process_every_n_frame` to skip frames
- Use smaller model variants (e.g., nano instead of large)

### Performance Optimization

1. **Use TensorRT models** (`.engine` files) for NVIDIA GPUs
2. **Adjust frame sampling** with `process_every_n_frame`
3. **Optimize resolution** based on detection requirements
4. **Use appropriate model size** (nano for speed, large for accuracy)

## Documentation

For detailed documentation:

- **User Guide**: [docs/user_guide/](docs/user_guide/) - Installation, configuration, usage
- **Developer Guide**: [docs/developer_guide/](docs/developer_guide/) - Architecture, contributing, testing
- **API Reference**: [docs/api_reference/](docs/api_reference/) - Code documentation

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/developer_guide/contributing.md) for:

- Development setup
- Coding standards
- Testing requirements
- Pull request process

## License

[Add your license information here]

## Support

- **Issues**: Report bugs and request features via GitHub Issues
- **Documentation**: Check the [docs/](docs/) directory
- **Discussions**: Use GitHub Discussions for questions and ideas