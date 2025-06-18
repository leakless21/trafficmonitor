# Traffic Monitor

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code Style](https://img.shields.io/badge/Code%20Style-Ruff-black.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/Tests-Pytest-orange.svg)](https://pytest.org/)

A sophisticated real-time traffic monitoring system using computer vision and machine learning for vehicle detection, tracking, license plate recognition, and traffic analytics.

## ✨ Features

- **Real-time Vehicle Detection**: YOLO-based vehicle detection with configurable confidence thresholds
- **Multi-Object Tracking**: Advanced vehicle tracking using BoxMOT with Re-ID model support
- **License Plate Recognition**: Automated license plate detection and OCR reading
- **Traffic Analytics**: Vehicle counting, classification, and traffic flow analysis
- **Multiprocessing Architecture**: Efficient parallel processing for real-time performance
- **Flexible Configuration**: YAML-based configuration management
- **Comprehensive Logging**: Structured logging for debugging and monitoring

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Git
- CUDA-compatible GPU (optional, for enhanced performance)

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your_repo/trafficmonitor.git
   cd trafficmonitor
   ```

2. **Set up the environment:**

   ```bash
   # Using Pixi (recommended)
   pixi install

   # Or using pip with virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download pre-trained models:**

   ```bash
   python scripts/download_model.py
   ```

4. **Configure the system:**
   ```bash
   # Edit configuration file
   cp src/traffic_monitor/config/settings.yaml.example src/traffic_monitor/config/settings.yaml
   # Adjust settings according to your needs
   ```

### Basic Usage

```bash
# Run the complete traffic monitoring system
python src/traffic_monitor/main_supervisor.py

# Or run individual components
python -m traffic_monitor.services.vehicle_detector
python -m traffic_monitor.services.vehicle_tracker
```

## 📁 Project Structure

```
trafficmonitor/
├── data/                           # Data storage
│   ├── db/                        # Database files
│   ├── models/                    # ML models (.pt, .onnx files)
│   └── videos/                    # Input video files
├── docs/                          # Documentation
│   ├── REQUIREMENTS.md            # System requirements
│   ├── ARCHITECTURE.md           # Architecture documentation
│   ├── COMPONENT_*.md            # Component-specific docs
│   └── GAP_ANALYSIS.md           # Known issues and TODOs
├── examples/                      # Example usage scripts
├── logs/                         # Application logs
├── scripts/                      # Utility scripts
│   ├── convert_to_onnx.py       # Model conversion
│   └── download_model.py        # Model downloading
├── src/traffic_monitor/          # Main application source
│   ├── config/                   # Configuration files
│   ├── services/                 # Business logic services
│   ├── utils/                    # Utility functions
│   └── main_supervisor.py       # Main entry point
└── test/                         # Test suite
    ├── services/                 # Service tests
    └── test_*.py                # Test files
```

## ⚙️ Configuration

The system uses YAML configuration files for easy customization:

### Main Configuration (`src/traffic_monitor/config/settings.yaml`)

```yaml
video:
  source: "data/videos/input/traffic.mp4"
  fps: 30

detection:
  model_path: "data/models/yolo11s.pt"
  confidence_threshold: 0.5
  device: "cpu" # or "cuda"

tracking:
  tracker_type: "bytetrack"
  reid_model_path: "data/models/reid.pt"

logging:
  level: "INFO"
  file: "logs/traffic_monitor.log"
```

### Tracker Configuration

Individual tracker configurations are available in `src/traffic_monitor/config/trackers/`:

- `bytetrack.yaml` - ByteTrack configuration
- `botsort.yaml` - BotSORT configuration
- `deepocsort.yaml` - DeepOCSORT configuration

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/traffic_monitor --cov-report=html

# Run specific test category
pytest test/services/
pytest test/test_integration.py
```

## 📊 Performance

- **Real-time Processing**: Optimized for real-time video processing
- **Multi-core Utilization**: Efficient multiprocessing architecture
- **Memory Management**: Configurable queue sizes and buffer management
- **GPU Acceleration**: CUDA support for enhanced performance

## 🔧 Development

### Code Quality

The project follows modern Python development practices:

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Type checking
mypy src/

# Run pre-commit hooks
pre-commit run --all-files
```

### Adding New Components

1. Create service in `src/traffic_monitor/services/`
2. Add configuration to `settings.yaml`
3. Write comprehensive tests
4. Update component documentation
5. Update this README if needed

## 📋 System Requirements

### Functional Requirements

- Video stream processing from multiple sources
- Real-time vehicle detection and tracking
- License plate recognition and OCR
- Traffic counting and analytics
- Configurable detection parameters

### Non-Functional Requirements

- **Performance**: Real-time processing capabilities
- **Scalability**: Multiprocessing architecture
- **Reliability**: Graceful error handling
- **Maintainability**: Clean, documented code
- **Deployment**: Container and cloud-ready

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/trafficmonitor.git
cd trafficmonitor

# Install development dependencies
pixi install --all-extras

# Setup pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest
```

## 📖 Documentation

- **[Requirements](docs/REQUIREMENTS.md)**: Detailed functional and non-functional requirements
- **[Architecture](docs/ARCHITECTURE.md)**: System architecture and design decisions
- **[Component Docs](docs/)**: Individual component documentation
- **[API Reference](docs/api/)**: Auto-generated API documentation
- **[Examples](examples/)**: Usage examples and tutorials

## 🐛 Known Issues

See [GAP_ANALYSIS.md](docs/GAP_ANALYSIS.md) for current known issues and planned improvements.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [BoxMOT](https://github.com/mikel-brostrom/boxmot) for multi-object tracking
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection
- [OpenCV](https://opencv.org/) for computer vision operations

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your_repo/trafficmonitor/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your_repo/trafficmonitor/discussions)
- **Email**: support@trafficmonitor.com

---

**Star ⭐ this repository if you find it helpful!**
