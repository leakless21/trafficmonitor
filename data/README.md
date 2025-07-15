# Traffic Monitor Data Directory

This directory contains all data files for the Traffic Monitor system including models, test data, and outputs.

## Structure

```
data/
├── README.md                 # This file
├── db/                       # Database files
│   ├── .gitkeep
│   └── traffic_monitor.db    # SQLite database (auto-created)
├── models/                   # Model files (gitignored - download separately)
│   ├── .gitkeep
│   ├── download_models.sh    # Script to download required models
│   ├── vehicle/              # Vehicle detection models
│   │   ├── 8n/              # YOLOv8 nano model
│   │   ├── 5nu/             # YOLOv5 nano ultralytics model
│   │   └── [other variants]
│   └── plate/               # License plate detection models
│       ├── 5nu/             # Plate detection model
│       └── [other variants]
├── samples/                  # Small sample data for testing
│   ├── videos/              # Sample video files
│   └── images/              # Sample image files
└── outputs/                  # Generated outputs (gitignored)
    ├── .gitkeep
    ├── videos/              # Processed video outputs
    └── reports/             # Analysis reports
```

## Model Management

### Required Models

The Traffic Monitor system requires the following models:

1. **Vehicle Detection Model**: YOLOv8 trained for vehicle detection
   - Default: `data/models/vehicle/8n/best.engine`
   - Alternatives: Various YOLO variants in respective directories

2. **License Plate Detection Model**: Specialized model for license plate detection
   - Default: `data/models/plate/5nu/best.engine`
   - Alternatives: Other plate detection models

### Downloading Models

Models are not included in the repository due to size constraints. Download them using:

```bash
# Download all required models
bash data/models/download_models.sh

# Or download specific models manually
# See individual model directories for download instructions
```

### Model Formats

- **TensorRT (.engine)**: Optimized for NVIDIA GPUs (recommended for production)
- **ONNX (.onnx)**: Cross-platform format (good for development/testing)
- **PyTorch (.pt)**: Original training format (largest file size)

## Sample Data

The `samples/` directory contains small test files for development and testing:

- **videos/**: Short video clips for testing video processing pipeline
- **images/**: Individual frames for testing detection and recognition

## Output Management

The `outputs/` directory contains generated files:

- **videos/**: Processed videos with annotations and visualizations
- **reports/**: Analysis reports and summaries in JSON/CSV format

### Cleanup

To clean up generated outputs:

```bash
# Clean all outputs
rm -rf data/outputs/videos/* data/outputs/reports/*

# Or use the make command
make clean
```

## Storage Considerations

### Large Files

- Model files can be 100MB-2GB each
- Video files can be very large (GB per video)
- Use `.gitignore` patterns to exclude large files from version control

### Optimization Tips

1. **Use TensorRT models** for best performance on NVIDIA GPUs
2. **Compress videos** using efficient codecs (H.264, H.265)
3. **Clean outputs regularly** to save disk space
4. **Use symbolic links** for shared model files across projects

## Environment Variables

You can override data paths using environment variables:

```bash
# Override model paths
export VEHICLE_MODEL_PATH="/path/to/custom/vehicle/model.engine"
export PLATE_MODEL_PATH="/path/to/custom/plate/model.engine"

# Override data directories
export DATA_DIR="/path/to/custom/data"
export OUTPUT_DIR="/path/to/custom/outputs"
```

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure models are downloaded and paths are correct
2. **Permission errors**: Check file permissions in data directory
3. **Disk space**: Monitor disk usage, especially in outputs directory
4. **GPU memory**: Large models may require significant GPU memory

### Disk Usage Monitoring

```bash
# Check total data directory size
du -sh data/

# Check individual subdirectory sizes
du -sh data/*/

# Find largest files
find data/ -type f -exec ls -lh {} \; | sort -k5 -hr | head -10
```