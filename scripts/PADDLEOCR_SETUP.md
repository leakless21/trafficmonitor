# PaddleOCR to CSV Script Setup Guide

## Overview

The `paddleocr_to_csv.py` script processes images using PaddleOCR and outputs results to CSV format similar to `train_anotaciones.csv`.

## Prerequisites

### 1. Install PaddlePaddle

Since PaddlePaddle is not available via conda-forge for Windows, you need to install it via pip:

```bash
# For CPU version
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# For GPU version (if you have CUDA)
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

### 2. Install via Pixi (Recommended)

```bash
# Install project dependencies
pixi install

# Install PaddlePaddle in the pixi environment (required for PaddleOCR)
pixi run pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# For GPU version (if you have CUDA):
# pixi run pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

### 3. Test Installation

```bash
# Test the setup
python scripts/test_paddleocr_simple.py
```

## Usage Examples

### Basic Usage

```bash
# Using pixi task
pixi run paddleocr_csv --input_dir data/test_images --output_csv results.csv

# Direct python call
python scripts/paddleocr_to_csv.py --input_dir data/test_images --output_csv results.csv
```

### Advanced Usage

```bash
# With different language
pixi run paddleocr_csv --input_dir images --output_csv output.csv --lang ch

# Using absolute paths in output
pixi run paddleocr_csv --input_dir data/plates --output_csv annotations.csv --absolute_paths
```

## Input/Output Format

### Expected Input

- Directory containing image files (JPG, PNG, JPEG, BMP, TIFF)
- Images should contain license plates or text to be recognized

### Output Format

CSV file with two columns:

```csv
image_path,plate_text
test_images/plate1.jpg,ABC123
test_images/plate2.png,XYZ789
subfolder/plate3.jpg,DEF456
```

## Configuration Options

| Parameter          | Description                            | Default |
| ------------------ | -------------------------------------- | ------- |
| `--input_dir`      | Directory containing images (required) | -       |
| `--output_csv`     | Output CSV file path (required)        | -       |
| `--use_gpu`        | Use GPU acceleration (deprecated)      | False   |
| `--lang`           | Language for OCR recognition           | 'en'    |
| `--absolute_paths` | Use absolute paths in CSV              | False   |

## Supported Languages

- `en`: English
- `ch`: Chinese
- `french`: French
- `german`: German
- `korean`: Korean
- `japan`: Japanese

## Performance Notes

- **Confidence Threshold**: Set to 0.5 by default
- **Text Cleaning**: Removes spaces and special characters
- **Progress Tracking**: Uses tqdm for progress display
- **Logging**: Creates detailed logs in `paddleocr_processing.log`

## Troubleshooting

### Common Issues

1. **"No module named 'paddle'"**

   ```bash
   pixi run pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
   ```

2. **"Unknown argument: show_log"**

   - This is resolved in the latest version of the script

3. **"DeprecationWarning: use_angle_cls"**
   - The script uses the updated parameter `use_textline_orientation`

### Testing the Setup

```bash
# Run the test script
python scripts/test_paddleocr_simple.py

# Should output:
# ✅ PaddleOCR import successful
# ✅ Logging setup successful
# ✅ Found X image files in test directory
# 🎉 All tests passed! Script is ready to use.
```

## Integration with Traffic Monitor

The script is integrated into the traffic monitor project:

- Added to `pyproject.toml` dependencies
- Available as pixi task: `paddleocr_csv`
- Includes comprehensive unit tests
- Documented in main scripts README

## Example Output

```
2025-06-25 01:45:06,225 - INFO - Starting OCR processing for directory: data/test_images
2025-06-25 01:45:06,225 - INFO - Initializing PaddleOCR...
Processing images: 100%|██████████| 10/10 [00:15<00:00,  1.50s/it]
2025-06-25 01:45:21,350 - INFO - Processing complete! Processed 8 images with detected text
2025-06-25 01:45:21,350 - INFO - Results saved to: results.csv
```
