# Scripts Directory

This directory contains utility scripts for the traffic monitor project.

## Available Scripts

### 1. batch_plate_crop.py

Batch processing script for cropping license plates from images.

### 2. paddleocr_to_csv.py

**LEGACY**: PaddleOCR processing script that reads license plates from images and outputs results to CSV format similar to `train_anotaciones.csv`.

#### Features:

- Uses PaddleOCR for text recognition
- Processes multiple image formats (JPG, PNG, JPEG, BMP, TIFF)
- Outputs CSV with `image_path,plate_text` format
- Supports GPU acceleration (optional)
- Multiple language support
- Confidence filtering for better accuracy
- Progress tracking with tqdm
- Comprehensive logging

#### Usage:

```bash
# Basic usage
python scripts/paddleocr_to_csv.py --input_dir path/to/images --output_csv results.csv

# With GPU acceleration
python scripts/paddleocr_to_csv.py --input_dir data/test_images --output_csv annotations.csv --use_gpu

# With different language
python scripts/paddleocr_to_csv.py --input_dir images --output_csv output.csv --lang ch

# Using pixi task
pixi run paddleocr_csv --input_dir data/test_images --output_csv results.csv
```

### 3. ocr_dataset_processor.py

**NEW & RECOMMENDED**: Advanced OCR dataset processor supporting multiple OCR engines with comprehensive features and better error handling.

#### Features:

- **Multiple OCR Engines**: FastPlateOCR (default) and PaddleOCR
- **Enhanced CSV Output**: Includes confidence scores, processing times, and engine info
- **Robust Error Handling**: Graceful handling of failed images and engine errors
- **Performance Monitoring**: Detailed statistics and processing time tracking
- **Batch Processing**: Intermediate saves and progress tracking
- **Flexible Configuration**: Customizable confidence thresholds and device selection
- **Comprehensive Logging**: Detailed logs with different verbosity levels

#### Usage:

```bash
# Use FastPlateOCR (recommended, default)
python scripts/ocr_dataset_processor.py --input_dir lp_data/CarTGMTCrop --output_csv results.csv

# Use PaddleOCR with specific language
python scripts/ocr_dataset_processor.py --input_dir data/test_images --output_csv results.csv --engine paddleocr --lang en

# Custom confidence threshold
python scripts/ocr_dataset_processor.py --input_dir images --output_csv annotations.csv --conf_threshold 0.7

# Use absolute paths and custom batch size
python scripts/ocr_dataset_processor.py --input_dir data/plates --output_csv results.csv --absolute_paths --batch_size 50

# High-performance processing with GPU (FastPlateOCR)
python scripts/ocr_dataset_processor.py --input_dir large_dataset --output_csv results.csv --device cuda

# PaddleOCR with GPU acceleration
python scripts/ocr_dataset_processor.py --input_dir dataset --output_csv results.csv --engine paddleocr --use_gpu
```

`pixi run ocr_dataset`

#### Arguments:

- `--input_dir`: Directory containing images to process (required)
- `--output_csv`: Output CSV file path (required)
- `--engine`: OCR engine to use ('fast_plate_ocr' or 'paddleocr', default: 'fast_plate_ocr')
- `--conf_threshold`: Confidence threshold for OCR results (default: 0.5)
- `--device`: Device for FastPlateOCR ('auto', 'cpu', 'cuda', default: 'auto')
- `--lang`: Language for PaddleOCR ('en', 'ch', 'french', 'german', 'korean', 'japan', default: 'en')
- `--use_gpu`: Use GPU acceleration for PaddleOCR
- `--absolute_paths`: Use absolute paths in CSV instead of relative paths
- `--batch_size`: Batch size for intermediate saves (default: 100)
- `--no_intermediate`: Disable intermediate result saving

#### Output CSV Format:

```csv
image_path,plate_text,confidence,processing_time,engine
test_images/plate1.jpg,ABC123,0.9500,0.1234,fast_plate_ocr
test_images/plate2.png,XYZ789,0.8750,0.0987,fast_plate_ocr
subfolder/plate3.jpg,DEF456,0.9200,0.1156,fast_plate_ocr
```

#### Performance Comparison:

| Engine       | Speed      | Accuracy   | GPU Support | Memory Usage |
| ------------ | ---------- | ---------- | ----------- | ------------ |
| FastPlateOCR | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ CUDA     | Low          |
| PaddleOCR    | ⭐⭐⭐     | ⭐⭐⭐⭐   | ✅ CUDA     | Medium       |

**Recommendation**: Use FastPlateOCR for best performance and accuracy on license plate recognition tasks.

#### Examples:

```bash
# Process a large dataset with FastPlateOCR
python scripts/ocr_dataset_processor.py \
    --input_dir lp_data/preprocessed_dataset/train \
    --output_csv output/train_ocr_results.csv \
    --engine fast_plate_ocr \
    --conf_threshold 0.6 \
    --batch_size 200

# Compare engines on the same dataset
python scripts/ocr_dataset_processor.py --input_dir data/test_images --output_csv fast_results.csv --engine fast_plate_ocr
python scripts/ocr_dataset_processor.py --input_dir data/test_images --output_csv paddle_results.csv --engine paddleocr

# Process validation dataset with high quality threshold
python scripts/ocr_dataset_processor.py \
    --input_dir lp_data/preprocessed_dataset/valid \
    --output_csv output/validation_high_quality.csv \
    --conf_threshold 0.8 \
    --absolute_paths
```

### 4. test_paddleocr_simple.py

Simple test script for validating PaddleOCR installation and functionality.

### 5. convert_to_onnx.py

Script for converting models to ONNX format for optimized inference.

### 6. ocr_evaluation.py

**NEW**: Comprehensive OCR performance evaluation script with industry-standard metrics.

#### Evaluation Metrics:

- **Character-Level Metrics**:
  - Character Error Rate (CER) using edit distance
  - Character-level precision, recall, F1 score
  - Insertion, deletion, substitution error counts
- **Plate-Level Metrics**:
  - Plate-level accuracy (exact match)
  - Plate-level precision, recall, F1 score
  - Detection rate and false positive/negative rates
- **Performance Metrics**:
  - Latency analysis (avg, median, min, max)
  - Throughput (plates per second)
  - Processing time distribution
- **Visual Analysis**:
  - Character-level confusion matrix
  - Performance distribution plots
  - Error rate visualizations

#### Features:

- **Flexible Text Normalization**: Case-insensitive matching, character substitution (O→0, I→1)
- **Robust File Alignment**: Smart matching between predictions and ground truth
- **Multiple Output Formats**: JSON, TXT reports, PNG visualizations
- **Configurable Evaluation**: Custom normalization rules and comparison settings

#### Usage:

```bash
# Basic evaluation
python scripts/ocr_evaluation.py \
    --predictions output/cartgmt_results.csv \
    --ground_truth lp_data/preprocessed_dataset/valid_anotaciones.csv \
    --output_dir evaluation_results

# Case-sensitive evaluation
python scripts/ocr_evaluation.py \
    --predictions results.csv \
    --ground_truth annotations.csv \
    --output_dir evaluation \
    --case_sensitive

# Custom normalization settings
python scripts/ocr_evaluation.py \
    --predictions results.csv \
    --ground_truth annotations.csv \
    --output_dir evaluation \
    --no_normalize \
    --ignore_chars - . _ space

# Using pixi task
pixi run ocr_eval --predictions output/results.csv --ground_truth annotations.csv --output_dir eval
```

### 7. comparison_evaluation.py

**NEW**: Side-by-side comparison of multiple OCR engines with comparative analysis.

#### Features:

- **Multi-Engine Comparison**: Compare 2+ OCR engines simultaneously
- **Comparative Visualizations**: Side-by-side performance charts
- **Best Performer Analysis**: Automatic identification of top performers
- **Comprehensive Reports**: Detailed comparison tables and summaries

#### Usage:

```bash
# Compare FastPlateOCR vs PaddleOCR
python scripts/comparison_evaluation.py \
    --predictions_list output/fast_results.csv output/paddle_results.csv \
    --ground_truth lp_data/preprocessed_dataset/valid_anotaciones.csv \
    --names FastPlateOCR PaddleOCR \
    --output_dir comparison_results

# Using pixi task
pixi run ocr_compare \
    --predictions_list results1.csv results2.csv \
    --ground_truth annotations.csv \
    --names Engine1 Engine2
```

### 8. convert_to_onnx.py

Script for converting models to ONNX format for optimized inference.

### 9. download_model.py

Utility script for downloading pre-trained models.

## Demo Scripts

See the `examples/` directory for demonstration scripts:

- `examples/ocr_dataset_demo.py`: Comprehensive demo of the OCR dataset processor
- `examples/evaluation_demo.py`: Demo of the OCR evaluation system
- `examples/paddleocr_demo.py`: Legacy PaddleOCR demo

## Dependencies

### FastPlateOCR

```bash
pip install fast-plate-ocr
```

### PaddleOCR

```bash
# CPU version
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install paddleocr

# GPU version (if you have CUDA)
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install paddleocr
```

## Performance Notes

- **FastPlateOCR**: Optimized for license plate recognition, fastest processing
- **PaddleOCR**: General-purpose OCR, good for various text recognition tasks
- **GPU Acceleration**: Available for both engines, significantly improves performance
- **Batch Processing**: Use larger batch sizes for better performance on large datasets
- **Confidence Filtering**: Higher thresholds reduce false positives but may miss valid plates
