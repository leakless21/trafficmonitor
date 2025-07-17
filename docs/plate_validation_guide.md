# Plate Display Validation Guide

## Overview

This guide provides multiple methods to verify that all detected license plates are properly displayed in the traffic monitoring system's output video.

## Validation Methods

### 🚀 Method 1: Quick Validation (Recommended)

**Purpose**: Fast check using summary reports and logs
**Use when**: You want a quick overview of system performance

```bash
uv run python tools/validation/simple_plate_validator.py
```

**What it checks**:
- ✅ Summary report statistics
- ✅ Database OCR records  
- ✅ Log file analysis
- ✅ Cross-validation between sources

**Sample Output**:
```
✅ VALIDATION PASSED
   - Plates are being detected and processed
   - Event Fusion Service is working correctly
   - Visualization should display plate information
```

### 🧪 Method 2: Synthetic Data Test

**Purpose**: Controlled test with known plate data
**Use when**: You want to verify the fusion logic works correctly

```bash
uv run python tools/validation/test_plate_display_system.py
```

**What it tests**:
- ✅ Complete data flow (tracking → plate detection → OCR)
- ✅ Out-of-order message handling
- ✅ Partial data scenarios
- ✅ Event Fusion Service logic

**Sample Output**:
```
🎉 ALL TESTS PASSED!
   ✅ Event Fusion Service correctly merges plate data
   ✅ Enriched messages contain proper plate information
   ✅ Out-of-order message handling works
   ✅ Visualization will display plates correctly
```

### 🔍 Method 3: Comprehensive Video Analysis

**Purpose**: Deep analysis of actual video output
**Use when**: You need pixel-perfect validation

```bash
uv run python tools/validation/plate_display_validator.py path/to/output/video.mp4
```

**What it does**:
- 🔍 Extracts text from video frames using OCR
- 📊 Compares with database records
- 📋 Generates detailed validation report
- 🎬 Creates annotated video (optional)

**Additional Options**:
```bash
# Save detailed report
uv run python tools/validation/plate_display_validator.py video.mp4 --save-report

# Create annotated video showing validation status
uv run python tools/validation/plate_display_validator.py video.mp4 --create-annotated
```

## Understanding Results

### ✅ Success Indicators

1. **High OCR Success Rate**: >80% in summary reports
2. **Database Records**: Plates found in `ocr_results` table
3. **Log Mentions**: Plate texts appear in visualization logs
4. **Cross-validation**: Database and log data overlap

### ⚠️ Warning Signs

1. **Low OCR Success Rate**: <50% indicates detection issues
2. **Empty Database**: No OCR records suggests pipeline problems
3. **Missing Log Entries**: No plate mentions in visualization logs
4. **No Overlap**: Database and logs don't match

### ❌ Failure Indicators

1. **Zero Plates Detected**: No plates found anywhere
2. **Pipeline Broken**: Services not communicating
3. **Fusion Issues**: Data not being merged correctly

## Troubleshooting Guide

### No Plates Detected

**Symptoms**: All validation tools show 0 plates
**Causes**:
- Video has no visible license plates
- Plates too small/blurry for detection
- License plate detection model not loaded
- Detection confidence threshold too high

**Solutions**:
```bash
# Check detection model
ls -la data/models/plate/

# Lower detection threshold in config
# lp_detector:
#   conf_threshold: 0.5  # Lower from 0.7

# Test with known plate images
```

### Plates Detected but Not Displayed

**Symptoms**: Database has plates, but validation shows missing
**Causes**:
- Event Fusion Service not running
- Visualization service not receiving enriched data
- TTL timeout too short
- Message routing issues

**Solutions**:
```bash
# Check Event Fusion Service logs
grep "EventFusionService" logs/traffic_monitor.log

# Verify queue routing in main_supervisor.py
# Check fusion_output_queue → visualization_input_queue

# Increase TTL in config
# event_fusion:
#   ttl_sec: 2.0  # Increase from 1.0
```

### OCR Working but Display Missing

**Symptoms**: OCR results in database, but not in video
**Causes**:
- Visualization service using legacy queues
- Event Fusion Service not merging data
- Font/rendering issues in visualization

**Solutions**:
```bash
# Check visualization service receives enriched messages
grep "EnrichedTrackedVehicleMessage" logs/traffic_monitor.log

# Verify visualization service processes plate data
grep "plate_text" logs/traffic_monitor.log

# Check font configuration in visualizer config
```

## Best Practices

### For Development
1. **Always run synthetic test first** to verify fusion logic
2. **Use offline mode** for complete data validation
3. **Check logs regularly** for fusion service metrics
4. **Test with known plate images** for baseline validation

### For Production
1. **Monitor OCR success rates** in summary reports
2. **Set up automated validation** in CI/CD pipeline
3. **Alert on low display rates** (<80%)
4. **Regular spot checks** with video analysis

### For Debugging
1. **Enable debug logging** for detailed message flow
2. **Use real-time mode** to check latency issues
3. **Examine individual frames** for rendering problems
4. **Cross-reference timestamps** between services

## Configuration for Better Plate Detection

### Optimize Detection
```yaml
lp_detector:
  conf_threshold: 0.6        # Lower for more detections
  model_path: "data/models/plate/best.pt"

vehicle_detector:
  conf_threshold: 0.5        # Detect more vehicles
```

### Optimize OCR
```yaml
ocr_reader:
  backend: fast_plate_ocr
  conf_threshold: 0.7        # Balance accuracy vs coverage
  device: "cuda"             # Use GPU if available
```

### Optimize Fusion
```yaml
event_fusion:
  ttl_sec: 2.0              # Wait longer for complete data
  max_buffer_size: 2000     # Handle more concurrent objects
```

## Validation Checklist

Before deploying to production:

- [ ] ✅ Synthetic test passes (100% success rate)
- [ ] ✅ Quick validation shows >80% OCR success
- [ ] ✅ Database contains OCR records
- [ ] ✅ Logs show plate mentions
- [ ] ✅ Video analysis confirms display
- [ ] ✅ Performance meets requirements
- [ ] ✅ Error handling works correctly

## Automated Validation

For CI/CD integration:

```bash
#!/bin/bash
# validation_pipeline.sh

echo "Running plate display validation..."

# 1. Quick validation
if ! uv run python tools/validation/simple_plate_validator.py; then
    echo "❌ Quick validation failed"
    exit 1
fi

# 2. Synthetic test
if ! uv run python tools/validation/test_plate_display_system.py; then
    echo "❌ Synthetic test failed"
    exit 1
fi

# 3. Video analysis (if output exists)
if [ -f "data/outputs/videos/latest/output.mp4" ]; then
    if ! uv run python tools/validation/plate_display_validator.py data/outputs/videos/latest/output.mp4; then
        echo "⚠️  Video analysis failed (non-critical)"
    fi
fi

echo "✅ All validations passed!"
```

The validation tools ensure that the Event Fusion Service correctly merges plate data and that all detected plates appear in the final visualization output.