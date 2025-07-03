# End-to-End Benchmark Guide

This guide explains how to use the comprehensive E2E benchmarking system for the Traffic Monitor thesis project.

## Overview

The E2E benchmark system provides:

- **System-level evaluation** that combines all components (detection, tracking, plate recognition, counting)
- **Performance profiling** with CPU/GPU resource monitoring
- **Automated CI/CD integration** with GitHub Actions
- **Reproducible results** with deterministic configurations

## Quick Start

### 1. Run a Basic Benchmark

```bash
# Fast benchmark (for development/CI)
python tools/benchmark_e2e.py \
  --config configs/benchmark/fast.yaml \
  --videos configs/benchmark/eval_videos.yaml \
  --output output/benchmarks/test_run

# Production benchmark (for final evaluation)
python tools/benchmark_e2e.py \
  --config configs/benchmark/prod.yaml \
  --videos configs/benchmark/eval_videos.yaml \
  --output output/benchmarks/prod_run \
  --profile
```

### 2. Run Performance-Only Benchmark

For quick performance checks without ground truth requirements:

```bash
# Fast performance benchmark (no ground truth needed)
python tools/benchmark_performance.py \
  --config configs/benchmark/perf_fast.yaml \
  --videos configs/benchmark/eval_videos.yaml \
  --output output/benchmarks/perf_fast \
  --profile \
  --warmup

# Standard performance benchmark with multiple iterations
python tools/benchmark_performance.py \
  --config configs/benchmark/perf_standard.yaml \
  --videos configs/benchmark/eval_videos.yaml \
  --output output/benchmarks/perf_standard \
  --profile \
  --iterations 3 \
  --verbose
```

### 3. Check Results Against Thresholds

```bash
# Check E2E results against thresholds
python tools/assert_thresholds.py output/benchmarks/test_run/metrics.json

# Check performance-only results
python tools/assert_performance_thresholds.py output/benchmarks/perf_fast/performance_metrics.json --fast-mode

# Strict mode (warnings become failures)
python tools/assert_thresholds.py output/benchmarks/test_run/metrics.json --strict
```

## Configuration

### Benchmark Profiles

**Fast Configuration** (`configs/benchmark/fast.yaml`)

- Lower resolution (640x360)
- Frame skipping (every 3rd frame)
- Reduced confidence thresholds
- Minimal logging
- Real-time processing mode

**Production Configuration** (`configs/benchmark/prod.yaml`)

- Full resolution (1280x720)
- Process every frame
- Higher confidence thresholds
- Detailed logging
- Offline processing mode (preserves all frames)

**Performance Fast Configuration** (`configs/benchmark/perf_fast.yaml`)

- Lower resolution (640x480)
- Frame skipping (every 3rd frame)
- Reduced detection count limits
- Minimal logging
- Optimized for speed testing

**Performance Standard Configuration** (`configs/benchmark/perf_standard.yaml`)

- Standard resolution (1280x720)
- Process every frame
- Balanced settings
- Performance monitoring enabled
- Realistic workload simulation

### Video Configuration

Edit `configs/benchmark/eval_videos.yaml` to add evaluation videos:

```yaml
videos:
  - name: "traffic_day"
    path: "data/eval/videos/traffic_day.mp4"
    description: "Daytime traffic scenario"
    duration_seconds: 120
    expected_vehicles: 15
    expected_plates: 10
```

### Ground Truth Format

Create ground truth files in `data/eval/ground_truth/{video_name}.events.json`:

```json
[
  {
    "event": "VehiclePassed",
    "track_id": 1,
    "plate": "51F12345",
    "ts_enter": 3.84,
    "ts_exit": 4.77,
    "vehicle_class": "car",
    "confidence": 0.95
  },
  {
    "event": "QueueLength",
    "timestamp": 5.0,
    "length": 2
  }
]
```

## Performance Thresholds

### E2E Benchmark Thresholds

The system enforces these minimum performance requirements for complete E2E evaluation:

| Metric       | Critical Threshold | Target Threshold |
| ------------ | ------------------ | ---------------- |
| Overall F1   | ≥ 0.65             | ≥ 0.70           |
| Vehicle F1   | ≥ 0.70             | ≥ 0.75           |
| Plate F1     | ≥ 0.60             | ≥ 0.65           |
| Count MAE    | ≤ 2.0              | ≤ 1.5            |
| Mean Latency | ≤ 150ms            | ≤ 120ms          |
| P95 Latency  | ≤ 250ms            | -                |
| FPS          | ≥ 5.0              | ≥ 10.0           |

### Performance-Only Thresholds

For performance benchmarks without ground truth:

| Metric          | Critical Threshold | Target Threshold |
| --------------- | ------------------ | ---------------- |
| FPS             | ≥ 3.0              | ≥ 8.0            |
| Frame Time      | ≤ 500ms            | ≤ 150ms          |
| P95 Latency     | ≤ 1000ms           | ≤ 300ms          |
| Memory Usage    | ≤ 2048MB           | ≤ 1024MB         |
| GPU Utilization | ≤ 95%              | ≤ 80%            |
| FPS Stability   | -                  | ≤ 2.0 (std dev)  |

_Fast mode applies relaxed multipliers to these thresholds_

To modify thresholds, edit `THRESHOLDS` in `tools/assert_thresholds.py` or `PerformanceThresholds` in `tools/assert_performance_thresholds.py`.

## Output Files

After running a benchmark, you'll find:

```
output/benchmarks/{run_id}/
├── metrics.json               # E2E benchmark results
├── performance_metrics.json   # Performance-only results
├── profiling.csv              # Detailed timing data
├── performance_profiling.csv  # Performance benchmark timing
├── {video_name}.pred.json     # Predictions per video
└── system_info.json           # Hardware/software info
```

### metrics.json Structure (E2E Benchmark)

```json
{
  "benchmark_info": {
    "timestamp": "2025-01-01T10:00:00",
    "total_time_seconds": 45.2,
    "videos_processed": 1,
    "system_info": {...}
  },
  "metrics": {
    "vehicle_identification": {
      "precision": 0.85,
      "recall": 0.78,
      "f1": 0.81
    },
    "plate_recognition": {
      "precision": 0.72,
      "recall": 0.68,
      "f1": 0.70
    },
    "counting": {
      "mae": 1.2,
      "rmse": 1.8,
      "smape": 15.5
    },
    "timing": {
      "mean_latency_ms": 98.5,
      "p95_latency_ms": 145.2,
      "fps": 15.3
    },
    "overall": {
      "f1": 0.75
    }
  }
}
```

### performance_metrics.json Structure (Performance-Only)

```json
{
  "benchmark_info": {
    "type": "performance_only",
    "timestamp": "2025-01-01T10:00:00",
    "total_time_seconds": 25.8,
    "iterations": 3,
    "videos_per_iteration": 2,
    "warmup_enabled": true,
    "system_info": {...}
  },
  "performance_metrics": {
    "fps": {
      "mean": 12.5,
      "median": 12.8,
      "min": 9.2,
      "max": 15.1,
      "std": 1.8
    },
    "frame_time_ms": {
      "mean": 85.3,
      "median": 82.1,
      "std": 12.4
    },
    "p95_latency_ms": {
      "mean": 145.2,
      "max": 168.5
    }
  },
  "profiler_stats": {
    "memory": {
      "peak_mb": 892.4,
      "mean_mb": 765.2
    },
    "gpu": {
      "peak_utilization": 78.5,
      "mean_utilization": 65.2
    }
  }
}
```

## CI/CD Integration

### GitHub Actions

The benchmark automatically runs on:

- **Pull Requests**: Fast configuration with performance checks
- **Main branch pushes**: Production configuration with strict thresholds
- **Manual dispatch**: Choose configuration and strict mode

### Local Pre-commit

Add to your development workflow:

```bash
# Before committing changes
python tools/benchmark_e2e.py \
  --config configs/benchmark/fast.yaml \
  --videos configs/benchmark/eval_videos.yaml \
  --output output/benchmarks/pre_commit

python tools/assert_thresholds.py output/benchmarks/pre_commit/metrics.json
```

## Advanced Usage

### Custom Evaluation

```python
from traffic_monitor.eval.e2e_evaluator import E2EEvaluator

evaluator = E2EEvaluator(
    iou_threshold=0.5,      # Spatial matching threshold
    temporal_threshold=1.0   # Temporal matching threshold (seconds)
)

metrics = evaluator.evaluate(
    gt_path="data/eval/ground_truth/my_video.events.json",
    pred_path="output/benchmarks/run/my_video.pred.json"
)

print(f"Overall F1: {metrics.overall_f1:.3f}")
```

### Profiling Integration

```python
from traffic_monitor.utils.profiler import Profiler

profiler = Profiler(enabled=True)

with profiler.section("my_component"):
    # Your code here
    pass

stats = profiler.get_stats()
print(f"Mean time: {stats['my_component_mean_ms']:.1f}ms")
```

## Troubleshooting

### Common Issues

**"Ground truth not found"**

- Ensure ground truth files exist in `data/eval/ground_truth/`
- Check that video names in `eval_videos.yaml` match ground truth filenames

**"Benchmark failed to complete"**

- Check that all required models are downloaded (`python tools/download_model.py`)
- Verify video files exist and are readable
- Check logs in `logs/benchmark.log`

**"Thresholds failed"**

- Review specific failing metrics in the output
- Consider if the failure indicates a real regression
- Adjust thresholds if justified by research requirements

### Performance Tips

1. **For faster iteration**: Use `fast.yaml` configuration
2. **For development**: Add `--verbose` flag for detailed logging
3. **For CI**: Use GitHub Actions matrix to test multiple configurations
4. **For analysis**: Enable `--profile` for detailed resource usage data

## Integration with Thesis

### Validation Strategy

1. **Component Benchmarks**: Individual detection, tracking, OCR metrics
2. **E2E Benchmarks**: System-level performance combining all components
3. **Ablation Studies**: Compare different tracker/OCR combinations
4. **Resource Analysis**: CPU/GPU utilization under different loads

### Reporting

Use the benchmark results for:

- **Tables**: Component vs E2E metrics comparison
- **Figures**: Performance vs accuracy tradeoffs
- **Charts**: Resource utilization over time
- **Validation**: Reproducible results for thesis defense

The E2E benchmark provides the comprehensive evaluation framework needed for rigorous academic validation of your traffic monitoring system.
