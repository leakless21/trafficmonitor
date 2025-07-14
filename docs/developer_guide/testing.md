# Testing

Unit and integration tests are located in the `test/` directory. To run tests, use `pytest`:

```bash
pytest
```

## End-to-End Benchmark Guide

This guide explains how to use the comprehensive E2E benchmarking system for the Traffic Monitor thesis project.

### Overview

The E2E benchmark system provides:

- **System-level evaluation** that combines all components (detection, tracking, plate recognition, counting)
- **Performance profiling** with CPU/GPU resource monitoring
- **Automated CI/CD integration** with GitHub Actions
- **Reproducible results** with deterministic configurations

### Quick Start

#### 1. Run a Basic Benchmark

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

#### 2. Check Results Against Thresholds

```bash
# Check results against thresholds
python tools/assert_thresholds.py output/benchmarks/test_run/metrics.json

# Strict mode (warnings become failures)
python tools/assert_thresholds.py output/benchmarks/test_run/metrics.json --strict
```

