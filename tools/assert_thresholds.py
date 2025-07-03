#!/usr/bin/env python3
"""
Threshold Assertion Script for CI/CD Pipeline.

This script checks if benchmark metrics meet minimum performance requirements
and exits with appropriate codes for CI gating.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
from loguru import logger


# Performance thresholds for CI gating
THRESHOLDS = {
    # Vehicle identification thresholds
    "vehicle_f1": ("min", 0.70),  # Minimum F1 score for vehicle identification
    "vehicle_precision": ("min", 0.75),  # Minimum precision
    "vehicle_recall": ("min", 0.65),  # Minimum recall
    
    # Plate recognition thresholds  
    "plate_f1": ("min", 0.60),  # Minimum F1 score for plate recognition
    "plate_precision": ("min", 0.70),  # Minimum precision for plates
    
    # Counting accuracy thresholds
    "count_mae": ("max", 2.0),  # Maximum mean absolute error for counting
    "count_smape": ("max", 20.0),  # Maximum symmetric MAPE percentage
    
    # Queue estimation thresholds
    "queue_mae": ("max", 1.5),  # Maximum queue length error
    
    # Performance thresholds
    "mean_latency_ms": ("max", 150.0),  # Maximum mean latency (ms)
    "p95_latency_ms": ("max", 250.0),  # Maximum 95th percentile latency (ms)
    "fps": ("min", 5.0),  # Minimum frames per second
    
    # Overall system threshold
    "overall_f1": ("min", 0.65),  # Minimum overall F1 score
}

# Warning thresholds (don't fail CI but log warnings)
WARNING_THRESHOLDS = {
    "vehicle_f1": ("min", 0.75),
    "plate_f1": ("min", 0.65), 
    "count_mae": ("max", 1.5),
    "mean_latency_ms": ("max", 120.0),
    "fps": ("min", 10.0),
    "overall_f1": ("min", 0.70),
}


def load_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)
        
        # Flatten nested structure for easier access
        flattened = {}
        
        metrics = data.get('metrics', {})
        for category, values in metrics.items():
            if isinstance(values, dict):
                for key, value in values.items():
                    flattened[f"{key}"] = value
            else:
                flattened[category] = values
        
        return flattened
        
    except Exception as e:
        logger.error(f"Failed to load metrics from {metrics_path}: {e}")
        return {}


def check_threshold(value: float, threshold_type: str, threshold_value: float) -> bool:
    """Check if a value meets the threshold requirement."""
    if threshold_type == "min":
        return value >= threshold_value
    elif threshold_type == "max":
        return value <= threshold_value
    else:
        raise ValueError(f"Unknown threshold type: {threshold_type}")


def evaluate_thresholds(metrics: Dict[str, Any], thresholds: Dict[str, Tuple[str, float]], 
                       threshold_name: str = "threshold") -> Tuple[bool, list, list]:
    """
    Evaluate metrics against thresholds.
    
    Returns:
        (all_passed, failures, warnings)
    """
    failures = []
    warnings = []
    
    for metric_name, (threshold_type, threshold_value) in thresholds.items():
        if metric_name not in metrics:
            warning_msg = f"Metric '{metric_name}' not found in results"
            warnings.append(warning_msg)
            continue
        
        value = metrics[metric_name]
        passed = check_threshold(value, threshold_type, threshold_value)
        
        if not passed:
            failure_msg = f"{metric_name}: {value:.3f} fails {threshold_type} {threshold_value}"
            if threshold_name == "threshold":
                failures.append(failure_msg)
            else:
                warnings.append(failure_msg)
        else:
            logger.debug(f"{metric_name}: {value:.3f} passes {threshold_type} {threshold_value}")
    
    all_passed = len(failures) == 0
    return all_passed, failures, warnings


def print_summary_table(metrics: Dict[str, Any]):
    """Print a formatted summary table of key metrics."""
    logger.info("=" * 70)
    logger.info("BENCHMARK METRICS SUMMARY")
    logger.info("=" * 70)
    
    # Key metrics to display
    key_metrics = [
        ("Overall F1", "f1", "higher"),
        ("Vehicle F1", "f1", "higher"), 
        ("Plate F1", "f1", "higher"),
        ("Count MAE", "mae", "lower"),
        ("Mean Latency (ms)", "mean_latency_ms", "lower"),
        ("P95 Latency (ms)", "p95_latency_ms", "lower"),
        ("FPS", "fps", "higher"),
    ]
    
    for display_name, metric_key, direction in key_metrics:
        value = metrics.get(metric_key, 0.0)
        
        # Get threshold for coloring
        threshold_info = THRESHOLDS.get(metric_key)
        status = "✓"
        if threshold_info:
            threshold_type, threshold_value = threshold_info
            passed = check_threshold(value, threshold_type, threshold_value)
            status = "✓" if passed else "✗"
        
        logger.info(f"{display_name:<25} {value:>8.3f} {status}")
    
    logger.info("=" * 70)


def main():
    """Main threshold checking function."""
    parser = argparse.ArgumentParser(description="Check benchmark thresholds for CI")
    parser.add_argument("metrics_file", help="Path to metrics.json file")
    parser.add_argument("--strict", action="store_true", 
                       help="Treat warnings as failures")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--output-format", choices=["text", "json", "github"],
                       default="text", help="Output format for CI systems")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # Load metrics
    metrics_path = Path(args.metrics_file)
    if not metrics_path.exists():
        logger.error(f"Metrics file not found: {metrics_path}")
        return 2
    
    metrics = load_metrics(metrics_path)
    if not metrics:
        logger.error("Failed to load metrics")
        return 2
    
    # Print summary
    if args.output_format == "text":
        print_summary_table(metrics)
    
    # Check thresholds
    passed, failures, warnings = evaluate_thresholds(metrics, THRESHOLDS, "critical")
    
    # Check warning thresholds
    if WARNING_THRESHOLDS:
        _, warning_failures, extra_warnings = evaluate_thresholds(
            metrics, WARNING_THRESHOLDS, "warning"
        )
        warnings.extend(warning_failures)
        warnings.extend(extra_warnings)
    
    # Output results based on format
    if args.output_format == "json":
        result = {
            "passed": passed,
            "failures": failures,
            "warnings": warnings,
            "metrics": metrics
        }
        print(json.dumps(result, indent=2))
    
    elif args.output_format == "github":
        # GitHub Actions output format
        if failures:
            print("::error::Benchmark thresholds failed")
            for failure in failures:
                print(f"::error::{failure}")
        
        if warnings:
            for warning in warnings:
                print(f"::warning::{warning}")
        
        # Set output variables
        print(f"::set-output name=passed::{str(passed).lower()}")
        print(f"::set-output name=overall_f1::{metrics.get('f1', 0.0):.3f}")
    
    else:  # text format
        if failures:
            logger.error("CRITICAL THRESHOLD FAILURES:")
            for failure in failures:
                logger.error(f"  ✗ {failure}")
        
        if warnings:
            logger.warning("PERFORMANCE WARNINGS:")
            for warning in warnings:
                logger.warning(f"  ⚠ {warning}")
        
        if passed and not (args.strict and warnings):
            logger.success("All critical thresholds passed! ✅")
        elif args.strict and warnings:
            logger.error("Strict mode enabled: treating warnings as failures")
            passed = False
    
    # Determine exit code
    if not passed:
        return 1  # Critical failure
    elif args.strict and warnings:
        return 1  # Strict mode treats warnings as failures
    elif warnings:
        return 0  # Warnings but passing
    else:
        return 0  # All good


if __name__ == "__main__":
    sys.exit(main()) 