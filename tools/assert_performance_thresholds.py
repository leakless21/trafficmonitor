#!/usr/bin/env python3
"""
Performance Threshold Assertion Script for Traffic Monitor.

Checks performance benchmark results against defined thresholds for CI/CD gating.
This version focuses purely on performance metrics without ground truth requirements.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


class PerformanceThresholds:
    """Performance threshold definitions for different benchmark modes."""
    
    # Critical thresholds - must pass for CI success
    CRITICAL_THRESHOLDS = {
        "fps": {
            "min": 3.0,  # Minimum acceptable FPS
            "description": "Minimum frames per second"
        },
        "frame_time_ms": {
            "max": 500.0,  # Maximum acceptable frame time
            "description": "Maximum frame processing time"
        },
        "p95_latency_ms": {
            "max": 1000.0,  # P95 latency threshold
            "description": "95th percentile latency"
        },
        "memory_usage_mb": {
            "max": 2048.0,  # Maximum memory usage
            "description": "Peak memory usage"
        },
        "gpu_utilization": {
            "max": 95.0,  # Maximum GPU utilization
            "description": "Peak GPU utilization percentage"
        }
    }
    
    # Warning thresholds - generate warnings but don't fail CI
    WARNING_THRESHOLDS = {
        "fps": {
            "min": 8.0,  # Target FPS
            "description": "Target frames per second"
        },
        "frame_time_ms": {
            "max": 150.0,  # Target frame time
            "description": "Target frame processing time"
        },
        "p95_latency_ms": {
            "max": 300.0,  # Target P95 latency
            "description": "Target 95th percentile latency"
        },
        "fps_stability_std": {
            "max": 2.0,  # FPS standard deviation
            "description": "FPS stability (lower is better)"
        },
        "memory_usage_mb": {
            "max": 1024.0,  # Target memory usage
            "description": "Target memory usage"
        },
        "gpu_utilization": {
            "max": 80.0,  # Target GPU utilization
            "description": "Target GPU utilization percentage"
        }
    }
    
    # Fast mode has more relaxed thresholds
    FAST_MODE_MULTIPLIERS = {
        "fps": {"min": 0.7},  # 30% relaxed for fast mode
        "frame_time_ms": {"max": 1.5},  # 50% more lenient
        "p95_latency_ms": {"max": 2.0},  # 100% more lenient
    }


class ThresholdResult:
    """Result of a threshold check."""
    
    def __init__(self, metric: str, value: float, threshold: float, 
                 passed: bool, is_critical: bool, description: str):
        self.metric = metric
        self.value = value
        self.threshold = threshold
        self.passed = passed
        self.is_critical = is_critical
        self.description = description


class PerformanceThresholdChecker:
    """Checks performance metrics against defined thresholds."""
    
    def __init__(self, fast_mode: bool = False, custom_thresholds: Optional[Dict] = None):
        self.fast_mode = fast_mode
        self.custom_thresholds = custom_thresholds or {}
        
    def _get_threshold_value(self, metric: str, threshold_type: str, base_thresholds: Dict) -> float:
        """Get threshold value with fast mode adjustments."""
        base_value = base_thresholds.get(metric, {}).get(threshold_type)
        if base_value is None:
            return None
            
        # Apply fast mode multipliers
        if self.fast_mode and metric in PerformanceThresholds.FAST_MODE_MULTIPLIERS:
            multiplier = PerformanceThresholds.FAST_MODE_MULTIPLIERS[metric].get(threshold_type, 1.0)
            if threshold_type == "min":
                return base_value * multiplier
            else:  # max thresholds
                return base_value * multiplier
                
        return base_value
    
    def _check_metric_threshold(self, metric: str, value: float, 
                              threshold_value: float, threshold_type: str,
                              is_critical: bool, description: str) -> ThresholdResult:
        """Check a single metric against its threshold."""
        if threshold_type == "min":
            passed = value >= threshold_value
        else:  # "max"
            passed = value <= threshold_value
            
        return ThresholdResult(
            metric=metric,
            value=value,
            threshold=threshold_value,
            passed=passed,
            is_critical=is_critical,
            description=description
        )
    
    def check_performance_metrics(self, metrics: Dict[str, Any]) -> List[ThresholdResult]:
        """Check all performance metrics against thresholds."""
        results = []
        
        # Extract nested metrics
        fps_data = metrics.get("fps", {})
        frame_time_data = metrics.get("frame_time_ms", {})
        p95_data = metrics.get("p95_latency_ms", {})
        
        # Check FPS metrics
        if "mean" in fps_data:
            # Critical FPS check
            critical_fps = self._get_threshold_value("fps", "min", PerformanceThresholds.CRITICAL_THRESHOLDS)
            if critical_fps is not None:
                results.append(self._check_metric_threshold(
                    "fps", fps_data["mean"], critical_fps, "min", True,
                    PerformanceThresholds.CRITICAL_THRESHOLDS["fps"]["description"]
                ))
            
            # Warning FPS check
            warning_fps = self._get_threshold_value("fps", "min", PerformanceThresholds.WARNING_THRESHOLDS)
            if warning_fps is not None:
                results.append(self._check_metric_threshold(
                    "fps", fps_data["mean"], warning_fps, "min", False,
                    PerformanceThresholds.WARNING_THRESHOLDS["fps"]["description"]
                ))
        
        # Check frame time metrics
        if "mean" in frame_time_data:
            # Critical frame time check
            critical_frame_time = self._get_threshold_value("frame_time_ms", "max", PerformanceThresholds.CRITICAL_THRESHOLDS)
            if critical_frame_time is not None:
                results.append(self._check_metric_threshold(
                    "frame_time_ms", frame_time_data["mean"], critical_frame_time, "max", True,
                    PerformanceThresholds.CRITICAL_THRESHOLDS["frame_time_ms"]["description"]
                ))
            
            # Warning frame time check
            warning_frame_time = self._get_threshold_value("frame_time_ms", "max", PerformanceThresholds.WARNING_THRESHOLDS)
            if warning_frame_time is not None:
                results.append(self._check_metric_threshold(
                    "frame_time_ms", frame_time_data["mean"], warning_frame_time, "max", False,
                    PerformanceThresholds.WARNING_THRESHOLDS["frame_time_ms"]["description"]
                ))
        
        # Check P95 latency
        if "mean" in p95_data:
            # Critical P95 check
            critical_p95 = self._get_threshold_value("p95_latency_ms", "max", PerformanceThresholds.CRITICAL_THRESHOLDS)
            if critical_p95 is not None:
                results.append(self._check_metric_threshold(
                    "p95_latency_ms", p95_data["mean"], critical_p95, "max", True,
                    PerformanceThresholds.CRITICAL_THRESHOLDS["p95_latency_ms"]["description"]
                ))
            
            # Warning P95 check
            warning_p95 = self._get_threshold_value("p95_latency_ms", "max", PerformanceThresholds.WARNING_THRESHOLDS)
            if warning_p95 is not None:
                results.append(self._check_metric_threshold(
                    "p95_latency_ms", p95_data["mean"], warning_p95, "max", False,
                    PerformanceThresholds.WARNING_THRESHOLDS["p95_latency_ms"]["description"]
                ))
        
        # Check FPS stability
        if "std" in fps_data:
            warning_stability = PerformanceThresholds.WARNING_THRESHOLDS["fps_stability_std"]["max"]
            results.append(self._check_metric_threshold(
                "fps_stability_std", fps_data["std"], warning_stability, "max", False,
                PerformanceThresholds.WARNING_THRESHOLDS["fps_stability_std"]["description"]
            ))
        
        return results
    
    def check_profiler_metrics(self, profiler_stats: Dict[str, Any]) -> List[ThresholdResult]:
        """Check profiler resource metrics against thresholds."""
        results = []
        
        # Check memory usage
        memory_stats = profiler_stats.get("memory", {})
        if "peak_mb" in memory_stats:
            # Critical memory check
            critical_memory = PerformanceThresholds.CRITICAL_THRESHOLDS["memory_usage_mb"]["max"]
            results.append(self._check_metric_threshold(
                "memory_usage_mb", memory_stats["peak_mb"], critical_memory, "max", True,
                PerformanceThresholds.CRITICAL_THRESHOLDS["memory_usage_mb"]["description"]
            ))
            
            # Warning memory check
            warning_memory = PerformanceThresholds.WARNING_THRESHOLDS["memory_usage_mb"]["max"]
            results.append(self._check_metric_threshold(
                "memory_usage_mb", memory_stats["peak_mb"], warning_memory, "max", False,
                PerformanceThresholds.WARNING_THRESHOLDS["memory_usage_mb"]["description"]
            ))
        
        # Check GPU utilization if available
        gpu_stats = profiler_stats.get("gpu", {})
        if "peak_utilization" in gpu_stats:
            # Critical GPU check
            critical_gpu = PerformanceThresholds.CRITICAL_THRESHOLDS["gpu_utilization"]["max"]
            results.append(self._check_metric_threshold(
                "gpu_utilization", gpu_stats["peak_utilization"], critical_gpu, "max", True,
                PerformanceThresholds.CRITICAL_THRESHOLDS["gpu_utilization"]["description"]
            ))
            
            # Warning GPU check
            warning_gpu = PerformanceThresholds.WARNING_THRESHOLDS["gpu_utilization"]["max"]
            results.append(self._check_metric_threshold(
                "gpu_utilization", gpu_stats["peak_utilization"], warning_gpu, "max", False,
                PerformanceThresholds.WARNING_THRESHOLDS["gpu_utilization"]["description"]
            ))
        
        return results


def print_results_text(results: List[ThresholdResult], benchmark_info: Dict[str, Any]):
    """Print threshold check results in human-readable format."""
    print("=" * 70)
    print("PERFORMANCE THRESHOLD CHECK RESULTS")
    print("=" * 70)
    
    print(f"Benchmark Type: {benchmark_info.get('type', 'Unknown')}")
    print(f"Timestamp: {benchmark_info.get('timestamp', 'Unknown')}")
    print(f"Total Time: {benchmark_info.get('total_time_seconds', 0):.1f}s")
    print()
    
    # Separate critical and warning results
    critical_results = [r for r in results if r.is_critical]
    warning_results = [r for r in results if not r.is_critical]
    
    # Print critical results
    print("CRITICAL THRESHOLDS:")
    print("-" * 50)
    critical_passed = 0
    for result in critical_results:
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"{status} {result.metric}: {result.value:.2f} (threshold: {result.threshold:.2f})")
        print(f"    {result.description}")
        if result.passed:
            critical_passed += 1
    
    print()
    print("WARNING THRESHOLDS:")
    print("-" * 50)
    warning_passed = 0
    for result in warning_results:
        status = "✓ PASS" if result.passed else "⚠ WARN" 
        print(f"{status} {result.metric}: {result.value:.2f} (threshold: {result.threshold:.2f})")
        print(f"    {result.description}")
        if result.passed:
            warning_passed += 1
    
    print()
    print("SUMMARY:")
    print(f"Critical: {critical_passed}/{len(critical_results)} passed")
    print(f"Warnings: {warning_passed}/{len(warning_results)} passed")
    
    all_critical_passed = critical_passed == len(critical_results)
    if all_critical_passed:
        print("✓ All critical thresholds PASSED - CI can proceed")
    else:
        print("✗ Some critical thresholds FAILED - CI should be blocked")
    
    print("=" * 70)


def print_results_github(results: List[ThresholdResult], benchmark_info: Dict[str, Any]):
    """Print results in GitHub Actions format."""
    print("::group::Performance Threshold Results")
    
    critical_failures = [r for r in results if r.is_critical and not r.passed]
    warnings = [r for r in results if not r.is_critical and not r.passed]
    
    for failure in critical_failures:
        print(f"::error::Critical threshold failed: {failure.metric} = {failure.value:.2f} "
              f"(threshold: {failure.threshold:.2f}) - {failure.description}")
    
    for warning in warnings:
        print(f"::warning::Warning threshold exceeded: {warning.metric} = {warning.value:.2f} "
              f"(threshold: {warning.threshold:.2f}) - {warning.description}")
    
    if not critical_failures:
        print("::notice::All critical performance thresholds passed")
    
    print("::endgroup::")


def print_results_json(results: List[ThresholdResult], benchmark_info: Dict[str, Any]) -> Dict[str, Any]:
    """Return results in JSON format."""
    result_data = []
    for result in results:
        result_data.append({
            "metric": result.metric,
            "value": result.value,
            "threshold": result.threshold,
            "passed": result.passed,
            "is_critical": result.is_critical,
            "description": result.description
        })
    
    critical_passed = sum(1 for r in results if r.is_critical and r.passed)
    critical_total = sum(1 for r in results if r.is_critical)
    
    return {
        "benchmark_info": benchmark_info,
        "threshold_results": result_data,
        "summary": {
            "critical_passed": critical_passed,
            "critical_total": critical_total,
            "all_critical_passed": critical_passed == critical_total,
            "warning_count": sum(1 for r in results if not r.is_critical and not r.passed)
        }
    }


def main():
    """Main threshold checking execution."""
    parser = argparse.ArgumentParser(description="Performance Threshold Checker")
    parser.add_argument("results_file", help="Performance benchmark results JSON file")
    parser.add_argument("--format", choices=["text", "json", "github"], default="text",
                       help="Output format")
    parser.add_argument("--fast-mode", action="store_true", 
                       help="Use relaxed thresholds for fast mode")
    parser.add_argument("--output", help="Output file for JSON format")
    
    args = parser.parse_args()
    
    try:
        # Load benchmark results
        with open(args.results_file, 'r') as f:
            results_data = json.load(f)
        
        benchmark_info = results_data.get("benchmark_info", {})
        performance_metrics = results_data.get("performance_metrics", {})
        profiler_stats = results_data.get("profiler_stats", {})
        
        # Auto-detect fast mode if not specified
        fast_mode = args.fast_mode
        if not fast_mode and "fast" in str(args.results_file):
            fast_mode = True
        
        # Initialize threshold checker
        checker = PerformanceThresholdChecker(fast_mode=fast_mode)
        
        # Run threshold checks
        results = []
        results.extend(checker.check_performance_metrics(performance_metrics))
        results.extend(checker.check_profiler_metrics(profiler_stats))
        
        if not results:
            print("No performance metrics found to check")
            return 1
        
        # Output results
        if args.format == "text":
            print_results_text(results, benchmark_info)
        elif args.format == "github":
            print_results_github(results, benchmark_info)
        elif args.format == "json":
            json_results = print_results_json(results, benchmark_info)
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(json_results, f, indent=2)
            else:
                print(json.dumps(json_results, indent=2))
        
        # Exit with appropriate code
        critical_failures = [r for r in results if r.is_critical and not r.passed]
        return 1 if critical_failures else 0
        
    except Exception as e:
        print(f"Error checking thresholds: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 