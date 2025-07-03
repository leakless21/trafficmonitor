"""
Test cases for E2E benchmark system.

This module tests the end-to-end benchmarking functionality to ensure
reliable evaluation and CI integration.
"""

import pytest
import json
import tempfile
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from traffic_monitor.eval.e2e_evaluator import E2EEvaluator, VehicleEvent, QueueEvent, EvaluationMetrics
from traffic_monitor.utils.profiler import Profiler, get_system_info


class TestE2EEvaluator:
    """Test the E2E evaluation logic."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.evaluator = E2EEvaluator()
    
    def test_vehicle_matching(self):
        """Test vehicle temporal matching algorithm."""
        gt_vehicles = [
            VehicleEvent(1, "ABC123", 5.0, 6.0, "car", 1.0),
            VehicleEvent(2, None, 8.0, 9.0, "truck", 1.0)
        ]
        
        pred_vehicles = [
            VehicleEvent(1, "ABC123", 5.1, 6.1, "car", 0.9),  # Good match
            VehicleEvent(3, "XYZ789", 10.0, 11.0, "car", 0.8)  # No match
        ]
        
        matches, unmatched_gt, unmatched_pred = self.evaluator.match_vehicles(gt_vehicles, pred_vehicles)
        
        assert len(matches) == 1
        assert matches[0] == (0, 0)  # First GT matches first pred
        assert 1 in unmatched_gt  # Second GT vehicle unmatched
        assert 1 in unmatched_pred  # Second pred vehicle unmatched
    
    def test_vehicle_identification_metrics(self):
        """Test vehicle identification F1 calculation."""
        gt_vehicles = [
            VehicleEvent(1, "ABC123", 5.0, 6.0, "car", 1.0),
            VehicleEvent(2, None, 8.0, 9.0, "truck", 1.0)
        ]
        
        pred_vehicles = [
            VehicleEvent(1, "ABC123", 5.1, 6.1, "car", 0.9)  # One correct match
        ]
        
        metrics = self.evaluator.evaluate_vehicle_identification(gt_vehicles, pred_vehicles)
        
        assert metrics['true_positives'] == 1
        assert metrics['false_negatives'] == 1
        assert metrics['false_positives'] == 0
        assert metrics['precision'] == 1.0
        assert metrics['recall'] == 0.5
        assert metrics['f1'] == pytest.approx(0.667, rel=1e-2)
    
    def test_plate_recognition_metrics(self):
        """Test plate recognition evaluation."""
        gt_vehicles = [
            VehicleEvent(1, "ABC123", 5.0, 6.0, "car", 1.0),
            VehicleEvent(2, "DEF456", 8.0, 9.0, "truck", 1.0),
            VehicleEvent(3, None, 10.0, 11.0, "car", 1.0)  # No plate
        ]
        
        pred_vehicles = [
            VehicleEvent(1, "ABC123", 5.1, 6.1, "car", 0.9),  # Correct plate
            VehicleEvent(2, "DEF789", 8.1, 9.1, "truck", 0.8),  # Wrong plate
            VehicleEvent(3, None, 10.1, 11.1, "car", 0.7)  # No plate
        ]
        
        metrics = self.evaluator.evaluate_plate_recognition(gt_vehicles, pred_vehicles)
        
        assert metrics['correct'] == 1
        assert metrics['total_gt'] == 2  # Two vehicles with plates in GT
        assert metrics['total_pred'] == 1  # One vehicle with plate in pred
        assert metrics['precision'] == 1.0  # 1 correct out of 1 predicted
        assert metrics['recall'] == 0.5  # 1 correct out of 2 ground truth
    
    def test_counting_metrics(self):
        """Test vehicle counting evaluation."""
        gt_vehicles = [VehicleEvent(i, None, i, i+1, "car", 1.0) for i in range(5)]
        pred_vehicles = [VehicleEvent(i, None, i, i+1, "car", 1.0) for i in range(3)]
        
        metrics = self.evaluator.evaluate_counting(gt_vehicles, pred_vehicles)
        
        assert metrics['gt_count'] == 5
        assert metrics['pred_count'] == 3
        assert metrics['mae'] == 2
        assert metrics['rmse'] == 4  # (5-3)^2
        assert metrics['smape'] == pytest.approx(50.0, rel=1e-2)  # 2*|5-3|/(5+3)*100
    
    def test_ground_truth_loading(self):
        """Test loading ground truth from JSON file."""
        # Create temporary ground truth file
        gt_data = [
            {
                "event": "VehiclePassed",
                "track_id": 1,
                "plate": "ABC123",
                "ts_enter": 5.0,
                "ts_exit": 6.0,
                "vehicle_class": "car",
                "confidence": 0.9
            },
            {
                "event": "QueueLength",
                "timestamp": 10.0,
                "length": 3
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(gt_data, f)
            gt_path = Path(f.name)
        
        try:
            vehicles, queue = self.evaluator.load_ground_truth(gt_path)
            
            assert len(vehicles) == 1
            assert vehicles[0].track_id == 1
            assert vehicles[0].plate == "ABC123"
            assert vehicles[0].vehicle_class == "car"
            
            assert len(queue) == 1
            assert queue[0].timestamp == 10.0
            assert queue[0].length == 3
            
        finally:
            gt_path.unlink()


class TestProfiler:
    """Test the profiler functionality."""
    
    def test_profiler_timing(self):
        """Test basic timing functionality."""
        profiler = Profiler(enabled=True)
        
        with profiler.section("test_section"):
            import time
            time.sleep(0.01)  # 10ms
        
        stats = profiler.get_stats()
        assert "test_section_mean_ms" in stats
        assert stats["test_section_mean_ms"] >= 10.0  # At least 10ms
    
    def test_profiler_disabled(self):
        """Test profiler when disabled."""
        profiler = Profiler(enabled=False)
        
        with profiler.section("test_section"):
            pass
        
        stats = profiler.get_stats()
        assert len(stats) == 0
    
    def test_system_info(self):
        """Test system information collection."""
        info = get_system_info()
        
        assert 'cpu_count' in info
        assert 'memory_total_gb' in info
        assert 'python_version' in info
        assert 'platform' in info
        assert info['cpu_count'] > 0
        assert info['memory_total_gb'] > 0


class TestBenchmarkThresholds:
    """Test the threshold checking functionality."""
    
    def test_threshold_checking_success(self):
        """Test threshold checking with passing metrics."""
        # Create temporary metrics file
        metrics = {
            "metrics": {
                "overall": {"f1": 0.75},
                "vehicle_identification": {"f1": 0.80, "precision": 0.85, "recall": 0.75},
                "plate_recognition": {"f1": 0.70, "precision": 0.80},
                "counting": {"mae": 1.0, "smape": 15.0},
                "queue_length": {"mae": 1.0},
                "timing": {"mean_latency_ms": 100.0, "p95_latency_ms": 180.0, "fps": 12.0}
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metrics, f)
            metrics_path = Path(f.name)
        
        try:
            # Run threshold checker
            result = subprocess.run([
                sys.executable, "tools/assert_thresholds.py", 
                str(metrics_path), "--output-format", "json"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            assert result.returncode == 0  # Should pass
            
            output = json.loads(result.stdout)
            assert output["passed"] == True
            assert len(output["failures"]) == 0
            
        finally:
            metrics_path.unlink()
    
    def test_threshold_checking_failure(self):
        """Test threshold checking with failing metrics."""
        # Create metrics that fail thresholds
        metrics = {
            "metrics": {
                "overall": {"f1": 0.50},  # Below 0.65 threshold
                "vehicle_identification": {"f1": 0.60, "precision": 0.70, "recall": 0.50},  # Below 0.70 threshold
                "plate_recognition": {"f1": 0.40, "precision": 0.50},  # Below 0.60 threshold
                "counting": {"mae": 3.0, "smape": 25.0},  # Above 2.0 threshold
                "queue_length": {"mae": 2.0},  # Above 1.5 threshold
                "timing": {"mean_latency_ms": 200.0, "p95_latency_ms": 300.0, "fps": 3.0}  # Multiple failures
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(metrics, f)
            metrics_path = Path(f.name)
        
        try:
            # Run threshold checker
            result = subprocess.run([
                sys.executable, "tools/assert_thresholds.py", 
                str(metrics_path), "--output-format", "json"
            ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
            
            assert result.returncode == 1  # Should fail
            
            output = json.loads(result.stdout)
            assert output["passed"] == False
            assert len(output["failures"]) > 0
            
        finally:
            metrics_path.unlink()


@pytest.mark.integration
def test_benchmark_runner_mock():
    """Integration test for the benchmark runner with mock pipeline."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        
        # Run benchmark with mock pipeline
        result = subprocess.run([
            sys.executable, "tools/benchmark_e2e.py",
            "--config", "configs/benchmark/fast.yaml",
            "--videos", "configs/benchmark/eval_videos.yaml", 
            "--output", str(output_dir),
            "--verbose"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
        
        # Check that benchmark completed
        assert result.returncode == 0, f"Benchmark failed: {result.stderr}"
        
        # Check output files exist
        assert (output_dir / "metrics.json").exists()
        assert (output_dir / "test_video_1.pred.json").exists()
        
        # Validate metrics format
        with open(output_dir / "metrics.json") as f:
            metrics = json.load(f)
        
        assert "benchmark_info" in metrics
        assert "metrics" in metrics
        assert "overall" in metrics["metrics"]
        assert "f1" in metrics["metrics"]["overall"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 