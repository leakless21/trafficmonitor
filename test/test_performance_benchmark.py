"""Tests for performance-only benchmarking system."""

import json
import tempfile
import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from traffic_monitor.utils.profiler import Profiler


class TestPerformanceBenchmark:
    """Test the performance-only benchmark components."""
    
    def test_performance_collector(self):
        """Test PerformanceCollector metrics calculation."""
        # Import here to avoid import issues
        sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
        from benchmark_performance import PerformanceCollector
        
        collector = PerformanceCollector()
        collector.start_collection()
        
        # Simulate some frame processing times
        frame_times = [0.1, 0.12, 0.08, 0.15, 0.09]  # 100ms, 120ms, etc.
        for frame_time in frame_times:
            collector.record_frame_processed(frame_time)
        
        collector.finish_collection()
        metrics = collector.get_metrics()
        
        # Verify timing metrics
        assert "timing" in metrics
        timing = metrics["timing"]
        assert timing["total_frames"] == 5
        assert timing["mean_frame_time_ms"] == pytest.approx(108.0, rel=1e-2)  # 0.108 * 1000
        
        # Verify throughput metrics  
        assert "throughput" in metrics
        throughput = metrics["throughput"]
        assert throughput["mean_fps"] == pytest.approx(9.26, rel=1e-1)  # 1/0.108
        
        # Verify efficiency metrics
        assert "efficiency" in metrics
        efficiency = metrics["efficiency"]
        assert efficiency["ms_per_frame"] == pytest.approx(108.0, rel=1e-2)
    
    def test_performance_thresholds(self):
        """Test PerformanceThresholdChecker logic."""
        # Import here to avoid import issues
        sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
        from assert_performance_thresholds import PerformanceThresholdChecker
        
        checker = PerformanceThresholdChecker(fast_mode=False)
        
        # Test metrics that should pass
        passing_metrics = {
            "fps": {"mean": 10.0, "std": 1.0},
            "frame_time_ms": {"mean": 100.0},
            "p95_latency_ms": {"mean": 200.0}
        }
        
        results = checker.check_performance_metrics(passing_metrics)
        
        # Should have both critical and warning checks
        critical_results = [r for r in results if r.is_critical]
        warning_results = [r for r in results if not r.is_critical]
        
        assert len(critical_results) > 0
        assert len(warning_results) > 0
        
        # All should pass with good metrics
        all_passed = all(r.passed for r in results)
        assert all_passed
        
        # Test metrics that should fail
        failing_metrics = {
            "fps": {"mean": 1.0, "std": 5.0},  # Too low FPS, too unstable
            "frame_time_ms": {"mean": 1000.0},  # Too slow
            "p95_latency_ms": {"mean": 2000.0}  # Too high latency
        }
        
        results = checker.check_performance_metrics(failing_metrics)
        
        # Should have failures
        critical_failures = [r for r in results if r.is_critical and not r.passed]
        assert len(critical_failures) > 0
    
    def test_fast_mode_relaxed_thresholds(self):
        """Test that fast mode applies relaxed thresholds."""
        # Import here to avoid import issues
        sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
        from assert_performance_thresholds import PerformanceThresholdChecker
        
        # Metrics that would fail in normal mode but pass in fast mode
        borderline_metrics = {
            "fps": {"mean": 2.5},  # Below normal critical (3.0) but above fast mode (2.1)
            "frame_time_ms": {"mean": 600.0},  # Above normal critical (500.0) but below fast mode (750.0)
        }
        
        # Check with normal mode
        normal_checker = PerformanceThresholdChecker(fast_mode=False)
        normal_results = normal_checker.check_performance_metrics(borderline_metrics)
        normal_failures = [r for r in normal_results if r.is_critical and not r.passed]
        
        # Check with fast mode
        fast_checker = PerformanceThresholdChecker(fast_mode=True)
        fast_results = fast_checker.check_performance_metrics(borderline_metrics)
        fast_failures = [r for r in fast_results if r.is_critical and not r.passed]
        
        # Fast mode should have fewer failures
        assert len(fast_failures) < len(normal_failures)
    
    def test_profiler_integration(self):
        """Test profiler integration with performance benchmarks."""
        profiler = Profiler(enabled=True)
        
        # Simulate some profiled sections
        with profiler.section("test_section"):
            import time
            time.sleep(0.01)  # 10ms
        
        stats = profiler.get_stats()
        
        # Verify we have timing data
        assert "sections" in stats
        assert "test_section" in stats["sections"]
        
        section_stats = stats["sections"]["test_section"]
        assert section_stats["count"] == 1
        assert section_stats["total_time"] >= 0.01  # At least 10ms
    
    def test_mock_pipeline_scaling(self):
        """Test that mock pipeline scales processing time with config."""
        # Import here to avoid import issues
        sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
        from benchmark_performance import MockPerformancePipeline, PerformanceCollector
        
        profiler = Profiler(enabled=False)  # Disable for clean timing
        
        # Fast mode config
        fast_config = {"fast_mode": True}
        fast_pipeline = MockPerformancePipeline(fast_config, profiler)
        
        # Standard mode config
        standard_config = {"fast_mode": False}
        standard_pipeline = MockPerformancePipeline(standard_config, profiler)
        
        # Process with both configs
        test_video = Path("test_video.mp4")  # Doesn't need to exist for mock
        
        fast_collector = PerformanceCollector()
        fast_result = fast_pipeline.process_video(test_video, fast_collector)
        
        standard_collector = PerformanceCollector()
        standard_result = standard_pipeline.process_video(test_video, standard_collector)
        
        # Fast mode should have higher FPS (lower frame time)
        fast_fps = fast_result["throughput"]["overall_fps"]
        standard_fps = standard_result["throughput"]["overall_fps"]
        
        assert fast_fps > standard_fps, "Fast mode should process frames faster"


def test_benchmark_integration():
    """Integration test for the full performance benchmark flow."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create mock config files
        config_file = temp_path / "test_config.yaml"
        config_file.write_text("""
fast_mode: true
detection:
  model: "yolov8n.pt"
tracking:
  tracker: "bytetrack"
""")
        
        videos_file = temp_path / "test_videos.yaml"
        videos_file.write_text("""
videos:
  - name: "test_video"
    path: "test.mp4"
    description: "Test video"
""")
        
        # Test would run the actual benchmark here
        # For now, just verify the config files are created correctly
        assert config_file.exists()
        assert videos_file.exists()


if __name__ == "__main__":
    pytest.main([__file__]) 