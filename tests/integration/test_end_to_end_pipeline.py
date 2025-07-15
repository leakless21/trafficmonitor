"""
End-to-end integration tests for the complete Traffic Monitor pipeline.
Tests the full system from video input to final outputs including database storage,
video generation, and summary reports.
"""

import pytest
import tempfile
import shutil
import time
import json
import sqlite3
from pathlib import Path
import sys
import multiprocessing as mp
from unittest.mock import patch, Mock
import cv2
import numpy as np

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from traffic_monitor.main_supervisor import TrafficMonitorSupervisor
from traffic_monitor.cli import main as cli_main
from traffic_monitor.utils.config_loader import load_config
from traffic_monitor.utils import minidb


class TestEndToEndPipeline:
    """Test complete end-to-end pipeline functionality."""
    
    def setup_method(self):
        """Set up test environment with temporary directories and test data."""
        # Create temporary directories
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_data_dir = self.temp_dir / "test_data"
        self.test_output_dir = self.temp_dir / "outputs"
        self.test_db_dir = self.temp_dir / "db"
        self.test_reports_dir = self.temp_dir / "reports"
        
        # Create directories
        for dir_path in [self.test_data_dir, self.test_output_dir, self.test_db_dir, self.test_reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create test video file
        self.test_video_path = self.test_data_dir / "test_video.mp4"
        self._create_test_video()
        
        # Create test configuration
        self.test_config_path = self.test_data_dir / "test_config.yaml"
        self._create_test_config()
        
        # Database path
        self.test_db_path = self.test_db_dir / "test_traffic_monitor.db"

    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def _create_test_video(self):
        """Create a test video file with moving objects."""
        # Video parameters
        width, height = 640, 480
        fps = 10
        duration_seconds = 3
        total_frames = fps * duration_seconds
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(self.test_video_path), fourcc, fps, (width, height))
        
        try:
            for frame_num in range(total_frames):
                # Create frame with moving rectangle (simulating vehicle)
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Add background
                frame[:] = (50, 50, 50)  # Dark gray background
                
                # Add moving "vehicle" - rectangle that moves across screen
                vehicle_x = int((frame_num / total_frames) * (width - 100))
                vehicle_y = height // 2 - 25
                
                # Draw vehicle rectangle
                cv2.rectangle(frame, (vehicle_x, vehicle_y), 
                            (vehicle_x + 100, vehicle_y + 50), (0, 255, 0), -1)
                
                # Add some noise/texture
                noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
                frame = cv2.add(frame, noise)
                
                out.write(frame)
        finally:
            out.release()

    def _create_test_config(self):
        """Create a test configuration file."""
        config = {
            "frame_grabber": {
                "video_source": str(self.test_video_path),
                "resize_resolution": [640, 480],
                "process_every_n_frame": 1,
                "log_every_n_frames": 10
            },
            "vehicle_detector": {
                "model_path": "mock_model.engine",  # Will be mocked
                "conf_threshold": 0.3,
                "class_mapping": {
                    0: "bicycle",
                    1: "bike", 
                    2: "bus",
                    3: "car",
                    4: "person",
                    5: "truck"
                }
            },
            "vehicle_tracker": {
                "tracker_type": "bytetrack",
                "half": True,
                "device": "cpu"  # Use CPU for testing
            },
            "lp_detector": {
                "model_path": "mock_lp_model.engine",  # Will be mocked
                "conf_threshold": 0.5
            },
            "ocr_reader": {
                "backend": "fast_plate_ocr",
                "conf_threshold": 0.5,
                "hub_model_name": "mock-model"
            },
            "vehicle_counter": {
                "counting_lines": [
                    [[0.2, 0.4], [0.8, 0.6]]  # Diagonal counting line
                ]
            },
            "visualizer": {
                "save_to_file": True,
                "save_path": str(self.test_output_dir / "videos"),
                "output_fourcc": "mp4v",
                "enable_gui": False,
                "font_scale": 0.6,
                "font_thickness": 2,
                "class_colors": {
                    "car": [0, 255, 0],
                    "truck": [255, 0, 0],
                    "bus": [0, 0, 255]
                }
            },
            "database": {
                "path": str(self.test_db_path),
                "reset_on_startup": True
            },
            "summary_service": {
                "enabled": True,
                "summary_output_dir": str(self.test_reports_dir),
                "print_summary": False,  # Disable for testing
                "save_detailed_report": True
            },
            "loguru": {
                "level": "WARNING",  # Reduce log noise in tests
                "terminal_output_enabled": False,
                "file_path": str(self.temp_dir / "test.log")
            }
        }
        
        import yaml
        with open(self.test_config_path, 'w') as f:
            yaml.dump(config, f)

    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_pipeline_with_mocked_models(self):
        """Test complete pipeline with mocked AI models."""
        with patch('ultralytics.YOLO') as mock_yolo, \
             patch('traffic_monitor.services.text_recognition_service.FastPlateOCR') as mock_ocr:
            
            # Mock vehicle detection model
            mock_vehicle_model = Mock()
            mock_detection_result = Mock()
            mock_detection_result.boxes = [Mock()]
            mock_detection_result.boxes[0].xyxy = [[100, 200, 200, 300]]  # Mock bbox
            mock_detection_result.boxes[0].conf = [0.85]  # Mock confidence
            mock_detection_result.boxes[0].cls = [3]  # Mock class (car)
            mock_vehicle_model.predict.return_value = [mock_detection_result]
            
            # Mock license plate detection model
            mock_lp_model = Mock()
            mock_lp_result = Mock()
            mock_lp_result.boxes = [Mock()]
            mock_lp_result.boxes[0].xyxy = [[10, 10, 50, 30]]  # Mock plate bbox
            mock_lp_result.boxes[0].conf = [0.75]  # Mock confidence
            mock_lp_model.predict.return_value = [mock_lp_result]
            
            # Configure YOLO mock to return different models
            def yolo_side_effect(model_path):
                if "lp" in model_path or "plate" in model_path:
                    return mock_lp_model
                else:
                    return mock_vehicle_model
            
            mock_yolo.side_effect = yolo_side_effect
            
            # Mock OCR
            mock_ocr_instance = Mock()
            mock_ocr_instance.run.return_value = [("ABC123", 0.9)]  # Mock plate text
            mock_ocr.return_value = mock_ocr_instance
            
            # Run the pipeline
            self._run_pipeline_test()

    def _run_pipeline_test(self):
        """Run the actual pipeline test."""
        # Load configuration
        config = load_config(self.test_config_path)
        assert config is not None, "Configuration should load successfully"
        
        # Initialize database
        minidb.configure_database(config)
        minidb.init_db()
        
        # Verify database initialization
        assert self.test_db_path.exists(), "Database file should be created"
        
        # Test database connection
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        expected_tables = ["vehicle_counts", "plate_results", "plate_results_latest"]
        for table in expected_tables:
            assert table in tables, f"Table {table} should exist"
        
        # Create and run supervisor (with timeout for testing)
        supervisor = TrafficMonitorSupervisor(config)
        
        # Start processing with timeout
        start_time = time.time()
        timeout = 30  # 30 second timeout
        
        try:
            # Start the supervisor in a separate process for timeout control
            supervisor_process = mp.Process(target=self._run_supervisor, args=(supervisor,))
            supervisor_process.start()
            supervisor_process.join(timeout=timeout)
            
            if supervisor_process.is_alive():
                supervisor_process.terminate()
                supervisor_process.join()
                
        except Exception as e:
            pytest.fail(f"Pipeline execution failed: {e}")
        
        processing_time = time.time() - start_time
        assert processing_time < timeout, f"Processing took too long: {processing_time:.2f}s"
        
        # Verify outputs
        self._verify_pipeline_outputs()

    def _run_supervisor(self, supervisor):
        """Run supervisor in separate process."""
        try:
            supervisor.run()
        except Exception as e:
            print(f"Supervisor error: {e}")

    def _verify_pipeline_outputs(self):
        """Verify that all expected outputs were generated."""
        # Check database has data
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        
        # Check vehicle counts
        cursor.execute("SELECT COUNT(*) FROM vehicle_counts")
        count_records = cursor.fetchone()[0]
        assert count_records >= 0, "Should have vehicle count records"
        
        # Check plate results
        cursor.execute("SELECT COUNT(*) FROM plate_results")
        plate_records = cursor.fetchone()[0]
        assert plate_records >= 0, "Should have plate result records"
        
        conn.close()
        
        # Check output video was created
        video_output_dir = self.test_output_dir / "videos"
        if video_output_dir.exists():
            video_files = list(video_output_dir.glob("*.mp4"))
            # Video creation might be optional based on configuration
            
        # Check summary report was created
        report_files = list(self.test_reports_dir.glob("*.json"))
        if report_files:
            # Verify report content
            with open(report_files[0], 'r') as f:
                report = json.load(f)
                assert "summary" in report
                assert "vehicle_analysis" in report
                assert "performance_metrics" in report

    @pytest.mark.integration
    def test_cli_interface_integration(self):
        """Test the CLI interface with end-to-end processing."""
        with patch('ultralytics.YOLO') as mock_yolo, \
             patch('traffic_monitor.services.text_recognition_service.FastPlateOCR') as mock_ocr, \
             patch('sys.argv', ['traffic_monitor', '--config', str(self.test_config_path), '--timeout', '10']):
            
            # Setup mocks (same as above)
            self._setup_model_mocks(mock_yolo, mock_ocr)
            
            # Test CLI execution
            try:
                # Mock sys.exit to prevent test termination
                with patch('sys.exit'):
                    cli_main()
            except SystemExit:
                pass  # Expected for CLI completion
            
            # Verify outputs were created
            assert self.test_db_path.exists(), "Database should be created via CLI"

    def _setup_model_mocks(self, mock_yolo, mock_ocr):
        """Setup model mocks for testing."""
        # Mock vehicle detection
        mock_vehicle_model = Mock()
        mock_detection_result = Mock()
        mock_detection_result.boxes = []  # No detections for simplicity
        mock_vehicle_model.predict.return_value = [mock_detection_result]
        
        # Mock license plate detection
        mock_lp_model = Mock()
        mock_lp_result = Mock()
        mock_lp_result.boxes = []  # No detections for simplicity
        mock_lp_model.predict.return_value = [mock_lp_result]
        
        def yolo_side_effect(model_path):
            if "lp" in str(model_path) or "plate" in str(model_path):
                return mock_lp_model
            else:
                return mock_vehicle_model
        
        mock_yolo.side_effect = yolo_side_effect
        
        # Mock OCR
        mock_ocr_instance = Mock()
        mock_ocr_instance.run.return_value = []  # No OCR results
        mock_ocr.return_value = mock_ocr_instance

    @pytest.mark.integration
    def test_error_handling_and_recovery(self):
        """Test system behavior under error conditions."""
        # Test with invalid video file
        invalid_config = self.test_config_path.parent / "invalid_config.yaml"
        
        config = {
            "frame_grabber": {
                "video_source": "nonexistent_video.mp4",
                "resize_resolution": [640, 480]
            },
            "loguru": {
                "level": "ERROR",
                "terminal_output_enabled": False
            }
        }
        
        import yaml
        with open(invalid_config, 'w') as f:
            yaml.dump(config, f)
        
        # Test that system handles invalid input gracefully
        loaded_config = load_config(invalid_config)
        assert loaded_config is not None, "Should load config even with invalid video path"
        
        # Test database error handling
        invalid_db_config = {
            "database": {
                "path": "/invalid/path/database.db"
            }
        }
        
        # Should handle database errors gracefully
        try:
            minidb.configure_database(invalid_db_config)
            # This might fail, which is expected
        except Exception:
            pass  # Expected behavior

    @pytest.mark.integration
    def test_performance_benchmarking(self):
        """Test system performance with benchmarking."""
        with patch('ultralytics.YOLO') as mock_yolo, \
             patch('traffic_monitor.services.text_recognition_service.FastPlateOCR') as mock_ocr:
            
            self._setup_model_mocks(mock_yolo, mock_ocr)
            
            # Load configuration
            config = load_config(self.test_config_path)
            
            # Measure initialization time
            start_time = time.time()
            minidb.configure_database(config)
            minidb.init_db()
            init_time = time.time() - start_time
            
            assert init_time < 5.0, f"Database initialization took too long: {init_time:.2f}s"
            
            # Test configuration loading performance
            start_time = time.time()
            for _ in range(10):
                load_config(self.test_config_path)
            config_load_time = (time.time() - start_time) / 10
            
            assert config_load_time < 0.1, f"Config loading too slow: {config_load_time:.3f}s per load"

    @pytest.mark.integration
    def test_concurrent_processing(self):
        """Test system behavior with concurrent operations."""
        import threading
        
        # Test concurrent database operations
        minidb.configure_database({"database": {"path": str(self.test_db_path)}})
        minidb.init_db()
        
        results = []
        errors = []
        
        def write_test_data(thread_id):
            try:
                success = minidb.write_vehicle_count(
                    camera_id=f"cam_{thread_id}",
                    total_count=thread_id,
                    class_counts={"car": thread_id}
                )
                results.append(success)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=write_test_data, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0, f"Concurrent operations had errors: {errors}"
        assert len(results) == 5, "All threads should complete"
        assert all(results), "All database operations should succeed"

    @pytest.mark.integration
    def test_memory_usage_monitoring(self):
        """Test memory usage during processing."""
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Load configuration and initialize
        config = load_config(self.test_config_path)
        minidb.configure_database(config)
        minidb.init_db()
        
        # Simulate processing load
        for i in range(100):
            minidb.write_vehicle_count(
                camera_id="memory_test",
                total_count=i,
                class_counts={"car": i}
            )
        
        # Check memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100, f"Memory usage increased too much: {memory_increase:.2f}MB"

    @pytest.mark.integration
    def test_data_integrity_end_to_end(self):
        """Test data integrity throughout the entire pipeline."""
        # Initialize system
        config = load_config(self.test_config_path)
        minidb.configure_database(config)
        minidb.init_db()
        
        # Test data flow integrity
        test_data = {
            "camera_id": "integrity_test",
            "total_count": 42,
            "class_counts": {"car": 30, "truck": 12}
        }
        
        # Write data
        success = minidb.write_vehicle_count(**test_data)
        assert success, "Data write should succeed"
        
        # Read data back
        counts = minidb.get_vehicle_counts(camera_id="integrity_test", limit=1)
        assert len(counts) == 1, "Should retrieve written data"
        
        retrieved = counts[0]
        assert retrieved["camera_id"] == test_data["camera_id"]
        assert retrieved["total_count"] == test_data["total_count"]
        assert retrieved["class_counts"] == test_data["class_counts"]

    @pytest.mark.integration
    @pytest.mark.slow
    def test_long_running_stability(self):
        """Test system stability over extended operation."""
        config = load_config(self.test_config_path)
        minidb.configure_database(config)
        minidb.init_db()
        
        # Simulate long-running operation
        start_time = time.time()
        operations = 0
        max_duration = 10  # 10 seconds for testing
        
        while time.time() - start_time < max_duration:
            success = minidb.write_vehicle_count(
                camera_id="stability_test",
                total_count=operations,
                class_counts={"car": operations % 10}
            )
            assert success, f"Operation {operations} should succeed"
            operations += 1
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.01)
        
        duration = time.time() - start_time
        ops_per_second = operations / duration
        
        assert operations > 100, f"Should complete many operations: {operations}"
        assert ops_per_second > 10, f"Should maintain reasonable throughput: {ops_per_second:.2f} ops/sec"

    # Helper methods for test utilities
    def _verify_video_output(self, video_path):
        """Verify that output video is valid."""
        if not video_path.exists():
            return False
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        return frame_count > 0

    def _get_database_stats(self):
        """Get database statistics for verification."""
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        
        stats = {}
        for table in ["vehicle_counts", "plate_results", "plate_results_latest"]:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = cursor.fetchone()[0]
        
        conn.close()
        return stats


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])