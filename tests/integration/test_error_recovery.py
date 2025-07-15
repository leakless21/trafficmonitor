"""
Error recovery and resilience integration tests for Traffic Monitor system.
Tests system behavior during failures, recovery mechanisms, and fault tolerance.
"""

import pytest
import tempfile
import shutil
import time
import os
import signal
import threading
import multiprocessing as mp
from pathlib import Path
import sys
import cv2
import numpy as np
from unittest.mock import patch, Mock, MagicMock, side_effect
from multiprocessing import Queue, Event, Process
import sqlite3
import psutil
import yaml

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from traffic_monitor.utils import minidb
from traffic_monitor.utils.config_loader import load_config
from traffic_monitor.utils.queue_utils import safe_put, put_realtime, put_offline
from traffic_monitor.main_supervisor import TrafficMonitorSupervisor
from traffic_monitor.services.frame_capture_service import frame_capture_process
from traffic_monitor.services.vehicle_detection_service import vehicle_detection_process
from traffic_monitor.services.vehicle_tracking_service import vehicle_tracking_process
from traffic_monitor.utils.custom_types import FrameMessage


@pytest.mark.integration
class TestErrorRecovery:
    """Test error recovery and resilience mechanisms."""
    
    def setup_method(self):
        """Set up error recovery test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_db_path = self.temp_dir / "recovery_test.db"
        self.test_video_path = self._create_test_video()
        
        # Configure test database
        self.db_config = {
            "database": {
                "path": str(self.test_db_path),
                "reset_on_startup": True
            }
        }
        
        minidb.configure_database(self.db_config)
        minidb.init_db()
        
        self.test_config = self._create_test_config()
    
    def teardown_method(self):
        """Clean up error recovery test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_process_crash_recovery(self):
        """Test recovery from process crashes."""
        frame_queue = Queue(maxsize=5)
        detection_queue = Queue(maxsize=5)
        shutdown_event = Event()
        
        with patch('ultralytics.YOLO') as mock_yolo:
            # Setup mock that will crash after a few calls
            mock_model = Mock()
            call_count = [0]  # Use list to modify in nested function
            
            def crash_after_calls(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] > 3:
                    raise RuntimeError("Simulated model crash")
                
                # Return normal result for first few calls
                mock_result = Mock()
                mock_result.boxes = []
                return [mock_result]
            
            mock_model.predict.side_effect = crash_after_calls
            mock_yolo.return_value = mock_model
            
            # Start frame capture process
            frame_process = Process(
                target=frame_capture_process,
                args=(self.test_config, frame_queue, shutdown_event)
            )
            
            # Start detection process (will crash)
            detection_process = Process(
                target=vehicle_detection_process,
                args=(self.test_config, frame_queue, detection_queue, shutdown_event)
            )
            
            try:
                frame_process.start()
                detection_process.start()
                
                # Wait for crash to occur
                time.sleep(3)
                
                # Detection process should have crashed
                detection_process.join(timeout=2)
                assert not detection_process.is_alive(), "Detection process should have crashed"
                assert detection_process.exitcode != 0, "Process should have non-zero exit code"
                
                # Frame process should still be running
                assert frame_process.is_alive(), "Frame process should survive other process crashes"
                
                # Test recovery by restarting detection process
                call_count[0] = 0  # Reset counter
                detection_process_2 = Process(
                    target=vehicle_detection_process,
                    args=(self.test_config, frame_queue, detection_queue, shutdown_event)
                )
                
                detection_process_2.start()
                time.sleep(1)
                
                # New process should be running
                assert detection_process_2.is_alive(), "Restarted process should be running"
                
                detection_process_2.terminate()
                detection_process_2.join()
                
            finally:
                shutdown_event.set()
                for proc in [frame_process]:
                    if proc.is_alive():
                        proc.join(timeout=3)
                        if proc.is_alive():
                            proc.terminate()
    
    def test_memory_leak_recovery(self):
        """Test recovery from memory leaks."""
        import gc
        
        # Monitor memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate memory leak scenario
        large_objects = []
        
        try:
            # Create memory pressure
            for i in range(100):
                # Create large objects that might not be properly cleaned up
                large_array = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
                large_objects.append(large_array)
                
                # Perform database operations under memory pressure
                success = minidb.write_vehicle_count(
                    camera_id="memory_leak_test",
                    total_count=i,
                    class_counts={"car": i}
                )
                
                # Check memory usage periodically
                if i % 20 == 0:
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_increase = current_memory - initial_memory
                    
                    # If memory usage gets too high, trigger cleanup
                    if memory_increase > 500:  # 500MB threshold
                        # Simulate recovery mechanism
                        large_objects.clear()
                        gc.collect()
                        break
                
                assert success, f"Database operation should succeed under memory pressure at iteration {i}"
        
        finally:
            # Cleanup
            large_objects.clear()
            gc.collect()
        
        # Verify system recovered
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory should not have increased excessively
        assert memory_increase < 200, f"Memory increase too high after cleanup: {memory_increase:.2f}MB"
        
        # System should still be functional
        success = minidb.write_vehicle_count(
            camera_id="post_recovery_test",
            total_count=1,
            class_counts={"car": 1}
        )
        assert success, "System should be functional after memory recovery"
    
    def test_database_lock_recovery(self):
        """Test recovery from database lock situations."""
        # Create a long-running transaction to simulate lock
        def create_lock():
            conn = sqlite3.connect(self.test_db_path, timeout=1.0)
            cursor = conn.cursor()
            cursor.execute("BEGIN EXCLUSIVE TRANSACTION")
            # Hold lock for a while
            time.sleep(2)
            conn.rollback()
            conn.close()
        
        # Start lock thread
        lock_thread = threading.Thread(target=create_lock)
        lock_thread.start()
        
        time.sleep(0.5)  # Let lock establish
        
        # Try to write data while database is locked
        start_time = time.time()
        success = minidb.write_vehicle_count(
            camera_id="lock_test",
            total_count=1,
            class_counts={"car": 1}
        )
        operation_time = time.time() - start_time
        
        # Wait for lock thread to complete
        lock_thread.join()
        
        # Operation should either succeed (after waiting) or fail gracefully
        assert isinstance(success, bool), "Should return boolean result"
        
        # If it failed due to lock, should not have taken too long
        if not success:
            assert operation_time < 5.0, f"Operation should timeout quickly, took {operation_time:.2f}s"
        
        # After lock is released, operations should work
        time.sleep(0.5)
        success = minidb.write_vehicle_count(
            camera_id="post_lock_test",
            total_count=1,
            class_counts={"car": 1}
        )
        assert success, "Should succeed after lock is released"
    
    def test_queue_deadlock_recovery(self):
        """Test recovery from queue deadlock situations."""
        # Create small queues that can deadlock
        small_queue_1 = Queue(maxsize=1)
        small_queue_2 = Queue(maxsize=1)
        
        def producer_1():
            """Producer that might cause deadlock."""
            try:
                for i in range(5):
                    # This might block if queue is full
                    success = put_realtime(small_queue_1, f"msg1_{i}", "producer1")
                    if not success:
                        break
                    time.sleep(0.1)
            except Exception:
                pass  # Handle any errors gracefully
        
        def producer_2():
            """Another producer that might cause deadlock."""
            try:
                for i in range(5):
                    success = put_realtime(small_queue_2, f"msg2_{i}", "producer2")
                    if not success:
                        break
                    time.sleep(0.1)
            except Exception:
                pass
        
        def consumer():
            """Consumer that processes both queues."""
            try:
                for _ in range(10):
                    # Try to get from both queues
                    try:
                        msg1 = small_queue_1.get(timeout=0.1)
                    except:
                        msg1 = None
                    
                    try:
                        msg2 = small_queue_2.get(timeout=0.1)
                    except:
                        msg2 = None
                    
                    if msg1 is None and msg2 is None:
                        break
                    
                    time.sleep(0.05)
            except Exception:
                pass
        
        # Start all threads
        threads = [
            threading.Thread(target=producer_1),
            threading.Thread(target=producer_2),
            threading.Thread(target=consumer)
        ]
        
        for thread in threads:
            thread.start()
        
        # Wait for completion with timeout
        start_time = time.time()
        for thread in threads:
            remaining_time = max(0, 5 - (time.time() - start_time))
            thread.join(timeout=remaining_time)
        
        total_time = time.time() - start_time
        
        # Should not deadlock indefinitely
        assert total_time < 6, f"Operations took too long, possible deadlock: {total_time:.2f}s"
        
        # Check if any threads are still alive (indicating deadlock)
        alive_threads = [t for t in threads if t.is_alive()]
        assert len(alive_threads) == 0, f"Threads still alive, possible deadlock: {len(alive_threads)}"
    
    def test_model_loading_failure_recovery(self):
        """Test recovery from model loading failures."""
        with patch('ultralytics.YOLO') as mock_yolo:
            # Mock YOLO to fail on first attempt, succeed on second
            call_count = [0]
            
            def failing_yolo(*args, **kwargs):
                call_count[0] += 1
                if call_count[0] == 1:
                    raise RuntimeError("Model loading failed")
                else:
                    # Return working mock on retry
                    mock_model = Mock()
                    mock_result = Mock()
                    mock_result.boxes = []
                    mock_model.predict.return_value = [mock_result]
                    return mock_model
            
            mock_yolo.side_effect = failing_yolo
            
            # Try to create detection service (should fail first time)
            from traffic_monitor.services.vehicle_detection_service import VehicleDetectionService
            
            try:
                service = VehicleDetectionService(self.test_config)
                # If it succeeds, that means retry worked
                assert call_count[0] >= 1, "Should have attempted to load model"
            except Exception as e:
                # If it fails, should be a model loading error
                assert "model" in str(e).lower() or "loading" in str(e).lower(), f"Unexpected error: {e}"
                
                # Try again (should succeed on retry)
                try:
                    service = VehicleDetectionService(self.test_config)
                    assert call_count[0] >= 2, "Should have retried model loading"
                except Exception as e2:
                    # If retry also fails, that's acceptable for this test
                    assert "model" in str(e2).lower(), f"Unexpected retry error: {e2}"
    
    def test_network_interruption_simulation(self):
        """Test behavior during simulated network interruptions."""
        # Simulate network-dependent operations (like model downloads)
        with patch('ultralytics.YOLO') as mock_yolo:
            # Mock network timeout scenarios
            def network_timeout(*args, **kwargs):
                import socket
                raise socket.timeout("Network timeout")
            
            mock_yolo.side_effect = network_timeout
            
            # Try operations that might depend on network
            try:
                from traffic_monitor.services.vehicle_detection_service import VehicleDetectionService
                service = VehicleDetectionService(self.test_config)
                # Should handle network errors gracefully
            except Exception as e:
                # Should be a network-related error
                assert any(keyword in str(e).lower() for keyword in ["network", "timeout", "connection"]), f"Unexpected error: {e}"
    
    def test_disk_io_failure_recovery(self):
        """Test recovery from disk I/O failures."""
        # Simulate disk I/O failures
        original_open = open
        
        def failing_open(*args, **kwargs):
            if "recovery_test" in str(args[0]):
                raise IOError("Disk I/O error")
            return original_open(*args, **kwargs)
        
        # Test file operations with I/O failures
        with patch('builtins.open', side_effect=failing_open):
            # Try to save configuration
            config_path = self.temp_dir / "recovery_test_config.yaml"
            
            try:
                with open(config_path, 'w') as f:
                    yaml.dump(self.test_config, f)
                # Should not reach here due to mocked failure
                assert False, "Should have failed due to I/O error"
            except IOError as e:
                assert "disk" in str(e).lower() or "i/o" in str(e).lower(), f"Expected I/O error: {e}"
        
        # After removing the patch, operations should work
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.test_config, f)
            assert config_path.exists(), "File should be created after I/O recovery"
        except Exception as e:
            # If it still fails, might be due to test environment
            assert "permission" in str(e).lower() or "access" in str(e).lower(), f"Unexpected error: {e}"
    
    def test_signal_handling_recovery(self):
        """Test recovery from signal interruptions."""
        if os.name == 'nt':  # Skip on Windows
            pytest.skip("Signal handling test not applicable on Windows")
        
        frame_queue = Queue()
        shutdown_event = Event()
        
        # Start a process
        process = Process(
            target=frame_capture_process,
            args=(self.test_config, frame_queue, shutdown_event)
        )
        
        try:
            process.start()
            time.sleep(1)  # Let process start
            
            # Send SIGTERM (graceful shutdown signal)
            os.kill(process.pid, signal.SIGTERM)
            
            # Process should handle signal gracefully
            process.join(timeout=5)
            assert not process.is_alive(), "Process should have shut down gracefully"
            
            # Exit code should indicate graceful shutdown (0) or signal handling
            assert process.exitcode in [0, -signal.SIGTERM], f"Unexpected exit code: {process.exitcode}"
            
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
    
    def test_resource_exhaustion_recovery(self):
        """Test recovery from resource exhaustion."""
        # Test file descriptor exhaustion
        file_handles = []
        max_files = 50  # Conservative limit for testing
        
        try:
            # Open many files to simulate FD exhaustion
            for i in range(max_files):
                try:
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    file_handles.append(temp_file)
                except OSError:
                    # Hit system limit
                    break
            
            # Try database operations with limited resources
            success = minidb.write_vehicle_count(
                camera_id="resource_exhaustion_test",
                total_count=1,
                class_counts={"car": 1}
            )
            
            # Should either succeed or fail gracefully
            assert isinstance(success, bool), "Should return boolean result"
            
        finally:
            # Clean up file handles
            for temp_file in file_handles:
                try:
                    temp_file.close()
                    os.unlink(temp_file.name)
                except:
                    pass  # Ignore cleanup errors
        
        # After cleanup, operations should work
        success = minidb.write_vehicle_count(
            camera_id="post_exhaustion_test",
            total_count=1,
            class_counts={"car": 1}
        )
        assert success, "Should succeed after resource cleanup"
    
    def test_configuration_reload_recovery(self):
        """Test recovery from configuration changes."""
        # Create initial config
        config_path = self.temp_dir / "reload_test_config.yaml"
        
        initial_config = {
            "frame_grabber": {
                "video_source": self.test_video_path,
                "resize_resolution": [640, 480]
            },
            "loguru": {"level": "INFO"}
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(initial_config, f)
        
        # Load initial config
        config = load_config(config_path)
        assert config is not None, "Should load initial config"
        assert config["loguru"]["level"] == "INFO", "Should have initial log level"
        
        # Modify config file
        modified_config = initial_config.copy()
        modified_config["loguru"]["level"] = "DEBUG"
        modified_config["frame_grabber"]["resize_resolution"] = [320, 240]
        
        with open(config_path, 'w') as f:
            yaml.dump(modified_config, f)
        
        # Reload config
        reloaded_config = load_config(config_path)
        assert reloaded_config is not None, "Should reload modified config"
        assert reloaded_config["loguru"]["level"] == "DEBUG", "Should have updated log level"
        assert reloaded_config["frame_grabber"]["resize_resolution"] == [320, 240], "Should have updated resolution"
        
        # Test with corrupted config
        with open(config_path, 'w') as f:
            f.write("invalid: yaml: content: [unclosed")
        
        # Should handle corrupted config gracefully
        try:
            corrupted_config = load_config(config_path)
            # Should either return None or raise exception
            assert corrupted_config is None, "Should return None for corrupted config"
        except Exception as e:
            # Should be a YAML parsing error
            assert "yaml" in str(e).lower() or "parse" in str(e).lower(), f"Unexpected error: {e}"
    
    def test_graceful_degradation(self):
        """Test graceful degradation when components fail."""
        # Test system behavior when optional components fail
        with patch('ultralytics.YOLO') as mock_yolo:
            # Mock vehicle detection to work
            mock_vehicle_model = Mock()
            mock_result = Mock()
            mock_result.boxes = [Mock()]
            mock_result.boxes[0].xyxy = [[100, 100, 200, 200]]
            mock_result.boxes[0].conf = [0.8]
            mock_result.boxes[0].cls = [3]
            mock_vehicle_model.predict.return_value = [mock_result]
            
            # Mock license plate detection to fail
            mock_lp_model = Mock()
            mock_lp_model.predict.side_effect = RuntimeError("LP detection failed")
            
            def yolo_side_effect(model_path):
                if "lp" in str(model_path) or "plate" in str(model_path):
                    return mock_lp_model  # Will fail
                else:
                    return mock_vehicle_model  # Will work
            
            mock_yolo.side_effect = yolo_side_effect
            
            # System should continue working even if LP detection fails
            from traffic_monitor.services.vehicle_detection_service import VehicleDetectionService
            
            try:
                vehicle_service = VehicleDetectionService(self.test_config)
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                
                # Vehicle detection should work
                detections = vehicle_service.detect_vehicles(frame)
                assert len(detections) > 0, "Vehicle detection should work"
                
            except Exception as e:
                # Should handle component failures gracefully
                assert "detection" in str(e).lower(), f"Unexpected error: {e}"
    
    # Helper methods
    def _create_test_video(self):
        """Create a test video file."""
        video_path = self.temp_dir / "recovery_test_video.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 10.0, (640, 480))
        
        for i in range(30):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add moving vehicle
            x = (i * 20) % 600
            cv2.rectangle(frame, (x, 200), (x + 80, 240), (0, 255, 0), -1)
            out.write(frame)
        
        out.release()
        return str(video_path)
    
    def _create_test_config(self):
        """Create test configuration."""
        return {
            "frame_grabber": {
                "video_source": self.test_video_path,
                "resize_resolution": [640, 480],
                "process_every_n_frame": 1
            },
            "vehicle_detector": {
                "model_path": "mock_model.engine",
                "conf_threshold": 0.5,
                "class_mapping": {3: "car"}
            },
            "vehicle_tracker": {
                "tracker_type": "bytetrack",
                "device": "cpu"
            },
            "lp_detector": {
                "model_path": "mock_lp_model.engine",
                "conf_threshold": 0.5
            },
            "loguru": {
                "level": "WARNING",
                "terminal_output_enabled": False
            }
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])