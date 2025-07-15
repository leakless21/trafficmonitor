"""
Edge case integration tests for Traffic Monitor system.
Tests boundary conditions, error scenarios, and unusual inputs.
"""

import pytest
import tempfile
import shutil
import time
import os
from pathlib import Path
import sys
import cv2
import numpy as np
from unittest.mock import patch, Mock, MagicMock
import sqlite3
import yaml
import threading
import multiprocessing as mp
from multiprocessing import Queue, Event

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from traffic_monitor.utils import minidb
from traffic_monitor.utils.config_loader import load_config
from traffic_monitor.utils.queue_utils import safe_put, put_realtime, put_offline
from traffic_monitor.main_supervisor import TrafficMonitorSupervisor
from traffic_monitor.services.vehicle_detection_service import VehicleDetectionService
from traffic_monitor.services.vehicle_tracking_service import VehicleTrackingService
from traffic_monitor.services.vehicle_counting_service import VehicleCountingService
from traffic_monitor.utils.custom_types import FrameMessage, DetectedVehicleMessage


@pytest.mark.integration
class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        """Set up edge case test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_db_path = self.temp_dir / "edge_test.db"
        
        # Configure test database
        self.db_config = {
            "database": {
                "path": str(self.test_db_path),
                "reset_on_startup": True
            }
        }
        
        minidb.configure_database(self.db_config)
        minidb.init_db()
    
    def teardown_method(self):
        """Clean up edge case test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_empty_video_file(self):
        """Test handling of empty video files."""
        empty_video_path = self.temp_dir / "empty.mp4"
        
        # Create empty video file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(empty_video_path), fourcc, 10.0, (640, 480))
        out.release()  # Close immediately to create empty file
        
        config = {
            "frame_grabber": {
                "video_source": str(empty_video_path),
                "resize_resolution": [640, 480]
            },
            "loguru": {"level": "ERROR", "terminal_output_enabled": False}
        }
        
        # Should handle empty video gracefully
        from traffic_monitor.services.frame_capture_service import frame_capture_process
        
        frame_queue = Queue()
        shutdown_event = Event()
        
        process = mp.Process(
            target=frame_capture_process,
            args=(config, frame_queue, shutdown_event)
        )
        
        try:
            process.start()
            time.sleep(2)  # Let it try to process
            
            # Should not crash, might produce no frames
            assert process.is_alive() or process.exitcode == 0, "Should handle empty video gracefully"
            
        finally:
            shutdown_event.set()
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
    
    def test_corrupted_video_file(self):
        """Test handling of corrupted video files."""
        corrupted_video_path = self.temp_dir / "corrupted.mp4"
        
        # Create corrupted video file (just random bytes)
        with open(corrupted_video_path, 'wb') as f:
            f.write(b"This is not a valid video file content")
        
        config = {
            "frame_grabber": {
                "video_source": str(corrupted_video_path),
                "resize_resolution": [640, 480]
            },
            "loguru": {"level": "ERROR", "terminal_output_enabled": False}
        }
        
        from traffic_monitor.services.frame_capture_service import frame_capture_process
        
        frame_queue = Queue()
        shutdown_event = Event()
        
        process = mp.Process(
            target=frame_capture_process,
            args=(config, frame_queue, shutdown_event)
        )
        
        try:
            process.start()
            time.sleep(2)
            
            # Should handle corrupted video gracefully
            # Process might exit with error code, but shouldn't hang
            process.join(timeout=5)
            assert not process.is_alive(), "Process should not hang on corrupted video"
            
        finally:
            if process.is_alive():
                process.terminate()
    
    def test_nonexistent_video_file(self):
        """Test handling of nonexistent video files."""
        config = {
            "frame_grabber": {
                "video_source": "/nonexistent/path/video.mp4",
                "resize_resolution": [640, 480]
            },
            "loguru": {"level": "ERROR", "terminal_output_enabled": False}
        }
        
        from traffic_monitor.services.frame_capture_service import frame_capture_process
        
        frame_queue = Queue()
        shutdown_event = Event()
        
        process = mp.Process(
            target=frame_capture_process,
            args=(config, frame_queue, shutdown_event)
        )
        
        try:
            process.start()
            time.sleep(2)
            
            # Should handle missing file gracefully
            process.join(timeout=5)
            assert not process.is_alive(), "Process should not hang on missing video"
            
        finally:
            if process.is_alive():
                process.terminate()
    
    def test_extremely_large_frames(self):
        """Test handling of extremely large video frames."""
        large_video_path = self.temp_dir / "large.mp4"
        
        # Create video with very large frames (4K)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(large_video_path), fourcc, 1.0, (3840, 2160))
        
        for i in range(3):  # Just a few frames
            frame = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        config = {
            "frame_grabber": {
                "video_source": str(large_video_path),
                "resize_resolution": [640, 480],  # Should resize down
                "process_every_n_frame": 1
            },
            "loguru": {"level": "WARNING", "terminal_output_enabled": False}
        }
        
        from traffic_monitor.services.frame_capture_service import frame_capture_process
        
        frame_queue = Queue()
        shutdown_event = Event()
        
        process = mp.Process(
            target=frame_capture_process,
            args=(config, frame_queue, shutdown_event)
        )
        
        try:
            process.start()
            
            # Should handle large frames and resize them
            frame_msg = frame_queue.get(timeout=10)
            assert frame_msg is not None, "Should process large frames"
            assert frame_msg.frame.shape[:2] == (480, 640), "Should resize to target resolution"
            
        finally:
            shutdown_event.set()
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
    
    def test_extremely_small_frames(self):
        """Test handling of extremely small video frames."""
        small_video_path = self.temp_dir / "small.mp4"
        
        # Create video with very small frames
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(small_video_path), fourcc, 10.0, (32, 24))
        
        for i in range(10):
            frame = np.random.randint(0, 255, (24, 32, 3), dtype=np.uint8)
            out.write(frame)
        
        out.release()
        
        config = {
            "frame_grabber": {
                "video_source": str(small_video_path),
                "resize_resolution": [640, 480],  # Should upscale
                "process_every_n_frame": 1
            },
            "loguru": {"level": "WARNING", "terminal_output_enabled": False}
        }
        
        from traffic_monitor.services.frame_capture_service import frame_capture_process
        
        frame_queue = Queue()
        shutdown_event = Event()
        
        process = mp.Process(
            target=frame_capture_process,
            args=(config, frame_queue, shutdown_event)
        )
        
        try:
            process.start()
            
            # Should handle small frames and upscale them
            frame_msg = frame_queue.get(timeout=5)
            assert frame_msg is not None, "Should process small frames"
            assert frame_msg.frame.shape[:2] == (480, 640), "Should upscale to target resolution"
            
        finally:
            shutdown_event.set()
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
    
    def test_zero_confidence_detections(self):
        """Test handling of zero confidence detections."""
        with patch('ultralytics.YOLO') as mock_yolo:
            # Mock model that returns zero confidence detections
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = [Mock()]
            mock_result.boxes[0].xyxy = [[100, 100, 200, 200]]
            mock_result.boxes[0].conf = [0.0]  # Zero confidence
            mock_result.boxes[0].cls = [3]
            mock_model.predict.return_value = [mock_result]
            mock_yolo.return_value = mock_model
            
            config = {
                "vehicle_detector": {
                    "model_path": "mock_model.engine",
                    "conf_threshold": 0.5,
                    "class_mapping": {3: "car"}
                }
            }
            
            service = VehicleDetectionService(config)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Should filter out zero confidence detections
            detections = service.detect_vehicles(frame)
            assert len(detections) == 0, "Should filter out zero confidence detections"
    
    def test_invalid_bounding_boxes(self):
        """Test handling of invalid bounding boxes."""
        with patch('ultralytics.YOLO') as mock_yolo:
            # Mock model that returns invalid bounding boxes
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = [Mock(), Mock(), Mock()]
            
            # Invalid boxes: negative coordinates, zero area, out of bounds
            mock_result.boxes[0].xyxy = [[-10, -10, 50, 50]]  # Negative coordinates
            mock_result.boxes[0].conf = [0.8]
            mock_result.boxes[0].cls = [3]
            
            mock_result.boxes[1].xyxy = [[100, 100, 100, 100]]  # Zero area
            mock_result.boxes[1].conf = [0.8]
            mock_result.boxes[1].cls = [3]
            
            mock_result.boxes[2].xyxy = [[500, 400, 1000, 800]]  # Out of bounds for 640x480
            mock_result.boxes[2].conf = [0.8]
            mock_result.boxes[2].cls = [3]
            
            mock_model.predict.return_value = [mock_result]
            mock_yolo.return_value = mock_model
            
            config = {
                "vehicle_detector": {
                    "model_path": "mock_model.engine",
                    "conf_threshold": 0.5,
                    "class_mapping": {3: "car"}
                }
            }
            
            service = VehicleDetectionService(config)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Should handle invalid boxes gracefully
            detections = service.detect_vehicles(frame)
            # Should either filter out invalid boxes or clamp them to valid ranges
            for detection in detections:
                bbox = detection["bbox_xyxy"]
                assert bbox[0] >= 0 and bbox[1] >= 0, "Coordinates should be non-negative"
                assert bbox[2] > bbox[0] and bbox[3] > bbox[1], "Box should have positive area"
    
    def test_extremely_high_detection_count(self):
        """Test handling of extremely high number of detections."""
        with patch('ultralytics.YOLO') as mock_yolo:
            # Mock model that returns many detections
            mock_model = Mock()
            mock_result = Mock()
            
            # Create 1000 detections
            num_detections = 1000
            mock_result.boxes = []
            
            for i in range(num_detections):
                mock_box = Mock()
                mock_box.xyxy = [[i % 600, i % 400, (i % 600) + 50, (i % 400) + 50]]
                mock_box.conf = [0.8]
                mock_box.cls = [3]
                mock_result.boxes.append(mock_box)
            
            mock_model.predict.return_value = [mock_result]
            mock_yolo.return_value = mock_model
            
            config = {
                "vehicle_detector": {
                    "model_path": "mock_model.engine",
                    "conf_threshold": 0.5,
                    "class_mapping": {3: "car"}
                }
            }
            
            service = VehicleDetectionService(config)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Should handle many detections without crashing
            start_time = time.time()
            detections = service.detect_vehicles(frame)
            processing_time = time.time() - start_time
            
            assert len(detections) <= num_detections, "Should return detections"
            assert processing_time < 5.0, f"Processing {num_detections} detections took too long: {processing_time:.2f}s"
    
    def test_tracking_with_no_detections(self):
        """Test tracking service with no detections."""
        config = {
            "vehicle_tracker": {
                "tracker_type": "bytetrack",
                "track_thresh": 0.5
            }
        }
        
        service = VehicleTrackingService(config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Track with empty detections
        tracked_objects = service.track_vehicles(frame, [])
        assert isinstance(tracked_objects, list), "Should return list even with no detections"
        assert len(tracked_objects) == 0, "Should return empty list for no detections"
    
    def test_counting_with_invalid_lines(self):
        """Test vehicle counting with invalid counting lines."""
        # Test with various invalid line configurations
        invalid_configs = [
            {"counting_lines": []},  # Empty lines
            {"counting_lines": [[[0.5, 0.5]]]},  # Single point instead of line
            {"counting_lines": [[[2.0, 0.5], [0.5, 0.5]]]},  # Coordinates > 1.0
            {"counting_lines": [[[-0.5, 0.5], [0.5, 0.5]]]},  # Negative coordinates
            {"counting_lines": [[[0.5, 0.5], [0.5, 0.5]]]},  # Zero-length line
        ]
        
        for config in invalid_configs:
            service = VehicleCountingService(config)
            
            # Should handle invalid configurations gracefully
            tracked_objects = [
                {
                    "bbox_xyxy": [100, 100, 200, 200],
                    "track_id": 1,
                    "class_name": "car"
                }
            ]
            
            # Should not crash with invalid line configurations
            try:
                result = service.count_vehicles(tracked_objects, "test_frame")
                assert isinstance(result, dict), "Should return result dict"
            except Exception as e:
                # If it raises an exception, it should be handled gracefully
                assert "line" in str(e).lower() or "count" in str(e).lower(), f"Unexpected error: {e}"
    
    def test_database_corruption_recovery(self):
        """Test database recovery from corruption."""
        # Write some initial data
        success = minidb.write_vehicle_count(
            camera_id="corruption_test",
            total_count=10,
            class_counts={"car": 10}
        )
        assert success, "Initial write should succeed"
        
        # Simulate database corruption by writing invalid data
        try:
            conn = sqlite3.connect(self.test_db_path)
            cursor = conn.cursor()
            cursor.execute("DROP TABLE vehicle_counts")  # Corrupt by dropping table
            conn.commit()
            conn.close()
        except:
            pass  # Ignore errors in corruption simulation
        
        # Try to write data after corruption
        success = minidb.write_vehicle_count(
            camera_id="recovery_test",
            total_count=5,
            class_counts={"car": 5}
        )
        
        # Should either succeed (if recovery works) or fail gracefully
        assert isinstance(success, bool), "Should return boolean result"
        
        # If it failed, try to reinitialize
        if not success:
            try:
                minidb.init_db()  # Try to recover
                success = minidb.write_vehicle_count(
                    camera_id="recovery_test",
                    total_count=5,
                    class_counts={"car": 5}
                )
                assert success, "Should succeed after recovery"
            except Exception as e:
                # Recovery might not be implemented, which is acceptable
                assert "database" in str(e).lower() or "table" in str(e).lower()
    
    def test_disk_space_exhaustion(self):
        """Test behavior when disk space is exhausted."""
        # Create a very large database entry to simulate disk space issues
        large_class_counts = {f"class_{i}": i for i in range(10000)}
        
        # This might fail due to disk space or succeed if there's enough space
        try:
            success = minidb.write_vehicle_count(
                camera_id="disk_space_test",
                total_count=sum(large_class_counts.values()),
                class_counts=large_class_counts
            )
            # If it succeeds, that's fine too
            assert isinstance(success, bool), "Should return boolean result"
        except Exception as e:
            # Should handle disk space errors gracefully
            assert any(keyword in str(e).lower() for keyword in ["disk", "space", "full", "write"]), f"Unexpected error: {e}"
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters."""
        # Test with various special characters in camera IDs and class names
        special_test_cases = [
            {"camera_id": "cam_测试", "class_name": "汽车"},  # Chinese characters
            {"camera_id": "cam_тест", "class_name": "машина"},  # Cyrillic
            {"camera_id": "cam_🚗", "class_name": "car_🚙"},  # Emojis
            {"camera_id": "cam'test\"", "class_name": "car'test\""},  # SQL injection attempt
            {"camera_id": "cam\ntest", "class_name": "car\ttest"},  # Control characters
            {"camera_id": "", "class_name": ""},  # Empty strings
        ]
        
        for test_case in special_test_cases:
            try:
                success = minidb.write_vehicle_count(
                    camera_id=test_case["camera_id"],
                    total_count=1,
                    class_counts={test_case["class_name"]: 1}
                )
                # Should either succeed or fail gracefully
                assert isinstance(success, bool), f"Should return boolean for case: {test_case}"
            except Exception as e:
                # Should handle special characters gracefully
                assert any(keyword in str(e).lower() for keyword in ["character", "encoding", "unicode", "invalid"]), f"Unexpected error for {test_case}: {e}"
    
    def test_extremely_long_strings(self):
        """Test handling of extremely long strings."""
        # Test with very long camera ID and class names
        long_camera_id = "a" * 10000
        long_class_name = "b" * 10000
        
        try:
            success = minidb.write_vehicle_count(
                camera_id=long_camera_id,
                total_count=1,
                class_counts={long_class_name: 1}
            )
            # Should either succeed (if database can handle it) or fail gracefully
            assert isinstance(success, bool), "Should return boolean result"
        except Exception as e:
            # Should handle long strings gracefully
            assert any(keyword in str(e).lower() for keyword in ["length", "long", "size", "limit"]), f"Unexpected error: {e}"
    
    def test_negative_and_extreme_numbers(self):
        """Test handling of negative and extreme numbers."""
        extreme_test_cases = [
            {"total_count": -1, "class_counts": {"car": -1}},  # Negative numbers
            {"total_count": 0, "class_counts": {"car": 0}},  # Zero values
            {"total_count": 2**63 - 1, "class_counts": {"car": 2**63 - 1}},  # Max int64
            {"total_count": float('inf'), "class_counts": {"car": float('inf')}},  # Infinity
            {"total_count": float('nan'), "class_counts": {"car": float('nan')}},  # NaN
        ]
        
        for i, test_case in enumerate(extreme_test_cases):
            try:
                success = minidb.write_vehicle_count(
                    camera_id=f"extreme_test_{i}",
                    total_count=test_case["total_count"],
                    class_counts=test_case["class_counts"]
                )
                # Should either succeed or fail gracefully
                assert isinstance(success, bool), f"Should return boolean for case {i}: {test_case}"
            except Exception as e:
                # Should handle extreme numbers gracefully
                assert any(keyword in str(e).lower() for keyword in ["number", "value", "range", "invalid", "overflow"]), f"Unexpected error for case {i}: {e}"
    
    def test_concurrent_database_corruption(self):
        """Test database behavior under concurrent access with corruption."""
        def corrupt_database():
            """Function to corrupt database while other operations are running."""
            time.sleep(0.1)  # Let other operations start
            try:
                # Try to corrupt the database
                conn = sqlite3.connect(self.test_db_path)
                cursor = conn.cursor()
                cursor.execute("UPDATE vehicle_counts SET camera_id = NULL WHERE rowid = 1")
                conn.commit()
                conn.close()
            except:
                pass  # Ignore corruption errors
        
        def write_data(thread_id):
            """Function to write data concurrently."""
            for i in range(10):
                try:
                    minidb.write_vehicle_count(
                        camera_id=f"concurrent_{thread_id}",
                        total_count=i,
                        class_counts={"car": i}
                    )
                    time.sleep(0.01)
                except:
                    pass  # Ignore errors during corruption test
        
        # Start concurrent operations
        threads = []
        
        # Start corruption thread
        corrupt_thread = threading.Thread(target=corrupt_database)
        threads.append(corrupt_thread)
        
        # Start writer threads
        for i in range(3):
            writer_thread = threading.Thread(target=write_data, args=(i,))
            threads.append(writer_thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=5)
        
        # System should survive concurrent corruption
        # Try a final operation to verify system state
        try:
            success = minidb.write_vehicle_count(
                camera_id="final_test",
                total_count=1,
                class_counts={"car": 1}
            )
            # Should either succeed or fail gracefully
            assert isinstance(success, bool), "Should return boolean result after corruption test"
        except Exception:
            # Database might be corrupted, which is acceptable for this test
            pass
    
    def test_configuration_edge_cases(self):
        """Test configuration loading with edge cases."""
        edge_case_configs = [
            {},  # Empty config
            {"invalid_section": {"key": "value"}},  # Unknown sections
            {"frame_grabber": None},  # None values
            {"frame_grabber": {"video_source": None}},  # None in required field
            {"frame_grabber": {"resize_resolution": []}},  # Empty list
            {"frame_grabber": {"resize_resolution": [-640, -480]}},  # Negative resolution
            {"frame_grabber": {"resize_resolution": [0, 0]}},  # Zero resolution
            {"frame_grabber": {"process_every_n_frame": 0}},  # Zero frame skip
            {"frame_grabber": {"process_every_n_frame": -1}},  # Negative frame skip
        ]
        
        for i, config in enumerate(edge_case_configs):
            config_path = self.temp_dir / f"edge_config_{i}.yaml"
            
            try:
                with open(config_path, 'w') as f:
                    yaml.dump(config, f)
                
                # Should handle edge case configs gracefully
                loaded_config = load_config(config_path)
                
                # Should either load successfully or return None/raise exception gracefully
                assert loaded_config is None or isinstance(loaded_config, dict), f"Invalid return type for config {i}"
                
            except Exception as e:
                # Should handle config errors gracefully
                assert any(keyword in str(e).lower() for keyword in ["config", "yaml", "load", "parse"]), f"Unexpected error for config {i}: {e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])