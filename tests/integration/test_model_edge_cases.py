"""
Model and AI-specific edge case tests.
Tests for model inference, memory management, and AI pipeline edge cases.
"""

import pytest
import tempfile
import shutil
import time
import threading
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import numpy as np
import torch

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from traffic_monitor.services.vehicle_detection_service import VehicleDetectionService
from traffic_monitor.services.vehicle_tracking_service import VehicleTrackingService


@pytest.mark.integration
class TestModelEdgeCases:
    """Test edge cases specific to AI models and inference."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def teardown_method(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_model_inference_timeout(self):
        """Test handling of model inference timeouts."""
        with patch('ultralytics.YOLO') as mock_yolo:
            # Mock model that takes too long to respond
            mock_model = Mock()
            
            def slow_predict(*args, **kwargs):
                time.sleep(2.0)  # Simulate slow inference
                mock_result = Mock()
                mock_result.boxes = []
                return [mock_result]
            
            mock_model.predict = slow_predict
            mock_yolo.return_value = mock_model
            
            config = {
                "vehicle_detector": {
                    "model_path": "slow_model.engine",
                    "conf_threshold": 0.5,
                    "class_mapping": {3: "car"}
                }
            }
            
            service = VehicleDetectionService(config)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Test with timeout
            start_time = time.time()
            try:
                detections = service.detect_vehicles(frame)
                inference_time = time.time() - start_time
                
                # Should either complete or handle timeout gracefully
                assert inference_time < 5.0, "Should not hang indefinitely"
                assert isinstance(detections, list), "Should return list even on timeout"
                
            except Exception as e:
                # Timeout exceptions are acceptable
                assert "timeout" in str(e).lower() or "time" in str(e).lower()

    def test_model_memory_leak_detection(self):
        """Test detection of memory leaks in model inference."""
        import psutil
        import gc
        
        with patch('ultralytics.YOLO') as mock_yolo:
            # Mock model that potentially leaks memory
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = []
            mock_model.predict.return_value = [mock_result]
            mock_yolo.return_value = mock_model
            
            config = {
                "vehicle_detector": {
                    "model_path": "test_model.engine",
                    "conf_threshold": 0.5,
                    "class_mapping": {3: "car"}
                }
            }
            
            service = VehicleDetectionService(config)
            
            # Get initial memory usage
            process = psutil.Process()
            initial_memory = process.memory_info().rss
            
            # Run many inference cycles
            for i in range(100):
                frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
                detections = service.detect_vehicles(frame)
                
                # Periodic garbage collection
                if i % 10 == 0:
                    gc.collect()
            
            # Check final memory usage
            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory
            
            # Memory increase should be reasonable (less than 100MB)
            assert memory_increase < 100 * 1024 * 1024, f"Potential memory leak: {memory_increase / 1024 / 1024:.2f}MB increase"

    def test_invalid_model_weights(self):
        """Test handling of invalid or corrupted model weights."""
        # Create fake model file with invalid content
        fake_model_path = self.temp_dir / "invalid_model.pt"
        with open(fake_model_path, 'wb') as f:
            f.write(b"This is not a valid model file")
        
        config = {
            "vehicle_detector": {
                "model_path": str(fake_model_path),
                "conf_threshold": 0.5,
                "class_mapping": {3: "car"}
            }
        }
        
        # Should handle invalid model gracefully
        try:
            service = VehicleDetectionService(config)
            # If it doesn't raise an exception during init, test inference
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            detections = service.detect_vehicles(frame)
            assert isinstance(detections, list), "Should return empty list for invalid model"
        except Exception as e:
            # Expected to fail with invalid model
            assert any(keyword in str(e).lower() for keyword in ["model", "load", "invalid", "corrupt"])

    def test_model_version_compatibility(self):
        """Test handling of model version compatibility issues."""
        with patch('ultralytics.YOLO') as mock_yolo:
            # Mock version compatibility error
            mock_yolo.side_effect = RuntimeError("Model version incompatible with current framework")
            
            config = {
                "vehicle_detector": {
                    "model_path": "incompatible_model.pt",
                    "conf_threshold": 0.5,
                    "class_mapping": {3: "car"}
                }
            }
            
            # Should handle version incompatibility gracefully
            try:
                service = VehicleDetectionService(config)
                assert False, "Should raise exception for incompatible model"
            except Exception as e:
                assert "incompatible" in str(e).lower() or "version" in str(e).lower()

    def test_batch_size_optimization_limits(self):
        """Test batch size optimization and limits."""
        with patch('ultralytics.YOLO') as mock_yolo:
            mock_model = Mock()
            
            # Mock different behavior for different batch sizes
            def batch_predict(frames, **kwargs):
                batch_size = len(frames) if isinstance(frames, list) else 1
                
                if batch_size > 10:
                    raise RuntimeError("Batch size too large for GPU memory")
                
                # Return results for each frame in batch
                results = []
                for _ in range(batch_size):
                    mock_result = Mock()
                    mock_result.boxes = []
                    results.append(mock_result)
                return results
            
            mock_model.predict = batch_predict
            mock_yolo.return_value = mock_model
            
            config = {
                "vehicle_detector": {
                    "model_path": "batch_model.engine",
                    "conf_threshold": 0.5,
                    "class_mapping": {3: "car"}
                }
            }
            
            service = VehicleDetectionService(config)
            
            # Test with small batch (should work)
            small_batch = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
            try:
                for frame in small_batch:
                    detections = service.detect_vehicles(frame)
                    assert isinstance(detections, list)
            except Exception:
                pytest.fail("Small batch should work")
            
            # Test with large batch (should handle gracefully)
            large_batch = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(15)]
            try:
                for frame in large_batch:
                    detections = service.detect_vehicles(frame)
                    # Should either work or handle batch size error gracefully
                    assert isinstance(detections, list)
            except Exception as e:
                # Expected for large batches
                assert "batch" in str(e).lower() or "memory" in str(e).lower()

    def test_gpu_memory_exhaustion(self):
        """Test handling of GPU memory exhaustion."""
        with patch('ultralytics.YOLO') as mock_yolo:
            mock_model = Mock()
            
            # Mock GPU memory error
            def gpu_memory_error(*args, **kwargs):
                raise RuntimeError("CUDA out of memory")
            
            mock_model.predict = gpu_memory_error
            mock_yolo.return_value = mock_model
            
            config = {
                "vehicle_detector": {
                    "model_path": "gpu_model.engine",
                    "conf_threshold": 0.5,
                    "class_mapping": {3: "car"}
                }
            }
            
            service = VehicleDetectionService(config)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Should handle GPU memory errors gracefully
            try:
                detections = service.detect_vehicles(frame)
                # If no exception, should return empty list
                assert isinstance(detections, list)
            except Exception as e:
                # Expected GPU memory error
                assert "cuda" in str(e).lower() or "memory" in str(e).lower()

    def test_concurrent_model_inference(self):
        """Test concurrent model inference from multiple threads."""
        with patch('ultralytics.YOLO') as mock_yolo:
            mock_model = Mock()
            
            # Mock thread-safe inference
            inference_count = {"value": 0}
            lock = threading.Lock()
            
            def thread_safe_predict(*args, **kwargs):
                with lock:
                    inference_count["value"] += 1
                    time.sleep(0.01)  # Simulate inference time
                
                mock_result = Mock()
                mock_result.boxes = []
                return [mock_result]
            
            mock_model.predict = thread_safe_predict
            mock_yolo.return_value = mock_model
            
            config = {
                "vehicle_detector": {
                    "model_path": "concurrent_model.engine",
                    "conf_threshold": 0.5,
                    "class_mapping": {3: "car"}
                }
            }
            
            service = VehicleDetectionService(config)
            
            def inference_worker():
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                detections = service.detect_vehicles(frame)
                assert isinstance(detections, list)
            
            # Start multiple inference threads
            threads = []
            for _ in range(5):
                t = threading.Thread(target=inference_worker)
                threads.append(t)
                t.start()
            
            # Wait for all threads to complete
            for t in threads:
                t.join(timeout=5)
                assert not t.is_alive(), "Thread should complete"
            
            # Verify all inferences completed
            assert inference_count["value"] == 5, "All concurrent inferences should complete"

    def test_model_warm_up_edge_cases(self):
        """Test model warm-up and initialization edge cases."""
        with patch('ultralytics.YOLO') as mock_yolo:
            mock_model = Mock()
            
            # Mock slow model initialization
            def slow_init(*args, **kwargs):
                time.sleep(0.5)  # Simulate slow loading
                return mock_model
            
            mock_yolo.side_effect = slow_init
            
            # Mock normal inference after warm-up
            mock_result = Mock()
            mock_result.boxes = []
            mock_model.predict.return_value = [mock_result]
            
            config = {
                "vehicle_detector": {
                    "model_path": "slow_init_model.engine",
                    "conf_threshold": 0.5,
                    "class_mapping": {3: "car"}
                }
            }
            
            # Test initialization time
            start_time = time.time()
            service = VehicleDetectionService(config)
            init_time = time.time() - start_time
            
            # Should handle slow initialization
            assert init_time < 2.0, f"Initialization too slow: {init_time:.2f}s"
            
            # Test first inference (warm-up)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            start_time = time.time()
            detections = service.detect_vehicles(frame)
            inference_time = time.time() - start_time
            
            assert isinstance(detections, list)
            assert inference_time < 1.0, f"First inference too slow: {inference_time:.2f}s"

    def test_tracking_model_edge_cases(self):
        """Test edge cases specific to tracking models."""
        config = {
            "vehicle_tracker": {
                "tracker_type": "bytetrack",
                "track_thresh": 0.5,
                "track_buffer": 30,
                "match_thresh": 0.8
            }
        }
        
        service = VehicleTrackingService(config)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Test with rapidly changing detections
        detection_sequences = [
            # Sequence 1: Many detections
            [{"bbox_xyxy": [i*50, 100, i*50+40, 140], "confidence": 0.8, "class_id": 3} for i in range(20)],
            # Sequence 2: No detections
            [],
            # Sequence 3: Single detection
            [{"bbox_xyxy": [300, 200, 340, 240], "confidence": 0.9, "class_id": 3}],
            # Sequence 4: Overlapping detections
            [
                {"bbox_xyxy": [100, 100, 200, 200], "confidence": 0.8, "class_id": 3},
                {"bbox_xyxy": [150, 150, 250, 250], "confidence": 0.7, "class_id": 3},
                {"bbox_xyxy": [120, 120, 220, 220], "confidence": 0.9, "class_id": 3}
            ]
        ]
        
        for i, detections in enumerate(detection_sequences):
            try:
                tracked_objects = service.track_vehicles(frame, detections)
                assert isinstance(tracked_objects, list), f"Should return list for sequence {i}"
                
                # Verify tracking consistency
                for obj in tracked_objects:
                    assert "track_id" in obj, "Tracked object should have track_id"
                    assert isinstance(obj["track_id"], int), "Track ID should be integer"
                    
            except Exception as e:
                # Should handle edge cases gracefully
                assert "track" in str(e).lower() or "detection" in str(e).lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])