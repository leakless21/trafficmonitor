"""
Integration tests for individual service interactions and edge cases.
Tests service-to-service communication, error handling, and boundary conditions.
"""

import pytest
import tempfile
import shutil
import time
import threading
import multiprocessing as mp
from pathlib import Path
import sys
import cv2
import numpy as np
from unittest.mock import patch, Mock, MagicMock
from multiprocessing import Queue, Event
import yaml

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from traffic_monitor.services.frame_capture_service import frame_capture_process
from traffic_monitor.services.vehicle_detection_service import vehicle_detection_process, VehicleDetectionService
from traffic_monitor.services.vehicle_tracking_service import vehicle_tracking_process, VehicleTrackingService
from traffic_monitor.services.license_plate_detection_service import license_plate_detection_process
from traffic_monitor.services.text_recognition_service import text_recognition_process
from traffic_monitor.services.vehicle_counting_service import vehicle_counting_process
from traffic_monitor.services.visualization_service import visualization_process
from traffic_monitor.services.event_distribution_service import event_distribution_process
from traffic_monitor.utils.custom_types import FrameMessage, DetectedVehicleMessage, TrackedVehicleMessage


@pytest.mark.integration
class TestServiceIntegration:
    """Test integration between different services."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_video_path = self._create_test_video()
        self.test_config = self._create_test_config()
        
        # Create queues for inter-service communication
        self.frame_queue = Queue(maxsize=10)
        self.detection_queue = Queue(maxsize=10)
        self.tracking_queue = Queue(maxsize=10)
        self.lp_detection_queue = Queue(maxsize=10)
        self.ocr_queue = Queue(maxsize=10)
        self.counting_queue = Queue(maxsize=10)
        self.shutdown_event = Event()
        
    def teardown_method(self):
        """Clean up test environment."""
        self.shutdown_event.set()
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_frame_capture_to_detection_integration(self):
        """Test frame capture service integration with vehicle detection."""
        with patch('ultralytics.YOLO') as mock_yolo:
            # Setup mock detection model
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = [Mock()]
            mock_result.boxes[0].xyxy = [[100, 100, 200, 200]]
            mock_result.boxes[0].conf = [0.85]
            mock_result.boxes[0].cls = [3]  # car class
            mock_model.predict.return_value = [mock_result]
            mock_yolo.return_value = mock_model
            
            # Start frame capture process
            frame_process = mp.Process(
                target=frame_capture_process,
                args=(self.test_config, self.frame_queue, self.shutdown_event)
            )
            
            # Start detection process
            detection_process = mp.Process(
                target=vehicle_detection_process,
                args=(self.test_config, self.frame_queue, self.detection_queue, self.shutdown_event)
            )
            
            try:
                frame_process.start()
                detection_process.start()
                
                # Wait for some processing
                time.sleep(2)
                
                # Check if detection results are produced
                detection_results = []
                timeout = time.time() + 5
                while time.time() < timeout and len(detection_results) < 3:
                    try:
                        result = self.detection_queue.get(timeout=1)
                        detection_results.append(result)
                    except:
                        break
                
                # Verify integration
                assert len(detection_results) > 0, "Should produce detection results"
                
                # Verify data structure
                for result in detection_results:
                    assert isinstance(result, DetectedVehicleMessage), "Should be DetectedVehicleMessage"
                    assert hasattr(result, 'frame_id'), "Should have frame_id"
                    assert hasattr(result, 'detections'), "Should have detections"
                    
            finally:
                self.shutdown_event.set()
                frame_process.join(timeout=3)
                detection_process.join(timeout=3)
                if frame_process.is_alive():
                    frame_process.terminate()
                if detection_process.is_alive():
                    detection_process.terminate()
    
    def test_detection_to_tracking_integration(self):
        """Test vehicle detection to tracking service integration."""
        # Create mock detection message
        detection_msg = DetectedVehicleMessage(
            frame_id="test_001",
            timestamp=time.time(),
            frame=np.zeros((480, 640, 3), dtype=np.uint8),
            detections=[
                {
                    "bbox_xyxy": [100, 100, 200, 200],
                    "confidence": 0.85,
                    "class_id": 3,
                    "class_name": "car"
                }
            ]
        )
        
        # Put detection message in queue
        self.detection_queue.put(detection_msg)
        
        # Start tracking process
        tracking_process = mp.Process(
            target=vehicle_tracking_process,
            args=(self.test_config, self.detection_queue, self.tracking_queue, self.shutdown_event)
        )
        
        try:
            tracking_process.start()
            
            # Wait for tracking result
            tracking_result = self.tracking_queue.get(timeout=5)
            
            # Verify tracking integration
            assert isinstance(tracking_result, TrackedVehicleMessage), "Should be TrackedVehicleMessage"
            assert tracking_result.frame_id == detection_msg.frame_id, "Frame IDs should match"
            assert len(tracking_result.tracked_objects) > 0, "Should have tracked objects"
            assert "track_id" in tracking_result.tracked_objects[0], "Should assign track IDs"
            
        finally:
            self.shutdown_event.set()
            tracking_process.join(timeout=3)
            if tracking_process.is_alive():
                tracking_process.terminate()
    
    def test_tracking_to_license_plate_detection_integration(self):
        """Test tracking to license plate detection integration."""
        with patch('ultralytics.YOLO') as mock_yolo:
            # Setup mock LP detection model
            mock_lp_model = Mock()
            mock_lp_result = Mock()
            mock_lp_result.boxes = [Mock()]
            mock_lp_result.boxes[0].xyxy = [[10, 10, 50, 30]]
            mock_lp_result.boxes[0].conf = [0.75]
            mock_lp_model.predict.return_value = [mock_lp_result]
            mock_yolo.return_value = mock_lp_model
            
            # Create mock tracking message
            tracking_msg = TrackedVehicleMessage(
                frame_id="test_001",
                timestamp=time.time(),
                frame=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
                tracked_objects=[
                    {
                        "bbox_xyxy": [100, 100, 300, 200],
                        "confidence": 0.85,
                        "class_id": 3,
                        "class_name": "car",
                        "track_id": 1
                    }
                ]
            )
            
            # Put tracking message in queue
            self.tracking_queue.put(tracking_msg)
            
            # Start LP detection process
            lp_process = mp.Process(
                target=license_plate_detection_process,
                args=(self.test_config, self.tracking_queue, self.lp_detection_queue, self.shutdown_event)
            )
            
            try:
                lp_process.start()
                
                # Wait for LP detection result
                lp_result = self.lp_detection_queue.get(timeout=5)
                
                # Verify LP detection integration
                assert hasattr(lp_result, 'vehicle_id'), "Should have vehicle_id"
                assert hasattr(lp_result, 'plate_detections'), "Should have plate_detections"
                
            finally:
                self.shutdown_event.set()
                lp_process.join(timeout=3)
                if lp_process.is_alive():
                    lp_process.terminate()
    
    def test_license_plate_to_ocr_integration(self):
        """Test license plate detection to OCR integration."""
        with patch('traffic_monitor.services.text_recognition_service.FastPlateOCR') as mock_ocr:
            # Setup mock OCR
            mock_ocr_instance = Mock()
            mock_ocr_instance.run.return_value = [("ABC123", 0.9)]
            mock_ocr.return_value = mock_ocr_instance
            
            # Create mock LP detection message
            lp_msg = Mock()
            lp_msg.vehicle_id = 1
            lp_msg.plate_detections = [
                {
                    "plate_crop": np.random.randint(0, 255, (30, 100, 3), dtype=np.uint8),
                    "confidence": 0.8
                }
            ]
            
            # Put LP message in queue
            self.lp_detection_queue.put(lp_msg)
            
            # Start OCR process
            ocr_process = mp.Process(
                target=text_recognition_process,
                args=(self.test_config, self.lp_detection_queue, self.ocr_queue, self.shutdown_event)
            )
            
            try:
                ocr_process.start()
                
                # Wait for OCR result
                ocr_result = self.ocr_queue.get(timeout=5)
                
                # Verify OCR integration
                assert hasattr(ocr_result, 'vehicle_id'), "Should have vehicle_id"
                assert hasattr(ocr_result, 'plate_text'), "Should have plate_text"
                assert ocr_result.vehicle_id == lp_msg.vehicle_id, "Vehicle IDs should match"
                
            finally:
                self.shutdown_event.set()
                ocr_process.join(timeout=3)
                if ocr_process.is_alive():
                    ocr_process.terminate()
    
    def test_tracking_to_counting_integration(self):
        """Test tracking to vehicle counting integration."""
        # Create mock tracking messages with movement
        tracking_msgs = []
        for i in range(5):
            msg = TrackedVehicleMessage(
                frame_id=f"test_{i:03d}",
                timestamp=time.time() + i * 0.1,
                frame=np.zeros((480, 640, 3), dtype=np.uint8),
                tracked_objects=[
                    {
                        "bbox_xyxy": [100 + i * 50, 100, 200 + i * 50, 200],
                        "confidence": 0.85,
                        "class_id": 3,
                        "class_name": "car",
                        "track_id": 1
                    }
                ]
            )
            tracking_msgs.append(msg)
        
        # Put tracking messages in queue
        for msg in tracking_msgs:
            self.tracking_queue.put(msg)
        
        # Start counting process
        counting_process = mp.Process(
            target=vehicle_counting_process,
            args=(self.test_config, self.tracking_queue, self.counting_queue, self.shutdown_event)
        )
        
        try:
            counting_process.start()
            
            # Wait for counting results
            counting_results = []
            timeout = time.time() + 5
            while time.time() < timeout and len(counting_results) < 3:
                try:
                    result = self.counting_queue.get(timeout=1)
                    counting_results.append(result)
                except:
                    break
            
            # Verify counting integration
            assert len(counting_results) > 0, "Should produce counting results"
            
            for result in counting_results:
                assert hasattr(result, 'camera_id'), "Should have camera_id"
                assert hasattr(result, 'total_count'), "Should have total_count"
                assert hasattr(result, 'class_counts'), "Should have class_counts"
                
        finally:
            self.shutdown_event.set()
            counting_process.join(timeout=3)
            if counting_process.is_alive():
                counting_process.terminate()
    
    def test_multi_service_pipeline_integration(self):
        """Test complete multi-service pipeline integration."""
        with patch('ultralytics.YOLO') as mock_yolo, \
             patch('traffic_monitor.services.text_recognition_service.FastPlateOCR') as mock_ocr:
            
            # Setup mocks
            self._setup_mocks(mock_yolo, mock_ocr)
            
            # Start all services
            processes = []
            
            # Frame capture
            frame_proc = mp.Process(
                target=frame_capture_process,
                args=(self.test_config, self.frame_queue, self.shutdown_event)
            )
            processes.append(frame_proc)
            
            # Vehicle detection
            detection_proc = mp.Process(
                target=vehicle_detection_process,
                args=(self.test_config, self.frame_queue, self.detection_queue, self.shutdown_event)
            )
            processes.append(detection_proc)
            
            # Vehicle tracking
            tracking_proc = mp.Process(
                target=vehicle_tracking_process,
                args=(self.test_config, self.detection_queue, self.tracking_queue, self.shutdown_event)
            )
            processes.append(tracking_proc)
            
            # Vehicle counting
            counting_proc = mp.Process(
                target=vehicle_counting_process,
                args=(self.test_config, self.tracking_queue, self.counting_queue, self.shutdown_event)
            )
            processes.append(counting_proc)
            
            try:
                # Start all processes
                for proc in processes:
                    proc.start()
                
                # Let pipeline run for a short time
                time.sleep(3)
                
                # Check for results in counting queue
                counting_results = []
                timeout = time.time() + 5
                while time.time() < timeout and len(counting_results) < 2:
                    try:
                        result = self.counting_queue.get(timeout=1)
                        counting_results.append(result)
                    except:
                        break
                
                # Verify end-to-end pipeline
                assert len(counting_results) > 0, "Pipeline should produce counting results"
                
            finally:
                self.shutdown_event.set()
                for proc in processes:
                    proc.join(timeout=3)
                    if proc.is_alive():
                        proc.terminate()
    
    def test_error_handling_between_services(self):
        """Test error handling and recovery between services."""
        # Test with corrupted frame data
        corrupted_frame_msg = FrameMessage(
            frame_id="corrupted_001",
            timestamp=time.time(),
            frame=None  # Corrupted frame
        )
        
        self.frame_queue.put(corrupted_frame_msg)
        
        with patch('ultralytics.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_model.predict.side_effect = Exception("Model error")
            mock_yolo.return_value = mock_model
            
            # Start detection process
            detection_process = mp.Process(
                target=vehicle_detection_process,
                args=(self.test_config, self.frame_queue, self.detection_queue, self.shutdown_event)
            )
            
            try:
                detection_process.start()
                
                # Wait briefly
                time.sleep(1)
                
                # Add valid frame after error
                valid_frame_msg = FrameMessage(
                    frame_id="valid_001",
                    timestamp=time.time(),
                    frame=np.zeros((480, 640, 3), dtype=np.uint8)
                )
                self.frame_queue.put(valid_frame_msg)
                
                # Process should continue running despite errors
                time.sleep(1)
                assert detection_process.is_alive(), "Process should survive errors"
                
            finally:
                self.shutdown_event.set()
                detection_process.join(timeout=3)
                if detection_process.is_alive():
                    detection_process.terminate()
    
    def test_queue_overflow_handling(self):
        """Test service behavior when queues overflow."""
        # Fill queue to capacity
        small_queue = Queue(maxsize=2)
        
        # Fill queue
        for i in range(2):
            small_queue.put(f"message_{i}")
        
        # Try to add more messages (should handle gracefully)
        detection_service = VehicleDetectionService(self.test_config)
        
        # Service should handle queue overflow gracefully
        # This tests the put_realtime functionality
        from traffic_monitor.utils.queue_utils import put_realtime
        
        # Should not block
        success = put_realtime(small_queue, "overflow_message", "test_service")
        # In real-time mode, this might succeed or fail depending on implementation
        assert isinstance(success, bool), "Should return boolean result"
    
    def test_service_shutdown_coordination(self):
        """Test coordinated shutdown of multiple services."""
        processes = []
        
        # Start multiple services
        detection_proc = mp.Process(
            target=vehicle_detection_process,
            args=(self.test_config, self.frame_queue, self.detection_queue, self.shutdown_event)
        )
        processes.append(detection_proc)
        
        tracking_proc = mp.Process(
            target=vehicle_tracking_process,
            args=(self.test_config, self.detection_queue, self.tracking_queue, self.shutdown_event)
        )
        processes.append(tracking_proc)
        
        try:
            # Start processes
            for proc in processes:
                proc.start()
            
            # Verify all processes are running
            time.sleep(0.5)
            for proc in processes:
                assert proc.is_alive(), "Process should be running"
            
            # Signal shutdown
            self.shutdown_event.set()
            
            # Wait for graceful shutdown
            shutdown_timeout = time.time() + 5
            while time.time() < shutdown_timeout:
                if all(not proc.is_alive() for proc in processes):
                    break
                time.sleep(0.1)
            
            # Verify all processes shut down
            for i, proc in enumerate(processes):
                assert not proc.is_alive(), f"Process {i} should have shut down gracefully"
                
        finally:
            # Force terminate if needed
            for proc in processes:
                if proc.is_alive():
                    proc.terminate()
                proc.join(timeout=1)
    
    def test_service_performance_under_load(self):
        """Test service performance under high message load."""
        # Generate many frame messages
        frame_messages = []
        for i in range(50):
            msg = FrameMessage(
                frame_id=f"load_test_{i:03d}",
                timestamp=time.time() + i * 0.01,
                frame=np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)  # Smaller frames for speed
            )
            frame_messages.append(msg)
        
        with patch('ultralytics.YOLO') as mock_yolo:
            # Setup fast mock
            mock_model = Mock()
            mock_result = Mock()
            mock_result.boxes = []  # No detections for speed
            mock_model.predict.return_value = [mock_result]
            mock_yolo.return_value = mock_model
            
            # Start detection process
            detection_process = mp.Process(
                target=vehicle_detection_process,
                args=(self.test_config, self.frame_queue, self.detection_queue, self.shutdown_event)
            )
            
            try:
                detection_process.start()
                
                # Send messages rapidly
                start_time = time.time()
                for msg in frame_messages:
                    self.frame_queue.put(msg)
                
                # Collect results
                results = []
                timeout = time.time() + 10
                while time.time() < timeout and len(results) < len(frame_messages):
                    try:
                        result = self.detection_queue.get(timeout=1)
                        results.append(result)
                    except:
                        break
                
                processing_time = time.time() - start_time
                throughput = len(results) / processing_time
                
                # Verify performance
                assert len(results) > len(frame_messages) * 0.8, "Should process most messages"
                assert throughput > 10, f"Throughput too low: {throughput:.2f} fps"
                
            finally:
                self.shutdown_event.set()
                detection_process.join(timeout=3)
                if detection_process.is_alive():
                    detection_process.terminate()
    
    # Helper methods
    def _create_test_video(self):
        """Create a test video file."""
        video_path = self.temp_dir / "test_video.mp4"
        
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
                "process_every_n_frame": 1,
                "log_every_n_frames": 10
            },
            "vehicle_detector": {
                "model_path": "mock_model.engine",
                "conf_threshold": 0.3,
                "class_mapping": {0: "bicycle", 1: "bike", 2: "bus", 3: "car", 4: "person", 5: "truck"}
            },
            "vehicle_tracker": {
                "tracker_type": "bytetrack",
                "half": True,
                "device": "cpu"
            },
            "lp_detector": {
                "model_path": "mock_lp_model.engine",
                "conf_threshold": 0.5
            },
            "ocr_reader": {
                "backend": "fast_plate_ocr",
                "conf_threshold": 0.5
            },
            "vehicle_counter": {
                "counting_lines": [[[0.2, 0.4], [0.8, 0.6]]]
            },
            "loguru": {
                "level": "WARNING",
                "terminal_output_enabled": False
            }
        }
    
    def _setup_mocks(self, mock_yolo, mock_ocr):
        """Setup mocks for testing."""
        # Vehicle detection mock
        mock_vehicle_model = Mock()
        mock_detection_result = Mock()
        mock_detection_result.boxes = [Mock()]
        mock_detection_result.boxes[0].xyxy = [[100, 100, 200, 200]]
        mock_detection_result.boxes[0].conf = [0.85]
        mock_detection_result.boxes[0].cls = [3]
        mock_vehicle_model.predict.return_value = [mock_detection_result]
        
        # License plate detection mock
        mock_lp_model = Mock()
        mock_lp_result = Mock()
        mock_lp_result.boxes = []
        mock_lp_model.predict.return_value = [mock_lp_result]
        
        def yolo_side_effect(model_path):
            if "lp" in str(model_path) or "plate" in str(model_path):
                return mock_lp_model
            else:
                return mock_vehicle_model
        
        mock_yolo.side_effect = yolo_side_effect
        
        # OCR mock
        mock_ocr_instance = Mock()
        mock_ocr_instance.run.return_value = []
        mock_ocr.return_value = mock_ocr_instance


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])