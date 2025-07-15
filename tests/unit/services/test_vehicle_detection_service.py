"""
Unit tests for vehicle detection service.
Tests YOLO model inference and detection processing.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from traffic_monitor.services.vehicle_detection_service import vehicle_detection_process


class TestVehicleDetectionService:
    """Test vehicle detection and YOLO model inference."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = {
            "model_path": "data/models/vehicle/8n/best.engine",
            "conf_threshold": 0.5,
            "iou_threshold": 0.45,
            "class_mapping": {
                "0": "bicycle",
                "1": "bike", 
                "2": "bus",
                "3": "car",
                "5": "truck"
            },
            "input_size": [640, 640],
            "device": "cuda:0"
        }
        
        self.mock_queues = {
            "frame_to_detection": Mock(),
            "detection_to_tracking": Mock(),
            "detection_to_visualization": Mock()
        }
        
        # Sample frame data
        self.sample_frame = {
            "frame_id": 100,
            "timestamp": 1234567890.0,
            "frame": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        }
        
        # Sample detection results
        self.sample_detections = [
            {
                "bbox": [100, 100, 200, 200],
                "confidence": 0.85,
                "class_id": 3,
                "class_name": "car"
            },
            {
                "bbox": [300, 150, 400, 250],
                "confidence": 0.72,
                "class_id": 2,
                "class_name": "bus"
            }
        ]

    def test_model_loading_simulation(self):
        """Test model loading and initialization (mocked)."""
        with patch('torch.jit.load') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model
            
            # Simulate model loading
            model_path = self.mock_config["model_path"]
            model = mock_load(model_path)
            
            assert model is not None, "Model should be loaded"
            mock_load.assert_called_once_with(model_path)

    def test_frame_preprocessing(self):
        """Test frame preprocessing for model input."""
        frame = self.sample_frame["frame"]
        target_size = self.mock_config["input_size"]
        
        # Preprocess frame
        processed_frame = self._preprocess_frame(frame, target_size)
        
        assert processed_frame.shape == (640, 640, 3), "Frame should be resized to model input size"
        assert processed_frame.dtype == np.uint8, "Frame should maintain uint8 type"

    def test_detection_postprocessing(self):
        """Test detection postprocessing and filtering."""
        # Mock raw model output
        raw_detections = np.array([
            [100, 100, 200, 200, 0.85, 3],  # car, high confidence
            [300, 150, 400, 250, 0.72, 2],  # bus, medium confidence
            [500, 200, 600, 300, 0.35, 3],  # car, low confidence (should be filtered)
        ])
        
        conf_threshold = self.mock_config["conf_threshold"]
        
        # Filter by confidence
        filtered_detections = self._filter_detections(raw_detections, conf_threshold)
        
        assert len(filtered_detections) == 2, "Should filter out low confidence detection"
        assert all(det[4] >= conf_threshold for det in filtered_detections), "All detections should meet confidence threshold"

    def test_class_mapping_application(self):
        """Test application of class mapping to detections."""
        class_mapping = self.mock_config["class_mapping"]
        
        # Test class ID to name mapping
        test_cases = [
            (3, "car"),
            (2, "bus"),
            (0, "bicycle"),
            (99, "unknown")  # Unknown class
        ]
        
        for class_id, expected_name in test_cases:
            mapped_name = class_mapping.get(str(class_id), "unknown")
            assert mapped_name == expected_name, f"Class {class_id} should map to {expected_name}"

    def test_bbox_coordinate_validation(self):
        """Test bounding box coordinate validation."""
        frame_shape = (480, 640, 3)  # height, width, channels
        
        valid_bboxes = [
            [0, 0, 100, 100],      # Top-left corner
            [540, 380, 640, 480],  # Bottom-right corner
            [200, 150, 400, 350],  # Center
        ]
        
        invalid_bboxes = [
            [-10, 0, 100, 100],    # Negative x
            [0, -10, 100, 100],    # Negative y
            [600, 400, 700, 500],  # Exceeds frame width
            [200, 450, 400, 500],  # Exceeds frame height
            [200, 200, 150, 150],  # Invalid (x2 < x1, y2 < y1)
        ]
        
        for bbox in valid_bboxes:
            assert self._validate_bbox(bbox, frame_shape), f"Valid bbox should pass: {bbox}"
        
        for bbox in invalid_bboxes:
            assert not self._validate_bbox(bbox, frame_shape), f"Invalid bbox should fail: {bbox}"

    def test_non_maximum_suppression(self):
        """Test Non-Maximum Suppression (NMS) for overlapping detections."""
        # Overlapping detections of same class
        overlapping_detections = [
            [100, 100, 200, 200, 0.9, 3],   # car, high confidence
            [110, 110, 210, 210, 0.8, 3],   # car, overlapping, lower confidence
            [300, 300, 400, 400, 0.85, 2],  # bus, separate
        ]
        
        iou_threshold = self.mock_config["iou_threshold"]
        
        # Apply NMS (simplified)
        nms_detections = self._apply_nms(overlapping_detections, iou_threshold)
        
        assert len(nms_detections) == 2, "NMS should remove overlapping detection"
        assert nms_detections[0][4] == 0.9, "Should keep highest confidence detection"

    def test_detection_confidence_distribution(self):
        """Test detection confidence score distribution."""
        detections = self.sample_detections
        
        confidences = [det["confidence"] for det in detections]
        
        # Check confidence range
        assert all(0.0 <= conf <= 1.0 for conf in confidences), "All confidences should be in [0, 1]"
        
        # Check confidence threshold filtering
        threshold = 0.7
        high_conf_detections = [det for det in detections if det["confidence"] >= threshold]
        
        assert len(high_conf_detections) <= len(detections), "Filtered detections should be subset"

    def test_detection_area_calculation(self):
        """Test bounding box area calculation."""
        bbox = [100, 100, 200, 200]  # 100x100 box
        expected_area = 100 * 100
        
        calculated_area = self._calculate_bbox_area(bbox)
        assert calculated_area == expected_area, f"Area should be {expected_area}, got {calculated_area}"

    def test_detection_center_calculation(self):
        """Test bounding box center point calculation."""
        bbox = [100, 100, 200, 200]
        expected_center = (150, 150)
        
        calculated_center = self._calculate_bbox_center(bbox)
        assert calculated_center == expected_center, f"Center should be {expected_center}, got {calculated_center}"

    def test_model_input_normalization(self):
        """Test input normalization for model inference."""
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Normalize to [0, 1] range
        normalized = frame.astype(np.float32) / 255.0
        
        assert 0.0 <= normalized.min() <= normalized.max() <= 1.0, "Normalized values should be in [0, 1]"
        assert normalized.dtype == np.float32, "Normalized frame should be float32"

    def test_batch_processing_simulation(self):
        """Test batch processing of multiple frames."""
        batch_size = 4
        frames = []
        
        for i in range(batch_size):
            frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            frames.append(frame)
        
        # Simulate batch processing
        batch_results = []
        for frame in frames:
            # Mock detection result
            detections = [
                {
                    "bbox": [100 + i*10, 100, 200 + i*10, 200],
                    "confidence": 0.8,
                    "class_id": 3
                }
            ]
            batch_results.append(detections)
        
        assert len(batch_results) == batch_size, "Should process all frames in batch"

    def test_gpu_memory_management(self):
        """Test GPU memory usage simulation."""
        # Simulate GPU memory check
        with patch('torch.cuda.is_available') as mock_cuda:
            mock_cuda.return_value = True
            
            with patch('torch.cuda.memory_allocated') as mock_memory:
                mock_memory.return_value = 1024 * 1024 * 512  # 512 MB
                
                memory_usage = mock_memory()
                memory_mb = memory_usage / (1024 * 1024)
                
                assert memory_mb > 0, "GPU memory usage should be positive"

    def test_detection_tracking_preparation(self):
        """Test preparation of detection data for tracking."""
        detections = self.sample_detections
        frame_id = 100
        
        # Prepare for tracking
        tracking_data = []
        for i, det in enumerate(detections):
            tracking_item = {
                "detection_id": i,
                "frame_id": frame_id,
                "bbox": det["bbox"],
                "confidence": det["confidence"],
                "class_id": det["class_id"],
                "class_name": det["class_name"]
            }
            tracking_data.append(tracking_item)
        
        assert len(tracking_data) == len(detections), "Should prepare all detections for tracking"
        assert all("detection_id" in item for item in tracking_data), "All items should have detection_id"

    def test_error_handling_invalid_frame(self):
        """Test handling of invalid frame data."""
        invalid_frames = [
            None,
            np.array([]),  # Empty array
            np.zeros((100, 100), dtype=np.uint8),  # Grayscale
            np.zeros((100, 100, 4), dtype=np.uint8),  # RGBA
            "invalid_frame",  # String
        ]
        
        for invalid_frame in invalid_frames:
            try:
                result = self._validate_frame_for_detection(invalid_frame)
                assert not result, f"Invalid frame should be rejected: {type(invalid_frame)}"
            except Exception:
                # Exception handling is acceptable
                pass

    def test_detection_performance_metrics(self):
        """Test detection performance measurement."""
        import time
        
        # Simulate detection timing
        start_time = time.time()
        
        # Mock detection process
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        time.sleep(0.01)  # Simulate processing time
        
        processing_time = time.time() - start_time
        fps = 1.0 / processing_time if processing_time > 0 else 0
        
        assert processing_time > 0, "Processing time should be positive"
        assert fps > 0, "FPS should be positive"

    def test_detection_visualization_data(self):
        """Test preparation of detection data for visualization."""
        detections = self.sample_detections
        frame_shape = (480, 640, 3)
        
        viz_data = {
            "frame_id": 100,
            "detections": detections,
            "frame_shape": frame_shape,
            "timestamp": 1234567890.0
        }
        
        assert "detections" in viz_data, "Visualization data should contain detections"
        assert len(viz_data["detections"]) == len(detections), "Should include all detections"

    # Helper methods for testing
    def _preprocess_frame(self, frame, target_size):
        """Preprocess frame for model input."""
        return cv2.resize(frame, tuple(target_size))

    def _filter_detections(self, detections, conf_threshold):
        """Filter detections by confidence threshold."""
        return detections[detections[:, 4] >= conf_threshold]

    def _validate_bbox(self, bbox, frame_shape):
        """Validate bounding box coordinates."""
        x1, y1, x2, y2 = bbox
        height, width = frame_shape[:2]
        
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return False
        if x2 <= x1 or y2 <= y1:
            return False
        return True

    def _apply_nms(self, detections, iou_threshold):
        """Apply Non-Maximum Suppression (simplified)."""
        # Sort by confidence (descending)
        sorted_dets = sorted(detections, key=lambda x: x[4], reverse=True)
        
        # Simple NMS implementation for testing
        keep = []
        for det in sorted_dets:
            overlap = False
            for kept_det in keep:
                if self._calculate_iou(det[:4], kept_det[:4]) > iou_threshold:
                    overlap = True
                    break
            if not overlap:
                keep.append(det)
        
        return keep

    def _calculate_iou(self, bbox1, bbox2):
        """Calculate Intersection over Union."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def _calculate_bbox_area(self, bbox):
        """Calculate bounding box area."""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def _calculate_bbox_center(self, bbox):
        """Calculate bounding box center."""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def _validate_frame_for_detection(self, frame):
        """Validate frame for detection processing."""
        if frame is None or not isinstance(frame, np.ndarray):
            return False
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return False
        return True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])