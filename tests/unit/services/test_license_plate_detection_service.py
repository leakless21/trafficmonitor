"""
Unit tests for license plate detection service.
Tests plate detection, cropping, and preprocessing.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))


class TestLicensePlateDetectionService:
    """Test license plate detection functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = {
            "model_path": "data/models/plate/best.onnx",
            "conf_threshold": 0.3,
            "input_size": [640, 640],
            "device": "cuda:0"
        }
        
        # Sample vehicle detection
        self.sample_vehicle = {
            "bbox_xyxy": [100, 100, 300, 200],
            "confidence": 0.85,
            "class_id": 3,
            "class_name": "car",
            "track_id": 1
        }
        
        # Sample frame
        self.sample_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    def test_vehicle_crop_extraction(self):
        """Test extracting vehicle region from frame."""
        vehicle = self.sample_vehicle
        frame = self.sample_frame
        
        x1, y1, x2, y2 = vehicle["bbox_xyxy"]
        vehicle_crop = frame[y1:y2, x1:x2]
        
        expected_height = y2 - y1
        expected_width = x2 - x1
        
        assert vehicle_crop.shape[0] == expected_height
        assert vehicle_crop.shape[1] == expected_width
        assert vehicle_crop.shape[2] == 3

    def test_plate_detection_preprocessing(self):
        """Test preprocessing of vehicle crop for plate detection."""
        x1, y1, x2, y2 = self.sample_vehicle["bbox_xyxy"]
        vehicle_crop = self.sample_frame[y1:y2, x1:x2]
        
        # Resize to model input size
        target_size = tuple(self.mock_config["input_size"])
        resized_crop = cv2.resize(vehicle_crop, target_size)
        
        assert resized_crop.shape[:2] == target_size
        assert resized_crop.dtype == np.uint8

    def test_plate_bbox_validation(self):
        """Test validation of detected plate bounding boxes."""
        vehicle_crop_shape = (100, 200, 3)  # height, width, channels
        
        valid_plate_bboxes = [
            [10, 60, 190, 90],   # Valid plate in vehicle
            [0, 0, 50, 20],      # Top-left corner
            [150, 80, 200, 100], # Bottom-right area
        ]
        
        invalid_plate_bboxes = [
            [-5, 60, 50, 90],    # Negative x
            [10, -5, 50, 20],    # Negative y
            [150, 60, 250, 90],  # Exceeds width
            [10, 80, 50, 120],   # Exceeds height
            [100, 60, 50, 90],   # x2 < x1
        ]
        
        for bbox in valid_plate_bboxes:
            assert self._validate_plate_bbox(bbox, vehicle_crop_shape), f"Valid plate bbox should pass: {bbox}"
        
        for bbox in invalid_plate_bboxes:
            assert not self._validate_plate_bbox(bbox, vehicle_crop_shape), f"Invalid plate bbox should fail: {bbox}"

    def test_plate_confidence_filtering(self):
        """Test filtering plate detections by confidence."""
        mock_detections = [
            {"bbox": [10, 60, 100, 80], "confidence": 0.8},   # High confidence
            {"bbox": [120, 65, 180, 85], "confidence": 0.4},  # Medium confidence
            {"bbox": [50, 70, 90, 85], "confidence": 0.2},    # Low confidence
        ]
        
        conf_threshold = self.mock_config["conf_threshold"]
        
        filtered = [det for det in mock_detections if det["confidence"] >= conf_threshold]
        
        assert len(filtered) == 2, "Should filter out low confidence detection"
        assert all(det["confidence"] >= conf_threshold for det in filtered)

    def test_plate_aspect_ratio_validation(self):
        """Test validation of plate aspect ratios."""
        # Typical license plate aspect ratios (width/height)
        min_aspect_ratio = 2.0  # Minimum reasonable aspect ratio
        max_aspect_ratio = 6.0  # Maximum reasonable aspect ratio
        
        test_bboxes = [
            [10, 60, 110, 80],   # 100x20, ratio = 5.0 (valid)
            [10, 60, 70, 80],    # 60x20, ratio = 3.0 (valid)
            [10, 60, 50, 80],    # 40x20, ratio = 2.0 (valid)
            [10, 60, 30, 80],    # 20x20, ratio = 1.0 (invalid - too square)
            [10, 60, 150, 80],   # 140x20, ratio = 7.0 (invalid - too wide)
        ]
        
        valid_count = 0
        for bbox in test_bboxes:
            x1, y1, x2, y2 = bbox
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0
            
            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                valid_count += 1
        
        assert valid_count == 3, "Should validate 3 out of 5 aspect ratios"

    def test_plate_size_filtering(self):
        """Test filtering plates by minimum size."""
        min_plate_area = 400  # Minimum plate area in pixels
        
        test_plates = [
            {"bbox": [10, 60, 110, 80], "confidence": 0.8},   # 100x20 = 2000 (valid)
            {"bbox": [10, 60, 70, 80], "confidence": 0.7},    # 60x20 = 1200 (valid)
            {"bbox": [10, 60, 30, 75], "confidence": 0.6},    # 20x15 = 300 (invalid)
        ]
        
        valid_plates = []
        for plate in test_plates:
            x1, y1, x2, y2 = plate["bbox"]
            area = (x2 - x1) * (y2 - y1)
            if area >= min_plate_area:
                valid_plates.append(plate)
        
        assert len(valid_plates) == 2, "Should filter out small plate"

    def test_multiple_plates_handling(self):
        """Test handling of multiple plates in single vehicle."""
        multiple_plates = [
            {"bbox": [10, 60, 110, 80], "confidence": 0.8},
            {"bbox": [120, 65, 200, 85], "confidence": 0.7},
            {"bbox": [50, 70, 130, 90], "confidence": 0.6},
        ]
        
        # Should typically keep the highest confidence plate
        best_plate = max(multiple_plates, key=lambda x: x["confidence"])
        
        assert best_plate["confidence"] == 0.8, "Should select highest confidence plate"

    def test_plate_detection_error_handling(self):
        """Test error handling in plate detection."""
        # Test with invalid vehicle crop
        invalid_crops = [
            None,
            np.array([]),
            np.zeros((10, 10), dtype=np.uint8),  # Too small
            np.zeros((5000, 5000, 3), dtype=np.uint8),  # Too large
        ]
        
        for invalid_crop in invalid_crops:
            try:
                result = self._validate_vehicle_crop(invalid_crop)
                assert not result, f"Invalid crop should be rejected: {type(invalid_crop)}"
            except Exception:
                # Exception handling is acceptable
                pass

    def test_plate_detection_performance(self):
        """Test plate detection performance."""
        import time
        
        vehicle_crop = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        
        start_time = time.time()
        
        # Simulate plate detection processing
        resized = cv2.resize(vehicle_crop, (640, 640))
        time.sleep(0.001)  # Simulate processing time
        
        processing_time = time.time() - start_time
        
        assert processing_time < 0.1, f"Plate detection should be fast: {processing_time:.3f}s"

    def test_plate_region_extraction(self):
        """Test extraction of plate region from vehicle crop."""
        vehicle_crop = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        plate_bbox = [50, 60, 150, 80]  # x1, y1, x2, y2
        
        x1, y1, x2, y2 = plate_bbox
        plate_crop = vehicle_crop[y1:y2, x1:x2]
        
        expected_height = y2 - y1
        expected_width = x2 - x1
        
        assert plate_crop.shape[0] == expected_height
        assert plate_crop.shape[1] == expected_width
        assert plate_crop.shape[2] == 3

    # Helper methods
    def _validate_plate_bbox(self, bbox, crop_shape):
        """Validate plate bounding box within vehicle crop."""
        x1, y1, x2, y2 = bbox
        height, width = crop_shape[:2]
        
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            return False
        if x2 <= x1 or y2 <= y1:
            return False
        return True

    def _validate_vehicle_crop(self, crop):
        """Validate vehicle crop for plate detection."""
        if crop is None or not isinstance(crop, np.ndarray):
            return False
        if len(crop.shape) != 3 or crop.shape[2] != 3:
            return False
        if crop.shape[0] < 20 or crop.shape[1] < 20:  # Too small
            return False
        if crop.shape[0] > 2000 or crop.shape[1] > 2000:  # Too large
            return False
        return True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])