#!/usr/bin/env python3
"""
Test script for batch plate detection and cropping functionality.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import cv2
import numpy as np
import pytest

# Add the scripts directory to the Python path
project_root = Path(__file__).parent.parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(scripts_dir))

try:
    from batch_plate_crop import (
        load_yolo_model, 
        get_image_files, 
        detect_plates, 
        crop_and_convert_plates,
        process_image
    )
except ImportError:
    pytest.skip("batch_plate_crop module not available", allow_module_level=True)

def create_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a synthetic test image."""
    # Create a simple test image with some features
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add some colored rectangles to simulate a scene
    cv2.rectangle(image, (50, 50), (200, 150), (100, 100, 100), -1)  # Gray rectangle
    cv2.rectangle(image, (250, 200), (400, 280), (0, 255, 0), -1)    # Green rectangle
    cv2.rectangle(image, (450, 300), (600, 380), (255, 0, 0), -1)    # Blue rectangle
    
    # Add some text to simulate a license plate
    cv2.rectangle(image, (300, 100), (500, 150), (255, 255, 255), -1)  # White rectangle for plate
    cv2.putText(image, "ABC123", (320, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return image

class TestBatchPlateCrop:
    """Test suite for batch plate cropping functionality."""
    
    def test_get_image_files(self):
        """Test getting image files from a directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / "image1.jpg").touch()
            (temp_path / "image2.png").touch()
            (temp_path / "image3.JPG").touch()  # Test case sensitivity
            (temp_path / "not_image.txt").touch()  # Should be ignored
            (temp_path / "subdir").mkdir()  # Should be ignored
            
            image_files = get_image_files(str(temp_path))
            
            assert len(image_files) == 3
            assert all(f.suffix.lower() in {'.jpg', '.png'} for f in image_files)
    
    def test_get_image_files_nonexistent_folder(self):
        """Test error handling for nonexistent folder."""
        with pytest.raises(FileNotFoundError):
            get_image_files("/nonexistent/folder")
    
    def test_crop_and_convert_plates(self):
        """Test cropping and converting plates to grayscale."""
        # Create test image
        image = create_test_image(640, 480)
        
        # Simulate detection results (bbox, confidence)
        detections = [
            ([100, 100, 300, 200], 0.9),  # Valid detection
            ([400, 300, 600, 400], 0.8),  # Another valid detection
        ]
        
        cropped_plates = crop_and_convert_plates(image, detections)
        
        assert len(cropped_plates) == 2
        
        for plate in cropped_plates:
            # Check that plates are grayscale (2D arrays)
            assert len(plate.shape) == 2
            # Check that plates have reasonable size
            assert plate.shape[0] > 0 and plate.shape[1] > 0
    
    def test_crop_and_convert_plates_invalid_bbox(self):
        """Test handling of invalid bounding boxes."""
        image = create_test_image(640, 480)
        
        # Invalid detections
        detections = [
            ([100, 100, 50, 200], 0.9),   # x2 < x1
            ([100, 200, 300, 100], 0.8),  # y2 < y1
            ([700, 100, 800, 200], 0.7),  # Outside image bounds
        ]
        
        cropped_plates = crop_and_convert_plates(image, detections)
        
        # Should filter out invalid detections
        assert len(cropped_plates) == 0
    
    def test_crop_and_convert_plates_edge_cases(self):
        """Test edge cases for cropping."""
        image = create_test_image(100, 100)  # Small image
        
        # Detection at image edge
        detections = [
            ([0, 0, 50, 50], 0.9),     # Top-left corner
            ([50, 50, 100, 100], 0.8), # Bottom-right corner
        ]
        
        cropped_plates = crop_and_convert_plates(image, detections)
        
        assert len(cropped_plates) == 2
        for plate in cropped_plates:
            assert plate.shape[0] == 50 and plate.shape[1] == 50
    
    @pytest.mark.skipif(not Path("data/models/plate_v8n.pt").exists(), 
                       reason="YOLO model not available")
    def test_load_yolo_model(self):
        """Test loading YOLO model (only if model exists)."""
        model_path = "data/models/plate_v8n.pt"
        model = load_yolo_model(model_path)
        assert model is not None
    
    def test_load_yolo_model_nonexistent(self):
        """Test error handling for nonexistent model."""
        with pytest.raises(FileNotFoundError):
            load_yolo_model("/nonexistent/model.pt")
    
    def test_process_image_integration(self):
        """Integration test for processing a single image."""
        if not Path("data/models/plate_v8n.pt").exists():
            pytest.skip("YOLO model not available")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test image file
            test_image = create_test_image()
            input_path = temp_path / "test_image.jpg"
            cv2.imwrite(str(input_path), test_image)
            
            # Create output directory
            output_path = temp_path / "output"
            output_path.mkdir()
            
            # Load model
            model = load_yolo_model("data/models/plate_v8n.pt")
            
            # Process image (might not detect plates on synthetic image, but should not crash)
            result = process_image(model, input_path, output_path, conf_threshold=0.1)
            
            # Should return 0 or positive number (number of plates saved)
            assert isinstance(result, int)
            assert result >= 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 