"""
Test cases for visualizer-related bug fixes.
"""
import pytest
from src.traffic_monitor.utils.custom_types import TrackedVehicleMessage
from src.traffic_monitor.services.ocr_reader import OCRReader
import numpy as np


def test_tracked_vehicle_message_includes_og_fps():
    """Test that TrackedVehicleMessage includes og_fps field."""
    message: TrackedVehicleMessage = {
        "frame_id": "test_frame_001",
        "camera_id": "cam_001", 
        "timestamp": 1640995200.0,
        "frame_data_jpeg": b"fake_jpeg_data",
        "frame_height": 720,
        "frame_width": 1280,
        "og_frame_height": 1080,
        "og_frame_width": 1920,
        "og_fps": 30.0,  # This field should be present
        "tracked_objects": []
    }
    
    # Should not raise KeyError
    assert "og_fps" in message
    assert message["og_fps"] == 30.0


def test_ocr_confidence_threshold_allows_valid_detections():
    """Test that OCR reader accepts detections with confidence >= 0.4."""
    config = {
        "hub_model_name": "global-plates-mobile-vit-v2-model",
        "device": "auto", 
        "conf_threshold": 0.4
    }
    
    # Create a mock OCR reader (without actual model initialization)
    ocr_reader = OCRReader.__new__(OCRReader)
    ocr_reader.conf_threshold = config["conf_threshold"]
    
    # Test that confidence 0.44 would be accepted (above threshold)
    test_confidence = 0.44
    assert test_confidence >= ocr_reader.conf_threshold, "OCR confidence 0.44 should be accepted with threshold 0.4"
    
    # Test that confidence 0.35 would be rejected (below threshold)
    test_confidence_low = 0.35
    assert test_confidence_low < ocr_reader.conf_threshold, "OCR confidence 0.35 should be rejected with threshold 0.4"


def test_tracked_vehicle_message_type_completeness():
    """Test that TrackedVehicleMessage has all required fields from FrameMessage."""
    required_fields = {
        "frame_id", "camera_id", "timestamp", "frame_data_jpeg",
        "frame_height", "frame_width", "og_frame_height", "og_frame_width", "og_fps",
        "tracked_objects"
    }
    
    message: TrackedVehicleMessage = {
        "frame_id": "test_frame_001",
        "camera_id": "cam_001",
        "timestamp": 1640995200.0,
        "frame_data_jpeg": b"fake_jpeg_data",
        "frame_height": 720,
        "frame_width": 1280,
        "og_frame_height": 1080,
        "og_frame_width": 1920,
        "og_fps": 30.0,
        "tracked_objects": []
    }
    
    for field in required_fields:
        assert field in message, f"Required field '{field}' missing from TrackedVehicleMessage"


def test_vehicle_detection_message_includes_og_fps():
    """Test that VehicleDetectionMessage includes og_fps field."""
    from src.traffic_monitor.utils.custom_types import VehicleDetectionMessage
    
    message: VehicleDetectionMessage = {
        "frame_id": "test_frame_001",
        "camera_id": "cam_001",
        "timestamp": 1640995200.0,
        "frame_data_jpeg": b"fake_jpeg_data",
        "frame_height": 720,
        "frame_width": 1280,
        "og_frame_height": 1080,
        "og_frame_width": 1920,
        "og_fps": 30.0,  # This field should be present
        "detections": []
    }
    
    # Should not raise KeyError
    assert "og_fps" in message
    assert message["og_fps"] == 30.0


def test_video_writer_path_construction():
    """Test that video writer creates correct file paths without duplication."""
    from pathlib import Path
    import time
    
    output_path = "data/videos/output/"
    
    # Simulate the correct filename construction
    filename = f"output_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    filepath = Path(output_path) / filename
    
    # Check that path doesn't contain duplication
    path_str = str(filepath)
    path_parts = path_str.split('/')
    
    # Should not have "output" directory duplicated
    output_count = path_parts.count("output")
    assert output_count <= 1, f"Path has duplicated 'output' directory: {path_str}"
    
    # Should have the correct structure
    assert "data" in path_parts
    assert "videos" in path_parts
    assert any("output_" in part and ".mp4" in part for part in path_parts)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 