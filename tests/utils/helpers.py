"""
Test helper functions and utilities.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List


def create_test_video(
    filename: str, 
    duration: int = 2, 
    fps: int = 10,
    resolution: Tuple[int, int] = (640, 480)
) -> None:
    """
    Create a simple test video with moving objects.
    
    Args:
        filename: Output video filename
        duration: Video duration in seconds
        fps: Frames per second
        resolution: Video resolution (width, height)
    """
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, resolution)
    
    width, height = resolution
    
    for frame_num in range(duration * fps):
        # Create a frame with a moving rectangle (simulating a vehicle)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Moving rectangle across the frame
        x = (frame_num * 10) % (width - 40)
        y = height // 2 - 40
        cv2.rectangle(frame, (x, y), (x + 40, y + 80), (0, 255, 0), -1)
        
        # Add some text (simulating a license plate)
        cv2.rectangle(frame, (x + 5, y + 60), (x + 35, y + 75), (255, 255, 255), -1)
        cv2.putText(frame, "ABC123", (x + 7, y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        out.write(frame)
    
    out.release()


def create_test_image(
    filename: str,
    resolution: Tuple[int, int] = (640, 480),
    add_vehicle: bool = True,
    add_plate: bool = True
) -> None:
    """
    Create a test image with optional vehicle and license plate.
    
    Args:
        filename: Output image filename
        resolution: Image resolution (width, height)
        add_vehicle: Whether to add a vehicle-like rectangle
        add_plate: Whether to add a license plate-like text
    """
    width, height = resolution
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    if add_vehicle:
        # Add vehicle-like rectangle
        cv2.rectangle(image, (200, 150), (400, 300), (100, 100, 255), -1)
        
        if add_plate:
            # Add license plate
            cv2.rectangle(image, (250, 250), (350, 280), (255, 255, 255), -1)
            cv2.putText(image, "TEST123", (255, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    cv2.imwrite(filename, image)


def assert_detection_valid(detection: dict) -> None:
    """
    Assert that a detection result has valid structure and values.
    
    Args:
        detection: Detection dictionary to validate
    """
    required_keys = ["bbox", "confidence", "class_id"]
    for key in required_keys:
        assert key in detection, f"Detection missing required key: {key}"
    
    bbox = detection["bbox"]
    assert len(bbox) == 4, "Bbox should have 4 coordinates"
    assert all(isinstance(coord, (int, float)) for coord in bbox), "Bbox coordinates should be numeric"
    
    confidence = detection["confidence"]
    assert 0.0 <= confidence <= 1.0, "Confidence should be between 0 and 1"
    
    class_id = detection["class_id"]
    assert isinstance(class_id, int), "Class ID should be an integer"
    assert class_id >= 0, "Class ID should be non-negative"


def cleanup_test_files(file_patterns: List[str]) -> None:
    """
    Clean up test files matching given patterns.
    
    Args:
        file_patterns: List of file patterns to delete
    """
    for pattern in file_patterns:
        for file_path in Path(".").glob(pattern):
            if file_path.exists():
                file_path.unlink()