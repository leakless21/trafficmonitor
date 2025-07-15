"""
Pytest configuration and shared fixtures for Traffic Monitor tests.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_frame():
    """Create a sample video frame for testing."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some content to make it more realistic
    cv2.rectangle(frame, (100, 100), (200, 200), (255, 255, 255), -1)
    cv2.putText(frame, "TEST", (120, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return frame


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return {
        "frame_grabber": {
            "video_source": "test_video.mp4",
            "resize_resolution": [640, 480],
            "process_every_n_frame": 1
        },
        "vehicle_detector": {
            "model_path": "test_model.engine",
            "conf_threshold": 0.5
        },
        "loguru": {
            "level": "DEBUG",
            "terminal_output_enabled": False
        }
    }


@pytest.fixture
def mock_detection():
    """Create a mock vehicle detection result."""
    return {
        "bbox": [100, 100, 200, 200],
        "confidence": 0.85,
        "class_id": 3,  # car
        "class_name": "car"
    }


@pytest.fixture
def mock_queue():
    """Create a mock queue for testing."""
    return Mock()