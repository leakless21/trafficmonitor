import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

from src.traffic_monitor.services.VisualizationService import VisualizationService
from src.traffic_monitor.utils.custom_types import TrackedVehicleMessage
from src.traffic_monitor.utils.logging_config import setup_logging

setup_logging()

class TestVisualizationServiceCountingLines:
    def test_VisualizationService_counting_lines_initialization(self):
        """Test that counting lines are properly stored in VisualizationService config."""
        config = {
            "counting_lines": [[[0.25, 0.5], [0.75, 0.5]], [[0.125, 0.375], [0.625, 0.875]]],
            "counting_line_color": [0, 255, 255],
            "counting_line_thickness": 3
        }
        VisualizationService = VisualizationService(config)
        
        assert len(VisualizationService.counting_lines_relative) == 2
        assert VisualizationService.counting_line_color == (0, 255, 255)
        assert VisualizationService.counting_line_thickness == 3

    def test_VisualizationService_no_counting_lines(self):
        """Test VisualizationService works correctly when no counting lines are configured."""
        config = {}
        VisualizationService = VisualizationService(config)
        
        assert len(VisualizationService.counting_lines_relative) == 0
        assert VisualizationService.counting_line_color == (0, 255, 255)  # Default yellow
        assert VisualizationService.counting_line_thickness == 3  # Default thickness

    def test_draw_counting_lines(self):
        """Test that counting lines are drawn correctly on the frame."""
        config = {
            "counting_lines": [[[0.25, 0.5], [0.75, 0.5]]],  # Horizontal line at 50% height, from 25% to 75% width
            "counting_line_color": [0, 255, 0],  # Green
            "counting_line_thickness": 5
        }
        VisualizationService = VisualizationService(config)
        
        # Create a test frame
        test_frame = np.zeros((400, 500, 3), dtype=np.uint8)
        frame_width, frame_height = 500, 400
        
        # Draw counting lines
        VisualizationService._draw_counting_lines(test_frame, frame_width, frame_height)
        
        # Check that line was drawn (pixels along the line should not be black)
        # Line should be at y=200 (50% of 400), from x=125 (25% of 500) to x=375 (75% of 500)
        line_pixels = test_frame[200, 125:375]  # Check pixels along the horizontal line
        assert np.any(line_pixels != [0, 0, 0])  # Some pixels should be non-black

    def test_process_frame_with_counting_lines(self):
        """Test that counting lines are included when processing a frame."""
        config = {
            "counting_lines": [[[0.0, 0.5], [1.0, 0.5]]],  # Horizontal line across full width at 50% height
            "counting_line_color": [255, 0, 0],  # Blue
            "counting_line_thickness": 2
        }
        VisualizationService = VisualizationService(config)
        
        # Create test frame message
        test_frame = np.zeros((200, 200, 3), dtype=np.uint8)
        _, jpeg_bytes = cv2.imencode('.jpg', test_frame)
        
        frame_msg = TrackedVehicleMessage(
            frame_id="1",
            timestamp=1234567890.0,
            camera_id="test_camera",
            frame_data_jpeg=jpeg_bytes.tobytes(),
            frame_height=200,
            frame_width=200,
            tracked_objects=[]
        )
        
        # Process frame
        result_frame = VisualizationService.process_frame(frame_msg)
        
        # Check that the frame is not all black (counting line should be drawn)
        assert result_frame.shape == (200, 200, 3)
        # The counting line should add some non-black pixels
        assert np.any(result_frame != [0, 0, 0])

    def test_counting_line_color_parsing(self):
        """Test different color format parsing for counting lines."""
        # Test BGR list format
        config = {"counting_line_color": [255, 128, 0]}
        VisualizationService = VisualizationService(config)
        assert VisualizationService.counting_line_color == (255, 128, 0)
        
        # Test string format
        config = {"counting_line_color": "(0, 255, 128)"}
        VisualizationService = VisualizationService(config)
        assert VisualizationService.counting_line_color == (0, 255, 128)
        
        # Test invalid format falls back to default
        config = {"counting_line_color": "invalid"}
        VisualizationService = VisualizationService(config)
        assert VisualizationService.counting_line_color == (255, 255, 255)  # Default white

    def test_multiple_counting_lines(self):
        """Test drawing multiple counting lines."""
        config = {
            "counting_lines": [
                [[0.1, 0.125], [0.3, 0.125]],     # Horizontal line 1
                [[0.2, 0.0625], [0.2, 0.1875]],  # Vertical line
                [[0.05, 0.1875], [0.35, 0.3125]] # Diagonal line
            ],
            "counting_line_color": [0, 255, 255],
            "counting_line_thickness": 2
        }
        VisualizationService = VisualizationService(config)
        
        # Create test frame (400x500 to make calculations easier)
        test_frame = np.zeros((400, 500, 3), dtype=np.uint8)
        frame_width, frame_height = 500, 400
        
        # Draw counting lines
        VisualizationService._draw_counting_lines(test_frame, frame_width, frame_height)
        
        # Verify that lines were drawn at expected positions
        # Check horizontal line: y=50 (12.5% of 400), x=50-150 (10%-30% of 500)
        assert np.any(test_frame[50, 50:151] != [0, 0, 0])
        # Check vertical line: x=100 (20% of 500), y=25-75 (6.25%-18.75% of 400)
        assert np.any(test_frame[25:76, 100] != [0, 0, 0])
        # Check that labels were added (should be non-black pixels around midpoints)
        assert np.any(test_frame[40:60, 90:110] != [0, 0, 0])  # Around first line label

    def test_relative_coordinate_conversion(self):
        """Test that relative coordinates are properly converted to absolute coordinates."""
        config = {
            "counting_lines": [[[0.0, 0.5], [1.0, 0.5]]],  # Full width line at middle height
            "counting_line_color": [255, 255, 255],
            "counting_line_thickness": 1
        }
        VisualizationService = VisualizationService(config)
        
        # Test different frame sizes
        for width, height in [(640, 480), (1920, 1080), (320, 240)]:
            test_frame = np.zeros((height, width, 3), dtype=np.uint8)
            VisualizationService._draw_counting_lines(test_frame, width, height)
            
            # Check that line was drawn at middle height
            middle_y = height // 2
            # Line should span from x=0 to x=width-1
            line_pixels = test_frame[middle_y, :]
            assert np.any(line_pixels != [0, 0, 0])  # Line should be visible 
