"""
Test resolution scaling for counting lines in vehicle counter and visualizer.
"""

import pytest
import numpy as np
import time
from typing import List, Union
from unittest.mock import MagicMock
from src.traffic_monitor.services.vehicle_counter import Counter
from src.traffic_monitor.services.visualizer import Visualizer
from src.traffic_monitor.utils.custom_types import VehicleCountMessage


class TestResolutionScaling:
    
    def test_vehicle_counter_absolute_coordinates(self):
        """Test vehicle counter with absolute coordinates and resolution change."""
        # Original video resolution: 1920x1080
        # Resized resolution: 1280x720
        # Counting line originally at y=750 (70% of 1080)
        
        counting_lines_coords: List[List[List[Union[int, float]]]] = [[[0, 750], [1920, 750]]]  # Absolute coordinates
        counter = Counter(counting_lines_coords)
        
        # Test with original resolution
        tracked_objects = [
            {
                "track_id": 1,
                "bbox_xyxy": [960, 740, 1000, 780],  # Vehicle crossing line at y=750
                "class_name": "car"
            }
        ]
        
        # Update with original dimensions
        result = counter.update(tracked_objects, 1080, 1920)
        
        # Verify normalization happened
        assert counter.relative_lines is not None
        assert len(counter.relative_lines) == 1
        
        # Check relative coordinates (750/1080 ≈ 0.694)
        relative_line = counter.relative_lines[0]
        assert abs(relative_line[0][1] - 0.6944444444444444) < 0.001  # 750/1080
        assert abs(relative_line[1][1] - 0.6944444444444444) < 0.001
        
        # Check absolute coordinates are properly scaled back
        assert counter.counting_lines_absolute is not None
        assert len(counter.counting_lines_absolute) == 1
    
    def test_vehicle_counter_relative_coordinates(self):
        """Test vehicle counter with relative coordinates."""
        # Relative coordinates (70% height, full width)
        counting_lines_coords: List[List[List[Union[int, float]]]] = [[[0.0, 0.7], [1.0, 0.7]]]
        counter = Counter(counting_lines_coords)
        
        tracked_objects = [
            {
                "track_id": 1,
                "bbox_xyxy": [640, 495, 680, 505],  # Vehicle crossing line at 70% of 720 = 504
                "class_name": "car"
            }
        ]
        
        # Update with 1280x720 resolution
        result = counter.update(tracked_objects, 720, 1280)
        
        # Verify relative coordinates are preserved
        assert counter.relative_lines is not None
        assert counter.relative_lines[0][0][1] == 0.7
        assert counter.relative_lines[0][1][1] == 0.7

    def test_visualizer_receives_counting_lines(self):
        """Test visualizer receives counting line information from VehicleCounter."""
        config = {
            "font": "FONT_HERSHEY_SIMPLEX",
            "class_colors": {}
        }
        
        visualizer = Visualizer(config)
        
        # Test that visualizer receives counting line info from VehicleCountMessage
        count_message: VehicleCountMessage = {
            "camera_id": "test",
            "timestamp": time.time(),
            "total_count": 1,
            "class_counts": {"car": 1},
            "counting_lines_absolute": [[[0, 500], [1280, 500]]],  # Scaled coordinates
            "line_display_color": [0, 255, 0],  # Green
            "line_thickness": 3
        }
        
        # Before update - no counting lines
        assert visualizer.counting_lines_absolute is None
        
        # Update with VehicleCountMessage
        visualizer.update_counting_lines(count_message)
        
        # After update - should have counting lines
        assert visualizer.counting_lines_absolute is not None
        assert visualizer.counting_lines_absolute == [[[0, 500], [1280, 500]]]
        assert visualizer.line_color == [0, 255, 0]
        assert visualizer.line_thickness == 3

    def test_edge_cases(self):
        """Test edge cases for coordinate normalization."""
        # Test empty counting lines
        counter = Counter([])
        result = counter.update([], 720, 1280)
        assert result is None
        
        # Test visualizer with empty counting line message
        config = {
            "font": "FONT_HERSHEY_SIMPLEX",
            "class_colors": {}
        }
        
        visualizer = Visualizer(config)
        count_message: VehicleCountMessage = {
            "camera_id": "test",
            "timestamp": time.time(),
            "total_count": 0,
            "class_counts": {},
            "counting_lines_absolute": None,
            "line_display_color": None,
            "line_thickness": None
        }
        
        # Should handle None values gracefully
        visualizer.update_counting_lines(count_message)
        assert visualizer.counting_lines_absolute is None

if __name__ == "__main__":
    pytest.main([__file__]) 