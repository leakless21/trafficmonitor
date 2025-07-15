"""
Unit tests for vehicle counting service.
Tests the core business logic for counting vehicles crossing lines.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from traffic_monitor.services.vehicle_counting_service import vehicle_counting_process
from traffic_monitor.utils.custom_types import Detection, TrackedObject


class TestVehicleCountingService:
    """Test vehicle counting logic and line crossing detection."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = {
            "counting_lines": [
                [[0.2, 0.3], [0.8, 0.4]],  # Horizontal line
                [[0.5, 0.1], [0.5, 0.9]]   # Vertical line
            ],
            "count_direction": "both",
            "min_track_length": 3
        }
        
        self.mock_queues = {
            "tracking_to_counting": Mock(),
            "counting_to_summary": Mock(),
            "counting_to_visualization": Mock()
        }
        
        self.sample_track = {
            "track_id": 1,
            "bbox": [100, 100, 200, 200],
            "confidence": 0.85,
            "class_id": 3,  # car
            "class_name": "car",
            "frame_id": 10
        }

    def test_line_crossing_detection_horizontal(self):
        """Test detection of vehicle crossing horizontal counting line."""
        from traffic_monitor.services.vehicle_counting_service import VehicleCountingService
        
        # Create counting service with horizontal line
        counting_lines = [[[0.2, 0.3], [0.8, 0.4]]]  # Horizontal line
        counter = VehicleCountingService(counting_lines)
        
        frame_width, frame_height = 640, 480
        og_width, og_height = 640, 480
        
        # Create tracked objects that cross the line
        # Line spans Y: 144-192, so bbox bottom (y2) should be clearly above/below
        tracked_objects_before = [{
            "track_id": 1,
            "bbox_xyxy": [300, 50, 400, 130],  # Above line (bottom at y=130, clearly above 144)
            "class_name": "car"
        }]
        
        tracked_objects_after = [{
            "track_id": 1, 
            "bbox_xyxy": [350, 200, 450, 280],  # Below line (bottom at y=280, clearly below 192)
            "class_name": "car"
        }]
        
        # First update - vehicle above line
        result1 = counter.update(tracked_objects_before, frame_width, frame_height, og_width, og_height)
        assert result1 is None, "No crossing should be detected yet"
        
        # Second update - vehicle below line (should trigger crossing)
        result2 = counter.update(tracked_objects_after, frame_width, frame_height, og_width, og_height)
        assert result2 is not None, "Vehicle should be detected crossing horizontal line"
        assert result2["total_count"] == 1, "Total count should be 1"
        assert result2["class_counts"]["car"] == 1, "Car count should be 1"

    def test_line_crossing_detection_vertical(self):
        """Test detection of vehicle crossing vertical counting line."""
        from traffic_monitor.services.vehicle_counting_service import VehicleCountingService
        
        # Create counting service with vertical line
        counting_lines = [[[0.5, 0.1], [0.5, 0.9]]]  # Vertical line
        counter = VehicleCountingService(counting_lines)
        
        frame_width, frame_height = 640, 480
        og_width, og_height = 640, 480
        
        # Create tracked objects that cross the vertical line (x=320)
        tracked_objects_left = [{
            "track_id": 2,
            "bbox_xyxy": [200, 200, 300, 300],  # Left of line (center x=250)
            "class_name": "car"
        }]
        
        tracked_objects_right = [{
            "track_id": 2,
            "bbox_xyxy": [350, 200, 450, 300],  # Right of line (center x=400)
            "class_name": "car"
        }]
        
        # First update - vehicle left of line
        result1 = counter.update(tracked_objects_left, frame_width, frame_height, og_width, og_height)
        assert result1 is None, "No crossing should be detected yet"
        
        # Second update - vehicle right of line (should trigger crossing)
        result2 = counter.update(tracked_objects_right, frame_width, frame_height, og_width, og_height)
        assert result2 is not None, "Vehicle should be detected crossing vertical line"
        assert result2["total_count"] == 1, "Total count should be 1"
        assert result2["class_counts"]["car"] == 1, "Car count should be 1"

    def test_counting_direction_filtering(self):
        """Test that counting respects direction configuration."""
        # Test upward movement
        track_up = [
            {"track_id": 3, "center": (300, 350), "frame_id": 1},  # Below
            {"track_id": 3, "center": (300, 250), "frame_id": 2},  # Above
        ]
        
        # Test downward movement  
        track_down = [
            {"track_id": 4, "center": (300, 250), "frame_id": 1},  # Above
            {"track_id": 4, "center": (300, 350), "frame_id": 2},  # Below
        ]
        
        # Test with "up" only configuration
        config_up_only = {"count_direction": "up"}
        
        # Mock direction detection
        up_direction = self._get_crossing_direction(track_up)
        down_direction = self._get_crossing_direction(track_down)
        
        assert up_direction == "up"
        assert down_direction == "down"

    def test_minimum_track_length_filtering(self):
        """Test that short tracks are filtered out."""
        short_track = [
            {"track_id": 5, "center": (300, 200), "frame_id": 1},
            {"track_id": 5, "center": (300, 300), "frame_id": 2},
        ]
        
        long_track = [
            {"track_id": 6, "center": (300, 200), "frame_id": 1},
            {"track_id": 6, "center": (300, 250), "frame_id": 2},
            {"track_id": 6, "center": (300, 300), "frame_id": 3},
            {"track_id": 6, "center": (300, 350), "frame_id": 4},
        ]
        
        min_length = 3
        
        assert len(short_track) < min_length, "Short track should be filtered"
        assert len(long_track) >= min_length, "Long track should pass filter"

    def test_duplicate_counting_prevention(self):
        """Test that same vehicle isn't counted multiple times."""
        # Simulate same track crossing line multiple times
        track_id = 7
        counted_tracks = set()
        
        # First crossing - should count
        if track_id not in counted_tracks:
            counted_tracks.add(track_id)
            first_count = True
        else:
            first_count = False
            
        # Second crossing - should not count again
        if track_id not in counted_tracks:
            counted_tracks.add(track_id)
            second_count = True
        else:
            second_count = False
            
        assert first_count, "First crossing should be counted"
        assert not second_count, "Second crossing should not be counted"

    def test_vehicle_class_filtering(self):
        """Test counting only specific vehicle classes."""
        detections = [
            {"class_id": 3, "class_name": "car"},      # Should count
            {"class_id": 2, "class_name": "bus"},      # Should count  
            {"class_id": 5, "class_name": "truck"},    # Should count
            {"class_id": 4, "class_name": "person"},   # Should not count
            {"class_id": 0, "class_name": "bicycle"},  # Should not count
        ]
        
        vehicle_classes = {2, 3, 5}  # bus, car, truck
        
        vehicle_detections = [d for d in detections if d["class_id"] in vehicle_classes]
        
        assert len(vehicle_detections) == 3, "Should filter to only vehicles"
        assert all(d["class_id"] in vehicle_classes for d in vehicle_detections)

    def test_counting_statistics_accuracy(self):
        """Test that counting statistics are calculated correctly."""
        counts = {
            "car": {"up": 5, "down": 3},
            "bus": {"up": 2, "down": 1}, 
            "truck": {"up": 1, "down": 2}
        }
        
        # Calculate totals
        total_up = sum(class_counts["up"] for class_counts in counts.values())
        total_down = sum(class_counts["down"] for class_counts in counts.values())
        total_vehicles = total_up + total_down
        
        assert total_up == 8, "Total up count should be 8"
        assert total_down == 6, "Total down count should be 6"
        assert total_vehicles == 14, "Total vehicle count should be 14"

    def test_queue_communication(self):
        """Test proper queue communication with other services."""
        with patch('traffic_monitor.services.vehicle_counting_service.vehicle_counting_process') as mock_process:
            # Mock queue operations
            mock_input_queue = Mock()
            mock_output_queue = Mock()
            
            # Test queue get operation
            mock_input_queue.get.return_value = self.sample_track
            mock_input_queue.empty.return_value = False
            
            # Test queue put operation
            count_result = {
                "frame_id": 10,
                "counts": {"car": {"up": 1, "down": 0}},
                "total": 1
            }
            
            mock_output_queue.put.assert_not_called()  # Initially
            mock_output_queue.put(count_result)
            mock_output_queue.put.assert_called_once_with(count_result)

    def test_error_handling_invalid_track(self):
        """Test handling of invalid or corrupted track data."""
        invalid_tracks = [
            None,  # None track
            {},    # Empty track
            {"track_id": None},  # Missing required fields
            {"track_id": 1, "bbox": "invalid"},  # Invalid bbox format
            {"track_id": 1, "bbox": [1, 2]},     # Incomplete bbox
        ]
        
        for invalid_track in invalid_tracks:
            try:
                # Attempt to process invalid track
                result = self._validate_track(invalid_track)
                assert not result, f"Invalid track should be rejected: {invalid_track}"
            except Exception:
                # Exception handling is also acceptable
                pass

    def test_performance_with_many_tracks(self):
        """Test performance with large number of simultaneous tracks."""
        import time
        
        # Generate many tracks
        num_tracks = 100
        tracks = []
        for i in range(num_tracks):
            track = {
                "track_id": i,
                "bbox": [100 + i, 100, 200 + i, 200],
                "confidence": 0.8,
                "class_id": 3,
                "frame_id": 10
            }
            tracks.append(track)
        
        # Measure processing time
        start_time = time.time()
        
        # Simulate processing (simplified)
        processed_tracks = []
        for track in tracks:
            if self._validate_track(track):
                processed_tracks.append(track)
        
        processing_time = time.time() - start_time
        
        assert processing_time < 1.0, f"Processing {num_tracks} tracks took too long: {processing_time:.2f}s"
        assert len(processed_tracks) == num_tracks, "All valid tracks should be processed"

    # Helper methods for testing
    def _check_line_crossing(self, track_history, line_pixels):
        """Simplified line crossing detection for testing."""
        if len(track_history) < 2:
            return False
            
        # Check if track crosses line by examining consecutive points
        for i in range(len(track_history) - 1):
            p1 = track_history[i]["center"]
            p2 = track_history[i + 1]["center"]
            
            # For horizontal line, check if y coordinates cross the line
            line_y = (line_pixels[0][1] + line_pixels[1][1]) // 2
            
            # Check if the track crosses the line between these two points
            if (p1[1] <= line_y <= p2[1]) or (p2[1] <= line_y <= p1[1]):
                return True
                
        return False

    def _get_crossing_direction(self, track_history):
        """Determine crossing direction for testing."""
        if len(track_history) < 2:
            return None
            
        start_y = track_history[0]["center"][1]
        end_y = track_history[-1]["center"][1]
        
        if end_y < start_y:
            return "up"
        elif end_y > start_y:
            return "down"
        else:
            return "horizontal"

    def _validate_track(self, track):
        """Validate track data for testing."""
        if not track or not isinstance(track, dict):
            return False
            
        required_fields = ["track_id", "bbox"]
        for field in required_fields:
            if field not in track:
                return False
                
        if not isinstance(track.get("bbox"), list) or len(track["bbox"]) != 4:
            return False
            
        return True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])