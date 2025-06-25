import pytest
import multiprocessing as mp
import time
from queue import Empty
from unittest.mock import patch, MagicMock

from src.traffic_monitor.services.vehicle_counter import vehicle_counter_process, Counter
from src.traffic_monitor.utils.custom_types import TrackedVehicleMessage, VehicleCountMessage
from src.traffic_monitor.utils.logging_config import setup_logging

setup_logging()

class TestCounter:
    def test_counter_initialization(self):
        counting_lines = [[[0.0, 0.694], [1.0, 0.694]]]  # Relative coordinates
        counter = Counter(counting_lines)
        assert len(counter.counting_lines_relative) == 1
        assert counter.vehicle_last_positions == {}
        assert counter.counted_track_ids == set()
        assert counter.counts == {}

    def test_counter_vehicle_crossing(self):
        counting_lines = [[[0.25, 0.25], [0.5, 0.25]]]  # Relative horizontal line at 25% height
        counter = Counter(counting_lines)
        frame_width, frame_height = 400, 400  # Square frame for easier testing
        
        # First frame - vehicle above line (at 20% height)
        tracked_objects_1 = [{
            "track_id": 1,
            "bbox_xyxy": [140, 60, 160, 80],  # Center at (150, 80) - above 25% line at y=100
            "class_name": "car"
        }]
        result = counter.update(tracked_objects_1, frame_width, frame_height)
        assert result is None  # No crossing yet
        
        # Second frame - vehicle below line (crossed) at 30% height
        tracked_objects_2 = [{
            "track_id": 1,
            "bbox_xyxy": [140, 100, 160, 120],  # Center at (150, 120) - below 25% line at y=100
            "class_name": "car"
        }]
        result = counter.update(tracked_objects_2, frame_width, frame_height)
        assert result is not None
        assert result["total_count"] == 1
        assert result["class_counts"]["car"] == 1

    def test_vehicle_counter_process_config(self):
        """Test vehicle counter process initialization with relative coordinates."""
        config = {
            "counting_lines": [[[0.0, 0.694], [1.0, 0.694]]],  # Relative coordinates
            "loguru": {
                "level": "DEBUG"
            }
        }
        
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        shutdown_event = mp.Event()
        
        # Create test message with frame dimensions
        test_message = TrackedVehicleMessage(
            frame_id="test_001",
            timestamp=time.time(),
            camera_id="test_camera",
            frame_data_jpeg=b"fake_jpeg_data",
            frame_height=1080,
            frame_width=1920,
            tracked_objects=[{
                "track_id": 1,
                "bbox_xyxy": [100, 500, 200, 600],
                "confidence": 0.9,
                "class_id": 2,
                "class_name": "car"
            }]
        )
        
        input_queue.put(test_message)
        input_queue.put(None)  # Signal to stop
        
        # This should not crash and should process the message
        vehicle_counter_process(config, input_queue, output_queue, shutdown_event)
        
        # Check that no exceptions were raised (process completed)
        assert True

    def test_multiple_vehicles_crossing(self):
        """Test counting multiple vehicles crossing a relative line."""
        counting_lines = [[[0.0, 0.5], [1.0, 0.5]]]  # Horizontal line at 50% height
        counter = Counter(counting_lines)
        frame_width, frame_height = 640, 480
        
        # Frame 1: Two vehicles above the line
        tracked_objects_1 = [
            {
                "track_id": 1,
                "bbox_xyxy": [100, 180, 140, 220],  # Center at (120, 220) - above 50% line at y=240
                "class_name": "car"
            },
            {
                "track_id": 2,
                "bbox_xyxy": [300, 160, 340, 200],  # Center at (320, 200) - above 50% line at y=240
                "class_name": "truck"
            }
        ]
        result = counter.update(tracked_objects_1, frame_width, frame_height)
        assert result is None  # No crossings yet
        
        # Frame 2: Both vehicles cross the line
        tracked_objects_2 = [
            {
                "track_id": 1,
                "bbox_xyxy": [100, 260, 140, 300],  # Center at (120, 300) - below 50% line at y=240
                "class_name": "car"
            },
            {
                "track_id": 2,
                "bbox_xyxy": [300, 260, 340, 300],  # Center at (320, 300) - below 50% line at y=240
                "class_name": "truck"
            }
        ]
        result = counter.update(tracked_objects_2, frame_width, frame_height)
        assert result is not None
        assert result["total_count"] == 2
        assert result["class_counts"]["car"] == 1
        assert result["class_counts"]["truck"] == 1

    def test_relative_coordinates_different_resolutions(self):
        """Test that relative coordinates work correctly across different frame resolutions."""
        counting_lines = [[[0.0, 0.5], [1.0, 0.5]]]  # Horizontal line at 50% height
        counter = Counter(counting_lines)
        
        # Test with different resolutions
        test_cases = [
            (640, 480),   # 4:3 VGA
            (1920, 1080), # 16:9 Full HD
            (1280, 720),  # 16:9 HD
            (320, 240)    # Small resolution
        ]
        
        for frame_width, frame_height in test_cases:
            counter_test = Counter(counting_lines)
            mid_height = frame_height // 2
            
            # Vehicle starts above the middle line
            tracked_objects_1 = [{
                "track_id": 1,
                "bbox_xyxy": [frame_width//4, mid_height-50, frame_width//4+40, mid_height-10],
                "class_name": "car"
            }]
            result = counter_test.update(tracked_objects_1, frame_width, frame_height)
            assert result is None
            
            # Vehicle crosses to below the middle line
            tracked_objects_2 = [{
                "track_id": 1,
                "bbox_xyxy": [frame_width//4, mid_height+10, frame_width//4+40, mid_height+50],
                "class_name": "car"
            }]
            result = counter_test.update(tracked_objects_2, frame_width, frame_height)
            assert result is not None
            assert result["total_count"] == 1

    def test_vehicle_counter_message_structure(self):
        """Test that VehicleCounter can handle TrackedVehicleMessage with all required fields."""
        
        # Create a complete TrackedVehicleMessage with all required fields
        test_message = {
            "frame_id": "test-frame-123",
            "camera_id": "test-camera",
            "timestamp": 1640995200.0,
            "frame_data_jpeg": b"fake_jpeg_data",
            "frame_height": 720,
            "frame_width": 1280,
            "og_frame_height": 1080,  # Original frame dimensions
            "og_frame_width": 1920,   # Original frame dimensions
            "tracked_objects": [
                {
                    "bbox_xyxy": [100, 100, 200, 200],
                    "track_id": 1,
                    "confidence": 0.9,
                    "class_id": 2,
                    "class_name": "car"
                },
                {
                    "bbox_xyxy": [300, 150, 400, 250],
                    "track_id": 2,
                    "confidence": 0.8,
                    "class_id": 1,
                    "class_name": "bicycle"
                }
            ]
        }
        
        # Test that Counter.update() can process this message structure
        counting_lines = [[[0.3, 0.3], [0.7, 0.7]]]
        counter = Counter(counting_lines)
        
        # This should not raise a KeyError for og_frame_width/og_frame_height
        try:
            result = counter.update(
                tracked_objects=test_message["tracked_objects"],
                frame_width=test_message["frame_width"],
                frame_height=test_message["frame_height"],
                og_width=test_message["og_frame_width"],
                og_height=test_message["og_frame_height"]
            )
            # Test passes if no exception is raised
            assert True, "VehicleCounter successfully processed message with og_frame dimensions"
        except KeyError as e:
            pytest.fail(f"VehicleCounter failed to process message due to missing field: {e}")

class TestVehicleCounterProcess:
    def test_vehicle_counter_process_logging_setup(self):
        """Test that vehicle_counter_process sets up logging correctly"""
        config = {
            "counting_lines": [[[0, 750], [1920, 750]]],
            "loguru": {
                "level": "DEBUG",
                "terminal_output_enabled": True
            }
        }
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        shutdown_event = mp.Event()
        
        # Mock setup_logging to verify it's called
        with patch('src.traffic_monitor.services.vehicle_counter.setup_logging') as mock_setup:
            with patch('src.traffic_monitor.services.vehicle_counter.logger') as mock_logger:
                # Add a None message to shut down the process
                input_queue.put(None)
                
                vehicle_counter_process(config, input_queue, output_queue, shutdown_event)
                
                # Verify setup_logging was called with loguru config
                mock_setup.assert_called_once_with(config["loguru"])
                
                # Verify process start logging
                mock_logger.info.assert_called()

    def test_vehicle_counter_process_handles_empty_queue(self):
        """Test that vehicle_counter_process handles empty queue gracefully"""
        config = {
            "counting_lines": [[[0, 750], [1920, 750]]],
            "loguru": {"level": "DEBUG"}
        }
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        shutdown_event = mp.Event()
        
        with patch('src.traffic_monitor.services.vehicle_counter.setup_logging'):
            with patch('src.traffic_monitor.services.vehicle_counter.logger'):
                # Set shutdown event immediately
                shutdown_event.set()
                
                # Should exit gracefully without errors
                vehicle_counter_process(config, input_queue, output_queue, shutdown_event)

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 