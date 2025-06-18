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
        counting_lines = [[[0, 750], [1920, 750]]]
        counter = Counter(counting_lines)
        assert len(counter.counting_lines) == 1
        assert counter.vehicle_last_positions == {}
        assert counter.counted_track_ids == set()
        assert counter.counts == {}

    def test_counter_vehicle_crossing(self):
        counting_lines = [[[100, 100], [200, 100]]]
        counter = Counter(counting_lines)
        
        # First frame - vehicle above line
        tracked_objects_1 = [{
            "track_id": 1,
            "bbox_xyxy": [140, 50, 160, 70],  # Center at (150, 70)
            "class_name": "car"
        }]
        result = counter.update(tracked_objects_1)
        assert result is None  # No crossing yet
        
        # Second frame - vehicle below line (crossed)
        tracked_objects_2 = [{
            "track_id": 1,
            "bbox_xyxy": [140, 120, 160, 140],  # Center at (150, 140)
            "class_name": "car"
        }]
        result = counter.update(tracked_objects_2)
        assert result is not None
        assert result["total_count"] == 1
        assert result["count_by_class"]["car"] == 1

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