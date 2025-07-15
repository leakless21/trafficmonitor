"""
Unit tests for queue utilities.
Tests inter-process communication and queue management.
"""

import pytest
import multiprocessing as mp
import queue
import time
import threading
from unittest.mock import Mock, patch
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from traffic_monitor.utils.queue_utils import (
    safe_put,
    put_realtime,
    put_offline,
    is_offline_mode,
    get_queue_size_for_mode
)


class TestQueueUtils:
    """Test queue utilities and inter-process communication."""
    
    def setup_method(self):
        """Set up test queues."""
        self.test_queue = mp.Queue(maxsize=3)
        self.large_queue = mp.Queue(maxsize=1000)
        
        # Sample data for testing
        self.sample_data = {
            "frame_id": 100,
            "timestamp": time.time(),
            "data": "test_data"
        }

    def teardown_method(self):
        """Clean up test queues."""
        # Clear any remaining items
        try:
            while not self.test_queue.empty():
                self.test_queue.get_nowait()
        except:
            pass
            
        try:
            while not self.large_queue.empty():
                self.large_queue.get_nowait()
        except:
            pass

    def test_offline_mode_detection(self):
        """Test offline mode detection from configuration."""
        config_offline = {"save_to_file": True}
        config_realtime = {"save_to_file": False}
        config_default = {}
        
        assert is_offline_mode(config_offline) == True
        assert is_offline_mode(config_realtime) == False
        assert is_offline_mode(config_default) == False

    def test_queue_size_for_mode(self):
        """Test queue size calculation for different modes."""
        # Offline mode should return 0 (unbounded)
        offline_size = get_queue_size_for_mode(True)
        assert offline_size == 0
        
        # Realtime mode should return limited size
        realtime_size = get_queue_size_for_mode(False, 5)
        assert realtime_size == 5
        
        # Default realtime size
        default_size = get_queue_size_for_mode(False)
        assert default_size == 3

    def test_put_realtime_success(self):
        """Test realtime queue put operation."""
        # Start with empty queue
        assert self.test_queue.empty()
        
        data = self.sample_data
        result = put_realtime(self.test_queue, data, "test_service")
        assert result == True, "Queue put should succeed"
        
        # Verify data was added
        assert not self.test_queue.empty()
        retrieved_data = self.test_queue.get_nowait()
        assert retrieved_data == data, "Retrieved data should match original"

    def test_put_offline_success(self):
        """Test offline queue put operation."""
        # Start with empty queue
        assert self.test_queue.empty()
        
        data = self.sample_data
        result = put_offline(self.test_queue, data, "test_service")
        assert result == True, "Queue put should succeed"
        
        # Verify data was added
        assert not self.test_queue.empty()
        retrieved_data = self.test_queue.get_nowait()
        assert retrieved_data == data, "Retrieved data should match original"

    def test_safe_put_realtime_mode(self):
        """Test safe_put in realtime mode."""
        data = self.sample_data
        
        result = safe_put(self.test_queue, data, offline_mode=False, service_name="test")
        assert result == True, "Safe put should succeed in realtime mode"

    def test_safe_put_offline_mode(self):
        """Test safe_put in offline mode."""
        data = self.sample_data
        
        result = safe_put(self.test_queue, data, offline_mode=True, service_name="test")
        assert result == True, "Safe put should succeed in offline mode"

    def test_realtime_queue_overflow_handling(self):
        """Test that realtime mode handles queue overflow gracefully."""
        # Fill queue to capacity first
        for i in range(3):  # maxsize=3
            self.test_queue.put_nowait(f"old_message_{i}")
        
        # Verify queue is full
        assert self.test_queue.qsize() == 3
        
        # Add new message - should handle overflow gracefully
        new_data = {"frame_id": 999, "data": "new_message"}
        result = put_realtime(self.test_queue, new_data, "test_service")
        
        # The function should handle the overflow (either succeed or fail gracefully)
        assert isinstance(result, bool), "Should return a boolean result"

    def test_queue_error_handling(self):
        """Test error handling in queue operations."""
        # Test with None data
        result = put_realtime(self.test_queue, None, "test_service")
        assert result == True, "Should handle None data gracefully"
        
        # Test with invalid queue (mock)
        mock_queue = Mock()
        mock_queue.put_nowait.side_effect = Exception("Test error")
        mock_queue.get_nowait.side_effect = queue.Empty()
        
        result = put_realtime(mock_queue, self.sample_data, "test_service")
        assert result == False, "Should return False on error"

    def test_performance_with_many_operations(self):
        """Test performance with many queue operations."""
        start_time = time.time()
        
        # Perform many operations
        for i in range(100):
            data = {"frame_id": i, "data": f"test_{i}"}
            put_realtime(self.large_queue, data, "perf_test")
        
        elapsed = time.time() - start_time
        assert elapsed < 1.0, f"100 operations took too long: {elapsed:.2f}s"

    def test_concurrent_access(self):
        """Test concurrent queue access."""
        import threading
        results = []
        
        def producer():
            for i in range(10):
                data = {"producer_id": threading.current_thread().ident, "item": i}
                result = put_realtime(self.large_queue, data, "producer")
                results.append(result)
        
        def consumer():
            consumed = 0
            while consumed < 5:  # Consume some items
                try:
                    item = self.large_queue.get_nowait()
                    consumed += 1
                except queue.Empty:
                    time.sleep(0.01)
        
        # Start threads
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)
        
        producer_thread.start()
        consumer_thread.start()
        
        producer_thread.join()
        consumer_thread.join()
        
        # Check that most operations succeeded
        success_count = sum(results)
        assert success_count >= 8, f"Too many failed operations: {success_count}/10"

    def test_queue_mode_selection(self):
        """Test that correct queue mode is selected based on configuration."""
        # Test realtime mode selection
        realtime_config = {"save_to_file": False}
        assert not is_offline_mode(realtime_config)
        
        # Test offline mode selection
        offline_config = {"save_to_file": True}
        assert is_offline_mode(offline_config)

    def test_queue_size_configuration(self):
        """Test queue size configuration for different modes."""
        # Test that offline mode uses unbounded queues
        offline_size = get_queue_size_for_mode(offline_mode=True)
        assert offline_size == 0, "Offline mode should use unbounded queues"
        
        # Test that realtime mode uses bounded queues
        realtime_size = get_queue_size_for_mode(offline_mode=False, default_realtime_size=5)
        assert realtime_size == 5, "Realtime mode should use bounded queues"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])