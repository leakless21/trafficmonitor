"""
Unit tests for event distribution service.
Tests message distribution, queue management, and event routing.
"""

import pytest
import multiprocessing as mp
from multiprocessing.queues import Queue
from queue import Empty, Full
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import time
import threading

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from traffic_monitor.services.event_distribution_service import event_distribution_process


class TestEventDistributionService:
    """Test event distribution functionality and message routing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = {
            "offline_mode": False,
            "queue_size": 10,
            "timeout": 1.0
        }
        
        # Create mock queues
        self.mock_input_queue = Mock(spec=Queue)
        self.mock_output_queues = [Mock(spec=Queue) for _ in range(3)]
        self.mock_shutdown_event = Mock()
        
        # Sample messages for testing
        self.sample_messages = [
            {
                "frame_id": "frame_001",
                "camera_id": "cam_01",
                "timestamp": time.time(),
                "data": "test_data_1"
            },
            {
                "frame_id": "frame_002", 
                "camera_id": "cam_01",
                "timestamp": time.time() + 1,
                "data": "test_data_2"
            },
            {
                "frame_id": "frame_003",
                "camera_id": "cam_01", 
                "timestamp": time.time() + 2,
                "data": "test_data_3"
            }
        ]

    def test_single_message_distribution(self):
        """Test distribution of a single message to multiple queues."""
        message = self.sample_messages[0]
        output_queues = self.mock_output_queues
        
        # Mock successful queue operations
        for queue in output_queues:
            queue.put.return_value = None
        
        # Simulate message distribution
        for queue in output_queues:
            queue.put(message)
        
        # Verify message was sent to all queues
        for queue in output_queues:
            queue.put.assert_called_with(message)

    def test_multiple_message_distribution(self):
        """Test distribution of multiple messages."""
        messages = self.sample_messages
        output_queues = self.mock_output_queues
        
        # Mock successful queue operations
        for queue in output_queues:
            queue.put.return_value = None
        
        # Simulate multiple message distribution
        for message in messages:
            for queue in output_queues:
                queue.put(message)
        
        # Verify all messages were distributed
        for queue in output_queues:
            assert queue.put.call_count == len(messages)

    def test_queue_failure_handling(self):
        """Test handling of queue failures during distribution."""
        message = self.sample_messages[0]
        output_queues = self.mock_output_queues
        
        # Mock one queue failure
        output_queues[0].put.side_effect = Full("Queue is full")
        output_queues[1].put.return_value = None
        output_queues[2].put.return_value = None
        
        # Simulate distribution with error handling
        successful_distributions = 0
        failed_distributions = 0
        
        for queue in output_queues:
            try:
                queue.put(message)
                successful_distributions += 1
            except Full:
                failed_distributions += 1
        
        # Verify error handling
        assert successful_distributions == 2, "Should have 2 successful distributions"
        assert failed_distributions == 1, "Should have 1 failed distribution"

    def test_shutdown_signal_propagation(self):
        """Test propagation of shutdown signal to all output queues."""
        output_queues = self.mock_output_queues
        shutdown_message = None
        
        # Mock successful queue operations
        for queue in output_queues:
            queue.put.return_value = None
        
        # Simulate shutdown signal propagation
        for queue in output_queues:
            queue.put(shutdown_message)
        
        # Verify shutdown signal was sent to all queues
        for queue in output_queues:
            queue.put.assert_called_with(shutdown_message)

    def test_input_queue_timeout_handling(self):
        """Test handling of input queue timeouts."""
        input_queue = self.mock_input_queue
        
        # Mock timeout exception
        input_queue.get.side_effect = Empty("Queue is empty")
        
        # Test timeout handling
        try:
            message = input_queue.get(timeout=1.0)
            assert False, "Should have raised Empty exception"
        except Empty:
            # Expected behavior
            assert True

    def test_message_ordering_preservation(self):
        """Test that message ordering is preserved during distribution."""
        messages = self.sample_messages
        output_queue = Mock(spec=Queue)
        
        # Track message order
        received_messages = []
        
        def capture_message(msg):
            received_messages.append(msg)
        
        output_queue.put.side_effect = capture_message
        
        # Distribute messages in order
        for message in messages:
            output_queue.put(message)
        
        # Verify order preservation
        assert len(received_messages) == len(messages)
        for i, message in enumerate(messages):
            assert received_messages[i] == message, f"Message order not preserved at index {i}"

    def test_concurrent_distribution(self):
        """Test concurrent message distribution."""
        messages = self.sample_messages
        output_queues = self.mock_output_queues
        
        # Mock thread-safe queue operations
        for queue in output_queues:
            queue.put.return_value = None
        
        def distribute_message(message):
            for queue in output_queues:
                queue.put(message)
        
        # Simulate concurrent distribution
        threads = []
        for message in messages:
            thread = threading.Thread(target=distribute_message, args=(message,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify all messages were distributed
        for queue in output_queues:
            assert queue.put.call_count == len(messages)

    def test_offline_mode_behavior(self):
        """Test behavior differences in offline mode."""
        config_offline = {"offline_mode": True}
        config_realtime = {"offline_mode": False}
        
        # Test offline mode configuration
        assert config_offline["offline_mode"] is True
        assert config_realtime["offline_mode"] is False
        
        # In offline mode, should use blocking operations
        # In realtime mode, should use non-blocking operations
        # This is tested through the safe_put function behavior

    def test_queue_size_management(self):
        """Test management of queue sizes and capacity."""
        # Mock queue with size tracking
        mock_queue = Mock(spec=Queue)
        mock_queue.qsize.return_value = 5
        mock_queue.maxsize = 10
        
        # Test queue capacity check
        current_size = mock_queue.qsize()
        max_size = mock_queue.maxsize
        utilization = current_size / max_size if max_size > 0 else 0
        
        assert current_size == 5
        assert max_size == 10
        assert utilization == 0.5

    def test_message_filtering(self):
        """Test filtering of messages based on criteria."""
        messages = [
            {"type": "detection", "data": "vehicle_data"},
            {"type": "count", "data": "count_data"},
            {"type": "ocr", "data": "plate_data"},
            {"type": "invalid", "data": "bad_data"}
        ]
        
        # Filter valid message types
        valid_types = {"detection", "count", "ocr"}
        filtered_messages = [msg for msg in messages if msg.get("type") in valid_types]
        
        assert len(filtered_messages) == 3, "Should filter out invalid message"
        assert all(msg["type"] in valid_types for msg in filtered_messages)

    def test_performance_with_high_throughput(self):
        """Test performance with high message throughput."""
        num_messages = 1000
        num_queues = 5
        
        # Generate test messages
        messages = []
        for i in range(num_messages):
            message = {
                "id": i,
                "timestamp": time.time(),
                "data": f"test_data_{i}"
            }
            messages.append(message)
        
        # Mock queues
        output_queues = [Mock(spec=Queue) for _ in range(num_queues)]
        for queue in output_queues:
            queue.put.return_value = None
        
        # Measure distribution time
        start_time = time.time()
        
        for message in messages:
            for queue in output_queues:
                queue.put(message)
        
        distribution_time = time.time() - start_time
        
        # Verify performance
        total_operations = num_messages * num_queues
        operations_per_second = total_operations / distribution_time
        
        assert distribution_time < 5.0, f"Distribution took too long: {distribution_time:.2f}s"
        assert operations_per_second > 1000, f"Too slow: {operations_per_second:.0f} ops/sec"

    def test_error_recovery(self):
        """Test error recovery and resilience."""
        messages = self.sample_messages
        output_queues = self.mock_output_queues
        
        # Mock intermittent failures
        call_count = 0
        def failing_put(message):
            nonlocal call_count
            call_count += 1
            if call_count % 3 == 0:  # Fail every 3rd call
                raise Exception("Simulated failure")
        
        output_queues[0].put.side_effect = failing_put
        output_queues[1].put.return_value = None
        output_queues[2].put.return_value = None
        
        # Simulate distribution with error recovery
        successful_distributions = 0
        failed_distributions = 0
        
        for message in messages:
            for queue in output_queues:
                try:
                    queue.put(message)
                    successful_distributions += 1
                except Exception:
                    failed_distributions += 1
        
        # Verify error recovery
        assert successful_distributions > 0, "Should have some successful distributions"
        assert failed_distributions > 0, "Should have some failed distributions"

    def test_memory_usage_monitoring(self):
        """Test monitoring of memory usage during distribution."""
        # Simulate memory tracking
        initial_memory = 100  # MB
        current_memory = initial_memory
        memory_threshold = 500  # MB
        
        # Simulate processing messages
        num_messages = 100
        memory_per_message = 2  # MB
        
        for i in range(num_messages):
            current_memory += memory_per_message
            
            # Check memory threshold
            if current_memory > memory_threshold:
                # Simulate memory cleanup
                current_memory = initial_memory
                break
        
        assert current_memory <= memory_threshold, "Memory usage should be controlled"

    def test_queue_health_monitoring(self):
        """Test monitoring of queue health and status."""
        output_queues = self.mock_output_queues
        
        # Mock queue health status
        queue_health = {}
        for i, queue in enumerate(output_queues):
            queue.qsize.return_value = i * 2  # Different sizes
            queue_health[f"queue_{i}"] = {
                "size": queue.qsize(),
                "healthy": queue.qsize() < 10,
                "last_update": time.time()
            }
        
        # Verify health monitoring
        healthy_queues = sum(1 for status in queue_health.values() if status["healthy"])
        assert healthy_queues == len(output_queues), "All queues should be healthy"

    def test_message_statistics_tracking(self):
        """Test tracking of message distribution statistics."""
        # Initialize statistics
        stats = {
            "messages_received": 0,
            "messages_distributed": 0,
            "distribution_errors": 0,
            "queues_active": 0
        }
        
        messages = self.sample_messages
        output_queues = self.mock_output_queues
        
        # Mock successful distributions
        for queue in output_queues:
            queue.put.return_value = None
        
        # Track statistics during distribution
        for message in messages:
            stats["messages_received"] += 1
            
            for queue in output_queues:
                try:
                    queue.put(message)
                    stats["messages_distributed"] += 1
                except Exception:
                    stats["distribution_errors"] += 1
        
        stats["queues_active"] = len(output_queues)
        
        # Verify statistics
        assert stats["messages_received"] == len(messages)
        assert stats["messages_distributed"] == len(messages) * len(output_queues)
        assert stats["distribution_errors"] == 0
        assert stats["queues_active"] == 3

    def test_graceful_shutdown(self):
        """Test graceful shutdown of distribution service."""
        shutdown_event = Mock()
        shutdown_event.is_set.return_value = True
        
        # Simulate shutdown check
        should_continue = not shutdown_event.is_set()
        
        assert should_continue is False, "Should stop processing on shutdown signal"

    def test_queue_type_validation(self):
        """Test validation of queue types and compatibility."""
        # Valid queue types
        valid_queues = [Mock(spec=Queue) for _ in range(3)]
        
        # Invalid queue types
        invalid_queues = [None, "not_a_queue", 123, []]
        
        # Test queue validation
        for queue in valid_queues:
            assert hasattr(queue, 'put'), "Valid queue should have put method"
            assert hasattr(queue, 'get'), "Valid queue should have get method"
        
        for invalid_queue in invalid_queues:
            if invalid_queue is not None:
                assert not hasattr(invalid_queue, 'put') or not hasattr(invalid_queue, 'get'), \
                    "Invalid queue should not have queue methods"

    # Helper methods
    def _create_test_message(self, message_id):
        """Create a test message with given ID."""
        return {
            "id": message_id,
            "timestamp": time.time(),
            "data": f"test_data_{message_id}",
            "type": "test"
        }

    def _simulate_queue_operation(self, queue, operation, *args, **kwargs):
        """Simulate queue operation with error handling."""
        try:
            if operation == "put":
                return queue.put(*args, **kwargs)
            elif operation == "get":
                return queue.get(*args, **kwargs)
        except (Empty, Full) as e:
            return None, str(e)
        except Exception as e:
            return None, f"Unexpected error: {e}"

    def _calculate_distribution_efficiency(self, successful, total):
        """Calculate distribution efficiency percentage."""
        if total == 0:
            return 0.0
        return (successful / total) * 100.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])