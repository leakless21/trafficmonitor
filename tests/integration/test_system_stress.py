"""
System stress tests for Traffic Monitor.
Tests system behavior under high load, resource constraints, and edge conditions.
"""

import pytest
import tempfile
import shutil
import time
import threading
import multiprocessing as mp
from pathlib import Path
import sys
import psutil
import os
import cv2
import numpy as np
from unittest.mock import patch, Mock

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from traffic_monitor.utils import minidb
from traffic_monitor.utils.config_loader import load_config
from traffic_monitor.utils.queue_utils import safe_put, put_realtime, put_offline


class TestSystemStress:
    """Test system behavior under stress conditions."""
    
    def setup_method(self):
        """Set up stress test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_db_path = self.temp_dir / "stress_test.db"
        
        # Configure test database
        self.db_config = {
            "database": {
                "path": str(self.test_db_path),
                "reset_on_startup": True
            }
        }
        
        minidb.configure_database(self.db_config)
        minidb.init_db()

    def teardown_method(self):
        """Clean up stress test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.stress
    @pytest.mark.slow
    def test_high_volume_database_operations(self):
        """Test database performance under high volume operations."""
        num_operations = 10000
        batch_size = 100
        
        start_time = time.time()
        successful_operations = 0
        failed_operations = 0
        
        # Perform high volume database writes
        for i in range(0, num_operations, batch_size):
            batch_start = time.time()
            
            for j in range(batch_size):
                operation_id = i + j
                success = minidb.write_vehicle_count(
                    camera_id=f"stress_cam_{operation_id % 10}",
                    total_count=operation_id,
                    class_counts={
                        "car": operation_id % 50,
                        "truck": operation_id % 20,
                        "bus": operation_id % 10
                    }
                )
                
                if success:
                    successful_operations += 1
                else:
                    failed_operations += 1
            
            batch_time = time.time() - batch_start
            if batch_time > 5.0:  # If batch takes too long, break
                break
        
        total_time = time.time() - start_time
        operations_per_second = successful_operations / total_time
        
        # Verify performance metrics
        assert successful_operations > num_operations * 0.9, f"Too many failed operations: {failed_operations}"
        assert operations_per_second > 100, f"Database throughput too low: {operations_per_second:.2f} ops/sec"
        assert total_time < 60, f"Operations took too long: {total_time:.2f}s"

    @pytest.mark.stress
    def test_concurrent_database_access(self):
        """Test database behavior with many concurrent connections."""
        num_threads = 20
        operations_per_thread = 100
        results = []
        errors = []
        
        def database_worker(thread_id):
            """Worker function for concurrent database access."""
            thread_results = []
            thread_errors = []
            
            for i in range(operations_per_thread):
                try:
                    success = minidb.write_vehicle_count(
                        camera_id=f"thread_{thread_id}",
                        total_count=i,
                        class_counts={"car": i}
                    )
                    thread_results.append(success)
                except Exception as e:
                    thread_errors.append(e)
            
            results.extend(thread_results)
            errors.extend(thread_errors)
        
        # Start concurrent threads
        threads = []
        start_time = time.time()
        
        for thread_id in range(num_threads):
            thread = threading.Thread(target=database_worker, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        execution_time = time.time() - start_time
        
        # Verify results
        assert len(errors) < len(results) * 0.1, f"Too many errors in concurrent access: {len(errors)}"
        assert execution_time < 30, f"Concurrent operations took too long: {execution_time:.2f}s"
        
        # Verify data integrity
        counts = minidb.get_vehicle_counts(limit=1000)
        assert len(counts) > 0, "Should have data from concurrent operations"

    @pytest.mark.stress
    def test_memory_pressure_handling(self):
        """Test system behavior under memory pressure."""
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create memory pressure by processing large amounts of data
        large_data_sets = []
        max_memory_increase = 500  # MB
        
        try:
            for i in range(100):
                # Create large data structures
                large_array = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
                large_data_sets.append(large_array)
                
                # Perform database operations under memory pressure
                success = minidb.write_vehicle_count(
                    camera_id="memory_pressure_test",
                    total_count=i,
                    class_counts={"car": i}
                )
                assert success, f"Database operation should succeed under memory pressure at iteration {i}"
                
                # Check memory usage
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                if memory_increase > max_memory_increase:
                    break  # Stop before consuming too much memory
        
        finally:
            # Clean up large data structures
            large_data_sets.clear()
        
        # Verify system remained stable
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        assert memory_increase < max_memory_increase * 1.2, f"Memory usage increased too much: {memory_increase:.2f}MB"

    @pytest.mark.stress
    def test_queue_overflow_handling(self):
        """Test queue behavior under overflow conditions."""
        from multiprocessing import Queue
        
        # Test with small queue size
        small_queue = Queue(maxsize=5)
        
        # Test real-time mode (should drop messages)
        successful_puts = 0
        failed_puts = 0
        
        for i in range(20):  # Try to put more than queue capacity
            success = put_realtime(small_queue, f"message_{i}", "stress_test")
            if success:
                successful_puts += 1
            else:
                failed_puts += 1
        
        # In real-time mode, some messages should be dropped to prevent blocking
        assert successful_puts > 0, "Some messages should be successfully queued"
        assert small_queue.qsize() <= 5, "Queue should not exceed maximum size"
        
        # Test offline mode (should block but not lose messages)
        offline_queue = Queue(maxsize=3)
        
        # Fill queue to capacity
        for i in range(3):
            success = put_offline(offline_queue, f"offline_msg_{i}", "stress_test")
            assert success, f"Offline put {i} should succeed"
        
        # Queue should be at capacity
        assert offline_queue.qsize() == 3, "Offline queue should be at capacity"

    @pytest.mark.stress
    def test_long_running_stability(self):
        """Test system stability over extended periods."""
        duration_seconds = 30  # 30 seconds for testing
        operation_interval = 0.1  # 100ms between operations
        
        start_time = time.time()
        operations_completed = 0
        errors_encountered = 0
        
        while time.time() - start_time < duration_seconds:
            try:
                # Perform various operations
                success = minidb.write_vehicle_count(
                    camera_id="stability_test",
                    total_count=operations_completed,
                    class_counts={"car": operations_completed % 100}
                )
                
                if success:
                    operations_completed += 1
                else:
                    errors_encountered += 1
                
                # Query data periodically
                if operations_completed % 10 == 0:
                    counts = minidb.get_vehicle_counts(camera_id="stability_test", limit=5)
                    assert isinstance(counts, list), "Query should return list"
                
                time.sleep(operation_interval)
                
            except Exception as e:
                errors_encountered += 1
                if errors_encountered > 10:  # Too many errors, abort
                    break
        
        actual_duration = time.time() - start_time
        operations_per_second = operations_completed / actual_duration
        error_rate = errors_encountered / max(1, operations_completed + errors_encountered)
        
        # Verify stability metrics
        assert operations_completed > 100, f"Should complete many operations: {operations_completed}"
        assert error_rate < 0.05, f"Error rate too high: {error_rate:.2%}"
        assert operations_per_second > 5, f"Throughput too low: {operations_per_second:.2f} ops/sec"

    @pytest.mark.stress
    def test_resource_exhaustion_recovery(self):
        """Test system recovery from resource exhaustion."""
        # Test file descriptor exhaustion simulation
        file_handles = []
        max_files = 100  # Reasonable limit for testing
        
        try:
            # Open many temporary files to simulate FD exhaustion
            for i in range(max_files):
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                file_handles.append(temp_file)
                
                # Try database operation with limited resources
                if i % 10 == 0:  # Test every 10 files
                    success = minidb.write_vehicle_count(
                        camera_id="resource_test",
                        total_count=i,
                        class_counts={"car": i}
                    )
                    # Should still work even with many open files
                    assert success, f"Database should work with {i} open files"
        
        finally:
            # Clean up file handles
            for temp_file in file_handles:
                try:
                    temp_file.close()
                    os.unlink(temp_file.name)
                except:
                    pass  # Ignore cleanup errors
        
        # Verify system recovered
        success = minidb.write_vehicle_count(
            camera_id="recovery_test",
            total_count=1,
            class_counts={"car": 1}
        )
        assert success, "System should recover after resource cleanup"

    @pytest.mark.stress
    def test_configuration_stress(self):
        """Test configuration loading under stress."""
        config_path = self.temp_dir / "stress_config.yaml"
        
        # Create large configuration
        large_config = {
            "frame_grabber": {"video_source": "test.mp4"},
            "loguru": {"level": "INFO"}
        }
        
        # Add many configuration sections
        for i in range(1000):
            large_config[f"section_{i}"] = {
                f"key_{j}": f"value_{i}_{j}" for j in range(10)
            }
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(large_config, f)
        
        # Test repeated loading
        load_times = []
        for i in range(50):
            start_time = time.time()
            config = load_config(config_path)
            load_time = time.time() - start_time
            load_times.append(load_time)
            
            assert config is not None, f"Config should load on iteration {i}"
            assert len(config) > 1000, "Should load all configuration sections"
        
        avg_load_time = sum(load_times) / len(load_times)
        max_load_time = max(load_times)
        
        assert avg_load_time < 1.0, f"Average config load time too slow: {avg_load_time:.3f}s"
        assert max_load_time < 2.0, f"Maximum config load time too slow: {max_load_time:.3f}s"

    @pytest.mark.stress
    def test_multiprocess_coordination(self):
        """Test coordination between multiple processes under stress."""
        from multiprocessing import Queue, Event, Process
        
        num_processes = 4
        messages_per_process = 100
        
        # Shared resources
        message_queue = Queue()
        coordination_event = Event()
        results = []
        
        def worker_process(process_id, queue, event):
            """Worker process for stress testing."""
            messages_sent = 0
            
            # Wait for coordination signal
            event.wait()
            
            # Send messages rapidly
            for i in range(messages_per_process):
                message = {
                    "process_id": process_id,
                    "message_id": i,
                    "timestamp": time.time()
                }
                
                success = safe_put(queue, message, offline_mode=False, service_name=f"StressWorker{process_id}")
                if success:
                    messages_sent += 1
            
            return messages_sent
        
        # Start worker processes
        processes = []
        for i in range(num_processes):
            process = Process(target=worker_process, args=(i, message_queue, coordination_event))
            processes.append(process)
            process.start()
        
        # Coordinate simultaneous start
        time.sleep(0.1)  # Let processes initialize
        coordination_event.set()
        
        # Collect messages
        messages_received = 0
        timeout = time.time() + 10  # 10 second timeout
        
        while time.time() < timeout and messages_received < num_processes * messages_per_process:
            try:
                message = message_queue.get(timeout=1)
                if message:
                    messages_received += 1
            except:
                break
        
        # Wait for processes to complete
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
        
        # Verify coordination worked
        expected_messages = num_processes * messages_per_process
        message_loss_rate = (expected_messages - messages_received) / expected_messages
        
        assert message_loss_rate < 0.1, f"Too many messages lost: {message_loss_rate:.2%}"
        assert messages_received > expected_messages * 0.8, f"Too few messages received: {messages_received}/{expected_messages}"

    @pytest.mark.stress
    def test_error_cascade_prevention(self):
        """Test that errors in one component don't cascade to others."""
        # Simulate error in one component
        error_count = 0
        success_count = 0
        
        for i in range(100):
            try:
                if i % 10 == 0:  # Simulate 10% error rate
                    # Force an error condition
                    result = minidb.write_vehicle_count(
                        camera_id=None,  # Invalid camera_id
                        total_count=i,
                        class_counts={"car": i}
                    )
                    if not result:
                        error_count += 1
                else:
                    # Normal operation
                    result = minidb.write_vehicle_count(
                        camera_id="cascade_test",
                        total_count=i,
                        class_counts={"car": i}
                    )
                    if result:
                        success_count += 1
                    else:
                        error_count += 1
                        
            except Exception:
                error_count += 1
        
        # Verify error isolation
        total_operations = success_count + error_count
        error_rate = error_count / total_operations
        
        assert success_count > 80, f"Too few successful operations: {success_count}"
        assert error_rate < 0.2, f"Error rate too high: {error_rate:.2%}"
        
        # Verify system still functional after errors
        final_test = minidb.write_vehicle_count(
            camera_id="final_test",
            total_count=1,
            class_counts={"car": 1}
        )
        assert final_test, "System should remain functional after error cascade test"

    # Helper methods
    def _monitor_system_resources(self, duration_seconds=10):
        """Monitor system resources during test execution."""
        process = psutil.Process(os.getpid())
        
        resource_samples = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            sample = {
                "timestamp": time.time(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "cpu_percent": process.cpu_percent(),
                "open_files": len(process.open_files()) if hasattr(process, 'open_files') else 0
            }
            resource_samples.append(sample)
            time.sleep(0.5)
        
        return resource_samples

    def _create_stress_test_data(self, size_mb=10):
        """Create test data of specified size for stress testing."""
        # Create array that uses approximately size_mb megabytes
        elements = (size_mb * 1024 * 1024) // 4  # 4 bytes per int32
        return np.random.randint(0, 1000, elements, dtype=np.int32)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "stress"])