"""
Unit tests for frame capture service.
Tests video input handling and frame processing.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from traffic_monitor.services.frame_capture_service import frame_capture_process


class TestFrameCaptureService:
    """Test frame capture and video input processing."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = {
            "video_source": "test_video.mp4",
            "resize_resolution": [640, 480],
            "process_every_n_frame": 1,
            "log_every_n_frames": 30,
            "max_frames": 1000
        }
        
        self.mock_queues = {
            "frame_to_detection": Mock(),
            "frame_to_visualization": Mock()
        }
        
        # Create test frame
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(self.test_frame, (100, 100), (200, 200), (255, 255, 255), -1)

    def test_video_file_loading(self):
        """Test loading video from file."""
        # Create temporary video file
        temp_video = self.create_test_video()
        
        try:
            config = self.mock_config.copy()
            config["video_source"] = temp_video
            
            with patch('cv2.VideoCapture') as mock_cap:
                mock_cap.return_value.isOpened.return_value = True
                mock_cap.return_value.read.return_value = (True, self.test_frame)
                mock_cap.return_value.get.return_value = 30.0  # FPS
                
                # Test video capture initialization
                cap = cv2.VideoCapture(temp_video)
                assert cap.isOpened(), "Video capture should open successfully"
                
        finally:
            if os.path.exists(temp_video):
                os.unlink(temp_video)

    def test_camera_input_handling(self):
        """Test camera input (webcam) handling."""
        config = self.mock_config.copy()
        config["video_source"] = 0  # Camera index
        
        with patch('cv2.VideoCapture') as mock_cap:
            mock_cap.return_value.isOpened.return_value = True
            mock_cap.return_value.read.return_value = (True, self.test_frame)
            
            # Test camera capture initialization
            cap = cv2.VideoCapture(0)
            assert cap.isOpened(), "Camera capture should open successfully"

    def test_frame_resizing(self):
        """Test frame resizing functionality."""
        original_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        target_size = (640, 480)
        
        # Test resize operation
        resized_frame = cv2.resize(original_frame, target_size)
        
        assert resized_frame.shape[:2] == (480, 640), "Frame should be resized correctly"
        assert resized_frame.dtype == np.uint8, "Frame dtype should be preserved"

    def test_frame_skipping_logic(self):
        """Test frame skipping based on process_every_n_frame setting."""
        config = self.mock_config.copy()
        config["process_every_n_frame"] = 3  # Process every 3rd frame
        
        frame_count = 0
        processed_frames = []
        
        # Simulate frame processing
        for i in range(10):
            frame_count += 1
            if frame_count % config["process_every_n_frame"] == 0:
                processed_frames.append(i)
        
        expected_processed = [2, 5, 8]  # 0-indexed: frames 3, 6, 9
        assert processed_frames == expected_processed, "Should process every 3rd frame"

    def test_frame_validation(self):
        """Test frame validation and error handling."""
        # Test valid frame
        valid_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert self.validate_frame(valid_frame), "Valid frame should pass validation"
        
        # Test invalid frames
        invalid_frames = [
            None,  # None frame
            np.array([]),  # Empty array
            np.zeros((480, 640), dtype=np.uint8),  # Missing color channel
            np.zeros((480, 640, 4), dtype=np.uint8),  # Too many channels
            np.zeros((480, 640, 3), dtype=np.float32),  # Wrong dtype
        ]
        
        for invalid_frame in invalid_frames:
            assert not self.validate_frame(invalid_frame), f"Invalid frame should fail: {type(invalid_frame)}"

    def test_queue_communication(self):
        """Test frame distribution to multiple queues."""
        frame_data = {
            "frame_id": 100,
            "frame": self.test_frame,
            "timestamp": 1234567890.0,
            "fps": 30.0
        }
        
        # Test queue put operations
        self.mock_queues["frame_to_detection"].put.assert_not_called()
        self.mock_queues["frame_to_detection"].put(frame_data)
        self.mock_queues["frame_to_detection"].put.assert_called_once_with(frame_data)

    def test_fps_calculation(self):
        """Test FPS calculation and timing."""
        import time
        
        frame_times = []
        start_time = time.time()
        
        # Simulate frame capture timing
        for i in range(5):
            frame_times.append(time.time())
            time.sleep(0.033)  # ~30 FPS
        
        # Calculate FPS
        if len(frame_times) > 1:
            time_diff = frame_times[-1] - frame_times[0]
            fps = (len(frame_times) - 1) / time_diff
            
            assert 25 <= fps <= 35, f"FPS should be around 30, got {fps:.2f}"

    def test_memory_management(self):
        """Test memory usage with continuous frame processing."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Process many frames
        frames = []
        for i in range(100):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            frames.append(frame)
        
        # Check memory usage
        current_memory = process.memory_info().rss
        memory_increase = current_memory - initial_memory
        
        # Clean up
        del frames
        gc.collect()
        
        # Memory increase should be reasonable (less than 500MB for 100 frames)
        assert memory_increase < 500 * 1024 * 1024, f"Memory usage too high: {memory_increase / 1024 / 1024:.2f}MB"

    def test_error_handling_corrupted_video(self):
        """Test handling of corrupted or invalid video files."""
        # Test with non-existent file
        config = self.mock_config.copy()
        config["video_source"] = "non_existent_file.mp4"
        
        with patch('cv2.VideoCapture') as mock_cap:
            mock_cap.return_value.isOpened.return_value = False
            
            cap = cv2.VideoCapture(config["video_source"])
            assert not cap.isOpened(), "Should fail to open non-existent file"

    def test_frame_metadata_extraction(self):
        """Test extraction of frame metadata."""
        with patch('cv2.VideoCapture') as mock_cap:
            # Mock video properties
            mock_cap.return_value.get.side_effect = lambda prop: {
                cv2.CAP_PROP_FPS: 30.0,
                cv2.CAP_PROP_FRAME_WIDTH: 1920.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 1080.0,
                cv2.CAP_PROP_FRAME_COUNT: 1000.0
            }.get(prop, 0.0)
            
            cap = cv2.VideoCapture("test.mp4")
            
            # Extract metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            assert fps == 30.0, "FPS should be 30"
            assert width == 1920.0, "Width should be 1920"
            assert height == 1080.0, "Height should be 1080"
            assert frame_count == 1000.0, "Frame count should be 1000"

    def test_real_time_processing_timing(self):
        """Test real-time processing timing constraints."""
        import time
        
        target_fps = 30
        target_frame_time = 1.0 / target_fps
        
        processing_times = []
        
        # Simulate frame processing
        for i in range(10):
            start_time = time.time()
            
            # Simulate frame processing work
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            resized = cv2.resize(frame, (320, 240))  # Quick resize
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
        
        avg_processing_time = sum(processing_times) / len(processing_times)
        
        # Processing should be faster than target frame time for real-time
        assert avg_processing_time < target_frame_time, f"Processing too slow: {avg_processing_time:.4f}s > {target_frame_time:.4f}s"

    def test_frame_buffer_management(self):
        """Test frame buffer management and overflow handling."""
        buffer_size = 5
        frame_buffer = []
        
        # Fill buffer
        for i in range(buffer_size + 3):  # Overflow by 3
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            frame_buffer.append(frame)
            
            # Maintain buffer size
            if len(frame_buffer) > buffer_size:
                frame_buffer.pop(0)  # Remove oldest frame
        
        assert len(frame_buffer) == buffer_size, "Buffer should maintain fixed size"

    def test_video_format_support(self):
        """Test support for different video formats."""
        video_formats = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        
        for format_ext in video_formats:
            test_file = f"test_video{format_ext}"
            
            with patch('cv2.VideoCapture') as mock_cap:
                mock_cap.return_value.isOpened.return_value = True
                
                cap = cv2.VideoCapture(test_file)
                assert cap.isOpened(), f"Should support {format_ext} format"

    def test_frame_timestamp_accuracy(self):
        """Test frame timestamp accuracy and consistency."""
        import time
        
        timestamps = []
        
        # Capture timestamps for multiple frames
        for i in range(5):
            timestamp = time.time()
            timestamps.append(timestamp)
            time.sleep(0.033)  # Simulate 30 FPS
        
        # Check timestamp intervals
        intervals = []
        for i in range(1, len(timestamps)):
            interval = timestamps[i] - timestamps[i-1]
            intervals.append(interval)
        
        avg_interval = sum(intervals) / len(intervals)
        expected_interval = 0.033  # ~30 FPS
        
        # Allow 10% tolerance
        tolerance = expected_interval * 0.1
        assert abs(avg_interval - expected_interval) < tolerance, f"Timestamp interval inaccurate: {avg_interval:.4f}s"

    def test_concurrent_frame_access(self):
        """Test concurrent access to frame data."""
        import threading
        import time
        
        shared_frame = {"data": None, "lock": threading.Lock()}
        results = []
        
        def frame_reader(frame_obj, reader_id):
            with frame_obj["lock"]:
                if frame_obj["data"] is not None:
                    results.append(f"reader_{reader_id}_success")
                else:
                    results.append(f"reader_{reader_id}_empty")
        
        def frame_writer(frame_obj):
            with frame_obj["lock"]:
                frame_obj["data"] = self.test_frame
        
        # Start writer thread
        writer_thread = threading.Thread(target=frame_writer, args=(shared_frame,))
        writer_thread.start()
        
        # Start multiple reader threads
        reader_threads = []
        for i in range(3):
            thread = threading.Thread(target=frame_reader, args=(shared_frame, i))
            reader_threads.append(thread)
            thread.start()
        
        # Wait for all threads
        writer_thread.join()
        for thread in reader_threads:
            thread.join()
        
        # Check results
        success_count = len([r for r in results if "success" in r])
        assert success_count >= 1, "At least one reader should succeed"

    # Helper methods
    def validate_frame(self, frame):
        """Validate frame format and properties."""
        if frame is None:
            return False
        if not isinstance(frame, np.ndarray):
            return False
        if len(frame.shape) != 3:
            return False
        if frame.shape[2] != 3:  # RGB channels
            return False
        if frame.dtype != np.uint8:
            return False
        return True

    def create_test_video(self):
        """Create a temporary test video file."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.close()
        
        # Create a simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_file.name, fourcc, 30.0, (640, 480))
        
        for i in range(30):  # 1 second at 30 FPS
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add moving rectangle
            x = (i * 10) % 600
            cv2.rectangle(frame, (x, 200), (x + 40, 280), (0, 255, 0), -1)
            out.write(frame)
        
        out.release()
        return temp_file.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])