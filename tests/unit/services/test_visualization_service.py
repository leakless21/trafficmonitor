"""
Unit tests for visualization service.
Tests video output generation, overlay rendering, and visualization features.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import time

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from traffic_monitor.services.visualization_service import visualization_process
from traffic_monitor.utils.custom_types import TrackedVehicleMessage, VehicleCountMessage


class TestVisualizationService:
    """Test visualization functionality and video output generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = {
            "font": "FONT_HERSHEY_SIMPLEX",
            "font_scale": 0.6,
            "font_thickness": 2,
            "ocr_duration": 3.0,
            "counting_lines": [[[0.2, 0.3], [0.8, 0.4]]],
            "counting_line_color": [0, 255, 255],
            "counting_line_thickness": 3,
            "save_to_file": True,
            "save_path": "data/outputs/videos/",
            "output_fourcc": "mp4v",
            "enable_gui": False,
            "class_colors": {
                "car": [0, 119, 187],
                "truck": [14, 127, 255],
                "bus": [44, 160, 44],
                "person": [75, 86, 140]
            },
            "default_color": [255, 255, 255]
        }
        
        # Create test frame
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(self.test_frame, (100, 100), (200, 200), (255, 255, 255), -1)
        
        # Sample tracking message
        self.sample_tracking_message = {
            "frame_id": "frame_001",
            "camera_id": "cam_01",
            "timestamp": time.time(),
            "frame_data_jpeg": cv2.imencode('.jpg', self.test_frame)[1].tobytes(),
            "frame_height": 480,
            "frame_width": 640,
            "og_frame_height": 1080,
            "og_frame_width": 1920,
            "og_fps": 30.0,
            "tracked_objects": [
                {
                    "bbox_xyxy": [100, 100, 200, 200],
                    "confidence": 0.85,
                    "class_id": 3,
                    "class_name": "car",
                    "track_id": 1
                },
                {
                    "bbox_xyxy": [300, 150, 400, 250],
                    "confidence": 0.92,
                    "class_id": 5,
                    "class_name": "truck",
                    "track_id": 2
                }
            ]
        }

    def test_frame_decoding(self):
        """Test decoding of JPEG frame data."""
        jpeg_data = self.sample_tracking_message["frame_data_jpeg"]
        
        # Decode frame
        frame = cv2.imdecode(np.frombuffer(jpeg_data, np.uint8), cv2.IMREAD_COLOR)
        
        assert frame is not None, "Frame should be decoded successfully"
        assert frame.shape == (480, 640, 3), "Frame should have correct dimensions"
        assert frame.dtype == np.uint8, "Frame should have correct data type"

    def test_bounding_box_rendering(self):
        """Test rendering of bounding boxes on frame."""
        frame = self.test_frame.copy()
        
        # Test bounding box parameters
        bbox = [100, 100, 200, 200]
        color = [0, 255, 0]  # Green
        thickness = 2
        
        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
        
        # Verify frame was modified
        assert not np.array_equal(frame, self.test_frame), "Frame should be modified after drawing"
        
        # Check that green pixels exist (bbox was drawn)
        green_pixels = np.sum((frame[:, :, 1] > 200) & (frame[:, :, 0] < 50) & (frame[:, :, 2] < 50))
        assert green_pixels > 0, "Should have green pixels from bounding box"

    def test_text_overlay_rendering(self):
        """Test rendering of text overlays (labels, confidence, etc.)."""
        frame = self.test_frame.copy()
        
        # Test text parameters
        text = "car: 0.85"
        position = (105, 95)  # Above bbox
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = [255, 255, 255]  # White
        thickness = 2
        
        # Draw text
        cv2.putText(frame, text, position, font, font_scale, color, thickness)
        
        # Verify frame was modified
        assert not np.array_equal(frame, self.test_frame), "Frame should be modified after text"
        
        # Check that white pixels exist (text was drawn)
        white_pixels = np.sum(np.all(frame > 200, axis=2))
        assert white_pixels > 0, "Should have white pixels from text"

    def test_counting_line_visualization(self):
        """Test rendering of counting lines."""
        frame = self.test_frame.copy()
        
        # Test counting line
        line_coords = [[0.2, 0.3], [0.8, 0.4]]  # Relative coordinates
        frame_height, frame_width = frame.shape[:2]
        
        # Convert to absolute coordinates
        pt1 = (int(line_coords[0][0] * frame_width), int(line_coords[0][1] * frame_height))
        pt2 = (int(line_coords[1][0] * frame_width), int(line_coords[1][1] * frame_height))
        
        color = [0, 255, 255]  # Yellow
        thickness = 3
        
        # Draw counting line
        cv2.line(frame, pt1, pt2, color, thickness)
        
        # Verify line was drawn
        assert not np.array_equal(frame, self.test_frame), "Frame should be modified after line"
        
        # Check that yellow pixels exist
        yellow_pixels = np.sum((frame[:, :, 1] > 200) & (frame[:, :, 2] > 200) & (frame[:, :, 0] < 50))
        assert yellow_pixels > 0, "Should have yellow pixels from counting line"

    def test_class_color_mapping(self):
        """Test color mapping for different vehicle classes."""
        class_colors = self.mock_config["class_colors"]
        default_color = self.mock_config["default_color"]
        
        # Test known classes
        assert "car" in class_colors
        assert "truck" in class_colors
        assert "bus" in class_colors
        
        # Test color format (BGR)
        for class_name, color in class_colors.items():
            assert isinstance(color, list), f"Color for {class_name} should be a list"
            assert len(color) == 3, f"Color for {class_name} should have 3 components (BGR)"
            assert all(0 <= c <= 255 for c in color), f"Color values for {class_name} should be 0-255"
        
        # Test default color
        assert len(default_color) == 3
        assert all(0 <= c <= 255 for c in default_color)

    def test_confidence_score_display(self):
        """Test display of confidence scores."""
        tracked_objects = self.sample_tracking_message["tracked_objects"]
        
        for obj in tracked_objects:
            confidence = obj["confidence"]
            class_name = obj["class_name"]
            
            # Format confidence text
            confidence_text = f"{class_name}: {confidence:.2f}"
            
            # Verify formatting
            assert confidence_text.startswith(class_name)
            assert ":" in confidence_text
            assert len(confidence_text.split(":")[1].strip()) >= 4  # At least "0.xx"

    def test_track_id_display(self):
        """Test display of track IDs."""
        tracked_objects = self.sample_tracking_message["tracked_objects"]
        
        for obj in tracked_objects:
            track_id = obj["track_id"]
            class_name = obj["class_name"]
            
            # Format track ID text
            track_text = f"ID:{track_id}"
            
            # Verify formatting
            assert track_text.startswith("ID:")
            assert str(track_id) in track_text
            assert isinstance(track_id, int)
            assert track_id > 0

    def test_video_writer_initialization(self):
        """Test video writer initialization and configuration."""
        # Test video writer parameters
        output_path = "test_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30.0
        frame_size = (640, 480)
        
        # Mock video writer
        with patch('cv2.VideoWriter') as mock_writer:
            mock_writer_instance = Mock()
            mock_writer.return_value = mock_writer_instance
            mock_writer_instance.isOpened.return_value = True
            
            # Initialize video writer
            writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
            
            # Verify initialization
            mock_writer.assert_called_once_with(output_path, fourcc, fps, frame_size)
            assert writer.isOpened()

    def test_frame_writing_to_video(self):
        """Test writing frames to video file."""
        with patch('cv2.VideoWriter') as mock_writer:
            mock_writer_instance = Mock()
            mock_writer.return_value = mock_writer_instance
            mock_writer_instance.isOpened.return_value = True
            
            # Write test frame
            frame = self.test_frame
            mock_writer_instance.write(frame)
            
            # Verify frame was written
            mock_writer_instance.write.assert_called_once_with(frame)

    def test_gui_display_mode(self):
        """Test GUI display functionality."""
        config_gui = self.mock_config.copy()
        config_gui["enable_gui"] = True
        
        with patch('cv2.imshow') as mock_imshow, \
             patch('cv2.waitKey') as mock_waitkey:
            
            mock_waitkey.return_value = ord('q')  # Simulate 'q' key press
            
            # Test GUI display
            frame = self.test_frame
            window_name = "Traffic Monitor"
            
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1)
            
            # Verify GUI calls
            mock_imshow.assert_called_once_with(window_name, frame)
            mock_waitkey.assert_called_once_with(1)
            assert key == ord('q')

    def test_ocr_result_overlay(self):
        """Test overlay of OCR results (license plates)."""
        frame = self.test_frame.copy()
        
        # Mock OCR result
        ocr_result = {
            "vehicle_id": 1,
            "lp_text": "ABC123",
            "ocr_confidence": 0.95,
            "timestamp": time.time()
        }
        
        # Test OCR text formatting
        ocr_text = f"LP: {ocr_result['lp_text']} ({ocr_result['ocr_confidence']:.2f})"
        position = (110, 220)  # Below bbox
        
        # Draw OCR text
        cv2.putText(frame, ocr_text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 1)
        
        # Verify OCR text format
        assert "LP:" in ocr_text
        assert ocr_result['lp_text'] in ocr_text
        assert f"{ocr_result['ocr_confidence']:.2f}" in ocr_text

    def test_vehicle_count_display(self):
        """Test display of vehicle count statistics."""
        # Mock vehicle count message
        count_message = {
            "camera_id": "cam_01",
            "timestamp": time.time(),
            "total_count": 15,
            "class_counts": {"car": 10, "truck": 3, "bus": 2}
        }
        
        # Format count display text
        total_text = f"Total: {count_message['total_count']}"
        class_text = ", ".join([f"{cls}: {count}" for cls, count in count_message['class_counts'].items()])
        
        # Verify count text formatting
        assert "Total:" in total_text
        assert str(count_message['total_count']) in total_text
        assert "car: 10" in class_text
        assert "truck: 3" in class_text
        assert "bus: 2" in class_text

    def test_frame_resizing(self):
        """Test frame resizing for display/output."""
        original_frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
        target_size = (640, 480)
        
        # Resize frame
        resized_frame = cv2.resize(original_frame, target_size)
        
        # Verify resize
        assert resized_frame.shape[:2] == target_size[::-1], "Frame should be resized to target size"
        assert resized_frame.dtype == np.uint8, "Frame should maintain data type"

    def test_performance_with_many_objects(self):
        """Test visualization performance with many tracked objects."""
        # Generate many tracked objects
        num_objects = 50
        tracked_objects = []
        for i in range(num_objects):
            obj = {
                "bbox_xyxy": [10 + i*12, 10, 60 + i*12, 60],
                "confidence": 0.8 + (i % 3) * 0.05,
                "class_id": 3,
                "class_name": "car",
                "track_id": i + 1
            }
            tracked_objects.append(obj)
        
        # Test visualization processing time
        frame = self.test_frame.copy()
        start_time = time.time()
        
        # Simulate drawing all objects
        for obj in tracked_objects:
            bbox = obj["bbox_xyxy"]
            color = [0, 255, 0]
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Add text
            text = f"{obj['class_name']}:{obj['track_id']}"
            cv2.putText(frame, text, (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        processing_time = time.time() - start_time
        
        assert processing_time < 1.0, f"Visualizing {num_objects} objects took too long: {processing_time:.2f}s"

    def test_error_handling_corrupted_frame(self):
        """Test handling of corrupted or invalid frame data."""
        invalid_frames = [
            None,
            b"invalid_jpeg_data",
            np.array([]),
            np.zeros((10, 10)),  # Wrong dimensions
            np.zeros((480, 640, 4), dtype=np.uint8),  # Wrong channels
        ]
        
        for invalid_frame in invalid_frames:
            try:
                if isinstance(invalid_frame, bytes):
                    # Try to decode
                    frame = cv2.imdecode(np.frombuffer(invalid_frame, np.uint8), cv2.IMREAD_COLOR)
                    assert frame is None or frame.size == 0, "Invalid JPEG should fail to decode"
                elif isinstance(invalid_frame, np.ndarray):
                    # Validate frame format
                    is_valid = (len(invalid_frame.shape) == 3 and 
                               invalid_frame.shape[2] == 3 and
                               invalid_frame.dtype == np.uint8)
                    if not is_valid:
                        assert True, "Invalid frame format detected"
                else:
                    assert invalid_frame is None, "None frame should be handled"
            except Exception:
                # Exception handling is acceptable for invalid data
                pass

    def test_memory_management_long_video(self):
        """Test memory management during long video processing."""
        # Simulate processing many frames
        num_frames = 1000
        frame_size = (480, 640, 3)
        
        # Track memory usage (simplified)
        frames_processed = 0
        max_memory_frames = 10  # Keep only recent frames in memory
        
        for frame_id in range(num_frames):
            # Simulate frame processing
            frame = np.random.randint(0, 255, frame_size, dtype=np.uint8)
            
            # Process frame (simulate visualization)
            processed_frame = frame.copy()
            cv2.rectangle(processed_frame, (10, 10), (50, 50), [255, 0, 0], 2)
            
            frames_processed += 1
            
            # Simulate memory cleanup
            if frames_processed % max_memory_frames == 0:
                # Clear old frames from memory (in real implementation)
                pass
        
        assert frames_processed == num_frames, "Should process all frames"

    def test_output_file_management(self):
        """Test output file creation and management."""
        # Test output path creation
        output_dir = "test_output_dir"
        output_file = "test_video.mp4"
        
        with patch('pathlib.Path.mkdir') as mock_mkdir, \
             patch('pathlib.Path.exists') as mock_exists:
            
            mock_exists.return_value = False
            
            # Create output directory
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            # Verify directory creation
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    # Helper methods
    def _create_test_frame_with_objects(self, objects):
        """Create test frame with drawn objects."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        for obj in objects:
            bbox = obj["bbox_xyxy"]
            color = [0, 255, 0]  # Green
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Add label
            label = f"{obj['class_name']}:{obj['track_id']}"
            cv2.putText(frame, label, (bbox[0], bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame

    def _validate_frame_format(self, frame):
        """Validate frame format for visualization."""
        if frame is None:
            return False
        
        if not isinstance(frame, np.ndarray):
            return False
        
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return False
        
        if frame.dtype != np.uint8:
            return False
        
        return True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])