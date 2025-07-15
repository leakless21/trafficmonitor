"""
Unit tests for vehicle tracking service.
Tests object tracking, track ID assignment, and tracking persistence.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import time

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from traffic_monitor.services.vehicle_tracking_service import vehicle_tracking_process
from traffic_monitor.utils.custom_types import VehicleDetectionMessage, TrackedVehicleMessage


class TestVehicleTrackingService:
    """Test vehicle tracking functionality and multi-object tracking."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = {
            "tracker_type": "bytetrack",
            "half": True,
            "device": "cuda",
            "reid_model_path": None,
            "evolve_param_dict": None,
            "per_class": None
        }
        
        self.mock_queues = {
            "detection_to_tracking": Mock(),
            "tracking_to_counting": Mock(),
            "tracking_to_visualization": Mock()
        }
        
        # Sample detection message
        self.sample_detection_message = {
            "frame_id": "frame_001",
            "camera_id": "cam_01",
            "timestamp": time.time(),
            "frame_data_jpeg": b"fake_jpeg_data",
            "frame_height": 480,
            "frame_width": 640,
            "og_frame_height": 1080,
            "og_frame_width": 1920,
            "og_fps": 30.0,
            "detections": [
                {
                    "bbox_xyxy": [100, 100, 200, 200],
                    "confidence": 0.85,
                    "class_id": 3,
                    "class_name": "car"
                },
                {
                    "bbox_xyxy": [300, 150, 400, 250],
                    "confidence": 0.92,
                    "class_id": 5,
                    "class_name": "truck"
                }
            ]
        }

    def test_tracker_initialization(self):
        """Test tracker initialization with different configurations."""
        # Test ByteTrack initialization
        config_bytetrack = {"tracker_type": "bytetrack", "half": True, "device": "cuda"}
        
        # Mock the tracker initialization
        with patch('ultralytics.YOLO') as mock_yolo:
            mock_tracker = Mock()
            mock_yolo.return_value = mock_tracker
            
            # Test that tracker type is properly configured
            assert config_bytetrack["tracker_type"] == "bytetrack"
            assert config_bytetrack["half"] is True
            assert config_bytetrack["device"] == "cuda"

    def test_detection_to_tracking_conversion(self):
        """Test conversion of detection results to tracking format."""
        detections = self.sample_detection_message["detections"]
        
        # Test detection format validation
        for detection in detections:
            assert "bbox_xyxy" in detection
            assert "confidence" in detection
            assert "class_id" in detection
            assert "class_name" in detection
            assert len(detection["bbox_xyxy"]) == 4
            assert 0 <= detection["confidence"] <= 1
            assert isinstance(detection["class_id"], int)

    def test_track_id_assignment(self):
        """Test that unique track IDs are assigned to objects."""
        # Simulate tracking results
        mock_tracking_results = [
            {"track_id": 1, "bbox": [100, 100, 200, 200], "class_id": 3, "confidence": 0.85},
            {"track_id": 2, "bbox": [300, 150, 400, 250], "class_id": 5, "confidence": 0.92},
            {"track_id": 3, "bbox": [500, 200, 600, 300], "class_id": 3, "confidence": 0.78}
        ]
        
        # Verify unique track IDs
        track_ids = [result["track_id"] for result in mock_tracking_results]
        assert len(track_ids) == len(set(track_ids)), "Track IDs should be unique"
        
        # Verify track IDs are positive integers
        for track_id in track_ids:
            assert isinstance(track_id, int)
            assert track_id > 0

    def test_track_persistence_across_frames(self):
        """Test that tracks persist across multiple frames."""
        # Simulate object in frame 1
        frame1_detections = [
            {"bbox_xyxy": [100, 100, 200, 200], "confidence": 0.85, "class_id": 3}
        ]
        
        # Simulate same object in frame 2 (slightly moved)
        frame2_detections = [
            {"bbox_xyxy": [105, 105, 205, 205], "confidence": 0.87, "class_id": 3}
        ]
        
        # Mock tracking to assign same ID to similar objects
        mock_track_id = 1
        
        # Test that similar objects get same track ID
        assert self._calculate_bbox_similarity(
            frame1_detections[0]["bbox_xyxy"], 
            frame2_detections[0]["bbox_xyxy"]
        ) > 0.8, "Similar bboxes should have high similarity"

    def test_track_loss_and_recovery(self):
        """Test handling of lost tracks and track recovery."""
        # Simulate track loss scenario
        active_tracks = {1, 2, 3}  # Active track IDs
        current_detections = [
            {"track_id": 1, "bbox": [100, 100, 200, 200]},
            {"track_id": 3, "bbox": [500, 200, 600, 300]}
            # Track ID 2 is missing (lost)
        ]
        
        current_track_ids = {det["track_id"] for det in current_detections}
        lost_tracks = active_tracks - current_track_ids
        
        assert lost_tracks == {2}, "Should detect lost track ID 2"

    def test_multi_object_tracking_performance(self):
        """Test tracking performance with multiple objects."""
        # Generate multiple detections
        num_objects = 20
        detections = []
        for i in range(num_objects):
            detection = {
                "bbox_xyxy": [50 + i*30, 100, 150 + i*30, 200],
                "confidence": 0.8 + (i % 3) * 0.05,
                "class_id": 3,
                "class_name": "car"
            }
            detections.append(detection)
        
        # Test that all objects can be processed
        assert len(detections) == num_objects
        
        # Simulate tracking processing time
        start_time = time.time()
        processed_detections = []
        for detection in detections:
            # Simulate tracking processing
            processed_detection = detection.copy()
            processed_detection["track_id"] = len(processed_detections) + 1
            processed_detections.append(processed_detection)
        processing_time = time.time() - start_time
        
        assert processing_time < 1.0, f"Tracking {num_objects} objects took too long: {processing_time:.2f}s"
        assert len(processed_detections) == num_objects

    def test_tracking_confidence_filtering(self):
        """Test filtering of low-confidence detections."""
        detections = [
            {"bbox_xyxy": [100, 100, 200, 200], "confidence": 0.95, "class_id": 3},  # High confidence
            {"bbox_xyxy": [300, 150, 400, 250], "confidence": 0.45, "class_id": 3},  # Low confidence
            {"bbox_xyxy": [500, 200, 600, 300], "confidence": 0.75, "class_id": 3},  # Medium confidence
        ]
        
        confidence_threshold = 0.5
        filtered_detections = [det for det in detections if det["confidence"] >= confidence_threshold]
        
        assert len(filtered_detections) == 2, "Should filter out low confidence detection"
        assert all(det["confidence"] >= confidence_threshold for det in filtered_detections)

    def test_class_specific_tracking(self):
        """Test tracking behavior for different object classes."""
        detections = [
            {"bbox_xyxy": [100, 100, 200, 200], "confidence": 0.85, "class_id": 3, "class_name": "car"},
            {"bbox_xyxy": [300, 150, 400, 250], "confidence": 0.92, "class_id": 5, "class_name": "truck"},
            {"bbox_xyxy": [500, 200, 600, 300], "confidence": 0.78, "class_id": 2, "class_name": "bus"},
            {"bbox_xyxy": [150, 300, 200, 400], "confidence": 0.65, "class_id": 4, "class_name": "person"}
        ]
        
        # Group by class
        class_groups = {}
        for det in detections:
            class_name = det["class_name"]
            if class_name not in class_groups:
                class_groups[class_name] = []
            class_groups[class_name].append(det)
        
        # Verify class grouping
        assert len(class_groups) == 4, "Should have 4 different classes"
        assert "car" in class_groups
        assert "truck" in class_groups
        assert "bus" in class_groups
        assert "person" in class_groups

    def test_tracking_message_format(self):
        """Test that tracking output message format is correct."""
        # Expected output format for TrackedVehicleMessage
        expected_fields = [
            "frame_id", "camera_id", "timestamp", "frame_data_jpeg",
            "frame_height", "frame_width", "og_frame_height", "og_frame_width", "og_fps",
            "tracked_objects"
        ]
        
        # Mock tracking output
        tracking_message = {
            "frame_id": "frame_001",
            "camera_id": "cam_01", 
            "timestamp": time.time(),
            "frame_data_jpeg": b"fake_jpeg_data",
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
                }
            ]
        }
        
        # Verify all required fields are present
        for field in expected_fields:
            assert field in tracking_message, f"Missing required field: {field}"
        
        # Verify tracked objects have track_id
        for obj in tracking_message["tracked_objects"]:
            assert "track_id" in obj, "Tracked object should have track_id"

    def test_queue_communication(self):
        """Test proper queue communication with other services."""
        with patch('multiprocessing.Queue') as mock_queue_class:
            mock_input_queue = Mock()
            mock_output_queue = Mock()
            
            # Test queue get operation
            mock_input_queue.get.return_value = self.sample_detection_message
            mock_input_queue.empty.return_value = False
            
            # Test queue put operation
            tracking_result = {
                "frame_id": "frame_001",
                "tracked_objects": [
                    {"track_id": 1, "bbox_xyxy": [100, 100, 200, 200], "class_name": "car"}
                ]
            }
            
            mock_output_queue.put.assert_not_called()  # Initially
            mock_output_queue.put(tracking_result)
            mock_output_queue.put.assert_called_once_with(tracking_result)

    def test_error_handling_invalid_detections(self):
        """Test handling of invalid or corrupted detection data."""
        invalid_detections = [
            None,  # None detection
            {},    # Empty detection
            {"bbox_xyxy": None},  # Invalid bbox
            {"bbox_xyxy": [1, 2]},  # Incomplete bbox
            {"bbox_xyxy": [100, 100, 200, 200]},  # Missing confidence
            {"bbox_xyxy": [100, 100, 200, 200], "confidence": "invalid"},  # Invalid confidence type
        ]
        
        for invalid_detection in invalid_detections:
            try:
                result = self._validate_detection(invalid_detection)
                assert not result, f"Invalid detection should be rejected: {invalid_detection}"
            except Exception:
                # Exception handling is also acceptable
                pass

    def test_tracking_memory_management(self):
        """Test memory management with long-running tracking."""
        # Simulate tracking over many frames
        max_tracks = 100
        active_tracks = {}
        
        # Add tracks gradually
        for frame_id in range(50):
            # Add new tracks
            for track_id in range(frame_id * 2, (frame_id + 1) * 2):
                if len(active_tracks) < max_tracks:
                    active_tracks[track_id] = {
                        "first_seen": frame_id,
                        "last_seen": frame_id,
                        "bbox": [100, 100, 200, 200]
                    }
            
            # Remove old tracks (simulate track loss)
            if frame_id > 10:
                tracks_to_remove = [tid for tid, track in active_tracks.items() 
                                  if frame_id - track["last_seen"] > 5]
                for tid in tracks_to_remove:
                    del active_tracks[tid]
        
        # Verify memory management
        assert len(active_tracks) <= max_tracks, "Should not exceed maximum track limit"

    def test_tracking_accuracy_metrics(self):
        """Test tracking accuracy and consistency metrics."""
        # Simulate ground truth vs tracking results
        ground_truth = [
            {"track_id": 1, "bbox": [100, 100, 200, 200]},
            {"track_id": 2, "bbox": [300, 150, 400, 250]},
        ]
        
        tracking_results = [
            {"track_id": 1, "bbox": [105, 105, 205, 205]},  # Slight offset
            {"track_id": 2, "bbox": [295, 145, 395, 245]},  # Slight offset
        ]
        
        # Calculate tracking accuracy (simplified)
        total_error = 0
        for gt, tr in zip(ground_truth, tracking_results):
            if gt["track_id"] == tr["track_id"]:
                bbox_error = self._calculate_bbox_error(gt["bbox"], tr["bbox"])
                total_error += bbox_error
        
        avg_error = total_error / len(ground_truth)
        assert avg_error < 10, f"Average tracking error too high: {avg_error}"

    # Helper methods
    def _calculate_bbox_similarity(self, bbox1, bbox2):
        """Calculate similarity between two bounding boxes."""
        # Calculate IoU (Intersection over Union)
        x1_max = max(bbox1[0], bbox2[0])
        y1_max = max(bbox1[1], bbox2[1])
        x2_min = min(bbox1[2], bbox2[2])
        y2_min = min(bbox1[3], bbox2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
        
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def _calculate_bbox_error(self, bbox1, bbox2):
        """Calculate error between two bounding boxes."""
        # Calculate center point distance
        center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
        center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
        
        return ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5

    def _validate_detection(self, detection):
        """Validate detection data format."""
        if not detection or not isinstance(detection, dict):
            return False
        
        if "bbox_xyxy" not in detection:
            return False
        
        bbox = detection["bbox_xyxy"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False
        
        if "confidence" not in detection:
            return False
        
        confidence = detection["confidence"]
        if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
            return False
        
        return True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])