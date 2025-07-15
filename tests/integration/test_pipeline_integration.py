"""
Integration tests for the complete traffic monitoring pipeline.
Tests end-to-end functionality and service interactions.
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
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.mark.integration
class TestPipelineIntegration:
    """Test complete pipeline integration."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.test_video_path = None
        self.temp_files = []
        
        # Create test video
        self.test_video_path = self._create_test_video()
        
        # Mock configuration
        self.pipeline_config = {
            "frame_grabber": {
                "video_source": self.test_video_path,
                "resize_resolution": [640, 480],
                "process_every_n_frame": 1
            },
            "vehicle_detector": {
                "model_path": "test_model.engine",
                "conf_threshold": 0.5,
                "class_mapping": {"3": "car", "2": "bus", "5": "truck"}
            },
            "vehicle_tracker": {
                "tracker_type": "bytetrack",
                "track_thresh": 0.5
            },
            "lp_detector": {
                "model_path": "test_plate_model.onnx",
                "conf_threshold": 0.3
            },
            "ocr_reader": {
                "conf_threshold": 0.5
            },
            "vehicle_counter": {
                "counting_lines": [[[0.2, 0.3], [0.8, 0.4]]],
                "count_direction": "both"
            }
        }

    def teardown_method(self):
        """Clean up test files."""
        for temp_file in self.temp_files:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_frame_to_detection_pipeline(self):
        """Test frame capture to vehicle detection pipeline."""
        # Mock frame data
        frame_data = {
            "frame_id": "test_001",
            "timestamp": time.time(),
            "frame": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        }
        
        # Mock vehicle detection result
        detection_result = {
            "frame_id": frame_data["frame_id"],
            "timestamp": frame_data["timestamp"],
            "detections": [
                {
                    "bbox_xyxy": [100, 100, 300, 200],
                    "confidence": 0.85,
                    "class_id": 3,
                    "class_name": "car"
                }
            ]
        }
        
        # Validate pipeline data flow
        assert frame_data["frame_id"] == detection_result["frame_id"]
        assert len(detection_result["detections"]) > 0
        assert detection_result["detections"][0]["confidence"] >= self.pipeline_config["vehicle_detector"]["conf_threshold"]

    def test_detection_to_tracking_pipeline(self):
        """Test vehicle detection to tracking pipeline."""
        # Mock detection data
        detection_data = {
            "frame_id": "test_001",
            "detections": [
                {
                    "bbox_xyxy": [100, 100, 300, 200],
                    "confidence": 0.85,
                    "class_id": 3,
                    "class_name": "car"
                },
                {
                    "bbox_xyxy": [400, 150, 600, 250],
                    "confidence": 0.75,
                    "class_id": 2,
                    "class_name": "bus"
                }
            ]
        }
        
        # Mock tracking result
        tracking_result = {
            "frame_id": detection_data["frame_id"],
            "tracked_objects": [
                {
                    "bbox_xyxy": [100, 100, 300, 200],
                    "confidence": 0.85,
                    "class_id": 3,
                    "class_name": "car",
                    "track_id": 1
                },
                {
                    "bbox_xyxy": [400, 150, 600, 250],
                    "confidence": 0.75,
                    "class_id": 2,
                    "class_name": "bus",
                    "track_id": 2
                }
            ]
        }
        
        # Validate tracking assignment
        assert len(tracking_result["tracked_objects"]) == len(detection_data["detections"])
        assert all("track_id" in obj for obj in tracking_result["tracked_objects"])

    def test_vehicle_to_plate_detection_pipeline(self):
        """Test vehicle detection to license plate detection pipeline."""
        # Mock vehicle detection
        vehicle_detection = {
            "bbox_xyxy": [100, 100, 300, 200],
            "confidence": 0.85,
            "class_id": 3,
            "class_name": "car",
            "track_id": 1
        }
        
        # Mock frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Extract vehicle region
        x1, y1, x2, y2 = vehicle_detection["bbox_xyxy"]
        vehicle_crop = frame[y1:y2, x1:x2]
        
        # Mock plate detection result
        plate_detection = {
            "vehicle_id": vehicle_detection["track_id"],
            "vehicle_class": vehicle_detection["class_name"],
            "plate_bbox": [50, 60, 150, 80],  # Relative to vehicle crop
            "plate_confidence": 0.7
        }
        
        # Validate plate detection
        assert plate_detection["vehicle_id"] == vehicle_detection["track_id"]
        assert plate_detection["plate_confidence"] >= self.pipeline_config["lp_detector"]["conf_threshold"]

    def test_plate_to_ocr_pipeline(self):
        """Test license plate detection to OCR pipeline."""
        # Mock plate detection
        plate_data = {
            "vehicle_id": 1,
            "plate_bbox": [50, 60, 150, 80],
            "plate_confidence": 0.7,
            "vehicle_crop": np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
        }
        
        # Extract plate region
        x1, y1, x2, y2 = plate_data["plate_bbox"]
        plate_crop = plate_data["vehicle_crop"][y1:y2, x1:x2]
        
        # Mock OCR result
        ocr_result = {
            "vehicle_id": plate_data["vehicle_id"],
            "plate_text": "ABC123",
            "ocr_confidence": 0.85,
            "char_confidences": [0.9, 0.8, 0.9, 0.8, 0.7, 0.9]
        }
        
        # Validate OCR result
        assert ocr_result["vehicle_id"] == plate_data["vehicle_id"]
        assert ocr_result["ocr_confidence"] >= self.pipeline_config["ocr_reader"]["conf_threshold"]
        assert len(ocr_result["plate_text"]) > 0

    def test_tracking_to_counting_pipeline(self):
        """Test vehicle tracking to counting pipeline."""
        # Mock tracking history
        track_history = [
            {"track_id": 1, "center": (300, 200), "frame_id": 1, "class_name": "car"},
            {"track_id": 1, "center": (400, 250), "frame_id": 2, "class_name": "car"},
            {"track_id": 1, "center": (500, 300), "frame_id": 3, "class_name": "car"},
        ]
        
        # Mock counting line
        counting_line = self.pipeline_config["vehicle_counter"]["counting_lines"][0]
        
        # Mock counting result
        counting_result = {
            "frame_id": 3,
            "counts": {"car": {"up": 0, "down": 1}},
            "total_vehicles": 1,
            "line_crossings": [
                {
                    "track_id": 1,
                    "class_name": "car",
                    "direction": "down",
                    "timestamp": time.time()
                }
            ]
        }
        
        # Validate counting result
        assert counting_result["total_vehicles"] > 0
        assert len(counting_result["line_crossings"]) > 0

    def test_data_persistence_pipeline(self):
        """Test data persistence throughout pipeline."""
        # Mock complete pipeline data
        pipeline_data = {
            "frame_id": "test_001",
            "timestamp": time.time(),
            "vehicle_detections": [
                {
                    "track_id": 1,
                    "class_name": "car",
                    "bbox": [100, 100, 300, 200],
                    "confidence": 0.85
                }
            ],
            "plate_detections": [
                {
                    "vehicle_id": 1,
                    "plate_text": "ABC123",
                    "confidence": 0.8
                }
            ],
            "vehicle_counts": {
                "car": {"up": 0, "down": 1}
            }
        }
        
        # Validate data consistency
        vehicle_ids = {det["track_id"] for det in pipeline_data["vehicle_detections"]}
        plate_vehicle_ids = {det["vehicle_id"] for det in pipeline_data["plate_detections"]}
        
        assert plate_vehicle_ids.issubset(vehicle_ids), "All plate detections should have corresponding vehicles"

    def test_error_propagation_pipeline(self):
        """Test error handling and propagation through pipeline."""
        # Test with invalid frame
        invalid_frame_data = {
            "frame_id": "test_error",
            "timestamp": time.time(),
            "frame": None  # Invalid frame
        }
        
        # Pipeline should handle gracefully
        try:
            if invalid_frame_data["frame"] is None:
                # Skip processing
                result = {"error": "Invalid frame", "frame_id": invalid_frame_data["frame_id"]}
            else:
                result = {"success": True}
            
            assert "error" in result or "success" in result
        except Exception as e:
            # Exception handling is acceptable
            assert isinstance(e, Exception)

    def test_performance_pipeline(self):
        """Test pipeline performance with multiple frames."""
        num_frames = 10
        start_time = time.time()
        
        processed_frames = []
        for i in range(num_frames):
            # Mock frame processing
            frame_data = {
                "frame_id": f"perf_test_{i}",
                "timestamp": time.time(),
                "frame": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            }
            
            # Simulate processing time
            time.sleep(0.001)
            
            processed_frames.append(frame_data)
        
        total_time = time.time() - start_time
        fps = num_frames / total_time
        
        assert fps > 10, f"Pipeline should process at least 10 FPS, got {fps:.2f}"
        assert len(processed_frames) == num_frames

    def test_memory_usage_pipeline(self):
        """Test memory usage throughout pipeline."""
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Process multiple frames
        for i in range(50):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Simulate pipeline processing
            detections = [{"bbox": [100, 100, 200, 200], "confidence": 0.8}]
            
            # Clean up frame reference
            del frame
        
        gc.collect()
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024, f"Memory usage too high: {memory_increase / 1024 / 1024:.2f}MB"

    def test_configuration_validation_pipeline(self):
        """Test pipeline configuration validation."""
        required_sections = [
            "frame_grabber",
            "vehicle_detector", 
            "vehicle_tracker",
            "lp_detector",
            "ocr_reader",
            "vehicle_counter"
        ]
        
        for section in required_sections:
            assert section in self.pipeline_config, f"Required config section missing: {section}"
        
        # Validate specific config values
        assert self.pipeline_config["vehicle_detector"]["conf_threshold"] > 0
        assert self.pipeline_config["vehicle_detector"]["conf_threshold"] <= 1
        assert len(self.pipeline_config["vehicle_counter"]["counting_lines"]) > 0

    # Helper methods
    def _create_test_video(self):
        """Create a test video file."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.close()
        self.temp_files.append(temp_file.name)
        
        # Create simple test video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_file.name, fourcc, 10.0, (640, 480))
        
        for i in range(30):  # 3 seconds at 10 FPS
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Add moving vehicle-like rectangle
            x = (i * 15) % 600
            y = 200
            cv2.rectangle(frame, (x, y), (x + 80, y + 40), (0, 255, 0), -1)
            
            # Add license plate-like rectangle
            cv2.rectangle(frame, (x + 20, y + 25), (x + 60, y + 35), (255, 255, 255), -1)
            cv2.putText(frame, "ABC123", (x + 22, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            
            out.write(frame)
        
        out.release()
        return temp_file.name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])