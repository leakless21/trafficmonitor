"""
Unit tests for Event Fusion Service.

Tests the core functionality of the Event Fusion Service including:
- Message merging scenarios (complete, partial, out-of-order)
- TTL expiry handling with various timing scenarios
- Memory management (buffer overflow, cleanup)
- Circuit breaker state transitions
- Edge case handling from the plan document
"""

import pytest
import time
import multiprocessing as mp
from unittest.mock import Mock, patch
from queue import Queue as ThreadQueue
from typing import Dict, Any

from traffic_monitor.services.event_fusion_service import (
    EventFusionService, 
    CircuitBreaker, 
    FusionMetrics,
    event_fusion_process
)
from traffic_monitor.utils.custom_types import (
    TrackedVehicleMessage,
    PlateDetectionMessage,
    OCRResultMessage,
    EnrichedTrackedVehicleMessage
)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_initial_state(self):
        """Test circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker(failure_threshold=3, timeout=10.0)
        assert cb.state == "CLOSED"
        assert cb.failure_count == 0
        assert cb.can_execute() is True
    
    def test_failure_threshold(self):
        """Test circuit breaker opens after failure threshold."""
        cb = CircuitBreaker(failure_threshold=3, timeout=10.0)
        
        # Record failures below threshold
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "CLOSED"
        assert cb.can_execute() is True
        
        # Exceed threshold
        cb.record_failure()
        assert cb.state == "OPEN"
        assert cb.can_execute() is False
    
    def test_timeout_recovery(self):
        """Test circuit breaker recovery after timeout."""
        cb = CircuitBreaker(failure_threshold=2, timeout=0.1)
        
        # Trip the circuit breaker
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "OPEN"
        
        # Wait for timeout
        time.sleep(0.2)
        assert cb.can_execute() is True
        assert cb.state == "HALF_OPEN"
        
        # Success should close it
        cb.record_success()
        assert cb.state == "CLOSED"


class TestEventFusionService:
    """Test Event Fusion Service core functionality."""
    
    @pytest.fixture
    def fusion_config(self):
        """Default configuration for fusion service."""
        return {
            "ttl_sec": 1.0,
            "max_buffer_size": 100,
            "max_state_age_sec": 5.0,
            "max_frame_gap": 10,
            "offline_mode": False,
            "service_name": "TestEventFusionService"
        }
    
    @pytest.fixture
    def fusion_service(self, fusion_config):
        """Create fusion service instance."""
        service = EventFusionService(fusion_config)
        yield service
        service.shutdown()
    
    @pytest.fixture
    def sample_tracking_message(self):
        """Sample tracking message."""
        return {
            "frame_id": "100",
            "camera_id": "cam1",
            "timestamp": time.time(),
            "frame_data_jpeg": b"fake_jpeg_data",
            "frame_height": 720,
            "frame_width": 1280,
            "og_frame_height": 720,
            "og_frame_width": 1280,
            "og_fps": 30.0,
            "tracked_objects": [
                {
                    "bbox_xyxy": [100, 100, 200, 200],
                    "confidence": 0.9,
                    "class_id": 3,
                    "class_name": "car",
                    "track_id": 42
                }
            ]
        }
    
    @pytest.fixture
    def sample_plate_detection_message(self):
        """Sample plate detection message."""
        return {
            "frame_id": "100",
            "camera_id": "cam1",
            "timestamp": time.time(),
            "frame_data_jpeg": b"fake_jpeg_data",
            "frame_height": 720,
            "frame_width": 1280,
            "og_frame_height": 720,
            "og_frame_width": 1280,
            "og_fps": 30.0,
            "vehicle_id": 42,
            "vehicle_class": "car",
            "plate_bbox_original": [120, 150, 180, 170],
            "plate_confidence": 0.85
        }
    
    @pytest.fixture
    def sample_ocr_message(self):
        """Sample OCR message."""
        return {
            "frame_id": "100",
            "camera_id": "cam1",
            "timestamp": time.time(),
            "vehicle_id": 42,
            "lp_text": "ABC123",
            "ocr_confidence": 0.92
        }
    
    def test_tracking_message_processing(self, fusion_service, sample_tracking_message):
        """Test basic tracking message processing."""
        result = fusion_service.process_tracking_message(sample_tracking_message)
        assert result is True
        
        # Check state was created
        key = ("100", 42)  # Frame ID is now a string
        assert key in fusion_service.state
        
        obj = fusion_service.state[key]
        assert obj["track_id"] == 42
        assert obj["class_name"] == "car"
        assert obj["plate_detected"] is False
        assert obj["plate_text_read"] is False
    
    def test_plate_detection_message_processing(self, fusion_service, sample_tracking_message, sample_plate_detection_message):
        """Test plate detection message processing."""
        # Process tracking message first
        fusion_service.process_tracking_message(sample_tracking_message)
        
        # Process plate detection
        result = fusion_service.process_plate_detection_message(sample_plate_detection_message)
        assert result is True
        
        # Check state was updated
        key = ("100", 42)
        obj = fusion_service.state[key]
        assert obj["plate_detected"] is True
        assert obj["plate_bbox_xyxy"] == [120, 150, 180, 170]
        assert obj["plate_confidence"] == 0.85
    
    def test_ocr_message_processing(self, fusion_service, sample_tracking_message, sample_ocr_message):
        """Test OCR message processing."""
        # Process tracking message first
        fusion_service.process_tracking_message(sample_tracking_message)
        
        # Process OCR
        result = fusion_service.process_ocr_message(sample_ocr_message)
        assert result is True
        
        # Check state was updated
        key = ("100", 42)
        obj = fusion_service.state[key]
        assert obj["plate_text_read"] is True
        assert obj["plate_text"] == "ABC123"
        assert obj["ocr_confidence"] == 0.92
    
    def test_out_of_order_messages(self, fusion_service, sample_tracking_message, sample_plate_detection_message):
        """Test handling of out-of-order messages (Edge case #1)."""
        # Process plate detection BEFORE tracking
        result = fusion_service.process_plate_detection_message(sample_plate_detection_message)
        assert result is True
        
        # Should be stored in pending updates
        key = ("100", 42)
        assert key in fusion_service.pending_updates
        assert len(fusion_service.pending_updates[key]) == 1
        
        # Now process tracking message
        fusion_service.process_tracking_message(sample_tracking_message)
        
        # Pending update should be applied
        assert key not in fusion_service.pending_updates
        obj = fusion_service.state[key]
        assert obj["plate_detected"] is True
        assert obj["plate_bbox_xyxy"] == [120, 150, 180, 170]
    
    def test_complete_message_flow(self, fusion_service, sample_tracking_message, sample_plate_detection_message, sample_ocr_message):
        """Test complete message flow with all data types."""
        # Process all messages
        fusion_service.process_tracking_message(sample_tracking_message)
        fusion_service.process_plate_detection_message(sample_plate_detection_message)
        fusion_service.process_ocr_message(sample_ocr_message)
        
        # Check final state
        key = ("100", 42)
        obj = fusion_service.state[key]
        assert obj["track_id"] == 42
        assert obj["plate_detected"] is True
        assert obj["plate_text_read"] is True
        assert obj["plate_text"] == "ABC123"
        assert obj["plate_bbox_xyxy"] == [120, 150, 180, 170]
    
    def test_ttl_flush(self, fusion_service, sample_tracking_message):
        """Test TTL-based flushing (Edge case #2)."""
        # Use very short TTL
        fusion_service.ttl_sec = 0.1
        
        # Process tracking message
        fusion_service.process_tracking_message(sample_tracking_message)
        
        # Wait for TTL to expire
        time.sleep(0.2)
        
        # Flush should return the object
        ready_messages = fusion_service.flush_ready_objects()
        assert len(ready_messages) == 1
        
        # State should be cleared
        assert len(fusion_service.state) == 0
    
    def test_higher_confidence_updates(self, fusion_service, sample_tracking_message):
        """Test that higher confidence updates override lower ones."""
        # Process tracking message
        fusion_service.process_tracking_message(sample_tracking_message)
        
        # Process low confidence plate detection
        low_conf_plate = {
            "frame_id": "100",
            "camera_id": "cam1",
            "timestamp": time.time(),
            "frame_data_jpeg": b"fake_jpeg_data",
            "frame_height": 720,
            "frame_width": 1280,
            "og_frame_height": 720,
            "og_frame_width": 1280,
            "og_fps": 30.0,
            "vehicle_id": 42,
            "vehicle_class": "car",
            "plate_bbox_original": [120, 150, 180, 170],
            "plate_confidence": 0.5
        }
        fusion_service.process_plate_detection_message(low_conf_plate)
        
        # Process high confidence plate detection
        high_conf_plate = low_conf_plate.copy()
        high_conf_plate["plate_confidence"] = 0.9
        high_conf_plate["plate_bbox_original"] = [125, 155, 185, 175]
        fusion_service.process_plate_detection_message(high_conf_plate)
        
        # Should keep higher confidence
        key = ("100", 42)
        obj = fusion_service.state[key]
        assert obj["plate_confidence"] == 0.9
        assert obj["plate_bbox_xyxy"] == [125, 155, 185, 175]
    
    def test_memory_pressure_flush(self, fusion_service):
        """Test memory pressure handling (Edge case #8)."""
        # Set very low buffer size
        fusion_service.max_buffer_size = 2
        
        # Create multiple tracking messages
        for i in range(5):
            tracking_msg = {
                "frame_id": str(i),
                "camera_id": "cam1",
                "timestamp": time.time(),
                "frame_data_jpeg": b"fake_jpeg_data",
                "frame_height": 720,
                "frame_width": 1280,
                "og_frame_height": 720,
                "og_frame_width": 1280,
                "og_fps": 30.0,
                "tracked_objects": [
                    {
                        "bbox_xyxy": [100, 100, 200, 200],
                        "confidence": 0.9,
                        "class_id": 3,
                        "class_name": "car",
                        "track_id": i
                    }
                ]
            }
            fusion_service.process_tracking_message(tracking_msg)
        
        # Should trigger memory pressure flush
        ready_messages = fusion_service.flush_ready_objects()
        assert len(ready_messages) > 0
        assert len(fusion_service.state) <= fusion_service.max_buffer_size
    
    def test_invalid_message_validation(self, fusion_service):
        """Test validation of invalid messages."""
        # Missing required fields
        invalid_msg = {"frame_id": "100"}
        result = fusion_service.process_tracking_message(invalid_msg)
        assert result is False
        assert fusion_service.metrics.validation_failures > 0
        
        # Invalid message type
        result = fusion_service.process_tracking_message("not_a_dict")
        assert result is False
    
    def test_enriched_message_creation(self, fusion_service, sample_tracking_message, sample_plate_detection_message, sample_ocr_message):
        """Test creation of enriched messages."""
        # Process all message types
        fusion_service.process_tracking_message(sample_tracking_message)
        fusion_service.process_plate_detection_message(sample_plate_detection_message)
        fusion_service.process_ocr_message(sample_ocr_message)
        
        # Force flush
        fusion_service.ttl_sec = 0.0
        ready_messages = fusion_service.flush_ready_objects()
        
        assert len(ready_messages) == 1
        enriched_msg = ready_messages[0]
        
        # Check message structure
        assert enriched_msg["frame_id"] == "100"
        assert len(enriched_msg["tracked_objects"]) == 1
        
        obj = enriched_msg["tracked_objects"][0]
        assert obj["track_id"] == 42
        assert obj["plate_detected"] is True
        assert obj["plate_text_read"] is True
        assert obj["plate_text"] == "ABC123"
        assert obj["plate_bbox_xyxy"] == [120, 150, 180, 170]
    
    def test_metrics_collection(self, fusion_service, sample_tracking_message):
        """Test metrics collection and calculation."""
        initial_processed = fusion_service.metrics.messages_processed_per_sec
        
        # Process some messages
        for _ in range(5):
            fusion_service.process_tracking_message(sample_tracking_message)
        
        # Update metrics
        fusion_service._update_metrics()
        
        # Check metrics were updated
        assert len(fusion_service.processed_messages) == 5
        assert fusion_service.metrics.state_dict_size_current == 1  # Only one unique (frame_id, track_id)


class TestEventFusionProcess:
    """Test the event fusion process function."""
    
    def test_process_initialization(self):
        """Test that the process initializes correctly."""
        config = {
            "ttl_sec": 1.0,
            "max_buffer_size": 100,
            "offline_mode": False,
            "service_name": "TestEventFusionService",
            "loguru": {"level": "INFO"}
        }
        
        # Create queues
        tracking_queue = mp.Queue()
        plate_queue = mp.Queue()
        ocr_queue = mp.Queue()
        counting_queue = mp.Queue()
        output_queue = mp.Queue()
        shutdown_event = mp.Event()
        
        # Put a shutdown signal
        tracking_queue.put(None)
        
        # This should run without error and exit cleanly
        try:
            event_fusion_process(
                config, tracking_queue, plate_queue, ocr_queue, 
                counting_queue, output_queue, shutdown_event
            )
        except Exception as e:
            pytest.fail(f"Process failed with exception: {e}")


class TestEdgeCases:
    """Test edge cases from the plan document."""
    
    @pytest.fixture
    def fusion_service(self):
        """Create fusion service for edge case testing."""
        config = {
            "ttl_sec": 1.0,
            "max_buffer_size": 100,
            "max_state_age_sec": 5.0,
            "max_frame_gap": 10,
            "offline_mode": False,
            "service_name": "EdgeCaseTestService"
        }
        service = EventFusionService(config)
        yield service
        service.shutdown()
    
    def test_edge_case_frame_gap(self, fusion_service):
        """Test edge case #7: Frame sequence handling with UUIDs."""
        # Note: Frame gap detection is disabled for UUID frame IDs
        # since UUIDs don't have sequential ordering
        
        # Process frame 1
        msg1 = {
            "frame_id": "frame_001",
            "camera_id": "cam1",
            "timestamp": time.time(),
            "frame_data_jpeg": b"fake_jpeg_data",
            "frame_height": 720,
            "frame_width": 1280,
            "og_frame_height": 720,
            "og_frame_width": 1280,
            "og_fps": 30.0,
            "tracked_objects": [{"bbox_xyxy": [100, 100, 200, 200], "confidence": 0.9, "class_id": 3, "class_name": "car", "track_id": 1}]
        }
        fusion_service.process_tracking_message(msg1)
        
        # Process frame with different UUID (no gap warning expected)
        msg2 = msg1.copy()
        msg2["frame_id"] = "frame_999"  # Different UUID
        msg2["tracked_objects"][0]["track_id"] = 2
        
        # Should process without warnings since UUID frame IDs don't have sequence
        result = fusion_service.process_tracking_message(msg2)
        assert result is True
        
        # Both frames should be in state
        assert len(fusion_service.state) == 2
    
    def test_edge_case_duplicate_messages(self, fusion_service):
        """Test edge case #5: Duplicate/conflicting OCR results."""
        # Process tracking message
        tracking_msg = {
            "frame_id": "100",
            "camera_id": "cam1",
            "timestamp": time.time(),
            "frame_data_jpeg": b"fake_jpeg_data",
            "frame_height": 720,
            "frame_width": 1280,
            "og_frame_height": 720,
            "og_frame_width": 1280,
            "og_fps": 30.0,
            "tracked_objects": [{"bbox_xyxy": [100, 100, 200, 200], "confidence": 0.9, "class_id": 3, "class_name": "car", "track_id": 42}]
        }
        fusion_service.process_tracking_message(tracking_msg)
        
        # Process first OCR result
        ocr1 = {
            "frame_id": "100",
            "camera_id": "cam1",
            "timestamp": time.time(),
            "vehicle_id": 42,
            "lp_text": "ABC123",
            "ocr_confidence": 0.8
        }
        fusion_service.process_ocr_message(ocr1)
        
        # Process conflicting OCR result with higher confidence
        ocr2 = ocr1.copy()
        ocr2["lp_text"] = "XYZ789"
        ocr2["ocr_confidence"] = 0.95
        fusion_service.process_ocr_message(ocr2)
        
        # Should keep higher confidence result
        key = ("100", 42)
        obj = fusion_service.state[key]
        assert obj["plate_text"] == "XYZ789"
        assert obj["ocr_confidence"] == 0.95
    
    def test_edge_case_empty_plate_text(self, fusion_service):
        """Test edge case #13: No plate detected/OCR returns empty."""
        # Process tracking message
        tracking_msg = {
            "frame_id": "100",
            "camera_id": "cam1",
            "timestamp": time.time(),
            "frame_data_jpeg": b"fake_jpeg_data",
            "frame_height": 720,
            "frame_width": 1280,
            "og_frame_height": 720,
            "og_frame_width": 1280,
            "og_fps": 30.0,
            "tracked_objects": [{"bbox_xyxy": [100, 100, 200, 200], "confidence": 0.9, "class_id": 3, "class_name": "car", "track_id": 42}]
        }
        fusion_service.process_tracking_message(tracking_msg)
        
        # Wait for TTL expiry without any plate data
        fusion_service.ttl_sec = 0.1
        time.sleep(0.2)
        
        # Flush should return object with no plate data
        ready_messages = fusion_service.flush_ready_objects()
        assert len(ready_messages) == 1
        
        obj = ready_messages[0]["tracked_objects"][0]
        assert obj["plate_detected"] is False
        assert obj["plate_text_read"] is False
        assert obj["plate_text"] is None