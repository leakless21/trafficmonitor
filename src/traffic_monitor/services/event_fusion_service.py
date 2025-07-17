"""
Event Fusion Service - Centralizes data merging for plate visualization.

This service implements the Event Fusion architecture described in the 
plate_visualization_data_fusion_plan.md. It merges data from multiple producers
(tracking, plate detection, OCR, counting) into a single enriched message
for the visualization service.

Key features:
- Zero additional latency through intelligent TTL management
- Comprehensive error handling and circuit breaker patterns
- Memory management with configurable limits
- Performance monitoring and metrics collection
- Graceful degradation under failure conditions
"""

import multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
from queue import Empty
from typing import Dict, Any, List, Tuple, Optional
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
import threading
from loguru import logger

from ..utils.custom_types import (
    TrackedVehicleMessage, 
    EnrichedTrackedVehicleMessage,
    EnrichedTrackedObject,
    PlateDetectionMessage,
    OCRResultMessage,
    VehicleCountMessage
)
from ..utils.queue_utils import safe_put


@dataclass
class CircuitBreaker:
    """Circuit breaker for producer health monitoring."""
    failure_threshold: int = 5
    timeout: float = 30.0
    failure_count: int = 0
    last_failure_time: float = 0.0
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def record_success(self):
        """Record a successful operation."""
        self.failure_count = 0
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            
    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            
    def can_execute(self) -> bool:
        """Check if operations can be executed."""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True


@dataclass
class FusionMetrics:
    """Comprehensive metrics for the fusion service."""
    # Throughput metrics
    messages_processed_per_sec: float = 0.0
    frames_flushed_per_sec: float = 0.0
    
    # Quality metrics
    complete_merges_ratio: float = 0.0
    partial_flushes_ratio: float = 0.0
    dropped_messages_count: int = 0
    
    # Performance metrics
    avg_merge_latency_ms: float = 0.0
    buffer_size_current: int = 0
    state_dict_size_current: int = 0
    memory_usage_mb: float = 0.0
    
    # Error metrics
    out_of_order_messages: int = 0
    producer_timeouts: int = 0
    validation_failures: int = 0
    
    # Counters for ratio calculations
    total_merges: int = 0
    complete_merges: int = 0
    partial_flushes: int = 0


class EventFusionService:
    """
    Event Fusion Service that merges data from multiple producers into enriched messages.
    
    Implements the architecture from plate_visualization_data_fusion_plan.md with:
    - Intelligent message merging and buffering
    - TTL-based flush strategy
    - Circuit breaker pattern for producer health
    - Comprehensive metrics and monitoring
    - Memory management and backpressure handling
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Configuration
        self.ttl_sec = config.get("ttl_sec", 1.0)
        self.max_buffer_size = config.get("max_buffer_size", 1000)
        self.max_state_age_sec = config.get("max_state_age_sec", 5.0)
        self.max_frame_gap = config.get("max_frame_gap", 10)
        self.offline_mode = config.get("offline_mode", False)
        self.service_name = config.get("service_name", "EventFusionService")
        
        # Internal state with memory management
        self.state: Dict[Tuple[int, int], Dict] = {}  # (frame_id, track_id) → partial object
        self.frame_buffer: Dict[int, List] = {}  # frame_id → list of enriched objects
        self.pending_updates: Dict[Tuple[int, int], List] = defaultdict(list)  # for out-of-order messages
        
        # Timing and sequence tracking
        self.last_frame_id = 0
        self.frame_timestamps: Dict[int, float] = {}  # frame_id → creation timestamp
        
        # Producer health monitoring
        self.producer_health = {
            'tracking': CircuitBreaker(failure_threshold=5, timeout=30),
            'plate_detection': CircuitBreaker(failure_threshold=3, timeout=20),
            'ocr': CircuitBreaker(failure_threshold=3, timeout=20),
            'counting': CircuitBreaker(failure_threshold=5, timeout=30)
        }
        
        # Metrics
        self.metrics = FusionMetrics()
        self.last_metrics_time = time.time()
        self.processed_messages = deque(maxlen=100)  # For throughput calculation
        
        # Threading for periodic cleanup
        self.cleanup_thread = None
        self.shutdown_flag = threading.Event()
        
        logger.info(f"[{self.service_name}] Initialized with TTL={self.ttl_sec}s, "
                   f"max_buffer={self.max_buffer_size}, offline_mode={self.offline_mode}")
    
    def start_cleanup_thread(self):
        """Start the periodic cleanup thread."""
        self.cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        self.cleanup_thread.start()
        
    def _periodic_cleanup(self):
        """Periodic cleanup of stale state and metrics calculation."""
        while not self.shutdown_flag.is_set():
            try:
                current_time = time.time()
                
                # Clean up stale state
                stale_keys = []
                for key, obj_data in self.state.items():
                    if current_time - obj_data.get('created_at', 0) > self.max_state_age_sec:
                        stale_keys.append(key)
                
                for key in stale_keys:
                    logger.debug(f"[{self.service_name}] Cleaning up stale state for {key}")
                    del self.state[key]
                
                # Clean up old frame timestamps
                old_frame_ids = [fid for fid, ts in self.frame_timestamps.items() 
                               if current_time - ts > self.max_state_age_sec]
                for fid in old_frame_ids:
                    del self.frame_timestamps[fid]
                
                # Update metrics
                self._update_metrics()
                
                # Log metrics periodically
                if int(current_time) % 30 == 0:  # Every 30 seconds
                    self._log_metrics()
                
                time.sleep(1.0)  # Cleanup every second
                
            except Exception as e:
                logger.error(f"[{self.service_name}] Error in cleanup thread: {e}")
                time.sleep(5.0)  # Back off on error
    
    def _update_metrics(self):
        """Update performance metrics."""
        current_time = time.time()
        
        # Calculate throughput
        recent_messages = [ts for ts in self.processed_messages if current_time - ts < 1.0]
        self.metrics.messages_processed_per_sec = len(recent_messages)
        
        # Update current state metrics
        self.metrics.buffer_size_current = len(self.frame_buffer)
        self.metrics.state_dict_size_current = len(self.state)
        
        # Calculate ratios
        if self.metrics.total_merges > 0:
            self.metrics.complete_merges_ratio = self.metrics.complete_merges / self.metrics.total_merges
            self.metrics.partial_flushes_ratio = self.metrics.partial_flushes / self.metrics.total_merges
    
    def _log_metrics(self):
        """Log current metrics."""
        mode_str = "offline" if self.offline_mode else "real-time"
        frame_count = len(set(k[0] for k in self.state.keys()))
        logger.info(f"[{self.service_name}] Metrics ({mode_str}): "
                   f"throughput={self.metrics.messages_processed_per_sec:.1f}msg/s, "
                   f"state_size={self.metrics.state_dict_size_current}, "
                   f"frames_buffered={frame_count}, "
                   f"complete_ratio={self.metrics.complete_merges_ratio:.2f}, "
                   f"dropped={self.metrics.dropped_messages_count}")
    
    def _validate_message(self, message: Dict, message_type: str) -> bool:
        """Validate message integrity and required fields."""
        try:
            # Common validation
            if not isinstance(message, dict):
                logger.warning(f"[{self.service_name}] Invalid message type: {type(message)}")
                return False
            
            required_fields = ["frame_id", "timestamp"]
            for field in required_fields:
                if field not in message:
                    logger.warning(f"[{self.service_name}] Missing required field '{field}' in {message_type}")
                    return False
            
            # Type-specific validation
            if message_type == "tracking":
                if "tracked_objects" not in message or not isinstance(message["tracked_objects"], list):
                    return False
                    
            elif message_type == "plate_detection":
                required_plate_fields = ["vehicle_id", "plate_bbox_original", "plate_confidence"]
                for field in required_plate_fields:
                    if field not in message:
                        return False
                        
            elif message_type == "ocr":
                required_ocr_fields = ["vehicle_id", "lp_text", "ocr_confidence"]
                for field in required_ocr_fields:
                    if field not in message:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"[{self.service_name}] Error validating {message_type} message: {e}")
            self.metrics.validation_failures += 1
            return False
    
    def _create_enriched_object(self, base_obj: Dict) -> Dict:
        """Create an enriched tracked object with default values."""
        enriched = base_obj.copy()
        enriched.update({
            'plate_bbox_xyxy': None,
            'plate_text': None,
            'plate_confidence': None,
            'ocr_confidence': None,
            'plate_detected': False,
            'plate_text_read': False
        })
        return enriched
    
    def process_tracking_message(self, message: TrackedVehicleMessage) -> bool:
        """Process a tracking message (base message type)."""
        if not self._validate_message(message, "tracking"):
            self.metrics.validation_failures += 1
            return False
        
        try:
            frame_id = message["frame_id"]  # Keep as string (UUID)
            current_time = time.time()
            
            # Note: Frame sequence tracking disabled for UUID frame IDs
            # UUIDs don't have sequential ordering like integers
            
            # Store frame timestamp
            self.frame_timestamps[frame_id] = current_time
            
            # Process each tracked object
            for obj in message["tracked_objects"]:
                track_id = obj["track_id"]
                key = (frame_id, track_id)
                
                # Create enriched object
                enriched_obj = self._create_enriched_object(obj)
                enriched_obj['created_at'] = current_time
                enriched_obj['frame_data'] = {
                    'camera_id': message["camera_id"],
                    'timestamp': message["timestamp"],
                    'frame_data_jpeg': message["frame_data_jpeg"],
                    'frame_height': message["frame_height"],
                    'frame_width': message["frame_width"],
                    'og_frame_height': message["og_frame_height"],
                    'og_frame_width': message["og_frame_width"],
                    'og_fps': message["og_fps"]
                }
                
                # Store in state
                self.state[key] = enriched_obj
                
                # Process any pending updates for this object
                if key in self.pending_updates:
                    pending_count = len(self.pending_updates[key])
                    for update in self.pending_updates[key]:
                        self._apply_update(key, update)
                    del self.pending_updates[key]
                    logger.debug(f"[{self.service_name}] Applied {pending_count} pending updates for {key}")
            
            self.producer_health['tracking'].record_success()
            self.processed_messages.append(current_time)
            return True
            
        except Exception as e:
            logger.error(f"[{self.service_name}] Error processing tracking message: {e}")
            self.producer_health['tracking'].record_failure()
            return False
    
    def process_plate_detection_message(self, message: PlateDetectionMessage) -> bool:
        """Process a plate detection message."""
        if not self._validate_message(message, "plate_detection"):
            self.metrics.validation_failures += 1
            return False
        
        try:
            frame_id = message["frame_id"]  # Keep as string (UUID)
            track_id = message["vehicle_id"]
            key = (frame_id, track_id)
            
            update = {
                'type': 'plate_detection',
                'plate_bbox_xyxy': message["plate_bbox_original"],
                'plate_confidence': message["plate_confidence"],
                'plate_detected': True
            }
            
            if key in self.state:
                self._apply_update(key, update)
            else:
                # Store for later when tracking message arrives
                self.pending_updates[key].append(update)
                self.metrics.out_of_order_messages += 1
                logger.debug(f"[{self.service_name}] Storing pending plate detection for {key}")
            
            self.producer_health['plate_detection'].record_success()
            self.processed_messages.append(time.time())
            return True
            
        except Exception as e:
            logger.error(f"[{self.service_name}] Error processing plate detection message: {e}")
            self.producer_health['plate_detection'].record_failure()
            return False
    
    def process_ocr_message(self, message: OCRResultMessage) -> bool:
        """Process an OCR result message."""
        if not self._validate_message(message, "ocr"):
            self.metrics.validation_failures += 1
            return False
        
        try:
            frame_id = message["frame_id"]  # Keep as string (UUID)
            track_id = message["vehicle_id"]
            key = (frame_id, track_id)
            
            update = {
                'type': 'ocr',
                'plate_text': message["lp_text"],
                'ocr_confidence': message["ocr_confidence"],
                'plate_text_read': True
            }
            
            if key in self.state:
                self._apply_update(key, update)
            else:
                # Store for later when tracking message arrives
                self.pending_updates[key].append(update)
                self.metrics.out_of_order_messages += 1
                logger.debug(f"[{self.service_name}] Storing pending OCR result for {key}")
            
            self.producer_health['ocr'].record_success()
            self.processed_messages.append(time.time())
            return True
            
        except Exception as e:
            logger.error(f"[{self.service_name}] Error processing OCR message: {e}")
            self.producer_health['ocr'].record_failure()
            return False
    
    def _apply_update(self, key: Tuple[int, int], update: Dict):
        """Apply an update to a tracked object."""
        if key not in self.state:
            logger.warning(f"[{self.service_name}] Cannot apply update to missing key {key}")
            return
        
        obj = self.state[key]
        
        if update['type'] == 'plate_detection':
            # Only update if confidence is higher or no existing detection
            existing_conf = obj.get('plate_confidence') or 0.0
            new_conf = update['plate_confidence']
            if new_conf > existing_conf:
                obj.update({
                    'plate_bbox_xyxy': update['plate_bbox_xyxy'],
                    'plate_confidence': new_conf,
                    'plate_detected': True
                })
                logger.debug(f"[{self.service_name}] Updated plate detection for {key} "
                           f"(conf={new_conf:.3f})")
        
        elif update['type'] == 'ocr':
            # Only update if confidence is higher or no existing OCR
            existing_conf = obj.get('ocr_confidence') or 0.0
            new_conf = update['ocr_confidence']
            if new_conf > existing_conf:
                obj.update({
                    'plate_text': update['plate_text'],
                    'ocr_confidence': new_conf,
                    'plate_text_read': True
                })
                logger.debug(f"[{self.service_name}] Updated OCR for {key}: "
                           f"'{update['plate_text']}' (conf={new_conf:.3f})")
    
    def _should_flush_object(self, key: Tuple[str, int], obj: Dict) -> bool:
        """Determine if an object should be flushed."""
        current_time = time.time()
        created_at = obj.get('created_at', current_time)
        
        # CRITICAL FIX: More aggressive flushing to ensure objects get processed
        # The issue was objects were being held too long waiting for complete data
        
        # Always flush after TTL regardless of completeness
        if current_time - created_at > self.ttl_sec:
            return True
        
        # Memory pressure - flush if too many objects
        if len(self.state) > 100:  # Much lower threshold
            return True
        
        # Complete object (all expected data received)
        has_tracking = obj.get('track_id') is not None
        has_plate_detection = obj.get('plate_detected', False)
        has_ocr = obj.get('plate_text_read', False)
        
        # Flush immediately if we have tracking data (don't wait for OCR)
        # This ensures objects get processed even if OCR is delayed
        if has_tracking:
            return True
        
        return False
    
    def flush_ready_objects(self) -> List[EnrichedTrackedVehicleMessage]:
        """Flush objects that are ready for visualization."""
        ready_messages = []
        current_time = time.time()
        
        # In real-time mode, implement frame dropping for low latency
        if not self.offline_mode:
            self._drop_old_frames_realtime(current_time)
        
        # Group objects by frame_id
        frame_groups = defaultdict(list)
        keys_to_remove = []
        
        for key, obj in self.state.items():
            frame_id, track_id = key
            
            if self._should_flush_object(key, obj):
                frame_groups[frame_id].append(obj)
                keys_to_remove.append(key)
        
        # Create enriched messages for each frame
        for frame_id, objects in frame_groups.items():
            if not objects:
                continue
            
            # Use frame data from first object (all should be the same)
            frame_data = objects[0]['frame_data']
            
            # Create enriched tracked objects
            enriched_objects = []
            for obj in objects:
                enriched_obj = {
                    'bbox_xyxy': obj['bbox_xyxy'],
                    'confidence': obj['confidence'],
                    'class_id': obj['class_id'],
                    'class_name': obj['class_name'],
                    'track_id': obj['track_id'],
                    'plate_bbox_xyxy': obj['plate_bbox_xyxy'],
                    'plate_text': obj['plate_text'],
                    'plate_confidence': obj['plate_confidence'],
                    'ocr_confidence': obj['ocr_confidence'],
                    'plate_detected': obj['plate_detected'],
                    'plate_text_read': obj['plate_text_read']
                }
                enriched_objects.append(enriched_obj)
            
            # Create enriched message
            enriched_message: EnrichedTrackedVehicleMessage = {
                'frame_id': frame_id,  # Already a string (UUID)
                'camera_id': frame_data['camera_id'],
                'timestamp': frame_data['timestamp'],
                'frame_data_jpeg': frame_data['frame_data_jpeg'],
                'frame_height': frame_data['frame_height'],
                'frame_width': frame_data['frame_width'],
                'og_frame_height': frame_data['og_frame_height'],
                'og_frame_width': frame_data['og_frame_width'],
                'og_fps': frame_data['og_fps'],
                'tracked_objects': enriched_objects
            }
            
            ready_messages.append(enriched_message)
            
            # Update metrics
            self.metrics.total_merges += 1
            complete_objects = sum(1 for obj in objects 
                                 if obj['plate_detected'] and obj['plate_text_read'])
            if complete_objects == len(objects):
                self.metrics.complete_merges += 1
            else:
                self.metrics.partial_flushes += 1
        
        # Remove flushed objects from state
        for key in keys_to_remove:
            del self.state[key]
        
        if ready_messages:
            mode_str = "offline" if self.offline_mode else "real-time"
            logger.debug(f"[{self.service_name}] Flushed {len(ready_messages)} frames "
                        f"with {sum(len(msg['tracked_objects']) for msg in ready_messages)} objects ({mode_str} mode)")
        
        return ready_messages
    
    def _drop_old_frames_realtime(self, current_time: float):
        """Drop old frames in real-time mode to maintain low latency."""
        if self.offline_mode:
            return  # Don't drop frames in offline mode
        
        # Find frames older than real-time threshold
        frame_ages = {}
        for key, obj in self.state.items():
            frame_id, track_id = key
            created_at = obj.get('created_at', current_time)
            if frame_id not in frame_ages:
                frame_ages[frame_id] = created_at
            else:
                frame_ages[frame_id] = min(frame_ages[frame_id], created_at)
        
        # Sort frames by age (oldest first)
        sorted_frames = sorted(frame_ages.items(), key=lambda x: x[1])
        
        # Keep only the most recent frames in real-time mode
        max_frames_realtime = 2  # Keep only last 2 frames for real-time processing
        
        if len(sorted_frames) > max_frames_realtime:
            frames_to_drop = sorted_frames[:-max_frames_realtime]
            
            keys_to_remove = []
            for frame_id, _ in frames_to_drop:
                for key in list(self.state.keys()):
                    if key[0] == frame_id:
                        keys_to_remove.append(key)
            
            # Remove old frames
            for key in keys_to_remove:
                del self.state[key]
                
            if keys_to_remove:
                logger.debug(f"[{self.service_name}] Dropped {len(keys_to_remove)} objects from "
                           f"{len(frames_to_drop)} old frames in real-time mode")
                self.metrics.dropped_messages_count += len(keys_to_remove)
    
    def shutdown(self):
        """Shutdown the fusion service."""
        logger.info(f"[{self.service_name}] Shutting down...")
        self.shutdown_flag.set()
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5.0)
        
        # Final metrics log
        self._log_metrics()


def event_fusion_process(
    config: Dict[str, Any],
    tracking_queue: Queue,
    plate_detection_queue: Queue,
    ocr_queue: Queue,
    counting_queue: Queue,
    output_queue: Queue,
    shutdown_event: Event
):
    """
    Event Fusion Service process that merges data from multiple producers.
    
    Args:
        config: Service configuration
        tracking_queue: Input queue for tracking messages
        plate_detection_queue: Input queue for plate detection messages
        ocr_queue: Input queue for OCR messages
        counting_queue: Input queue for counting messages (for future use)
        output_queue: Output queue for enriched messages
        shutdown_event: Event to signal shutdown
    """
    from ..utils.logging_config import setup_logging
    setup_logging(config.get("loguru", {}))
    
    process_name = mp.current_process().name
    service_name = config.get("service_name", "EventFusionService")
    offline_mode = config.get("offline_mode", False)
    
    logger.info(f"[{service_name}] Starting process {process_name}")
    
    try:
        # Initialize fusion service
        fusion_service = EventFusionService(config)
        fusion_service.start_cleanup_thread()
        
        logger.info(f"[{service_name}] Fusion service initialized successfully")
        
        # Main processing loop
        while not shutdown_event.is_set():
            messages_processed = 0
            
            # Process tracking messages (highest priority)
            try:
                while True:
                    tracking_msg = tracking_queue.get_nowait()
                    if tracking_msg is None:
                        logger.info(f"[{service_name}] Received shutdown signal from tracking queue")
                        shutdown_event.set()
                        break
                    
                    fusion_service.process_tracking_message(tracking_msg)
                    messages_processed += 1
                    
            except Empty:
                pass
            
            # Process plate detection messages
            try:
                while True:
                    plate_msg = plate_detection_queue.get_nowait()
                    if plate_msg is None:
                        break
                    
                    fusion_service.process_plate_detection_message(plate_msg)
                    messages_processed += 1
                    
            except Empty:
                pass
            
            # Process OCR messages
            try:
                while True:
                    ocr_msg = ocr_queue.get_nowait()
                    if ocr_msg is None:
                        break
                    
                    fusion_service.process_ocr_message(ocr_msg)
                    messages_processed += 1
                    
            except Empty:
                pass
            
            # Process counting messages (for future use)
            try:
                while True:
                    count_msg = counting_queue.get_nowait()
                    if count_msg is None:
                        break
                    # For now, just consume and ignore
                    messages_processed += 1
                    
            except Empty:
                pass
            
            # Flush ready objects and send to visualization
            ready_messages = fusion_service.flush_ready_objects()
            for enriched_msg in ready_messages:
                success = safe_put(output_queue, enriched_msg, offline_mode, service_name)
                if not success:
                    logger.warning(f"[{service_name}] Failed to send enriched message")
                    fusion_service.metrics.dropped_messages_count += 1
            
            # If no messages processed, short sleep to prevent busy waiting
            if messages_processed == 0:
                if shutdown_event.is_set():
                    break
                time.sleep(0.001)
                
            # WORKAROUND: Force periodic flushing to prevent starvation
            # This ensures objects get flushed even without complete data
            if messages_processed == 0 and len(fusion_service.state) > 0:
                current_time = time.time()
                # Force flush objects older than 2 seconds
                old_keys = []
                for key, obj in fusion_service.state.items():
                    if current_time - obj.get('created_at', current_time) > 2.0:
                        old_keys.append(key)
                
                if old_keys:
                    logger.debug(f"[{service_name}] Force flushing {len(old_keys)} stale objects")
                    for key in old_keys:
                        obj = fusion_service.state[key]
                        frame_id = key[0]
                        
                        # Create minimal enriched message
                        enriched_obj = {
                            'bbox_xyxy': obj['bbox_xyxy'],
                            'confidence': obj['confidence'],
                            'class_id': obj['class_id'],
                            'class_name': obj['class_name'],
                            'track_id': obj['track_id'],
                            'plate_bbox_xyxy': obj.get('plate_bbox_xyxy'),
                            'plate_text': obj.get('plate_text'),
                            'plate_confidence': obj.get('plate_confidence'),
                            'ocr_confidence': obj.get('ocr_confidence'),
                            'plate_detected': obj.get('plate_detected', False),
                            'plate_text_read': obj.get('plate_text_read', False)
                        }
                        
                        frame_data = obj['frame_data']
                        enriched_message = {
                            'frame_id': frame_id,
                            'camera_id': frame_data['camera_id'],
                            'timestamp': frame_data['timestamp'],
                            'frame_data_jpeg': frame_data['frame_data_jpeg'],
                            'frame_height': frame_data['frame_height'],
                            'frame_width': frame_data['frame_width'],
                            'og_frame_height': frame_data['og_frame_height'],
                            'og_frame_width': frame_data['og_frame_width'],
                            'og_fps': frame_data['og_fps'],
                            'tracked_objects': [enriched_obj]
                        }
                        
                        success = safe_put(output_queue, enriched_message, offline_mode, service_name)
                        if success:
                            del fusion_service.state[key]
                            logger.debug(f"[{service_name}] Force flushed object {key}")
                        else:
                            logger.warning(f"[{service_name}] Failed to force flush object {key}")
        
    except Exception as e:
        logger.exception(f"[{service_name}] Critical error in fusion process: {e}")
        if output_queue:
            output_queue.put(None)  # Signal shutdown to downstream
    
    finally:
        logger.info(f"[{service_name}] Shutting down process {process_name}")
        if 'fusion_service' in locals():
            fusion_service.shutdown()