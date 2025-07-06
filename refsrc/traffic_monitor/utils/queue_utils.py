"""
Queue management utilities for real-time vs offline processing modes.

This module provides utilities for managing multiprocessing queues in different
operational modes:
- Real-time mode: Drop old frames to maintain low latency
- Offline mode: Preserve all frames for complete video processing

Based on best practices from OBS Studio, GStreamer, and video processing pipelines.
"""

from queue import Empty, Full
from multiprocessing.queues import Queue
from typing import Any
from loguru import logger


def is_offline_mode(config: dict) -> bool:
    """
    Determine if we're in offline (save-to-file) mode based on configuration.
    
    Args:
        config: Configuration dictionary that may contain save_to_file flag
        
    Returns:
        True if in offline mode (preserve all frames), False for real-time mode
    """
    return config.get("save_to_file", False)


def get_queue_size_for_mode(offline_mode: bool, default_realtime_size: int = 3) -> int:
    """
    Get appropriate queue size based on processing mode.
    
    Args:
        offline_mode: Whether we're in offline processing mode
        default_realtime_size: Queue size for real-time mode (default: 3)
        
    Returns:
        Queue size (0 for unbounded in offline mode, limited size for real-time)
    """
    if offline_mode:
        return 0  # Unbounded queue for offline mode
    else:
        return default_realtime_size  # Small queue for real-time mode


def put_realtime(queue: Queue, message: Any, service_name: str = "Unknown") -> bool:
    """
    Put message to queue using real-time (leaky) strategy.
    
    Drops oldest message if queue is full to maintain low latency.
    
    Args:
        queue: Multiprocessing queue
        message: Message to put
        service_name: Name of service for logging
        
    Returns:
        True if message was put successfully, False otherwise
    """
    try:
        # Drop old message if queue is full
        try:
            old_msg = queue.get_nowait()
            logger.trace(f"[{service_name}] Dropped old message to maintain real-time performance")
        except Empty:
            pass  # Queue was not full
        
        # Put new message without blocking
        queue.put_nowait(message)
        return True
        
    except Full:
        # This should be extremely rare with get_nowait() + put_nowait() pattern
        logger.warning(f"[{service_name}] Queue unexpectedly full, message dropped")
        return False
    except Exception as e:
        logger.error(f"[{service_name}] Error in put_realtime: {e}")
        return False


def put_offline(queue: Queue, message: Any, service_name: str = "Unknown") -> bool:
    """
    Put message to queue using offline (blocking) strategy.
    
    Blocks until message can be put to preserve all frames.
    
    Args:
        queue: Multiprocessing queue
        message: Message to put
        service_name: Name of service for logging
        
    Returns:
        True if message was put successfully, False on error
    """
    try:
        # Block until we can put the message (preserves all frames)
        queue.put(message)
        return True
    except Exception as e:
        logger.error(f"[{service_name}] Error in put_offline: {e}")
        return False


def safe_put(queue: Queue, message: Any, offline_mode: bool, service_name: str = "Unknown") -> bool:
    """
    Put message to queue using appropriate strategy for the current mode.
    
    Args:
        queue: Multiprocessing queue
        message: Message to put
        offline_mode: Whether we're in offline processing mode
        service_name: Name of service for logging
        
    Returns:
        True if message was put successfully, False otherwise
    """
    if offline_mode:
        return put_offline(queue, message, service_name)
    else:
        return put_realtime(queue, message, service_name)


def log_queue_stats(queue: Queue, service_name: str, frame_count: int, log_interval: int = 100):
    """
    Log queue statistics periodically for monitoring.
    
    Args:
        queue: Multiprocessing queue to monitor
        service_name: Name of service for logging
        frame_count: Current frame count
        log_interval: How often to log (every N frames)
    """
    if frame_count % log_interval == 0:
        try:
            qsize = queue.qsize()
            logger.debug(f"[{service_name}] Frame {frame_count}, Queue size: {qsize}")
        except Exception as e:
            logger.trace(f"[{service_name}] Could not get queue size: {e}") 