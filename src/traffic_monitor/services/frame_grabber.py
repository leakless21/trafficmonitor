import time
import cv2
import uuid
import base64
import multiprocessing as mp
from loguru import logger
from typing import Any, Dict
from queue import Full
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event
from ..utils.logging_config import setup_logging

def frame_grabber_process(
    config: Dict[str, Any],
    output_queue: Queue,
    shutdown_event: Event
    ):
    """
    Captures video frames from a specified source and places them into an output queue.

    This process continuously reads frames, encodes them to JPEG, assigns unique IDs,
    and adds metadata before putting them into the queue. It handles graceful shutdown
    and logs various events, including queue full scenarios.

    Args:
        config (Dict[str, Any]): Configuration dictionary, expected to contain 'video_source',
                                 'camera_id', and 'log_every_n_frames'.
        output_queue (Queue): A multiprocessing queue to which `FrameMessage` objects will be placed.
        shutdown_event (Event): A multiprocessing event used to signal the process to terminate.
    """
    # Initialize logging specifically for this child process to ensure proper log handling
    setup_logging()
    
    process_name = mp.current_process().name
    logger.info(f"[{process_name}] Frame Grabber process started. Config: {config}")
    
    video_source = config.get("video_source")
    if not video_source:
        logger.error(f"[{process_name}] No video source found in config. Exiting frame grabber process.")
        return
    
    logger.info(f"[{process_name}] Attempting to open video source: {video_source}")
    video_capture = cv2.VideoCapture(video_source)
    
    # Check if the video source was opened successfully
    if not video_capture.isOpened():
        logger.error(f"[{process_name}] Failed to open video source: {video_source}. Exiting.")
        return
    logger.info(f"[{process_name}] Video source opened successfully: {video_source}")

    frame_counter = 0
    # Configure logging frequency for frames, defaulting to every 30 frames
    log_every_n_frames = config.get("log_every_n_frames", 30)
    
    try:
        # Main loop: continue until a shutdown signal is received
        while not shutdown_event.is_set():
            ret, frame = video_capture.read()
            # If frame reading fails, log an error and break the loop
            if not ret:
                logger.error(f"[{process_name}] Failed to read frame from video source: {video_source}. Breaking loop.")
                break
            
            # Get frame dimensions
            height, width, _ = frame.shape
            
            # Encode the frame to JPEG format for efficient transfer
            success, encoded_image = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            if not success:
                logger.warning(f"[{process_name}] Failed to encode frame to JPEG. Skipping this frame.")
                continue
            jpeg_binary = encoded_image.tobytes()
            
            # Construct the frame message with metadata
            message = {
                "frame_id": str(uuid.uuid4()), # Generate a unique ID for each frame
                "camera_id": config.get("camera_id", "default_cam"), # Camera ID from config or default
                "timestamp": time.time(), # Current timestamp for the frame
                "frame_data_jpeg": jpeg_binary,
                "frame_width": width,
                "frame_height": height,
                "source": video_source # Original video source identifier
            }
            
            try:
                # Attempt to put the message into the output queue with a timeout
                output_queue.put(message, timeout=0.5)
                frame_counter += 1
                # Log frame processing status periodically
                if frame_counter % log_every_n_frames == 0:
                    logger.debug(f"[{process_name}] Frame {message['frame_id']} (count: {frame_counter}) put to queue. Current queue size: {output_queue.qsize()}.")
            except Full:
                # If the queue is full, log a warning and drop the current frame
                logger.warning(f"[{process_name}] Output queue is full. Frame {message['frame_id']} dropped.")
                continue
    except KeyboardInterrupt:
        logger.info(f"[{process_name}] KeyboardInterrupt received. Shutting down.")
        shutdown_event.set()
        if not output_queue.full():
            try:
                output_queue.put(None, timeout=0.1) # Signal downstream to stop gracefully
            except Exception as e:
                logger.warning(f"[{process_name}] Could not put None to output_queue on shutdown: {e}")
    except Exception as e:
        # Catch any unexpected exceptions and log them with traceback information
        logger.exception(f"[{process_name}] Error in frame grabber process: {e}")
        raise # Re-raise the exception to ensure the process terminates if an unrecoverable error occurs
    finally:
        # Ensure the video capture object is released when the process exits
        logger.info(f"[{process_name}] Frame grabber process cleaning up and exiting...")
        video_capture.release()
        # Optionally, signal downstream processes about shutdown by putting None to queue
        # This block is commented out as the shutdown signal is managed by shutdown_event.
        # if output_queue and not shutdown_event.is_set():
        #     try:
        #         output_queue.put(None, timeout=0.1) # Signal downstream to stop gracefully
        #     except Exception as e:
        #         logger.warning(f"[{process_name}] Could not put None to output_queue on shutdown: {e}")

if __name__ == "__main__":
    # This block is for direct testing of the frame_grabber_process function.
    # It sets up a mock environment to simulate queue and shutdown events.
    logger.remove()
    logger.add(lambda msg: print(msg), format="{time} {level} {message}")
    mock_shutdown_event = mp.Event()
    
    class MockQueue:
        """
        A simple mock queue for testing purposes, mimicking the put method.
        """
        def put(self, item, timeout=None):
            logger.info(f"Mock queue received item (simulated): {item['frame_id']}")
        def get(self, timeout=None):
            pass

    mock_output_queue = MockQueue()
    mock_config = {"video_source": 0, "camera_id": "test_cam", "log_every_n_frames": 1}

    logger.info("Direct test: Starting frame grabber process in a separate thread...")

    import threading
    test_thread = threading.Thread(
        target=frame_grabber_process, 
        args=(mock_config, mock_output_queue, mock_shutdown_event)
    )
    test_thread.name = "TestFrameGrabber"  # Assign a name for logging
    test_thread.start()
    
    time.sleep(5)  # Allow the process to run for 5 seconds
    logger.info("Direct test: Signaling shutdown event.")
    mock_shutdown_event.set()
    
    test_thread.join(timeout=5)  # Wait for the thread to complete its shutdown
    logger.info("Direct test: Frame grabber thread joined.")
    logger.info("Direct test finished.")
