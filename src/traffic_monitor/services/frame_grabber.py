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
    Frame grabber process function. 
    
    Args:
        config: Configuration dictionary
        output_queue: Multiprocessing queue for output
        shutdown_event: Multiprocessing event for shutdown signal
    """
    # Initialize logging for this child process
    setup_logging()
    
    process_name = mp.current_process().name
    logger.info(f"[{process_name}] Frame Grabber process started. Config: {config}")
    video_source = config.get("video_source")
    if not video_source:
        logger.error(f"[{process_name}] No video source found in config")
        return
    logger.info(f"[{process_name}] Video source found in config: {video_source}")
    video_capture = cv2.VideoCapture(video_source)
    if not video_capture.isOpened():
        logger.error(f"[{process_name}] Failed to open video source: {video_source}")
        return
    logger.info(f"[{process_name}] Video source opened successfully: {video_source}")

    frame_counter = 0
    log_every_n_frames = config.get("log_every_n_frames", 30)  # Default to every 30 frames
    
    try:
        while not shutdown_event.is_set():
            ret, frame = video_capture.read()
            if not ret:
                logger.error(f"[{process_name}] Failed to read frame from video source: {video_source}")
                break
            height, width, _ = frame.shape
            success, encoded_image = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            if not success:
                logger.warning(f"[{process_name}] Failed to encode frame to JPEG. Skipping frame.")
                continue
            jpeg_as_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
            message = {
                "frame_id": str(uuid.uuid4()),
                "camera_id": config.get("camera_id", "default_cam"),
                "timestamp": time.time(),
                "frame_data_jpeg": jpeg_as_base64,
                "frame_width": width,
                "frame_height": height,
                "source": video_source
            }
            try:
                output_queue.put(message, timeout=0.5)
                frame_counter += 1
                if frame_counter % log_every_n_frames == 0:
                    logger.debug(f"[{process_name}] Frame {message['frame_id']} put to queue. Queue size: {output_queue.qsize()} (Frame #{frame_counter})")
            except Full:
                logger.warning(f"[{process_name}] Queue is full. Frame {message['frame_id']} dropped.")
                continue
    except Exception as e:
        logger.exception(f"{process_name} Error in frame grabber process: {e}")
    finally:
        logger.info(f"{process_name} Frame grabber process cleaning up and exiting...")
        video_capture.release()
        # If output_queue exists and we need to signal downstream, we might put None here.
        # if output_queue:
        #     try:
        #         output_queue.put(None, timeout=0.1) # Signal downstream to stop
        #     except Exception:
        #         logger.warning(f"[{process_name}] Could not put None to output_queue on shutdown.")

if __name__ == "__main__":
    logger.remove()
    logger.add(lambda msg: print(msg), format="{time} {level} {message}")
    mock_shutdown_event = mp.Event()
    
    class MockQueue:
        def put(self, item, timeout=None):
            logger.info(f"Mock queue received: {item}")
        def get(self, timeout=None):
            pass

    mock_output_queue = MockQueue()
    mock_config = {"video_source": "dummy_source"}

    logger.info("Directly starting frame grabber process...")

    import threading
    test_thread = threading.Thread(
        target=frame_grabber_process, 
        args=(mock_config, mock_output_queue, mock_shutdown_event)
    )
    test_thread.name = "TestFrameGrabber"  # Manually set name for logger
    test_thread.start()
    
    time.sleep(5)  # Let it run for 5 seconds
    logger.info("Direct test: Setting shutdown event.")
    mock_shutdown_event.set()
    
    test_thread.join(timeout=5)  # Wait for the thread to finish
    logger.info("Direct test: Frame grabber thread joined.")
    logger.info("Direct test finished.")
