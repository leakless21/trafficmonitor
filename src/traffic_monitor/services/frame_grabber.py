import time
import multiprocessing as mp
from loguru import logger
from typing import Any, Dict

def frame_grabber_process(
    config: Dict[str, Any],
    output_queue,  # mp.Queue
    shutdown_event  # mp.Event
):
    """
    Frame grabber process function.
    
    Args:
        config: Configuration dictionary
        output_queue: Multiprocessing queue for output
        shutdown_event: Multiprocessing event for shutdown signal
    """
    process_name = mp.current_process().name
    logger.info(f"{process_name} Frame Grabber process started. Config: {config}")

    try:
        count = 0
        while not shutdown_event.is_set():
            count += 1
            logger.info(f"{process_name} Frame {count} grabbed. Putting to queue: {output_queue is not None}")
            # output_queue.put(f"Frame {count}")  # We'll add this when we have a queue
            time_slept = 0
            sleep_interval = 0.1
            total_sleep_time = 2.0
            while time_slept < total_sleep_time:
                time.sleep(sleep_interval)
                time_slept += sleep_interval
                
                # Check for shutdown during sleep to be more responsive
                if shutdown_event.is_set():
                    logger.info(f"{process_name} Shutdown event received during sleep. Exiting...")
                    return
            
    except Exception as e:
        logger.error(f"{process_name} Error in frame grabber process: {e}")
    finally:
        logger.info(f"{process_name} Frame grabber process cleaning up and exiting...")
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
    mock_config = {"source": "dummy_source"}

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
