import multiprocessing as mp
import time
import loguru
from .utils.logging_config import setup_logging

def main():
    setup_logging()
    loguru.logger.info("Starting main supervisor process...")
    shutdown_event = mp.Event()
    frame_grabber_output_queue = mp.Queue(maxsize=10)
    frame_grabber_process = mp.Process(
        target=frame_grabber_process,
        name="FrameGrabber",
        args=(
            {"source": "dummy_source"},
            None,
            shutdown_event
        )
    )
    frame_grabber_process.start()
    processes = [frame_grabber_process]
    try:
        while not shutdown_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        loguru.logger.info("Received keyboard interrupt. Shutting down...")
        shutdown_event.set()
    finally:
        loguru.logger.info("Waiting for FrameGrabber process to finish...")
        if frame_grabber_process.is_alive():
            frame_grabber_process.join(timeout=5)
        if frame_grabber_process.is_alive():
            loguru.logger.warning("FrameGrabber process did not finish in time. Sending SIGKILL.")
            frame_grabber_process.terminate()
            frame_grabber_process.join(timeout=5)
        else:
            loguru.logger.info("FrameGrabber process finished.")
        # At the very end of the finally block, or after it
        loguru.logger.info("Closing queues...")
        frame_grabber_output_queue.close()
        frame_grabber_output_queue.join_thread() # Wait for all items to be flushed
        loguru.logger.info("Queues closed.")
        loguru.logger.info("Supervisor cleanup complete.")
    loguru.logger.info("Supervisor finished.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
