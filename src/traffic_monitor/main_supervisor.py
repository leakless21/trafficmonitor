import multiprocessing as mp
import time
import loguru
from .utils.logging_config import setup_logging
from .services.frame_grabber import frame_grabber_process

def main():
    setup_logging()
    loguru.logger.info("Starting main supervisor process...")
    shutdown_event = mp.Event()

    fg_config = {"source": "dummy_video_from_supervisor", "service_name": "FrameGrabber"}

    frame_grabber_output_queue = mp.Queue(maxsize=10)

    fg_process = mp.Process(
        target=frame_grabber_process,
        name="FrameGrabber",
        args=(
            fg_config,
            frame_grabber_output_queue,
            shutdown_event
        )
    )

    fg_process.start()
    processes = [fg_process]

    try:
        while not shutdown_event.is_set():
            all_alive = True
            for process in processes:
                if not process.is_alive():
                    loguru.logger.error(f"Process {process.name} has died. Shutting down...")
                    all_alive = False
                    shutdown_event.set()
                    break
            if not all_alive:
                break
            time.sleep(0.5)

    except KeyboardInterrupt:
        loguru.logger.info("Received keyboard interrupt. Shutting down...")
        shutdown_event.set()

    finally:
        loguru.logger.info("Starting cleanup...")
        for process in processes:
            loguru.logger.info(f"Waiting for process {process.name} to finish...")
            if process.is_alive():
                process.join(timeout=5)
            if process.is_alive():
                loguru.logger.warning(f"Process {process.name} did not finish in time. Sending SIGKILL.")
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    loguru.logger.error(f"Process {process.name} failed to terminate.")
        else:
            loguru.logger.info("FrameGrabber process finished.")
        # At the very end of the finally block, or after it
        loguru.logger.info("Closing queues...")
        try:
            frame_grabber_output_queue.close()
            frame_grabber_output_queue.join_thread() # Wait for all items to be flushed
            loguru.logger.info("Queues closed.")
        except Exception as queue_error:
            loguru.logger.error(f"Error closing queues: {queue_error}")
        loguru.logger.info("Supervisor cleanup complete.")
    loguru.logger.info("Supervisor finished.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
