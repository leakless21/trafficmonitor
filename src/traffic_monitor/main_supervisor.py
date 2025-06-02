import multiprocessing
import time
import loguru
from .utils.logging_config import setup_logging

def main():
    setup_logging()
    loguru.logger.info("Starting main supervisor process...")
    shutdown_event = multiprocessing.Event()
    try:
        while not shutdown_event.is_set():
            time.sleep(0.5)
    except KeyboardInterrupt:
        loguru.logger.info("Received keyboard interrupt. Shutting down...")
        shutdown_event.set()
    finally:
        loguru.logger.info("Supervisor cleanup complete.")
    loguru.logger.info("Supervisor finished.")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
