import multiprocessing as mp
import time
import os
import yaml
from loguru import logger
from pathlib import Path
from queue import Empty
from .utils.logging_config import setup_logging
from .services.frame_grabber import frame_grabber_process
from .utils.config_loader import load_config
from .services.vehicle_detector import vehicle_detector_process


def main():
    setup_logging()
    logger.info("Starting main supervisor process...")
    shutdown_event = mp.Event()

    # Use absolute path to ensure it works regardless of working directory
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "src" / "traffic_monitor" / "config" / "settings.yaml"

    config = load_config(config_path)
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return
    else:
        logger.info(f"Loaded configuration: {config}")

    fg_config = config.get("frame_grabber", {})
    fg_config["service_name"] = "FrameGrabber"

    vd_config = config.get("vehicle_detector", {})
    vd_config["service_name"] = "VehicleDetector"

    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return

    frame_grabber_output_queue = mp.Queue(maxsize=10)
    vehicle_detector_output_queue = mp.Queue(maxsize=10)

    # FrameGrabber process
    fg_process = mp.Process(
        target=frame_grabber_process,
        name="FrameGrabber",
        args=(
            fg_config,
            frame_grabber_output_queue,
            shutdown_event
        )
    )
    # VehicleDetector process
    vd_process = mp.Process(
        target=vehicle_detector_process,
        name="VehicleDetector",
        args=(
            vd_config,
            frame_grabber_output_queue,
            vehicle_detector_output_queue,
            shutdown_event
        )
    )

    # Start the processes
    fg_process.start()
    vd_process.start()

    processes = [fg_process, vd_process]

    try:
        logger.info("Starting main loop...")
        while not shutdown_event.is_set():
            all_processes_alive = True
            for p in processes:
                if not p.is_alive():
                    logger.error(f"Process {p.name} has died. Shutting down.")
                    all_processes_alive = False
                    #shutdown_event.set()
                    break
            if not all_processes_alive:
                shutdown_event.set()
                break
            time.sleep(0.5)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down.")
        shutdown_event.set()
    
    finally:
        logger.info("Starting cleanup...")
        for process in processes:
            logger.info(f"Waiting for process {process.name} to finish...")
            if process.is_alive():
                process.join(timeout=5)
            if process.is_alive():
                logger.warning(f"Process {process.name} did not finish in time. Sending SIGKILL.")
                process.terminate()
                process.join(timeout=2)
            else:
                logger.info(f"Process {process.name} finished.")
        logger.info("Closing queues...")
        try:
            frame_grabber_output_queue.close()
            frame_grabber_output_queue.join_thread() # Wait for all items to be flushed
            vehicle_detector_output_queue.close()
            vehicle_detector_output_queue.join_thread() # Wait for all items to be flushed
            logger.info("Queues closed.")
        except Exception as queue_error:
            logger.error(f"Error closing queues: {queue_error}")
        logger.info("Supervisor cleanup complete.")
    logger.info("Supervisor finished.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
