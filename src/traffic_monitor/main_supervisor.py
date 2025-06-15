import multiprocessing as mp
import time
import os
import yaml
from loguru import logger
from pathlib import Path
from queue import Empty

from traffic_monitor.utils.custom_types import PlateDetectionMessage
from .utils.logging_config import setup_logging
from .services.frame_grabber import frame_grabber_process
from .utils.config_loader import load_config
from .services.vehicle_detector import vehicle_detector_process
from .services.vehicle_tracker import vehicle_tracker_process
from .services.lp_detector import lp_detector_process


def main():
    logger.info("Starting main supervisor process...")
    shutdown_event = mp.Event()

    # Use absolute path to ensure it works regardless of working directory
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / "src" / "traffic_monitor" / "config" / "settings.yaml"

    config = load_config(config_path)
    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return

    loguru_config = config.get("loguru", {})
    setup_logging(loguru_config)

    logger.info(f"Loaded configuration: {config}")

    fg_config = config.get("frame_grabber", {})
    fg_config["service_name"] = "FrameGrabber"

    vd_config = config.get("vehicle_detector", {})
    vd_config["service_name"] = "VehicleDetector"

    vt_config = config.get("vehicle_tracker", {})
    vt_config["service_name"] = "VehicleTracker"
    vt_config["class_mapping"] = config["vehicle_detector"]["class_mapping"]

    lp_config = config.get("lp_detector", {})
    lp_config["service_name"] = "LPDetector"

    if not config:
        logger.error("Failed to load configuration. Exiting.")
        return

    frame_grabber_output_queue = mp.Queue(maxsize=1000)
    vehicle_detector_output_queue = mp.Queue(maxsize=1000)
    vehicle_tracker_output_queue = mp.Queue(maxsize=1000)
    lp_detector_output_queue = mp.Queue(maxsize=1000)
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
    # VehicleTracker process
    vt_process = mp.Process(
        target=vehicle_tracker_process,
        name="VehicleTracker",
        args=(vt_config, vehicle_detector_output_queue, vehicle_tracker_output_queue, shutdown_event)
    )
    # LPDetector process
    lp_process = mp.Process(
        target=lp_detector_process,
        name="LPDetector",
        args=(lp_config, vehicle_tracker_output_queue, lp_detector_output_queue, shutdown_event)
    )
    # Start the processes
    fg_process.start()
    logger.info(f"MainProcess] FrameGrabber process started with PID {fg_process.pid}.")

    vd_process.start()
    logger.info(f"MainProcess] VehicleDetector process started with PID {vd_process.pid}.")

    vt_process.start()
    logger.info(f"MainProcess] VehicleTracker process started with PID {vt_process.pid}.")

    lp_process.start()
    logger.info(f"MainProcess] LPDetector process started with PID {lp_process.pid}.")

    processes = [fg_process, vd_process, vt_process, lp_process]

    try:
        logger.info("Starting main loop...")
        plates_detected = 0
        while True:
            try:
                plate_message: PlateDetectionMessage = lp_detector_output_queue.get(timeout=1)
                if plate_message is None:
                    logger.info(f"MainProcess] Received None message, shutting down.")
                    break
                plates_detected += 1
                logger.info(f"MainProcess] Detected plate {plates_detected} for vehicle {plate_message['vehicle_id']} with plate: {plate_message['plate_bbox_original']} (conf: {plate_message['plate_confidence']})")
            except Empty:
                if not any(process.is_alive() for process in processes):
                    logger.error("MainProcess] All processes are dead. Shutting down.")
                    break
                continue
        logger.info("Consumer loop finished. Shutting down.")
        shutdown_event.set()

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
            vehicle_tracker_output_queue.close()
            vehicle_tracker_output_queue.join_thread() # Wait for all items to be flushed
            lp_detector_output_queue.close()
            lp_detector_output_queue.join_thread() # Wait for all items to be flushed
            logger.info("Queues closed.")
        except Exception as queue_error:
            logger.error(f"Error closing queues: {queue_error}")
        logger.info("Supervisor cleanup complete.")
    logger.info("Supervisor finished.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
