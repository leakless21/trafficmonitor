import multiprocessing as mp
import time
import os
import yaml
from loguru import logger
from pathlib import Path
from queue import Empty

from traffic_monitor.utils.custom_types import OCRResultMessage
from .utils.logging_config import setup_logging
from .utils.minidb import configure_database, init_db
from .services.distributor import distributor_process
from .services.frame_grabber import frame_grabber_process
from .utils.config_loader import load_config
from .services.vehicle_detector import vehicle_detector_process
from .services.vehicle_tracker import vehicle_tracker_process
from .services.lp_detector import lp_detector_process
from .services.ocr_reader import ocr_reader_process
from .services.vehicle_counter import vehicle_counter_process
from .services.visualizer import visualize_process

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

    # Configure and initialize database
    configure_database(config)
    init_db()

    # Log condensed config info at debug level
    logger.debug("Configuration loaded", 
                video_source=config.get("frame_grabber", {}).get("video_source"),
                tracker_type=config.get("vehicle_tracker", {}).get("tracker_type"),
                log_level=loguru_config.get("level", "INFO"))

    # Build service configurations
    fg_config = config.get("frame_grabber", {})
    fg_config["service_name"] = "FrameGrabber"
    fg_config["loguru"] = loguru_config

    vd_config = config.get("vehicle_detector", {})
    vd_config["service_name"] = "VehicleDetector"
    vd_config["loguru"] = loguru_config

    vt_config = config.get("vehicle_tracker", {})
    vt_config["service_name"] = "VehicleTracker"
    vt_config["class_mapping"] = config["vehicle_detector"]["class_mapping"]
    vt_config["loguru"] = loguru_config

    db_config = config.get("database", {})

    lp_config = config.get("lp_detector", {})
    lp_config["service_name"] = "LPDetector"
    lp_config["loguru"] = loguru_config
    lp_config["database"] = db_config

    ocr_config = config.get("ocr_reader", {})
    ocr_config["service_name"] = "OCRReader"
    ocr_config["loguru"] = loguru_config
    ocr_config["database"] = db_config

    vc_config = config.get("vehicle_counter", {})
    vc_config["service_name"] = "VehicleCounter"
    vc_config["loguru"] = loguru_config
    vc_config["database"] = db_config

    vis_config = config.get("visualizer", {})
    vis_config["service_name"] = "Visualizer"
    vis_config["loguru"] = loguru_config
    vis_config["database"] = db_config

    # Import queue utilities for mode-aware queue management
    from .utils.queue_utils import is_offline_mode, get_queue_size_for_mode
    
    # Determine processing mode and queue sizes
    offline_mode = is_offline_mode(vis_config)
    queue_size = get_queue_size_for_mode(offline_mode)
    
    mode_desc = "offline (preserve all frames)" if offline_mode else "real-time (drop old frames)"
    logger.info(f"Queue management mode: {mode_desc}, queue size: {queue_size if queue_size > 0 else 'unbounded'}")
    
    # Create queues with mode-appropriate sizing
    frame_grabber_output_queue = mp.Queue(maxsize=queue_size)
    vehicle_detector_output_queue = mp.Queue(maxsize=queue_size)
    vehicle_tracker_output_queue = mp.Queue(maxsize=queue_size)
    lp_detector_output_queue = mp.Queue(maxsize=queue_size)
    ocr_reader_output_queue = mp.Queue(maxsize=queue_size)
    vehicle_counter_output_queue = mp.Queue(maxsize=queue_size)
    visualizer_input_queue = mp.Queue(maxsize=queue_size)

    lp_detector_input_queue = mp.Queue(maxsize=queue_size)
    vehicle_counter_input_queue = mp.Queue(maxsize=queue_size)

    # Pass offline mode to services that need it
    fg_config["offline_mode"] = offline_mode
    vd_config["offline_mode"] = offline_mode
    vt_config["offline_mode"] = offline_mode
    lp_config["offline_mode"] = offline_mode
    ocr_config["offline_mode"] = offline_mode
    vc_config["offline_mode"] = offline_mode

    # Create process configurations
    process_configs = [
        ("FrameGrabber", frame_grabber_process, (fg_config, frame_grabber_output_queue, shutdown_event)),
        ("VehicleDetector", vehicle_detector_process, (vd_config, frame_grabber_output_queue, vehicle_detector_output_queue, shutdown_event)),
        ("VehicleTracker", vehicle_tracker_process, (vt_config, vehicle_detector_output_queue, vehicle_tracker_output_queue, shutdown_event)),
        ("LPDetector", lp_detector_process, (lp_config, lp_detector_input_queue, lp_detector_output_queue, shutdown_event)),
        ("OCRReader", ocr_reader_process, (ocr_config, lp_detector_output_queue, ocr_reader_output_queue, shutdown_event)),
        ("VehicleCounter", vehicle_counter_process, (vc_config, vehicle_counter_input_queue, vehicle_counter_output_queue, shutdown_event)),
        ("Distributor", distributor_process, (offline_mode, vehicle_tracker_output_queue, [lp_detector_input_queue, vehicle_counter_input_queue, visualizer_input_queue], shutdown_event)),
        ("Visualizer", visualize_process, (vis_config, visualizer_input_queue, ocr_reader_output_queue, vehicle_counter_output_queue, shutdown_event)),
    ]

    # Start all processes
    processes = []
    for name, target, args in process_configs:
        process = mp.Process(target=target, name=name, args=args)
        process.start()
        processes.append(process)
        logger.info(f"Started {name} process with PID {process.pid}")

    try:
        logger.info("All processes started. Press Ctrl+C to quit.")
        while not shutdown_event.is_set():
            if not all(process.is_alive() for process in processes):
                logger.error("One or more processes died. Shutting down.")
                shutdown_event.set()
                break
            time.sleep(0.5)
        logger.info("Main loop finished. Shutting down.")
        shutdown_event.set()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down.")
        shutdown_event.set()
    
    finally:
        logger.info("Starting cleanup...")
        for process in processes:
            if process.is_alive():
                process.join(timeout=5)
            if process.is_alive():
                logger.warning(f"Process {process.name} did not finish in time. Terminating.")
                process.terminate()
                process.join(timeout=2)
            else:
                logger.debug(f"Process {process.name} finished cleanly")
        
        logger.info("Closing queues...")
        queues = [
            frame_grabber_output_queue, vehicle_detector_output_queue, vehicle_tracker_output_queue,
            lp_detector_output_queue, ocr_reader_output_queue, vehicle_counter_output_queue,
            visualizer_input_queue, lp_detector_input_queue, vehicle_counter_input_queue
        ]
        
        for queue in queues:
            try:
                queue.close()
                queue.join_thread()
            except Exception as e:
                logger.debug(f"Error closing queue: {e}")
        
        logger.info("Supervisor cleanup complete.")
        logger.info("Supervisor finished.")


if __name__ == "__main__":
    main()
