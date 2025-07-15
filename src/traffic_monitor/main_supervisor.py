import multiprocessing as mp
import time
from loguru import logger
from pathlib import Path

from .utils.logging_config import setup_logging
from .utils.minidb import configure_database, init_db
from .services.event_distribution_service import event_distribution_process
from .services.frame_capture_service import frame_capture_process
from .utils.config_loader import load_config
from .services.vehicle_detection_service import vehicle_detection_process
from .services.vehicle_tracking_service import vehicle_tracking_process
from .services.license_plate_detection_service import license_plate_detection_process
from .services.text_recognition_service import text_recognition_process
from .services.vehicle_counting_service import vehicle_counting_process
from .services.visualization_service import visualization_process
from .services.summary_service import summary_service_process

def main(config=None):
    # Ensure consistent multiprocessing start method across platforms (Linux default is 'fork')
    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
            logger.debug("Multiprocessing start method set to 'spawn' for cross-platform compatibility")
    except RuntimeError:
        # Start method already set elsewhere; proceed
        pass

    logger.info("Starting main supervisor process...")
    shutdown_event = mp.Event()

    # Use absolute path to ensure it works regardless of working directory
    project_root = Path(__file__).parent.parent.parent
    default_config_path = project_root / "configs" / "base" / "default.yaml"

    # -------------------------------------------------------------
    # Load configuration
    # -------------------------------------------------------------
    # "config" may be:
    #   1. A dict with overrides from the CLI.
    #   2. A path to a YAML file.
    #   3. None, if no config was specified at all.
    #
    # The goal is to always have a base config from settings.yaml and
    # merge any provided config on top of it.

    # Always load the default configuration as the base
    config_dict = load_config(default_config_path)
    if config_dict is None:
        logger.error("CRITICAL: Default settings.yaml could not be loaded. Exiting.")
        return

    if isinstance(config, (str, Path)):
        # If a path was provided, load that config and merge it
        logger.info(f"Loading config from path: {config}")
        override_config = load_config(config)
        if override_config:
            config_dict.update(override_config) # Simple update, can be improved
    elif isinstance(config, dict) and config:
        # If a dict was provided (from CLI), merge it
        logger.info("Merging CLI/interactive config with defaults.")
        # A simple update is not enough for nested dicts. We need a deep merge.
        def _deep_update(dest: dict, src: dict):
            """Recursively update dict dest with src (src overrides)."""
            for key, val in src.items():
                if isinstance(val, dict) and isinstance(dest.get(key), dict):
                    _deep_update(dest[key], val)
                else:
                    dest[key] = val
            return dest
        config_dict = _deep_update(config_dict, config)

    # Abort if configuration could not be loaded
    if config_dict is None:
        logger.error("Failed to load configuration. Exiting.")
        return

    loguru_config = config_dict.get("loguru", {})
    setup_logging(loguru_config)

    # Configure and initialize database
    configure_database(config_dict) if config_dict else configure_database({})
    init_db()

    # Log condensed config info at debug level
    logger.debug("Configuration loaded", 
                video_source=config_dict.get("frame_grabber", {}).get("video_source"),
                tracker_type=config_dict.get("vehicle_tracker", {}).get("tracker_type"),
                log_level=loguru_config.get("level", "INFO"))

    # Build service configurations
    fg_config = config_dict.get("frame_grabber", {}) if config_dict else {}
    fg_config["service_name"] = "FrameCaptureService"
    fg_config["loguru"] = loguru_config

    vd_config = config_dict.get("vehicle_detector", {}) if config_dict else {}
    vd_config["service_name"] = "VehicleDetectionService"
    vd_config["loguru"] = loguru_config

    vt_config = config_dict.get("vehicle_tracker", {}) if config_dict else {}
    vt_config["service_name"] = "VehicleTrackingService"
    vt_config["class_mapping"] = config_dict.get("vehicle_detector", {}).get("class_mapping", {})
    vt_config["loguru"] = loguru_config

    db_config = config_dict.get("database", {}) if config_dict else {}

    lp_config = config_dict.get("lp_detector", {}) if config_dict else {}
    lp_config["service_name"] = "LicensePlateDetectionService"
    lp_config["loguru"] = loguru_config
    lp_config["database"] = db_config

    ocr_config = config_dict.get("ocr_reader", {}) if config_dict else {}
    ocr_config["service_name"] = "TextRecognitionService"
    ocr_config["loguru"] = loguru_config
    ocr_config["database"] = db_config

    vc_config = config_dict.get("vehicle_counter", {}) if config_dict else {}
    vc_config["service_name"] = "VehicleCountingService"
    vc_config["loguru"] = loguru_config
    vc_config["database"] = db_config

    vis_config = config_dict.get("visualizer", {})
    vis_config["service_name"] = "VisualizationService"
    vis_config["loguru"] = loguru_config
    vis_config["database"] = db_config

    # Summary service configuration - include all relevant configs for reporting
    summary_config = config_dict.get("summary_service", {})
    summary_config["service_name"] = "SummaryService"
    summary_config["loguru"] = loguru_config
    summary_config["video_source"] = fg_config.get("video_source", "Unknown")
    summary_config["database"] = db_config

    # -----------------------------------------------------------------
    # Ensure both visualizer.save_path and summary_output_dir point to the
    # SAME session subfolder. When one (or both) are missing, we create
    # a fresh timestamped directory under data/outputs/videos.
    # -----------------------------------------------------------------
    from datetime import datetime
    default_output_root = Path("data/outputs/videos")
    default_output_root.mkdir(parents=True, exist_ok=True)

    save_path = vis_config.get("save_path")
    # If save_path is missing or equals the root directory, create subfolder
    if not save_path or Path(save_path).resolve() == default_output_root.resolve():
        # Create new session folder
        session_dir = default_output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir.mkdir(parents=True, exist_ok=True)
        vis_config["save_path"] = str(session_dir)
        summary_config.setdefault("summary_output_dir", str(session_dir))
    else:
        # save_path already looks like a custom folder; use it for reports as well
        summary_config.setdefault("summary_output_dir", save_path)

    # -------------------------------------------------------------
    # Determine processing mode BEFORE adding it into summary config
    # -------------------------------------------------------------
    from .utils.queue_utils import is_offline_mode, get_queue_size_for_mode
    offline_mode = is_offline_mode(vis_config)

    # Add configuration details from all services for comprehensive reporting
    summary_config.update({
        # Frame grabber config
        "resize_resolution": fg_config.get("resize_resolution"),
        "process_every_n_frame": fg_config.get("process_every_n_frame"),
        "offline_mode": offline_mode,
        
        # Vehicle detection config
        "model_path": vd_config.get("model_path"),
        "conf_threshold": vd_config.get("conf_threshold"),
        "class_mapping": vd_config.get("class_mapping"),
        "device": vd_config.get("device", vt_config.get("device")),
        
        # Vehicle tracking config
        "tracker_type": vt_config.get("tracker_type"),
        "half": vt_config.get("half"),
        "reid_model_path": vt_config.get("reid_model_path"),
        
        # License plate detection config
        "lp_model_path": lp_config.get("model_path"),
        
        # OCR config
        "backend": ocr_config.get("backend"),
        "lang": ocr_config.get("lang"),
        "use_gpu": ocr_config.get("use_gpu"),
        "hub_model_name": ocr_config.get("hub_model_name"),
        
        # Vehicle counting config
        "counting_lines": vc_config.get("counting_lines"),
        
        # Visualization config
        "save_to_file": vis_config.get("save_to_file"),
        "save_path": vis_config.get("save_path"),
        "output_fourcc": vis_config.get("output_fourcc")
    })

    # -------------------------------------------------------------
    # Queue sizing based on processing mode
    # -------------------------------------------------------------
    queue_size = get_queue_size_for_mode(offline_mode)
    
    mode_desc = "offline (preserve all frames)" if offline_mode else "real-time (drop old frames)"
    logger.info(f"Queue management mode: {mode_desc}, queue size: {queue_size if queue_size > 0 else 'unbounded'}")
    
    # Create queues with mode-appropriate sizing
    frame_capture_output_queue = mp.Queue(maxsize=queue_size)
    vehicle_detection_output_queue = mp.Queue(maxsize=queue_size)
    vehicle_tracking_output_queue = mp.Queue(maxsize=queue_size)
    license_plate_detection_output_queue = mp.Queue(maxsize=queue_size)
    text_recognition_output_queue = mp.Queue(maxsize=queue_size)
    vehicle_counting_output_queue = mp.Queue(maxsize=queue_size)
    # Dedicated queues to ensure the visualiser receives a *copy* of each message
    vehicle_counting_vis_queue = mp.Queue(maxsize=queue_size)
    text_recognition_vis_queue = mp.Queue(maxsize=queue_size)
    visualization_input_queue = mp.Queue(maxsize=queue_size)

    license_plate_detection_input_queue = mp.Queue(maxsize=queue_size)
    vehicle_counting_input_queue = mp.Queue(maxsize=queue_size)
    
    # Summary service queues (copy data from other queues for metrics collection)
    summary_tracking_queue = mp.Queue(maxsize=queue_size)
    summary_count_queue = mp.Queue(maxsize=queue_size)
    summary_ocr_queue = mp.Queue(maxsize=queue_size)

    # Pass offline mode to services that need it
    fg_config["offline_mode"] = offline_mode
    vd_config["offline_mode"] = offline_mode
    vt_config["offline_mode"] = offline_mode
    lp_config["offline_mode"] = offline_mode
    ocr_config["offline_mode"] = offline_mode
    vc_config["offline_mode"] = offline_mode

    # Create process configurations
    process_configs = [
        ("FrameCaptureService", frame_capture_process, (fg_config, frame_capture_output_queue, shutdown_event)),
        ("VehicleDetectionService", vehicle_detection_process, (vd_config, frame_capture_output_queue, vehicle_detection_output_queue, shutdown_event)),
        ("VehicleTrackingService", vehicle_tracking_process, (vt_config, vehicle_detection_output_queue, vehicle_tracking_output_queue, shutdown_event)),
        ("LicensePlateDetectionService", license_plate_detection_process, (lp_config, license_plate_detection_input_queue, license_plate_detection_output_queue, shutdown_event)),
        ("TextRecognitionService", text_recognition_process, (ocr_config, license_plate_detection_output_queue, text_recognition_output_queue, shutdown_event)),
        ("VehicleCountingService", vehicle_counting_process, (vc_config, vehicle_counting_input_queue, vehicle_counting_output_queue, shutdown_event)),
        ("EventDistributionService", event_distribution_process, (offline_mode, vehicle_tracking_output_queue, [license_plate_detection_input_queue, vehicle_counting_input_queue, visualization_input_queue, summary_tracking_queue], shutdown_event)),
        ("VisualizationService", visualization_process, (vis_config, visualization_input_queue, text_recognition_vis_queue, vehicle_counting_vis_queue, shutdown_event)),
    ]
    
    # Add summary service if enabled
    if summary_config.get("enabled", True):
        # Add additional event distribution processes to send copies to summary service
        process_configs.extend([
            ("CountDistributionService", event_distribution_process, (offline_mode, vehicle_counting_output_queue, [summary_count_queue, vehicle_counting_vis_queue], shutdown_event)),
            ("OCRDistributionService", event_distribution_process, (offline_mode, text_recognition_output_queue, [summary_ocr_queue, text_recognition_vis_queue], shutdown_event)),
            ("SummaryService", summary_service_process, (summary_config, summary_tracking_queue, summary_count_queue, summary_ocr_queue, shutdown_event))
        ])

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
            # If any process exited with a non-zero code → treat as error and shut down
            for process in processes:
                if process.exitcode not in (None, 0):
                    logger.error(f"Process {process.name} exited with code {process.exitcode}. Initiating shutdown.")
                    shutdown_event.set()
                    break

            # If every child has finished successfully, we're done
            if all(p.exitcode == 0 for p in processes):
                logger.info("All child processes finished gracefully. Exiting main loop.")
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
                process.join(timeout=10)
            if process.is_alive():
                logger.warning(f"Process {process.name} did not finish in time. Terminating.")
                process.terminate()
                process.join(timeout=10)
            else:
                logger.debug(f"Process {process.name} finished cleanly")
        
        logger.info("Closing queues...")
        queues = [
            frame_capture_output_queue, vehicle_detection_output_queue, vehicle_tracking_output_queue,
            license_plate_detection_output_queue, text_recognition_output_queue, vehicle_counting_output_queue,
            vehicle_counting_vis_queue, text_recognition_vis_queue,
            visualization_input_queue, license_plate_detection_input_queue, vehicle_counting_input_queue,
            summary_tracking_queue, summary_count_queue, summary_ocr_queue
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
