import yaml
import sys
import os
from loguru import logger
from typing import Dict, Any, Optional

def setup_logging(loguru_config: Dict[str, Any] | None = None):
    """
    Configures Loguru logger with parameters from the provided dictionary
    or from a settings.yaml file if no dictionary is provided.
    Includes process name for clarity in multiprocessing.
    """
    logger.remove()

    if loguru_config is None:
        try:
            with open("src/traffic_monitor/config/settings.yaml", "r") as f:
                settings = yaml.safe_load(f)
            loguru_config = settings.get("loguru", {})
        except FileNotFoundError:
            logger.warning("settings.yaml not found. Using default logging configuration.")
            loguru_config = {}
        except yaml.YAMLError as e:
            logger.error(f"Error reading settings.yaml: {e}. Using default logging configuration.")
            loguru_config = {}

    # Default values
    level = "INFO"
    log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                  "<level>{level: <8}</level> | "
                  "<cyan>{process.name: <15}</cyan> | "
                  "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    # File logging options
    log_file_path = None
    log_file_rotation = "10 MB"
    log_file_retention = "7 days"
    log_file_compression = "zip"
    terminal_output_enabled = True
    log_file_overwrite = True

    if loguru_config:
        level = loguru_config.get("level", level)
        log_format = loguru_config.get("format", log_format)
        log_file_path = loguru_config.get("file_path", log_file_path)
        log_file_rotation = loguru_config.get("file_rotation", log_file_rotation)
        log_file_retention = loguru_config.get("file_retention", log_file_retention)
        log_file_compression = loguru_config.get("file_compression", log_file_compression)
        terminal_output_enabled = loguru_config.get("terminal_output_enabled", terminal_output_enabled)
        log_file_overwrite = loguru_config.get("log_file_overwrite", log_file_overwrite)

    if log_file_overwrite and log_file_path and os.path.exists(log_file_path):
        try:
            os.remove(log_file_path)
            logger.info(f"Existing log file '{log_file_path}' removed for overwrite.")
        except OSError as e:
            logger.error(f"Error removing existing log file '{log_file_path}': {e}")

    if terminal_output_enabled:
        logger.add(
            sys.stdout, level=level,
            format=log_format
        )
    
    if log_file_path:
        logger.add(
            log_file_path, 
            level=level, 
            format=log_format, 
            rotation=log_file_rotation, 
            retention=log_file_retention, 
            compression=log_file_compression
        )

    logger.info("Logger initialized from setup_logging function.")