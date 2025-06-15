import yaml
import sys
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
    level = "DEBUG"
    log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                  "<level>{level: <8}</level> | "
                  "<cyan>{process.name: <15}</cyan> | "
                  "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

    if loguru_config:
        level = loguru_config.get("level", level)
        log_format = loguru_config.get("format", log_format)

    logger.add(
        sys.stdout, level=level,
        format=log_format
    )
    logger.info("Logger initialized from setup_logging function.")