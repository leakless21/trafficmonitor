import yaml
import sys
import os
import re
from loguru import logger
from typing import Dict, Any


class SensitiveDataFilter:
    """Filter to redact sensitive information from log messages"""
    
    def __init__(self):
        # Patterns for sensitive data
        self.patterns = [
            (re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'), '[CARD_REDACTED]'),  # Credit cards
            (re.compile(r'\b[A-Z0-9]{2,4}-\d{3,4}\b'), '[PLATE_REDACTED]'),  # License plates
            (re.compile(r'password["\s]*[:=]["\s]*[^"\s,}]+', re.IGNORECASE), 'password="[REDACTED]"'),  # Passwords
            (re.compile(r'api[_-]?key["\s]*[:=]["\s]*[^"\s,}]+', re.IGNORECASE), 'api_key="[REDACTED]"'),  # API keys
        ]
    
    def __call__(self, record):
        """Filter function for Loguru"""
        message = record["message"]
        for pattern, replacement in self.patterns:
            message = pattern.sub(replacement, message)
        record["message"] = message
        return record


def setup_logging(loguru_config: Dict[str, Any] | None = None):
    """
    Configures Loguru logger with parameters from the provided dictionary
    or from a settings.yaml file if no dictionary is provided.
    Includes process name for clarity in multiprocessing.
    """
    # Only remove existing handlers in the main process to avoid disrupting parent configuration
    current_process_name = os.getenv('MULTIPROCESSING_PROCESS_NAME', 'MainProcess')
    if current_process_name == 'MainProcess':
        logger.remove()

    if loguru_config is None:
        try:
            with open("configs/base/default.yaml", "r") as f:
                settings = yaml.safe_load(f)
            loguru_config = settings.get("loguru", {})
        except FileNotFoundError:
            logger.warning("settings.yaml not found. Using default logging configuration.")
            loguru_config = {}
        except yaml.YAMLError as e:
            logger.error(f"Error reading settings.yaml: {e}. Using default logging configuration.")
            loguru_config = {}
    
    # Ensure loguru_config is a dict
    if loguru_config is None:
        loguru_config = {}

    # Environment-based configuration
    level = os.getenv("LOG_LEVEL", loguru_config.get("level", "INFO")).upper()
    use_json = os.getenv("LOG_FORMAT", "").lower() == "json" or loguru_config.get("use_json", False)
    
    # Determine format based on environment
    if use_json:
        log_format = (
            '{"timestamp": "{time:YYYY-MM-DD HH:mm:ss.SSS}", '
            '"level": "{level}", '
            '"process": "{process.name}", '
            '"name": "{name}", '
            '"function": "{function}", '
            '"line": "{line}", '
            '"message": "{message}"}'
        )
    else:
        log_format = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                      "<level>{level: <8}</level> | "
                      "<cyan>{process.name: <15}</cyan> | "
                      "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    # File logging options
    log_file_path = loguru_config.get("file_path")
    log_file_rotation = loguru_config.get("file_rotation", "10 MB")
    log_file_retention = loguru_config.get("file_retention", "7 days")
    log_file_compression = loguru_config.get("file_compression", "zip")
    terminal_output_enabled = loguru_config.get("terminal_output_enabled", True)
    log_file_overwrite = loguru_config.get("log_file_overwrite", True)

    # Create sensitive data filter
    sensitive_filter = SensitiveDataFilter()

    # Handle file overwrite only in main process
    if current_process_name == 'MainProcess' and log_file_overwrite and log_file_path and os.path.exists(log_file_path):
        try:
            os.remove(log_file_path)
            logger.info(f"Existing log file '{log_file_path}' removed for overwrite.")
        except OSError as e:
            logger.error(f"Error removing existing log file '{log_file_path}': {e}")

    # Add terminal handler if enabled
    if terminal_output_enabled:
        logger.add(
            sys.stdout, 
            level=level,
            format=log_format,
            filter=sensitive_filter,
            serialize=use_json
        )
    
    # Add file handler if configured
    if log_file_path:
        logger.add(
            log_file_path, 
            level=level, 
            format=log_format, 
            rotation=log_file_rotation, 
            retention=log_file_retention, 
            compression=log_file_compression,
            filter=sensitive_filter,
            serialize=use_json
        )

    # Disable noisy third-party libraries
    logger.disable("matplotlib")
    logger.disable("urllib3")
    logger.disable("requests")
    logger.disable("PIL")
    
    # Set third-party libraries to WARNING level minimum
    import logging
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logger.info("Logger initialized from setup_logging function.")