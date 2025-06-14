import yaml
from pathlib import Path
from loguru import logger

def load_config(config_path: str | Path) -> dict | None:
    """
    Loads a YAML configuration file.

    Args:
        config_path: The path to the YAML configuration file.

    Returns:
        A dictionary with the configuration, or None if an error occurred.
    """
    try:
        path = Path(config_path)
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Config loaded successfully from {path}")
        return config
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        return None
    except yaml.YAMLError as e:
        logger.error(f"Error loading config: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error loading config: {e}")
        return None


if __name__ == "__main__":
    test_path = "src/traffic_monitor/config/settings.yaml"
    logger.info(f"--- Running direct test for config_loader.py ---")
    logger.info(f"Attempting to load: {test_path}")
    
    config = load_config(test_path)
    
    if config:
        logger.info("Test load successful. Config contents:")
        # Pretty print the dictionary for better readability
        import json
        logger.info(json.dumps(config, indent=2))
    else:
        logger.error("Test load failed.")    