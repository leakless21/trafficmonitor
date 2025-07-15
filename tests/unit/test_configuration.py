"""Test configuration loading and validation."""

import yaml
from pathlib import Path
import pytest


def test_default_config_exists():
    """Test that the default configuration file exists."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "base" / "default.yaml"
    assert config_path.exists(), f"Default configuration file not found: {config_path}"


def test_default_config_is_valid_yaml():
    """Test that the default configuration is valid YAML."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "base" / "default.yaml"
    
    if not config_path.exists():
        pytest.skip("Configuration file not found")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert isinstance(config, dict), "Configuration should be a dictionary"
    assert len(config) > 0, "Configuration should not be empty"


def test_required_config_sections_exist():
    """Test that required configuration sections exist."""
    config_path = Path(__file__).parent.parent.parent / "configs" / "base" / "default.yaml"
    
    if not config_path.exists():
        pytest.skip("Configuration file not found")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    required_sections = [
        "frame_grabber",
        "vehicle_detector", 
        "vehicle_tracker",
        "lp_detector",
        "ocr_reader",
        "vehicle_counter",
        "visualizer",
        "loguru",
        "database"
    ]
    
    for section in required_sections:
        assert section in config, f"Required configuration section '{section}' not found"


def test_environment_configs_exist():
    """Test that environment-specific configurations exist."""
    configs_dir = Path(__file__).parent.parent.parent / "configs" / "environments"
    
    dev_config = configs_dir / "development.yaml"
    prod_config = configs_dir / "production.yaml"
    
    assert dev_config.exists(), "Development configuration not found"
    assert prod_config.exists(), "Production configuration not found"


def test_environment_configs_are_valid():
    """Test that environment configurations are valid YAML."""
    configs_dir = Path(__file__).parent.parent.parent / "configs" / "environments"
    
    for config_file in ["development.yaml", "production.yaml"]:
        config_path = configs_dir / config_file
        
        if not config_path.exists():
            continue
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert isinstance(config, dict), f"{config_file} should be a dictionary"