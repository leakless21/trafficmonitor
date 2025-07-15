"""
Unit tests for config loader utility.
Tests YAML configuration loading, validation, and error handling.
"""

import pytest
import yaml
import tempfile
import os
from unittest.mock import Mock, patch, mock_open
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from traffic_monitor.utils.config_loader import load_config


class TestConfigLoader:
    """Test configuration loading functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Sample valid configuration
        self.valid_config = {
            "frame_grabber": {
                "video_source": "test_video.mp4",
                "resize_resolution": [640, 480],
                "process_every_n_frame": 1
            },
            "vehicle_detector": {
                "model_path": "test_model.engine",
                "conf_threshold": 0.5,
                "class_mapping": {
                    0: "bicycle",
                    1: "bike", 
                    2: "bus",
                    3: "car",
                    4: "person",
                    5: "truck"
                }
            },
            "loguru": {
                "level": "INFO",
                "terminal_output_enabled": True
            }
        }
        
        # Sample invalid configurations
        self.invalid_yaml = "invalid: yaml: content: ["
        self.empty_config = {}

    def test_load_valid_config_file(self):
        """Test loading a valid YAML configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.valid_config, f)
            config_path = f.name
        
        try:
            # Load configuration
            loaded_config = load_config(config_path)
            
            # Verify configuration was loaded correctly
            assert loaded_config is not None, "Configuration should be loaded"
            assert isinstance(loaded_config, dict), "Configuration should be a dictionary"
            assert "frame_grabber" in loaded_config, "Should contain frame_grabber section"
            assert "vehicle_detector" in loaded_config, "Should contain vehicle_detector section"
            assert loaded_config["frame_grabber"]["video_source"] == "test_video.mp4"
            
        finally:
            os.unlink(config_path)

    def test_load_config_with_pathlib_path(self):
        """Test loading configuration using pathlib.Path object."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.valid_config, f)
            config_path = Path(f.name)
        
        try:
            # Load configuration using Path object
            loaded_config = load_config(config_path)
            
            # Verify configuration was loaded
            assert loaded_config is not None, "Configuration should be loaded with Path object"
            assert isinstance(loaded_config, dict), "Configuration should be a dictionary"
            
        finally:
            config_path.unlink()

    def test_load_nonexistent_config_file(self):
        """Test handling of non-existent configuration file."""
        nonexistent_path = "nonexistent_config.yaml"
        
        # Attempt to load non-existent file
        loaded_config = load_config(nonexistent_path)
        
        # Should return None for non-existent file
        assert loaded_config is None, "Should return None for non-existent file"

    def test_load_invalid_yaml_file(self):
        """Test handling of invalid YAML syntax."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(self.invalid_yaml)
            config_path = f.name
        
        try:
            # Attempt to load invalid YAML
            loaded_config = load_config(config_path)
            
            # Should return None for invalid YAML
            assert loaded_config is None, "Should return None for invalid YAML"
            
        finally:
            os.unlink(config_path)

    def test_load_empty_config_file(self):
        """Test loading an empty configuration file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.empty_config, f)
            config_path = f.name
        
        try:
            # Load empty configuration
            loaded_config = load_config(config_path)
            
            # Should load successfully but be empty
            assert loaded_config is not None, "Empty config should still load"
            assert isinstance(loaded_config, dict), "Should be a dictionary"
            assert len(loaded_config) == 0, "Should be empty"
            
        finally:
            os.unlink(config_path)

    def test_config_data_types(self):
        """Test that configuration data types are preserved."""
        config_with_types = {
            "string_value": "test_string",
            "integer_value": 42,
            "float_value": 3.14,
            "boolean_value": True,
            "list_value": [1, 2, 3],
            "dict_value": {"nested": "value"},
            "null_value": None
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_with_types, f)
            config_path = f.name
        
        try:
            # Load configuration
            loaded_config = load_config(config_path)
            
            # Verify data types are preserved
            assert isinstance(loaded_config["string_value"], str)
            assert isinstance(loaded_config["integer_value"], int)
            assert isinstance(loaded_config["float_value"], float)
            assert isinstance(loaded_config["boolean_value"], bool)
            assert isinstance(loaded_config["list_value"], list)
            assert isinstance(loaded_config["dict_value"], dict)
            assert loaded_config["null_value"] is None
            
            # Verify values
            assert loaded_config["string_value"] == "test_string"
            assert loaded_config["integer_value"] == 42
            assert loaded_config["float_value"] == 3.14
            assert loaded_config["boolean_value"] is True
            assert loaded_config["list_value"] == [1, 2, 3]
            assert loaded_config["dict_value"]["nested"] == "value"
            
        finally:
            os.unlink(config_path)

    def test_nested_configuration_structure(self):
        """Test loading of deeply nested configuration structures."""
        nested_config = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "deep_value": "found_it"
                        }
                    }
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(nested_config, f)
            config_path = f.name
        
        try:
            # Load nested configuration
            loaded_config = load_config(config_path)
            
            # Verify nested structure is preserved
            assert loaded_config["level1"]["level2"]["level3"]["level4"]["deep_value"] == "found_it"
            
        finally:
            os.unlink(config_path)

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters in configuration."""
        unicode_config = {
            "unicode_string": "Hello 世界 🌍",
            "special_chars": "!@#$%^&*()_+-={}[]|\\:;\"'<>?,./"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
            yaml.dump(unicode_config, f, allow_unicode=True)
            config_path = f.name
        
        try:
            # Load configuration with unicode
            loaded_config = load_config(config_path)
            
            # Verify unicode is preserved
            assert loaded_config["unicode_string"] == "Hello 世界 🌍"
            assert loaded_config["special_chars"] == "!@#$%^&*()_+-={}[]|\\:;\"'<>?,./"
            
        finally:
            os.unlink(config_path)

    def test_large_configuration_file(self):
        """Test loading of large configuration files."""
        # Generate large configuration
        large_config = {}
        for i in range(1000):
            large_config[f"section_{i}"] = {
                f"key_{j}": f"value_{i}_{j}" for j in range(10)
            }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(large_config, f)
            config_path = f.name
        
        try:
            # Load large configuration
            import time
            start_time = time.time()
            loaded_config = load_config(config_path)
            load_time = time.time() - start_time
            
            # Verify large config loads successfully and quickly
            assert loaded_config is not None, "Large config should load"
            assert len(loaded_config) == 1000, "Should have all sections"
            assert load_time < 5.0, f"Loading took too long: {load_time:.2f}s"
            
        finally:
            os.unlink(config_path)

    def test_file_permission_errors(self):
        """Test handling of file permission errors."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(self.valid_config, f)
            config_path = f.name
        
        try:
            # Remove read permissions (Unix-like systems)
            if hasattr(os, 'chmod'):
                os.chmod(config_path, 0o000)
                
                # Attempt to load file without permissions
                loaded_config = load_config(config_path)
                
                # Should handle permission error gracefully
                assert loaded_config is None, "Should return None for permission error"
            
        finally:
            # Restore permissions and cleanup
            if hasattr(os, 'chmod'):
                os.chmod(config_path, 0o644)
            os.unlink(config_path)

    def test_yaml_anchors_and_references(self):
        """Test handling of YAML anchors and references."""
        yaml_with_anchors = """
        defaults: &defaults
          timeout: 30
          retries: 3
        
        service1:
          <<: *defaults
          name: "service1"
        
        service2:
          <<: *defaults
          name: "service2"
          timeout: 60
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_with_anchors)
            config_path = f.name
        
        try:
            # Load configuration with anchors
            loaded_config = load_config(config_path)
            
            # Verify anchors are resolved correctly
            assert loaded_config["service1"]["timeout"] == 30
            assert loaded_config["service1"]["retries"] == 3
            assert loaded_config["service1"]["name"] == "service1"
            
            assert loaded_config["service2"]["timeout"] == 60  # Overridden
            assert loaded_config["service2"]["retries"] == 3   # From anchor
            assert loaded_config["service2"]["name"] == "service2"
            
        finally:
            os.unlink(config_path)

    def test_config_with_comments(self):
        """Test loading configuration files with comments."""
        yaml_with_comments = """
        # This is a comment
        frame_grabber:
          video_source: "test.mp4"  # Inline comment
          # Another comment
          resize_resolution: [640, 480]
        
        # Section comment
        vehicle_detector:
          model_path: "model.engine"
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_with_comments)
            config_path = f.name
        
        try:
            # Load configuration with comments
            loaded_config = load_config(config_path)
            
            # Verify configuration loads correctly (comments are ignored)
            assert loaded_config is not None
            assert loaded_config["frame_grabber"]["video_source"] == "test.mp4"
            assert loaded_config["frame_grabber"]["resize_resolution"] == [640, 480]
            assert loaded_config["vehicle_detector"]["model_path"] == "model.engine"
            
        finally:
            os.unlink(config_path)

    def test_error_logging(self):
        """Test that errors are properly logged."""
        with patch('traffic_monitor.utils.config_loader.logger') as mock_logger:
            # Test file not found error
            load_config("nonexistent.yaml")
            mock_logger.error.assert_called()
            
            # Test YAML error
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write("invalid: yaml: [")
                config_path = f.name
            
            try:
                load_config(config_path)
                mock_logger.error.assert_called()
            finally:
                os.unlink(config_path)

    def test_config_validation_helpers(self):
        """Test helper functions for configuration validation."""
        config = self.valid_config
        
        # Test required sections exist
        required_sections = ["frame_grabber", "vehicle_detector", "loguru"]
        for section in required_sections:
            assert section in config, f"Required section '{section}' missing"
        
        # Test required fields in sections
        assert "video_source" in config["frame_grabber"]
        assert "model_path" in config["vehicle_detector"]
        assert "level" in config["loguru"]
        
        # Test data type validation
        assert isinstance(config["frame_grabber"]["resize_resolution"], list)
        assert isinstance(config["vehicle_detector"]["conf_threshold"], (int, float))
        assert isinstance(config["loguru"]["terminal_output_enabled"], bool)

    def test_config_merging_simulation(self):
        """Test simulation of configuration merging (base + override)."""
        base_config = {
            "service": {
                "timeout": 30,
                "retries": 3,
                "debug": False
            }
        }
        
        override_config = {
            "service": {
                "timeout": 60,
                "debug": True
            }
        }
        
        # Simulate config merging
        merged_config = base_config.copy()
        for section, values in override_config.items():
            if section in merged_config:
                merged_config[section].update(values)
            else:
                merged_config[section] = values
        
        # Verify merging
        assert merged_config["service"]["timeout"] == 60  # Overridden
        assert merged_config["service"]["retries"] == 3   # From base
        assert merged_config["service"]["debug"] is True  # Overridden

    # Helper methods
    def _create_temp_config(self, config_data):
        """Create a temporary configuration file."""
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(config_data, temp_file)
        temp_file.close()
        return temp_file.name

    def _validate_config_structure(self, config, required_sections):
        """Validate configuration structure."""
        if not isinstance(config, dict):
            return False
        
        for section in required_sections:
            if section not in config:
                return False
        
        return True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])