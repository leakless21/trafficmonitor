from unittest.mock import Mock
from src.traffic_monitor.services.vehicle_counter import Counter


def test_counting_line_config_initialization():
    """Test that counting line configuration is properly initialized."""
    # This is the configuration format from settings.yaml
    counting_lines_config = [
        [
            [0.31, 0.22],
            [0.85, 0.33]
        ]
    ]
    
    # This should not raise an exception
    counter = Counter(counting_lines_config)
    
    # Test the initialization with valid frame dimensions
    tracked_objects = []
    frame_width, frame_height = 1280, 720
    og_width, og_height = 1920, 1080
    
    # This should not fail and should return None (no count update)
    result = counter.update(tracked_objects, frame_width, frame_height, og_width, og_height)
    assert result is None
    
    # The line should be properly initialized
    assert counter.absolute_coords is not None
    assert counter.relative_coords is not None


def test_counting_line_config_format_detection():
    """Test that the counter can handle different coordinate formats."""
    counter = Counter([[]])  # Empty config for testing
    
    # Test with relative coordinates (float values 0-1)
    relative_line = [[0.31, 0.22], [0.85, 0.33]]
    result = counter._init_and_normalize_line(relative_line, 1080, 1920, 1280, 720)
    assert result is not None
    
    # Test with absolute coordinates (integer pixel values)
    absolute_line = [[595, 238], [1632, 356]]  # Based on 1920x1080 frame
    result = counter._init_and_normalize_line(absolute_line, 1080, 1920, 1280, 720)
    assert result is not None


def test_counting_line_config_invalid_format():
    """Test that invalid coordinate formats are handled gracefully."""
    counter = Counter([[]])
    
    # Test with invalid format (string coordinates)
    invalid_line = [["invalid", "coords"], ["bad", "format"]]
    result = counter._init_and_normalize_line(invalid_line, 1080, 1920, 1280, 720)
    assert result is None
    assert counter.relative_coords == []


if __name__ == "__main__":
    test_counting_line_config_initialization()
    test_counting_line_config_format_detection()
    test_counting_line_config_invalid_format()
    print("All tests passed!") 