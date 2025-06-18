import pytest
import yaml
from pathlib import Path


def test_yolo_class_mapping_correctness():
    """Test that vehicle detector class mapping uses correct YOLO COCO class IDs."""
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "src" / "traffic_monitor" / "config" / "settings.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    vehicle_detector_config = config.get("vehicle_detector", {})
    class_mapping = vehicle_detector_config.get("class_mapping", {})
    
    # Expected YOLO COCO class mapping for traffic monitoring
    expected_mapping = {
        0: "person",
        1: "bicycle", 
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }
    
    # Convert string keys to int for comparison
    actual_mapping = {int(k): v for k, v in class_mapping.items()}
    
    # Verify mapping is correct
    assert actual_mapping == expected_mapping, f"Class mapping mismatch. Expected: {expected_mapping}, Got: {actual_mapping}"
    
    # Verify all vehicle classes are present
    expected_classes = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
    actual_classes = set(actual_mapping.values())
    assert actual_classes == expected_classes, f"Missing vehicle classes. Expected: {expected_classes}, Got: {actual_classes}"


def test_class_mapping_ids_are_valid_coco_ids():
    """Test that all class IDs in the mapping are valid YOLO COCO class IDs."""
    
    # Valid YOLO COCO class IDs (0-79)
    valid_coco_ids = list(range(80))
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "src" / "traffic_monitor" / "config" / "settings.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    vehicle_detector_config = config.get("vehicle_detector", {})
    class_mapping = vehicle_detector_config.get("class_mapping", {})
    
    # Convert string keys to int
    class_ids = [int(k) for k in class_mapping.keys()]
    
    # Verify all IDs are valid COCO IDs
    for class_id in class_ids:
        assert class_id in valid_coco_ids, f"Invalid COCO class ID: {class_id}. Must be between 0-79."


def test_class_mapping_no_duplicates():
    """Test that there are no duplicate class names in the mapping."""
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "src" / "traffic_monitor" / "config" / "settings.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    vehicle_detector_config = config.get("vehicle_detector", {})
    class_mapping = vehicle_detector_config.get("class_mapping", {})
    
    class_names = list(class_mapping.values())
    unique_class_names = set(class_names)
    
    assert len(class_names) == len(unique_class_names), f"Duplicate class names found: {class_names}" 