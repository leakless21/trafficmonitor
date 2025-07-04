import pytest
import multiprocessing as mp
import numpy as np
from unittest.mock import MagicMock, patch
from queue import Empty, Full
from loguru import logger
import cv2
import time

from src.traffic_monitor.services.vehicle_detector import VehicleDetector, vehicle_detector_process
from src.traffic_monitor.utils.logging_config import setup_logging
from src.traffic_monitor.utils.custom_types import Detection

# Initialize logging for the test environment
setup_logging()

@pytest.fixture
def mock_vehicle_detector_config():
    """Provides a mock configuration dictionary for VehicleDetector tests."""
    return {
        "model_path": "data/models/yolov8n.pt",
        "conf_threshold": 0.7,
        "class_mapping": {
            0: "person",
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }
    }

@pytest.fixture
def mock_input_queue():
    """Provides a mock multiprocessing input queue."""
    mock_queue = MagicMock()
    mock_queue.get = MagicMock()
    mock_queue.put = MagicMock()
    mock_queue.empty = MagicMock(return_value=False)
    return mock_queue

@pytest.fixture
def mock_output_queue():
    """Provides a mock multiprocessing output queue."""
    mock_queue = MagicMock()
    mock_queue.get = MagicMock()
    mock_queue.put = MagicMock()
    mock_queue.empty = MagicMock(return_value=False)
    return mock_queue

@pytest.fixture
def mock_shutdown_event():
    """Provides a mock multiprocessing event."""
    mock_event = MagicMock()
    mock_event.is_set = MagicMock(return_value=False)
    mock_event.set = MagicMock()
    mock_event.clear = MagicMock()
    return mock_event

@pytest.fixture
def sample_frame_message():
    """
    Provides a sample FrameMessage for testing.
    """
    # Create a dummy JPEG image (e.g., a black 640x480 image)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, encoded_image = cv2.imencode('.jpg', dummy_frame)
    jpeg_binary = encoded_image.tobytes()

    return {
        "frame_id": "frame_test_1",
        "camera_id": "cam_test_A",
        "timestamp": time.time(),
        "frame_data_jpeg": jpeg_binary,
        "frame_width": 640,
        "frame_height": 480
    }

class MockYOLOModel:
    """
    A mock YOLO model to simulate ultralytics.YOLO behavior for VehicleDetector.
    """
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, frame, conf, verbose=False):
        class MockBoxes:
            def __init__(self, xyxy, cls, conf):
                self.xyxy = np.array([xyxy])
                self.cls = np.array([cls])
                self.conf = np.array([conf])

        class MockResult:
            def __init__(self, boxes):
                self.boxes = boxes
        
        # Simulate detection for a car and a person, but filter by confidence
        if frame is not None:
            detections = []
            # Example: a car detection
            car_conf = 0.85
            if car_conf >= conf:
                car_box = MockBoxes([100, 100, 200, 200], 2, car_conf)
                detections.append(car_box)
            
            # Example: a person detection
            person_conf = 0.90
            if person_conf >= conf:
                person_box = MockBoxes([300, 300, 350, 400], 0, person_conf)
                detections.append(person_box)
            
            if detections:
                return [MockResult(detections)]
        return []

@patch('ultralytics.YOLO', new=MockYOLOModel)
def test_vehicle_detector_init_success(mock_vehicle_detector_config):
    """
    Tests successful initialization of VehicleDetector.
    """
    logger.info("Running test_vehicle_detector_init_success")
    detector = VehicleDetector(
        mock_vehicle_detector_config["model_path"],
        mock_vehicle_detector_config["conf_threshold"],
        mock_vehicle_detector_config["class_mapping"]
    )
    assert detector is not None
    assert detector.conf_threshold == 0.7
    assert detector.class_mapping == {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    logger.info("Finished test_vehicle_detector_init_success")

@patch('ultralytics.YOLO', side_effect=Exception("YOLO load error"))
def test_vehicle_detector_init_failure(mock_yolo_model, mock_vehicle_detector_config):
    """
    Tests VehicleDetector initialization failure.
    """
    logger.info("Running test_vehicle_detector_init_failure")
    with pytest.raises(Exception, match="YOLO load error"):
        VehicleDetector(
            mock_vehicle_detector_config["model_path"],
            mock_vehicle_detector_config["conf_threshold"],
            mock_vehicle_detector_config["class_mapping"]
        )
    logger.info("Finished test_vehicle_detector_init_failure")

@patch('ultralytics.YOLO', new=MockYOLOModel)
def test_vehicle_detector_detect_success(mock_vehicle_detector_config):
    """
    Tests successful vehicle detection.
    """
    logger.info("Running test_vehicle_detector_detect_success")
    detector = VehicleDetector(
        mock_vehicle_detector_config["model_path"],
        mock_vehicle_detector_config["conf_threshold"],
        mock_vehicle_detector_config["class_mapping"]
    )
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame)
    
    assert len(detections) == 2 # Expecting 2 detections from MockYOLOModel
    
    car_detection = next((d for d in detections if d["class_name"] == "car"), None)
    person_detection = next((d for d in detections if d["class_name"] == "person"), None)

    assert car_detection is not None
    assert car_detection["bbox_xyxy"] == [100, 100, 200, 200]
    assert car_detection["confidence"] == 0.85
    assert car_detection["class_id"] == 2

    assert person_detection is not None
    assert person_detection["bbox_xyxy"] == [300, 300, 350, 400]
    assert person_detection["confidence"] == 0.90
    assert person_detection["class_id"] == 0
    
    logger.info("Finished test_vehicle_detector_detect_success")

@patch('ultralytics.YOLO', new=MockYOLOModel)
def test_vehicle_detector_detect_no_detection_below_threshold(mock_vehicle_detector_config):
    """
    Tests when no vehicles are detected due to low confidence.
    """
    logger.info("Running test_vehicle_detector_detect_no_detection_below_threshold")
    config = mock_vehicle_detector_config.copy()
    config["conf_threshold"] = 0.95 # Set a higher threshold
    detector = VehicleDetector(
        config["model_path"],
        config["conf_threshold"],
        config["class_mapping"]
    )
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame)
    
    assert len(detections) == 0 # No detections expected
    logger.info("Finished test_vehicle_detector_detect_no_detection_below_threshold")

@patch('ultralytics.YOLO', new=MockYOLOModel)
def test_vehicle_detector_detect_unmapped_class(mock_vehicle_detector_config):
    """
    Tests that detections for unmapped classes are ignored.
    """
    logger.info("Running test_vehicle_detector_detect_unmapped_class")
    config = mock_vehicle_detector_config.copy()
    # Temporarily remove 'person' from class_mapping to simulate unmapped class
    config["class_mapping"] = {2: "car"}
    detector = VehicleDetector(
        config["model_path"],
        config["conf_threshold"],
        config["class_mapping"]
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(dummy_frame)

    assert len(detections) == 1 # Only car should be detected
    assert detections[0]["class_name"] == "car"
    logger.info("Finished test_vehicle_detector_detect_unmapped_class")

# --- Tests for vehicle_detector_process ---

@patch('src.traffic_monitor.services.vehicle_detector.VehicleDetector')
@patch('cv2.imdecode', return_value=np.zeros((480, 640, 3), dtype=np.uint8))
def test_vehicle_detector_process_basic_flow(
    mock_imdecode,
    mock_vehicle_detector_class,
    mock_vehicle_detector_config,
    mock_input_queue,
    mock_output_queue,
    mock_shutdown_event,
    sample_frame_message
):
    """
    Tests the basic flow of vehicle_detector_process with successful detections.
    """
    logger.info("Running test_vehicle_detector_process_basic_flow")
    
    # Setup mock VehicleDetector instance
    mock_detector_instance = MagicMock()
    mock_vehicle_detector_class.return_value = mock_detector_instance
    mock_detector_instance.detect.return_value = [
        {"bbox_xyxy": [100, 100, 200, 200], "confidence": 0.85, "class_id": 2, "class_name": "car"}
    ] # Simulate one detection

    # Simulate input queue providing messages then None for shutdown
    mock_input_queue.get.side_effect = [
        sample_frame_message,
        None # Signal shutdown
    ]
    mock_shutdown_event.is_set.side_effect = [False, False, True] # Allow processing, then shut down

    process = mp.Process(
        target=vehicle_detector_process,
        args=(mock_vehicle_detector_config, mock_input_queue, mock_output_queue, mock_shutdown_event)
    )
    process.start()
    process.join(timeout=5)

    # Assertions
    mock_vehicle_detector_class.assert_called_once_with(
        mock_vehicle_detector_config["model_path"],
        mock_vehicle_detector_config["conf_threshold"],
        {int(k): v for k, v in mock_vehicle_detector_config["class_mapping"].items()}
    )
    assert mock_input_queue.get.call_count >= 2 # Called for message and None
    assert mock_output_queue.put.call_count == 2 # One detection message + One None for shutdown
    
    # Check the contents of the output message
    output_message = mock_output_queue.put.call_args_list[0][0]
    assert output_message["frame_id"] == "frame_test_1"
    assert len(output_message["detections"]) == 1
    assert output_message["detections"][0]["class_name"] == "car"

    # Check for the shutdown signal
    assert mock_output_queue.put.call_args_list[1][0][0] is None

    logger.info("Finished test_vehicle_detector_process_basic_flow")

@patch('src.traffic_monitor.services.vehicle_detector.VehicleDetector', side_effect=Exception("Detector init error"))
def test_vehicle_detector_process_init_failure(
    mock_detector_class,
    mock_vehicle_detector_config,
    mock_input_queue,
    mock_output_queue,
    mock_shutdown_event
):
    """
    Tests vehicle_detector_process when VehicleDetector initialization fails.
    """
    logger.info("Running test_vehicle_detector_process_init_failure")

    process = mp.Process(
        target=vehicle_detector_process,
        args=(mock_vehicle_detector_config, mock_input_queue, mock_output_queue, mock_shutdown_event)
    )
    process.start()
    process.join(timeout=5)

    mock_detector_class.assert_called_once()
    mock_input_queue.get.assert_not_called()
    mock_output_queue.put.assert_not_called() # No output if init fails before queue loop
    logger.info("Finished test_vehicle_detector_process_init_failure")

@patch('src.traffic_monitor.services.vehicle_detector.VehicleDetector')
@patch('cv2.imdecode', return_value=np.zeros((480, 640, 3), dtype=np.uint8))
def test_vehicle_detector_process_empty_input_queue(
    mock_imdecode,
    mock_detector_class,
    mock_vehicle_detector_config,
    mock_input_queue,
    mock_output_queue,
    mock_shutdown_event
):
    """
    Tests vehicle_detector_process when the input queue is empty.
    """
    logger.info("Running test_vehicle_detector_process_empty_input_queue")

    mock_detector_instance = MagicMock()
    mock_detector_class.return_value = mock_detector_instance
    mock_detector_instance.detect.return_value = []

    mock_input_queue.get.side_effect = Empty # Simulate empty queue
    mock_shutdown_event.is_set.side_effect = [False, True] # Allow one loop, then shut down

    process = mp.Process(
        target=vehicle_detector_process,
        args=(mock_vehicle_detector_config, mock_input_queue, mock_output_queue, mock_shutdown_event)
    )
    process.start()
    process.join(timeout=5)

    mock_input_queue.get.assert_called() # Should attempt to get from queue
    mock_output_queue.put.assert_not_called() # No detections, so no put expected before shutdown
    logger.info("Finished test_vehicle_detector_process_empty_input_queue")

@patch('src.traffic_monitor.services.vehicle_detector.VehicleDetector')
@patch('cv2.imdecode', return_value=np.zeros((480, 640, 3), dtype=np.uint8))
def test_vehicle_detector_process_output_queue_full(
    mock_imdecode,
    mock_detector_class,
    mock_vehicle_detector_config,
    mock_input_queue,
    mock_output_queue,
    mock_shutdown_event,
    sample_frame_message
):
    """
    Tests vehicle_detector_process when the output queue is full.
    """
    logger.info("Running test_vehicle_detector_process_output_queue_full")

    mock_detector_instance = MagicMock()
    mock_detector_class.return_value = mock_detector_instance
    mock_detector_instance.detect.return_value = [
        {"bbox_xyxy": [100, 100, 200, 200], "confidence": 0.85, "class_id": 2, "class_name": "car"}
    ]

    mock_input_queue.get.side_effect = [
        sample_frame_message,
        None # Signal shutdown
    ]
    mock_output_queue.put.side_effect = Full # Simulate queue full
    mock_shutdown_event.is_set.side_effect = [False, False, True] # Allow processing, then shut down

    process = mp.Process(
        target=vehicle_detector_process,
        args=(mock_vehicle_detector_config, mock_input_queue, mock_output_queue, mock_shutdown_event)
    )
    process.start()
    process.join(timeout=5)

    mock_output_queue.put.assert_called_once() # Should attempt to put at least once
    logger.info("Finished test_vehicle_detector_process_output_queue_full")

@patch('src.traffic_monitor.services.vehicle_detector.VehicleDetector')
@patch('cv2.imdecode', side_effect=Exception("Imdecode error"))
def test_vehicle_detector_process_imdecode_failure(
    mock_imdecode,
    mock_detector_class,
    mock_vehicle_detector_config,
    mock_input_queue,
    mock_output_queue,
    mock_shutdown_event,
    sample_frame_message
):
    """
    Tests vehicle_detector_process when cv2.imdecode fails.
    """
    logger.info("Running test_vehicle_detector_process_imdecode_failure")

    mock_detector_instance = MagicMock()
    mock_detector_class.return_value = mock_detector_instance
    mock_detector_instance.detect.return_value = [] # This won't be called if imdecode fails earlier

    mock_input_queue.get.side_effect = [
        sample_frame_message,
        None # Signal shutdown
    ]
    mock_shutdown_event.is_set.side_effect = [False, False, True] # Allow processing, then shut down

    process = mp.Process(
        target=vehicle_detector_process,
        args=(mock_vehicle_detector_config, mock_input_queue, mock_output_queue, mock_shutdown_event)
    )
    process.start()
    process.join(timeout=5)

    mock_output_queue.put.assert_called_once_with(None) # Should signal shutdown due to error
    logger.info("Finished test_vehicle_detector_process_imdecode_failure") 