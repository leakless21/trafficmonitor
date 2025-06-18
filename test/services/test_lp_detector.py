import pytest
import multiprocessing as mp
import numpy as np
from unittest.mock import MagicMock, patch, ANY
from queue import Empty, Full
from loguru import logger
import cv2

from src.traffic_monitor.services.lp_detector import LPDetector, lp_detector_process
from src.traffic_monitor.utils.logging_config import setup_logging

# Initialize logging for the test environment
setup_logging()

@pytest.fixture
def mock_lp_detector_config():
    """Provides a mock configuration dictionary for LPDetector tests."""
    return {
        "lp_detector": {
            "model_path": "data/models/plate_v8n.pt",
            "conf_threshold": 0.6
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
    # Create a dummy JPEG image (e.g., a black 100x100 image)
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    _, encoded_image = cv2.imencode('.jpg', dummy_frame)
    jpeg_binary = encoded_image.tobytes()

    return {
        "frame_id": "test_frame_123",
        "camera_id": "test_cam",
        "timestamp": 1678886400.0,
        "frame_data_jpeg": jpeg_binary,
        "frame_width": 100,
        "frame_height": 100,
        "tracked_objects": [
            {
                "track_id": 1,
                "bbox_xyxy": [10, 10, 50, 50],
                "class_name": "car",
                "class_id": 2
            },
            {
                "track_id": 2,
                "bbox_xyxy": [60, 60, 90, 90],
                "class_name": "truck",
                "class_id": 7
            }
        ]
    }

class MockYOLOModel:
    """
    A mock YOLO model to simulate ultralytics.YOLO behavior.
    """
    def __init__(self, *args, **kwargs):
        pass

    def predict(self, frame, conf, verbose=False):
        # Simulate a detection result
        class MockBoxes:
            def __init__(self, xyxy, conf):
                self.xyxy = np.array([xyxy])
                self.conf = np.array([conf])

        class MockResult:
            def __init__(self, boxes):
                # boxes should be a list of MockBoxes
                self.boxes = boxes

        if frame is None:
            return [] # No results for None frame
        
        # Simulate a successful detection for a specific crop area
        if np.array_equal(frame.shape, (40, 40, 3)): # For the first vehicle crop [10,10,50,50]
            return [MockResult([MockBoxes([15, 15, 35, 35], 0.75)])] # Plate within car
        elif np.array_equal(frame.shape, (30, 30, 3)): # For the second vehicle crop [60,60,90,90]
            return [MockResult([MockBoxes([65, 65, 85, 85], 0.8)])] # Plate within truck
        else:
            return [] # No plates found for other frames/crops

@patch('ultralytics.YOLO', new=MockYOLOModel)
def test_lp_detector_init_success(mock_lp_detector_config):
    """
    Tests successful initialization of LPDetector.
    """
    logger.info("Running test_lp_detector_init_success")
    with patch('os.path.exists', return_value=True):
        detector = LPDetector(mock_lp_detector_config["lp_detector"]["model_path"],
                              mock_lp_detector_config["lp_detector"]["conf_threshold"])
        assert detector is not None
        assert detector.conf_threshold == 0.6
    logger.info("Finished test_lp_detector_init_success")

@patch('ultralytics.YOLO', side_effect=Exception("Model load error"))
def test_lp_detector_init_failure(mock_yolo_model, mock_lp_detector_config):
    """
    Tests LPDetector initialization failure (e.g., model not found/corrupted).
    """
    logger.info("Running test_lp_detector_init_failure")
    with pytest.raises(Exception, match="Model load error"):
        LPDetector(mock_lp_detector_config["lp_detector"]["model_path"],
                   mock_lp_detector_config["lp_detector"]["conf_threshold"])
    logger.info("Finished test_lp_detector_init_failure")

@patch('ultralytics.YOLO', new=MockYOLOModel)
def test_lp_detector_find_plates_success(mock_lp_detector_config):
    """
    Tests successful license plate detection.
    """
    logger.info("Running test_lp_detector_find_plates_success")
    with patch('os.path.exists', return_value=True):
        detector = LPDetector(mock_lp_detector_config["lp_detector"]["model_path"],
                              mock_lp_detector_config["lp_detector"]["conf_threshold"])
        
        # Create a dummy frame to simulate a vehicle crop
        test_frame = np.zeros((40, 40, 3), dtype=np.uint8)
        result = detector.find_plates(test_frame)
        
        assert result is not None
        bbox, conf = result
        assert bbox == [15, 15, 35, 35]  # Based on MockYOLOModel's simulated output
        assert conf == 0.75
    logger.info("Finished test_lp_detector_find_plates_success")

@patch('ultralytics.YOLO', new=MockYOLOModel)
def test_lp_detector_find_plates_no_detection(mock_lp_detector_config):
    """
    Tests when no license plates are detected.
    """
    logger.info("Running test_lp_detector_find_plates_no_detection")
    with patch('os.path.exists', return_value=True):
        detector = LPDetector(mock_lp_detector_config["lp_detector"]["model_path"],
                              mock_lp_detector_config["lp_detector"]["conf_threshold"])
        
        # Create a dummy frame that won't trigger a detection in MockYOLOModel
        test_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        result = detector.find_plates(test_frame)
        
        assert result is None
    logger.info("Finished test_lp_detector_find_plates_no_detection")

# --- Tests for lp_detector_process ---

@patch('os.path.exists', return_value=True)
@patch('ultralytics.YOLO', new=MockYOLOModel)
@patch('cv2.imdecode', return_value=np.zeros((100, 100, 3), dtype=np.uint8))
def test_lp_detector_process_basic_flow(
    mock_imdecode,
    mock_yolo_model,
    mock_path_exists,
    mock_lp_detector_config,
    mock_input_queue,
    mock_output_queue,
    mock_shutdown_event,
    sample_frame_message
):
    """
    Tests the basic flow of lp_detector_process with successful detections.
    """
    logger.info("Running test_lp_detector_process_basic_flow")
    
    # Simulate input queue providing messages then None for shutdown
    mock_input_queue.get.side_effect = [
        sample_frame_message,
        None # Signal shutdown
    ]
    mock_shutdown_event.is_set.side_effect = [False, False, True] # Allow processing, then shut down

    process = mp.Process(
        target=lp_detector_process,
        args=(mock_lp_detector_config, mock_input_queue, mock_output_queue, mock_shutdown_event)
    )
    process.start()
    process.join(timeout=5) # Increased timeout

    # Assertions
    mock_path_exists.assert_called_with(mock_lp_detector_config["lp_detector"]["model_path"])
    assert mock_input_queue.get.call_count >= 2 # Called for message and None
    assert mock_output_queue.put.call_count == 3 # One for each vehicle, plus None for shutdown
    
    # Check the contents of the first output message
    assert len(mock_output_queue.put.call_args_list) >= 1 # Ensure at least one call
    first_output_call_args, _ = mock_output_queue.put.call_args_list[0]
    first_plate_message = first_output_call_args[0]
    assert "frame_id" in first_plate_message
    assert first_plate_message["frame_id"] == "test_frame_123"
    assert first_plate_message["vehicle_id"] == "1"
    assert first_plate_message["plate_bbox_original"] == [25, 25, 45, 45] # 10+15, 10+15, 10+35, 10+35
    assert first_plate_message["plate_confidence"] == 0.75

    # Check the contents of the second output message
    assert len(mock_output_queue.put.call_args_list) >= 2 # Ensure at least two calls
    second_output_call_args, _ = mock_output_queue.put.call_args_list[1]
    second_plate_message = second_output_call_args[0]
    assert "frame_id" in second_plate_message
    assert second_plate_message["frame_id"] == "test_frame_123"
    assert second_plate_message["vehicle_id"] == "2"
    assert second_plate_message["plate_bbox_original"] == [125, 125, 155, 155] # 60+65, 60+65, 60+85, 60+85
    assert second_plate_message["plate_confidence"] == 0.8

    # Check for the shutdown signal
    assert len(mock_output_queue.put.call_args_list) >= 3 # Ensure at least three calls
    assert mock_output_queue.put.call_args_list[2][0][0] is None

    logger.info("Finished test_lp_detector_process_basic_flow")

@patch('os.path.exists', return_value=False)
@patch('ultralytics.YOLO', new=MockYOLOModel)
def test_lp_detector_process_model_not_found(
    mock_yolo_model,
    mock_path_exists,
    mock_lp_detector_config,
    mock_input_queue,
    mock_output_queue,
    mock_shutdown_event
):
    """
    Tests lp_detector_process when the model file is not found.
    """
    logger.info("Running test_lp_detector_process_model_not_found")

    process = mp.Process(
        target=lp_detector_process,
        args=(mock_lp_detector_config, mock_input_queue, mock_output_queue, mock_shutdown_event)
    )
    process.start()
    process.join(timeout=5) # Increased timeout

    mock_path_exists.assert_called_once_with(mock_lp_detector_config["lp_detector"]["model_path"])
    mock_yolo_model.assert_not_called() # YOLO should not be initialized
    mock_input_queue.get.assert_not_called() # No messages should be processed
    mock_output_queue.put.assert_called_once_with(None) # Should signal shutdown
    logger.info("Finished test_lp_detector_process_model_not_found")

@patch('os.path.exists', return_value=True)
@patch('ultralytics.YOLO', side_effect=Exception("YOLO init error"))
def test_lp_detector_process_init_failure(
    mock_yolo_model,
    mock_path_exists,
    mock_lp_detector_config,
    mock_input_queue,
    mock_output_queue,
    mock_shutdown_event
):
    """
    Tests lp_detector_process when LPDetector initialization fails.
    """
    logger.info("Running test_lp_detector_process_init_failure")

    process = mp.Process(
        target=lp_detector_process,
        args=(mock_lp_detector_config, mock_input_queue, mock_output_queue, mock_shutdown_event)
    )
    process.start()
    process.join(timeout=5) # Increased timeout

    mock_path_exists.assert_called_once()
    mock_yolo_model.assert_called_once() # YOLO should be attempted to be initialized
    mock_input_queue.get.assert_not_called() # No messages should be processed
    mock_output_queue.put.assert_called_once_with(None) # Should signal shutdown
    logger.info("Finished test_lp_detector_process_init_failure")

@patch('os.path.exists', return_value=True)
@patch('ultralytics.YOLO', new=MockYOLOModel)
@patch('cv2.imdecode', return_value=np.zeros((100, 100, 3), dtype=np.uint8))
def test_lp_detector_process_empty_input_queue(
    mock_imdecode,
    mock_yolo_model,
    mock_path_exists,
    mock_lp_detector_config,
    mock_input_queue,
    mock_output_queue,
    mock_shutdown_event
):
    """
    Tests lp_detector_process when the input queue is empty and then shuts down.
    """
    logger.info("Running test_lp_detector_process_empty_input_queue")

    mock_input_queue.get.side_effect = Empty # Simulate empty queue
    mock_shutdown_event.is_set.side_effect = [False, True] # Allow one loop, then shut down

    process = mp.Process(
        target=lp_detector_process,
        args=(mock_lp_detector_config, mock_input_queue, mock_output_queue, mock_shutdown_event)
    )
    process.start()
    process.join(timeout=5) # Increased timeout

    mock_input_queue.get.assert_called() # Should attempt to get from queue
    mock_output_queue.put.assert_called_once_with(None) # Should signal shutdown
    logger.info("Finished test_lp_detector_process_empty_input_queue")

@patch('os.path.exists', return_value=True)
@patch('ultralytics.YOLO', new=MockYOLOModel)
@patch('cv2.imdecode', return_value=np.zeros((100, 100, 3), dtype=np.uint8))
def test_lp_detector_process_output_queue_full(
    mock_imdecode,
    mock_yolo_model,
    mock_path_exists,
    mock_lp_detector_config,
    mock_input_queue,
    mock_output_queue,
    mock_shutdown_event,
    sample_frame_message
):
    """
    Tests lp_detector_process when the output queue is full.
    """
    logger.info("Running test_lp_detector_process_output_queue_full")

    mock_input_queue.get.side_effect = [
        sample_frame_message, # First message to process
        None # Signal shutdown
    ]
    # Simulate queue full for the first call, then allow subsequent calls (second vehicle, then None for shutdown)
    mock_output_queue.put.side_effect = [Full, None, None]

    mock_shutdown_event.is_set.side_effect = [False, False, True] # Allow processing, then shut down

    process = mp.Process(
        target=lp_detector_process,
        args=(mock_lp_detector_config, mock_input_queue, mock_output_queue, mock_shutdown_event)
    )
    process.start()
    process.join(timeout=5)

    # Should attempt to put for both vehicles and then for shutdown
    assert mock_output_queue.put.call_count == 3

    # Check that the None signal for shutdown is still sent
    assert mock_output_queue.put.call_args_list[2][0][0] is None
    logger.info("Finished test_lp_detector_process_output_queue_full")

@patch('os.path.exists', return_value=True)
@patch('ultralytics.YOLO', new=MockYOLOModel)
@patch('cv2.imdecode', return_value=np.zeros((100, 100, 3), dtype=np.uint8))
def test_lp_detector_process_invalid_bbox(
    mock_imdecode,
    mock_yolo_model,
    mock_path_exists,
    mock_lp_detector_config,
    mock_input_queue,
    mock_output_queue,
    mock_shutdown_event,
    sample_frame_message
):
    """
    Tests lp_detector_process with invalid bounding box coordinates.
    """
    logger.info("Running test_lp_detector_process_invalid_bbox")

    # Create a message with an invalid bbox (x1 >= x2)
    invalid_bbox_message = sample_frame_message.copy()
    invalid_bbox_message["tracked_objects"] = [
        {
            "track_id": 1,
            "bbox_xyxy": [50, 10, 10, 50], # Invalid: x1 >= x2
            "class_name": "car",
            "class_id": 2
        }
    ]
    
    mock_input_queue.get.side_effect = [
        invalid_bbox_message,
        None # Signal shutdown
    ]
    mock_shutdown_event.is_set.side_effect = [False, False, True] # Allow processing, then shut down

    process = mp.Process(
        target=lp_detector_process,
        args=(mock_lp_detector_config, mock_input_queue, mock_output_queue, mock_shutdown_event)
    )
    process.start()
    process.join(timeout=5) # Increased timeout

    # No plate detection messages should be put, only the shutdown None
    mock_output_queue.put.assert_called_once_with(None)
    logger.info("Finished test_lp_detector_process_invalid_bbox") 