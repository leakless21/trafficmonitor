import pytest
import time
import cv2
import multiprocessing as mp
from unittest.mock import MagicMock, patch
from src.traffic_monitor.services.frame_capture_service import frame_capture_process
from src.traffic_monitor.utils.logging_config import setup_logging
from loguru import logger
from queue import Full, Empty
import numpy as np

# Initialize logging for the test environment
setup_logging()

@pytest.fixture
def mock_config():
    """Provides a mock configuration dictionary for tests."""
    return {
        "video_source": "dummy_video.mp4",
        "camera_id": "test_cam",
        "log_every_n_frames": 1,
        "process_every_n_frame": 1, # Default to no skipping
        "resize_resolution": [640, 480] # Use smaller resolution that fits our mock frame
    }

def test_frame_capture_process_no_video_source(
    mock_config
):
    """
    Tests handling of missing video source configuration.
    """
    logger.info("Running test_frame_capture_process_no_video_source")
    # This test directly calls frame_capture_process as it tests early exit before cv2.VideoCapture is called.
    config = {"camera_id": "test_cam"} # Missing video_source

    output_queue = mp.Queue()
    shutdown_event = mp.Event()

    frame_capture_process(config, output_queue, shutdown_event)

    assert output_queue.empty()
    logger.info("Finished test_frame_capture_process_no_video_source")

@patch('cv2.VideoCapture')
def test_frame_capture_process_capture_failure(
    mock_video_capture_class,
    mock_config
):
    """
    Tests behavior when video capture fails to open.
    """
    logger.info("Running test_frame_capture_process_capture_failure")

    mock_cap_instance = MagicMock()
    mock_video_capture_class.return_value = mock_cap_instance
    mock_cap_instance.isOpened.return_value = False

    output_queue = mp.Queue()
    shutdown_event = mp.Event()

    frame_capture_process(mock_config, output_queue, shutdown_event)

    assert output_queue.empty()
    logger.info("Finished test_frame_capture_process_capture_failure")

@patch('cv2.VideoCapture')
@patch('cv2.imencode')
def test_frame_capture_process_frame_skipping_logic(
    mock_imencode,
    mock_video_capture_class,
    mock_config
):
    """
    Tests that frame skipping logic works correctly by directly controlling the mock.
    """
    logger.info("Running test_frame_capture_process_frame_skipping_logic")

    # Configure frame skipping
    mock_config["process_every_n_frame"] = 2

    mock_cap_instance = MagicMock()
    mock_video_capture_class.return_value = mock_cap_instance
    mock_cap_instance.isOpened.return_value = True

    # Mock frame dimensions
    mock_cap_instance.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
        cv2.CAP_PROP_FRAME_WIDTH: 640.0,
        cv2.CAP_PROP_FPS: 30.0
    }.get(prop, 0.0)

    # Create mock frame
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Set up read to return 6 successful frames then fail (which will exit the loop)
    mock_cap_instance.read.side_effect = [
        (True, mock_frame),  # Frame 1 - should be processed (frame_counter=1, (1-1)%2=0)
        (True, mock_frame),  # Frame 2 - should be skipped (frame_counter=2, (2-1)%2=1)
        (True, mock_frame),  # Frame 3 - should be processed (frame_counter=3, (3-1)%2=0)
        (True, mock_frame),  # Frame 4 - should be skipped (frame_counter=4, (4-1)%2=1)
        (True, mock_frame),  # Frame 5 - should be processed (frame_counter=5, (5-1)%2=0)
        (False, None)        # End of video
    ]

    # Mock imencode
    mock_encoded = MagicMock()
    mock_encoded.tobytes.return_value = b'mock_jpeg_data'
    mock_imencode.return_value = (True, mock_encoded)

    output_queue = mp.Queue()
    shutdown_event = mp.Event()

    # Run the frame grabber process
    frame_capture_process(mock_config, output_queue, shutdown_event)

    # Count processed frames
    processed_frames = 0
    while not output_queue.empty():
        try:
            message = output_queue.get(timeout=0.1)
            assert "frame_id" in message
            assert message["camera_id"] == "test_cam"
            assert message["frame_data_jpeg"] == b'mock_jpeg_data'
            processed_frames += 1
        except Empty:
            break

    # With frame skipping every 2nd frame, we should process frames 1, 3, 5 = 3 frames
    assert processed_frames == 3, f"Expected 3 processed frames, got {processed_frames}"
    logger.info(f"Correctly processed {processed_frames} frames with frame skipping enabled")
    logger.info("Finished test_frame_capture_process_frame_skipping_logic")

@patch('cv2.VideoCapture')
@patch('cv2.imencode')
def test_frame_capture_process_no_frame_skipping(
    mock_imencode,
    mock_video_capture_class,
    mock_config
):
    """
    Tests that without frame skipping, all frames are processed.
    """
    logger.info("Running test_frame_capture_process_no_frame_skipping")

    # No frame skipping (default)
    mock_config["process_every_n_frame"] = 1

    mock_cap_instance = MagicMock()
    mock_video_capture_class.return_value = mock_cap_instance
    mock_cap_instance.isOpened.return_value = True

    # Mock frame dimensions
    mock_cap_instance.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
        cv2.CAP_PROP_FRAME_WIDTH: 640.0,
        cv2.CAP_PROP_FPS: 30.0
    }.get(prop, 0.0)

    # Create mock frame
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Set up read to return 3 successful frames then fail
    mock_cap_instance.read.side_effect = [
        (True, mock_frame),  # Frame 1 - should be processed
        (True, mock_frame),  # Frame 2 - should be processed
        (True, mock_frame),  # Frame 3 - should be processed
        (False, None)        # End of video
    ]

    # Mock imencode
    mock_encoded = MagicMock()
    mock_encoded.tobytes.return_value = b'mock_jpeg_data'
    mock_imencode.return_value = (True, mock_encoded)

    output_queue = mp.Queue()
    shutdown_event = mp.Event()

    # Run the frame grabber process
    frame_capture_process(mock_config, output_queue, shutdown_event)

    # Count processed frames
    processed_frames = 0
    while not output_queue.empty():
        try:
            message = output_queue.get(timeout=0.1)
            processed_frames += 1
        except Empty:
            break

    # Without frame skipping, all 3 frames should be processed
    assert processed_frames == 3, f"Expected 3 processed frames, got {processed_frames}"
    logger.info(f"Correctly processed {processed_frames} frames without frame skipping")
    logger.info("Finished test_frame_capture_process_no_frame_skipping")

@patch('cv2.VideoCapture')
@patch('cv2.imencode')
def test_frame_capture_process_queue_full_handling(
    mock_imencode,
    mock_video_capture_class,
    mock_config
):
    """
    Tests handling of a full output queue.
    """
    logger.info("Running test_frame_capture_process_queue_full_handling")

    mock_cap_instance = MagicMock()
    mock_video_capture_class.return_value = mock_cap_instance
    mock_cap_instance.isOpened.return_value = True

    # Mock frame dimensions
    mock_cap_instance.get.side_effect = lambda prop: {
        cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
        cv2.CAP_PROP_FRAME_WIDTH: 640.0,
        cv2.CAP_PROP_FPS: 30.0
    }.get(prop, 0.0)

    # Create mock frame
    mock_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Set up read to return frames
    mock_cap_instance.read.side_effect = [
        (True, mock_frame),  # Frame 1
        (True, mock_frame),  # Frame 2
        (False, None)        # End of video
    ]

    # Mock imencode
    mock_encoded = MagicMock()
    mock_encoded.tobytes.return_value = b'mock_jpeg_data'
    mock_imencode.return_value = (True, mock_encoded)

    # Use a small queue that will fill up
    output_queue = mp.Queue(maxsize=1)
    shutdown_event = mp.Event()

    # First, fill the queue
    output_queue.put({"dummy": "message"})

    # Now run the frame grabber process - it should handle the full queue gracefully
    frame_capture_process(mock_config, output_queue, shutdown_event)

    # The queue should still have the dummy message (and possibly one more if it fit)
    assert not output_queue.empty()
    logger.info("Finished test_frame_capture_process_queue_full_handling") 
