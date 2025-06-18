import pytest
import time
import cv2
import multiprocessing as mp
from unittest.mock import MagicMock, patch
from src.traffic_monitor.services.frame_grabber import frame_grabber_process
from src.traffic_monitor.utils.logging_config import setup_logging
from loguru import logger
from queue import Full, Empty

# Initialize logging for the test environment
setup_logging()

@pytest.fixture
def mock_config():
    """Provides a mock configuration dictionary for tests."""
    return {
        "video_source": "dummy_video.mp4",
        "camera_id": "test_cam",
        "log_every_n_frames": 1
    }


@patch('cv2.VideoCapture')
def test_frame_grabber_process_basic_flow(
    mock_video_capture_class,
    mock_config
):
    """
    Tests the basic flow of frame capturing and queuing.
    """
    logger.info("Running test_frame_grabber_process_basic_flow")

    # Create real multiprocessing objects
    output_queue = mp.Queue()
    shutdown_event = mp.Event()

    with patch.object(output_queue, 'put') as mock_queue_put:
        with patch.object(shutdown_event, 'is_set') as mock_shutdown_event_is_set:
            with patch.object(shutdown_event, 'set') as mock_shutdown_event_set:

                mock_cap_instance = MagicMock()
                mock_video_capture_class.return_value = mock_cap_instance
                mock_cap_instance.isOpened.return_value = True

                mock_frame = MagicMock()
                mock_frame.shape = (480, 640, 3)
                
                with patch('cv2.imencode', return_value=(True, MagicMock(tobytes=MagicMock(return_value=b'mock_jpeg_data')))):
                    mock_cap_instance.read.side_effect = [
                        (True, mock_frame),
                        (True, mock_frame),
                        (False, None) 
                    ]
                    
                    mock_shutdown_event_is_set.side_effect = [False, False, True]

                    process = mp.Process(
                        target=frame_grabber_process,
                        args=(mock_config, output_queue, shutdown_event)
                    )
                    process.start()

                    time.sleep(0.1) 
                    mock_shutdown_event_set.assert_not_called()
                    shutdown_event.set()
                    process.join(timeout=1)
                    
                    mock_video_capture_class.assert_called_once_with(mock_config["video_source"])
                    mock_cap_instance.isOpened.assert_called_once()
                    assert mock_cap_instance.read.call_count >= 2
                    assert mock_queue_put.call_count >= 2
                    
                    first_call_args, _ = mock_queue_put.call_args_list[0]
                    assert "frame_id" in first_call_args[0]
                    assert first_call_args[0]["camera_id"] == "test_cam"
                    assert first_call_args[0]["frame_data_jpeg"] == b'mock_jpeg_data'

    logger.info("Finished test_frame_grabber_process_basic_flow")

@patch('cv2.VideoCapture')
def test_frame_grabber_process_no_video_source(
    mock_video_capture_class
):
    """
    Tests handling of missing video source configuration.
    """
    logger.info("Running test_frame_grabber_process_no_video_source")
    config = {"camera_id": "test_cam"}

    output_queue = mp.Queue()
    shutdown_event = mp.Event()

    with patch.object(output_queue, 'put') as mock_queue_put:
        with patch.object(shutdown_event, 'set') as mock_shutdown_event_set:

            frame_grabber_process(config, output_queue, shutdown_event)

            mock_video_capture_class.assert_not_called()
            mock_queue_put.assert_not_called()
            mock_shutdown_event_set.assert_not_called()
    logger.info("Finished test_frame_grabber_process_no_video_source")

@patch('cv2.VideoCapture')
def test_frame_grabber_process_capture_failure(
    mock_video_capture_class,
    mock_config
):
    """
    Tests behavior when video capture fails to open.
    """
    logger.info("Running test_frame_grabber_process_capture_failure")
    mock_cap_instance = MagicMock()
    mock_video_capture_class.return_value = mock_cap_instance
    mock_cap_instance.isOpened.return_value = False

    output_queue = mp.Queue()
    shutdown_event = mp.Event()

    with patch.object(output_queue, 'put') as mock_queue_put:
        with patch.object(shutdown_event, 'set') as mock_shutdown_event_set:

            frame_grabber_process(mock_config, output_queue, shutdown_event)

            mock_video_capture_class.assert_called_once_with(mock_config["video_source"])
            mock_cap_instance.isOpened.assert_called_once()
            mock_cap_instance.read.assert_not_called()
            mock_queue_put.assert_not_called()
            mock_shutdown_event_set.assert_not_called()
    logger.info("Finished test_frame_grabber_process_capture_failure")

@patch('cv2.VideoCapture')
def test_frame_grabber_process_queue_full(
    mock_video_capture_class,
    mock_config
):
    """
    Tests handling of a full output queue.
    """
    logger.info("Running test_frame_grabber_process_queue_full")

    output_queue = mp.Queue()
    shutdown_event = mp.Event()

    with patch.object(output_queue, 'put', side_effect=Full) as mock_queue_put:
        with patch.object(shutdown_event, 'is_set') as mock_shutdown_event_is_set:
            with patch.object(shutdown_event, 'set') as mock_shutdown_event_set:

                mock_cap_instance = MagicMock()
                mock_video_capture_class.return_value = mock_cap_instance
                mock_cap_instance.isOpened.return_value = True

                mock_frame = MagicMock()
                mock_frame.shape = (480, 640, 3)

                mock_cap_instance.read.return_value = (True, mock_frame)
                mock_shutdown_event_is_set.side_effect = [False, True]
                
                with patch('cv2.imencode', return_value=(True, MagicMock(tobytes=MagicMock(return_value=b'mock_jpeg_data')))):
                    process = mp.Process(
                        target=frame_grabber_process,
                        args=(mock_config, output_queue, shutdown_event)
                    )
                    process.start()
                    time.sleep(0.1)
                    mock_shutdown_event_set.assert_not_called()
                    shutdown_event.set()
                    process.join(timeout=1)

                mock_queue_put.assert_called()
    logger.info("Finished test_frame_grabber_process_queue_full") 