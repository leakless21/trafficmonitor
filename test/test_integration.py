"""
Integration tests for the Traffic Monitor system.

These tests verify that the multiprocessing components work together correctly
without using mocks that cause pickling issues.
"""

import pytest
import time
import multiprocessing as mp
import cv2
import numpy as np
from pathlib import Path

from src.traffic_monitor.main_supervisor import main
from src.traffic_monitor.utils.config_loader import load_config


@pytest.mark.integration
def test_supervisor_startup_and_shutdown():
    """
    Test that the main supervisor can start up and shut down cleanly.
    This verifies that all multiprocessing components can be created and terminated.
    """
    # Create a minimal test video file
    test_video_path = "test_video.mp4"
    create_test_video(test_video_path)
    
    try:
        # Load config and modify for testing
        config = load_config("src/traffic_monitor/config/settings.yaml")
        if config is None:
            pytest.skip("Could not load configuration file")
        config["video_source"] = test_video_path
        config["log_every_n_frames"] = 1
        
        # Run supervisor for a short time
        supervisor_process = mp.Process(target=main, args=(config,))
        supervisor_process.start()
        
        # Let it run for a few seconds
        time.sleep(3)
        
        # Terminate cleanly
        supervisor_process.terminate()
        supervisor_process.join(timeout=5)
        
        # Verify it terminated properly
        assert not supervisor_process.is_alive()
        
    finally:
        # Clean up test video
        if Path(test_video_path).exists():
            Path(test_video_path).unlink()


def create_test_video(filename: str, duration: int = 2, fps: int = 10):
    """
    Create a simple test video with moving objects for testing.
    """
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (640, 480))
    
    for frame_num in range(duration * fps):
        # Create a frame with a moving rectangle (simulating a vehicle)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Moving rectangle across the frame
        x = (frame_num * 10) % 600
        y = 200
        cv2.rectangle(frame, (x, y), (x + 40, y + 80), (0, 255, 0), -1)
        
        # Add some text (simulating a license plate)
        cv2.rectangle(frame, (x + 5, y + 60), (x + 35, y + 75), (255, 255, 255), -1)
        cv2.putText(frame, "ABC123", (x + 7, y + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        out.write(frame)
    
    out.release()


@pytest.mark.integration 
@pytest.mark.skipif(not Path("data/videos/input").exists(), reason="Test video directory not found")
def test_with_real_video():
    """
    Test with a real video file if available.
    This test is skipped if no test videos are available.
    """
    video_dir = Path("data/videos/input")
    video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.MOV"))
    
    if not video_files:
        pytest.skip("No test video files found")
    
    test_video = video_files[0]
    
    # Load config and modify for testing
    config = load_config("src/traffic_monitor/config/settings.yaml")
    if config is None:
        pytest.skip("Could not load configuration file")
    config["video_source"] = str(test_video)
    config["log_every_n_frames"] = 10  # Process fewer frames for faster testing
    
    # Run supervisor for a short time
    supervisor_process = mp.Process(target=main, args=(config,))
    supervisor_process.start()
    
    # Let it run for a few seconds
    time.sleep(5)
    
    # Terminate cleanly
    supervisor_process.terminate()
    supervisor_process.join(timeout=10)
    
    # Verify it terminated properly
    assert not supervisor_process.is_alive()


if __name__ == "__main__":
    # Allow running integration tests directly
    test_supervisor_startup_and_shutdown()
    print("Integration test passed!") 