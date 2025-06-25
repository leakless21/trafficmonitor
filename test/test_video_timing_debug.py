import pytest
import cv2
import time
import multiprocessing as mp
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.traffic_monitor.main_supervisor import main
from src.traffic_monitor.utils.config_loader import load_config


def test_video_timing_analysis():
    """
    Test to analyze video timing and identify frame drop patterns.
    This test helps debug the speed-up issue by tracking frame processing times.
    """
    # Check if test video exists
    test_video_path = Path("data/videos/input/IMG_3637.MOV")
    if not test_video_path.exists():
        pytest.skip(f"Test video not found: {test_video_path}")
    
    # Analyze the input video properties
    cap = cv2.VideoCapture(str(test_video_path))
    if not cap.isOpened():
        pytest.skip(f"Could not open test video: {test_video_path}")
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / original_fps
    
    print(f"\nInput Video Analysis:")
    print(f"- FPS: {original_fps}")
    print(f"- Frame Count: {frame_count}")
    print(f"- Duration: {duration:.2f} seconds")
    print(f"- Expected frame interval: {1/original_fps:.3f} seconds")
    
    cap.release()
    
    # Track frame processing times
    frame_times = []
    start_time = time.time()
    
    # Mock the visualizer to capture frame timing
    original_process_frame = None
    
    def mock_process_frame(self, frame_msg):
        current_time = time.time()
        frame_times.append({
            'timestamp': frame_msg['timestamp'],
            'process_time': current_time,
            'frame_id': frame_msg['frame_id']
        })
        
        # Call original method if it exists
        if original_process_frame:
            return original_process_frame(self, frame_msg)
        else:
            # Simple mock frame processing
            import numpy as np
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Run the system for a short time to collect timing data
    with patch('src.traffic_monitor.services.visualizer.Visualizer.process_frame', mock_process_frame):
        config = load_config("src/traffic_monitor/config/settings.yaml")
        if config is None:
            pytest.skip("Could not load configuration file")
        
        # Run for 10 seconds max
        supervisor_process = mp.Process(target=main, args=())
        supervisor_process.start()
        
        time.sleep(10)  # Let it run for 10 seconds
        
        supervisor_process.terminate()
        supervisor_process.join(timeout=5)
    
    # Analyze the collected timing data
    if len(frame_times) < 2:
        pytest.skip("Not enough frames processed for analysis")
    
    print(f"\nFrame Processing Analysis:")
    print(f"- Frames processed: {len(frame_times)}")
    
    # Calculate frame intervals
    intervals = []
    for i in range(1, len(frame_times)):
        interval = frame_times[i]['process_time'] - frame_times[i-1]['process_time']
        intervals.append(interval)
    
    if intervals:
        avg_interval = sum(intervals) / len(intervals)
        expected_interval = 1.0 / original_fps
        
        print(f"- Average processing interval: {avg_interval:.3f} seconds")
        print(f"- Expected interval: {expected_interval:.3f} seconds")
        print(f"- Processing speed ratio: {expected_interval/avg_interval:.2f}x")
        
        # Check for frame drops (intervals much larger than expected)
        large_gaps = [i for i in intervals if i > expected_interval * 2]
        if large_gaps:
            print(f"- WARNING: {len(large_gaps)} large frame gaps detected!")
            print(f"- Largest gap: {max(large_gaps):.3f} seconds")
        
        # Check for speed-up (intervals much smaller than expected)
        small_gaps = [i for i in intervals if i < expected_interval * 0.5]
        if small_gaps:
            print(f"- WARNING: {len(small_gaps)} suspiciously fast intervals detected!")
            print(f"- Smallest gap: {min(small_gaps):.3f} seconds")


def test_output_video_timing():
    """
    Test to compare input and output video timing.
    """
    input_video = Path("data/videos/input/IMG_3637.MOV")
    output_dir = Path("data/videos/output")
    
    if not input_video.exists():
        pytest.skip(f"Input video not found: {input_video}")
    
    # Find the most recent output video
    if output_dir.exists():
        output_videos = list(output_dir.glob("output_*.mp4"))
        if output_videos:
            latest_output = max(output_videos, key=lambda p: p.stat().st_mtime)
            
            # Compare input and output video properties
            input_cap = cv2.VideoCapture(str(input_video))
            output_cap = cv2.VideoCapture(str(latest_output))
            
            if input_cap.isOpened() and output_cap.isOpened():
                input_fps = input_cap.get(cv2.CAP_PROP_FPS)
                input_frames = int(input_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                input_duration = input_frames / input_fps
                
                output_fps = output_cap.get(cv2.CAP_PROP_FPS)
                output_frames = int(output_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                output_duration = output_frames / output_fps
                
                print(f"\nVideo Comparison:")
                print(f"Input:  {input_frames} frames @ {input_fps} FPS = {input_duration:.2f}s")
                print(f"Output: {output_frames} frames @ {output_fps} FPS = {output_duration:.2f}s")
                print(f"Duration ratio: {output_duration/input_duration:.2f}x")
                
                if output_duration < input_duration * 0.9:
                    print("WARNING: Output video is significantly shorter - indicates frame dropping!")
                elif output_duration > input_duration * 1.1:
                    print("WARNING: Output video is significantly longer - indicates processing issues!")
                
                input_cap.release()
                output_cap.release()


if __name__ == "__main__":
    test_video_timing_analysis()
    test_output_video_timing() 