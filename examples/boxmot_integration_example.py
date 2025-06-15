#!/usr/bin/env python3
"""
BoxMOT Integration Example for Traffic Monitor Application

This example demonstrates how to integrate BoxMOT tracking into a Python application
for vehicle tracking in traffic monitoring scenarios.
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from boxmot import create_tracker, get_tracker_config
    BOXMOT_AVAILABLE = True
except ImportError:
    logger.warning("BoxMOT not installed. Install with: pip install boxmot")
    BOXMOT_AVAILABLE = False

class TrafficTracker:
    """
    Traffic monitoring tracker using BoxMOT for multi-object tracking
    """
    
    def __init__(
        self, 
        tracker_type: str = 'bytetrack',
        device: str = 'cpu',
        reid_weights: Optional[Path] = None,
        confidence_threshold: float = 0.5
    ):
        """
        Initialize the traffic tracker
        
        Args:
            tracker_type: Type of tracker to use ('bytetrack', 'botsort', 'strongsort', etc.)
            device: Device to run on ('cpu' or 'cuda')
            reid_weights: Path to ReID model weights (for appearance-based trackers)
            confidence_threshold: Minimum confidence for detections
        """
        if not BOXMOT_AVAILABLE:
            raise ImportError("BoxMOT is required. Install with: pip install boxmot")
        
        self.tracker_type = tracker_type
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Initialize tracker
        self.tracker = create_tracker(
            tracker_type=tracker_type,
            tracker_config=None,  # Use default config
            reid_weights=reid_weights,
            device=device,
            half=device == 'cuda',  # Use FP16 on GPU
            per_class=False
        )
        
        # Vehicle class IDs (COCO dataset)
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
        # Tracking statistics
        self.frame_count = 0
        self.total_tracks = 0
        self.active_tracks = set()
        
        logger.info(f"Initialized {tracker_type} tracker on {device}")
    
    def dummy_detector(self, frame: np.ndarray) -> np.ndarray:
        """
        Dummy detector for demonstration purposes.
        Replace this with your actual detection model (YOLO, etc.)
        
        Args:
            frame: Input video frame
            
        Returns:
            Detections array of shape (N, 6) with format [x1, y1, x2, y2, conf, cls]
        """
        # Generate some dummy vehicle detections for demonstration
        h, w = frame.shape[:2]
        
        # Simulate 2-4 vehicle detections per frame
        num_detections = np.random.randint(2, 5)
        detections = []
        
        for _ in range(num_detections):
            # Random bounding box
            x1 = np.random.randint(0, w // 2)
            y1 = np.random.randint(0, h // 2)
            x2 = x1 + np.random.randint(50, 200)
            y2 = y1 + np.random.randint(30, 100)
            
            # Random confidence and vehicle class
            conf = np.random.uniform(0.3, 0.95)
            cls = np.random.choice(list(self.vehicle_classes.keys()))
            
            detections.append([x1, y1, x2, y2, conf, cls])
        
        return np.array(detections, dtype=np.float32)
    
    def filter_detections(self, detections: np.ndarray) -> np.ndarray:
        """
        Filter detections based on confidence and vehicle classes
        
        Args:
            detections: Raw detections array
            
        Returns:
            Filtered detections array
        """
        if len(detections) == 0:
            return detections
        
        # Filter by confidence threshold
        conf_mask = detections[:, 4] >= self.confidence_threshold
        
        # Filter by vehicle classes only
        class_mask = np.isin(detections[:, 5], list(self.vehicle_classes.keys()))
        
        # Combine filters
        valid_mask = conf_mask & class_mask
        
        return detections[valid_mask]
    
    def update_tracking_stats(self, tracks: np.ndarray):
        """
        Update tracking statistics
        
        Args:
            tracks: Current frame tracks
        """
        if len(tracks) > 0:
            current_track_ids = set(tracks[:, 4].astype(int))
            
            # Count new tracks
            new_tracks = current_track_ids - self.active_tracks
            self.total_tracks += len(new_tracks)
            
            # Update active tracks
            self.active_tracks = current_track_ids
    
    def visualize_tracks(self, frame: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        """
        Visualize tracking results on the frame
        
        Args:
            frame: Input frame
            tracks: Tracking results
            
        Returns:
            Frame with visualizations
        """
        vis_frame = frame.copy()
        
        # Define colors for different vehicle types
        colors = {
            2: (0, 255, 0),    # car - green
            3: (255, 0, 0),    # motorcycle - blue
            5: (0, 255, 255),  # bus - yellow
            7: (255, 0, 255)   # truck - magenta
        }
        
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls, _ = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)
            cls = int(cls)
            
            # Get color for vehicle class
            color = colors.get(cls, (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and vehicle type
            vehicle_type = self.vehicle_classes.get(cls, 'unknown')
            label = f'ID:{track_id} {vehicle_type} {conf:.2f}'
            
            # Calculate text size and position
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                vis_frame, 
                (x1, y1 - text_height - 10), 
                (x1 + text_width, y1), 
                color, 
                -1
            )
            
            # Draw text
            cv2.putText(
                vis_frame, 
                label, 
                (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (255, 255, 255), 
                2
            )
        
        # Draw statistics
        stats_text = [
            f"Frame: {self.frame_count}",
            f"Active Tracks: {len(self.active_tracks)}",
            f"Total Tracks: {self.total_tracks}",
            f"Tracker: {self.tracker_type}"
        ]
        
        for i, text in enumerate(stats_text):
            cv2.putText(
                vis_frame,
                text,
                (10, 30 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return vis_frame
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a single frame through the tracking pipeline
        
        Args:
            frame: Input video frame
            
        Returns:
            Tuple of (tracks, visualized_frame)
        """
        self.frame_count += 1
        
        # Get detections (replace with your actual detector)
        detections = self.dummy_detector(frame)
        
        # Filter detections
        detections = self.filter_detections(detections)
        
        # Update tracker
        tracks = self.tracker.update(detections, frame)
        
        # Update statistics
        self.update_tracking_stats(tracks)
        
        # Visualize results
        vis_frame = self.visualize_tracks(frame, tracks)
        
        return tracks, vis_frame
    
    def process_video(
        self, 
        video_path: str, 
        output_path: Optional[str] = None,
        display: bool = True
    ):
        """
        Process a video file with tracking
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            display: Whether to display video during processing
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Processing video: {video_path}")
        logger.info(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Initialize video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                tracks, vis_frame = self.process_frame(frame)
                
                # Save frame if writer is available
                if writer:
                    writer.write(vis_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Traffic Tracking', vis_frame)
                    
                    # Check for exit key
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):  # Save screenshot
                        screenshot_path = f"tracking_screenshot_{self.frame_count}.jpg"
                        cv2.imwrite(screenshot_path, vis_frame)
                        logger.info(f"Screenshot saved: {screenshot_path}")
                
                # Progress update
                if self.frame_count % 100 == 0:
                    progress = (self.frame_count / total_frames) * 100
                    logger.info(f"Progress: {progress:.1f}% ({self.frame_count}/{total_frames})")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        logger.info(f"Processing complete. Total tracks: {self.total_tracks}")

def main():
    """
    Main function demonstrating BoxMOT integration
    """
    if not BOXMOT_AVAILABLE:
        logger.error("BoxMOT is not available. Please install with: pip install boxmot")
        return
    
    # Configuration
    config = {
        'tracker_type': 'bytetrack',  # Fast tracker for real-time performance
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'confidence_threshold': 0.5
    }
    
    logger.info(f"Using device: {config['device']}")
    
    # Initialize tracker
    tracker = TrafficTracker(**config)
    
    # Example 1: Process webcam feed
    logger.info("Starting webcam tracking (press 'q' to quit, 's' to save screenshot)")
    
    cap = cv2.VideoCapture(0)  # Use webcam
    
    if cap.isOpened():
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                tracks, vis_frame = tracker.process_frame(frame)
                
                # Display
                cv2.imshow('Traffic Tracking - Webcam', vis_frame)
                
                # Check for exit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_path = f"webcam_tracking_{tracker.frame_count}.jpg"
                    cv2.imwrite(screenshot_path, vis_frame)
                    logger.info(f"Screenshot saved: {screenshot_path}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
    else:
        logger.warning("Could not open webcam")
    
    # Example 2: Process video file (uncomment to use)
    # video_path = "path/to/your/traffic_video.mp4"
    # output_path = "tracked_output.mp4"
    # tracker.process_video(video_path, output_path, display=True)

if __name__ == "__main__":
    main()