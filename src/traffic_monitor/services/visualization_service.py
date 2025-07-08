import multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
from pathlib import Path
from queue import Empty
from typing import Any, Dict
from loguru import logger

import cv2
import numpy as np
import time
from collections import deque
import os

from ..utils.custom_types import TrackedVehicleMessage, VehicleCountMessage, OCRResultMessage, TrackedObject
from ..utils.logging_config import setup_logging

class VisualizationService:
    def __init__(self, config: dict):
        # Handle font using getattr for direct access to cv2 constants
        font_config = config.get("font", "FONT_HERSHEY_SIMPLEX")
        if isinstance(font_config, str):
            # Remove cv2. prefix if present
            font_name = font_config.replace("cv2.", "")
            self.font = getattr(cv2, font_name, cv2.FONT_HERSHEY_SIMPLEX)
        else:
            self.font = font_config  # Already an integer
            
        self.font_scale = config.get("font_scale", 0.6)
        self.font_thickness = config.get("font_thickness", 2)
        self.ocr_duration = config.get("ocr_duration", 3.0)

        # Parse colors safely
        self.colors = self._parse_colors(config.get("class_colors", {}))
        self.default_color = self._parse_color(config.get("default_color", [255, 255, 255]))

        self.counting_lines_relative = config.get("counting_lines", [])
        self.counting_line_color = self._parse_color(config.get("counting_line_color", [0, 255, 255]))  # Yellow by default
        self.counting_line_thickness = config.get("counting_line_thickness", 3)
        
        # key: track_id  -> {"text": str, "confidence": float}
        self.buffer = {}
        self.buffer_timeout = config.get("buffer_timeout", 0.5)  # seconds

        # Metrics for synchronization
        self.metrics = {
            "total_frames_processed": 0,
            "complete_frames": 0,
            "incomplete_frames": 0,
            "frames_missing_ocr": 0,
            "frames_missing_count": 0,
        }
        self.fps_calculator = deque(maxlen=60)
        
        # Add timing controls for consistent video output
        self.last_frame_timestamp = None
        self.frame_count = 0
        self.video_start_time = None
        
        logger.info(f"[VisualizationService] Visualization service initialized with font: {self.font}, font scale: {self.font_scale}, font thickness: {self.font_thickness}")
        logger.info(f"[VisualizationService] Loaded {len(self.counting_lines_relative)} counting line(s) for visualization")
        logger.debug(f"[VisualizationService] Parsed colors: {self.colors}")
        self.save_to_file = config.get("save_to_file", False)
        self.video_writer: cv2.VideoWriter | None = None

        if self.save_to_file:
            self.output_path = config.get("save_path", "data/videos/output/")
            self.output_fourcc = config.get("output_fourcc", "mp4v")
            logger.info(f"[Visualizer] Saving to file: {self.output_path} with fourcc: {self.output_fourcc}")
            
    def _parse_color(self, color_value):
        """Parse color value from various formats to tuple."""
        if isinstance(color_value, (list, tuple)):
            # Fastest path, no changes needed.
            return tuple(color_value)
        elif isinstance(color_value, str):
            # Optimized: Try strict format parsing before slower fallback.
            s = color_value
            if s.startswith('(') and s.endswith(')'):
                # Remove outer parenthesis once
                color_str = s[1:-1]
                try:
                    # Use map(int, ...) which is much faster than generator with strip in Python C API
                    return tuple(map(int, map(str.strip, color_str.split(','))))
                except Exception as e:
                    logger.warning(f"[Visualizer] Error parsing color '{color_value}': {e}, using default")
                    return (255, 255, 255)
            else:
                logger.warning(f"[Visualizer] Invalid color format: {color_value}, using default")
                return (255, 255, 255)
        else:
            logger.warning(f"[Visualizer] Unknown color format: {color_value}, using default")
            return (255, 255, 255)

    def _parse_colors(self, colors_config):
        """Parse all colors from config."""
        parsed_colors = {}
        for class_name, color_value in colors_config.items():
            parsed_colors[class_name] = self._parse_color(color_value)
        return parsed_colors

    def _buffer_message(self, message: dict):
        """Buffers incoming messages by frame_id."""
        frame_id = message.get("frame_id")
        if frame_id is None:
            logger.warning(f"Message is missing frame_id, cannot buffer: {message}")
            return

        if frame_id not in self.buffer:
            self.buffer[frame_id] = {"received_at": time.time()}

        # Use unique keys to identify message type
        if "tracked_objects" in message:
            self.buffer[frame_id]["tracking"] = message
        elif "lp_text" in message:
            if "ocr_results" not in self.buffer[frame_id]:
                self.buffer[frame_id]["ocr_results"] = {}
            self.buffer[frame_id]["ocr_results"][message["vehicle_id"]] = message
        elif "total_count" in message:
            self.buffer[frame_id]["vehicle_count"] = message

    def _get_ready_frame(self) -> dict | None:
        """
        Retrieves the next frame to be processed if it's ready.
        Frames are processed in ascending order of frame_id.
        """
        if not self.buffer:
            return None

        # Sort keys to process frames in order
        sorted_frame_ids = sorted(self.buffer.keys(), key=lambda x: int(x))
        
        next_frame_id = sorted_frame_ids[0]
        frame_data = self.buffer[next_frame_id]

        if "tracking" not in frame_data:
            return None # Essential data not yet present

        is_complete = "ocr_results" in frame_data and "vehicle_count" in frame_data
        is_timed_out = (time.time() - frame_data["received_at"]) > self.buffer_timeout

        if is_complete or is_timed_out:
            self.metrics["total_frames_processed"] += 1
            if is_complete:
                self.metrics["complete_frames"] += 1
            else:
                self.metrics["incomplete_frames"] += 1
                if "ocr_results" not in frame_data:
                    self.metrics["frames_missing_ocr"] += 1
                if "vehicle_count" not in frame_data:
                    self.metrics["frames_missing_count"] += 1
            
            return self.buffer.pop(next_frame_id)

        return None

    def _init_video_writer(self, frame_width: int, frame_height: int, og_fps: float):
        filename = f"output_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        filepath = Path(self.output_path) / filename
        
        # Ensure output directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        fourcc = cv2.VideoWriter.fourcc(*self.output_fourcc)
        self.video_writer = cv2.VideoWriter(str(filepath), fourcc, og_fps, (frame_width, frame_height))

        if self.video_writer.isOpened():
            logger.info(f"[Visualizer] Successfully initialized video writer to {filepath}")
        else:
            logger.error(f"[Visualizer] Failed to initialize video writer to {filepath}")
            logger.error(f"[Visualizer] Debug info - fourcc: {self.output_fourcc}, fps: {og_fps}, dimensions: {frame_width}x{frame_height}")
            self.video_writer = None

    def _draw_vehicle_info(self, image: np.ndarray, vehicle: TrackedObject, ocr_results: dict):
        x1, y1, x2, y2 = vehicle["bbox_xyxy"]
        class_name = vehicle["class_name"]
        track_id = vehicle["track_id"]

        color = self.colors.get(class_name, self.default_color)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.font_thickness)

        label = f"{class_name} {track_id}"
        if ocr_results and track_id in ocr_results:
            label += f" {ocr_results[track_id]['lp_text']}"

        (text_width, text_height), baseline = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
        cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1 - baseline), color, cv2.FILLED)
        cv2.putText(image, label, (x1, y1 - baseline), self.font, self.font_scale, (0, 0, 0), self.font_thickness)

    def _draw_stats(self, image: np.ndarray, vehicle_count: dict):
        # FPS calculation remains the same
        if len(self.fps_calculator) >= 10:
            fps = len(self.fps_calculator) / (self.fps_calculator[-1] - self.fps_calculator[0])
            fps_text = f"FPS: {fps:.1f}"
        else:
            fps_text = f"FPS: Initializing... ({len(self.fps_calculator)}/10)"
        cv2.putText(image, fps_text, (10, 30), self.font, self.font_scale, (255, 255, 255), self.font_thickness)

        # Draw vehicle counts from the buffered message
        if vehicle_count:
            total = vehicle_count.get("total_count", 0)
            by_class = vehicle_count.get("class_counts", {})
            count_text = f"Total: {total}"
            cv2.putText(image, count_text, (10, 70), self.font, self.font_scale, (255, 255, 255), self.font_thickness)
            
            index = 0
            for class_name, count in by_class.items():
                class_text = f"{class_name}: {count}"
                cv2.putText(image, class_text, (10, 100 + (index * 20)), self.font, self.font_scale, (255, 255, 255), self.font_thickness)
                index += 1

    def _draw_counting_lines(self, image: np.ndarray, frame_width: int, frame_height: int):
        """Draw counting lines on the frame, handling both relative and absolute coordinates."""
        if not self.counting_lines_relative:
            return
        
        # Check if coordinates are already absolute (same logic as vehicle counter)
        counting_lines_absolute = []
        for line in self.counting_lines_relative:
            if len(line) >= 2:
                # Check if coordinates are absolute (integers) or relative (floats)
                if isinstance(line[0][0], int):
                    # Coordinates are already absolute, use them directly
                    absolute_line = [[int(line[0][0]), int(line[0][1])], [int(line[1][0]), int(line[1][1])]]
                    counting_lines_absolute.append(absolute_line)
                    
                    # Log on first few frames
                    if hasattr(self, 'frame_count') and self.frame_count < 5:
                        logger.info(f"[VisualizationService] Using absolute coordinates directly: {absolute_line}")
                else:
                    # Coordinates are relative, convert to absolute
                    absolute_line = [
                        [int(line[0][0] * frame_width), int(line[0][1] * frame_height)],
                        [int(line[1][0] * frame_width), int(line[1][1] * frame_height)]
                    ]
                    counting_lines_absolute.append(absolute_line)
                    
                    # Log on first few frames
                    if hasattr(self, 'frame_count') and self.frame_count < 5:
                        logger.info(f"[VisualizationService] Converting relative {line} to absolute {absolute_line}")
        
        # Log the conversion on first frame (when frame count is low)
        if hasattr(self, 'frame_count') and self.frame_count < 5:
            logger.info(f"[VisualizationService] Drawing {len(counting_lines_absolute)} counting line(s) on {frame_width}x{frame_height} frame")
        
        for i, absolute_line in enumerate(counting_lines_absolute):
            if len(absolute_line) >= 2:
                start_abs = (absolute_line[0][0], absolute_line[0][1])
                end_abs = (absolute_line[1][0], absolute_line[1][1])
                
                # Log absolute coordinates on first few frames
                if hasattr(self, 'frame_count') and self.frame_count < 5:
                    logger.info(f"[VisualizationService] Drawing counting line {i+1}: ({start_abs[0]},{start_abs[1]}) to ({end_abs[0]},{end_abs[1]})")
                
                # Draw the line
                color = self.counting_line_color or (0, 255, 255)
                cv2.line(image, start_abs, end_abs, color, self.counting_line_thickness)
                
    def process_buffered_frame(self, frame_data: dict) -> np.ndarray:
        tracking_msg = frame_data["tracking"]
        jpeg_bytes = tracking_msg["frame_data_jpeg"]
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        current_time = time.time()

        self.fps_calculator.append(current_time)

        if self.save_to_file and self.video_writer is None:
            self._init_video_writer(tracking_msg["frame_width"], tracking_msg["frame_height"], tracking_msg["og_fps"])

        ocr_results = frame_data.get("ocr_results", {})
        for vehicle in tracking_msg["tracked_objects"]:
            self._draw_vehicle_info(frame, vehicle, ocr_results)

        self._draw_stats(frame, frame_data.get("vehicle_count", {}))
        self._draw_counting_lines(frame, tracking_msg["frame_width"], tracking_msg["frame_height"])

        if self.video_writer:
            self.video_writer.write(frame)
            self.frame_count += 1

        return frame
    
    def release(self):
        if self.video_writer:
            logger.debug("Releasing video writer")
            self.video_writer.release()
            self.video_writer = None

def visualization_process(config: dict, tracking_queue: Queue, OCR_queue: Queue, vehicle_count_queue: Queue, shutdown_event: Event):
    # Setup logging for this process
    try:
        setup_logging(config.get("loguru"))
        logger.info("VisualizationService process started")
    except Exception as e:
        print(f"Failed to setup logging: {e}")
    
    process_name = mp.current_process().name

    # Determine whether GUI display is enabled. Default True but disable automatically
    # if the environment does not have a DISPLAY (common on headless Linux servers).
    enable_gui: bool = config.get("enable_gui", True)
    if enable_gui and not os.environ.get("DISPLAY"):
        logger.warning("DISPLAY environment variable not found. Running in headless mode (enable_gui=False)")
        enable_gui = False

    visualizer = None
    try:
        # If GUI is enabled, verify that we can open a window; otherwise, switch to headless
        if enable_gui:
            logger.debug("Testing OpenCV GUI capabilities")
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            try:
                cv2.imshow("Traffic Monitor", test_img)
                cv2.waitKey(1)  # Process window events
                cv2.destroyWindow("Traffic Monitor")
                logger.info("OpenCV GUI available - running with window output")
            except Exception as window_error:
                logger.warning(f"OpenCV GUI not available: {window_error}. Falling back to headless mode.")
                enable_gui = False

        visualizer = VisualizationService(config)
        logger.info("VisualizationService initialized successfully")
        
        # Log the counting lines being used for visualization
        counting_lines = config.get("counting_lines", [])
        logger.info(f"[VisualizationService] Using counting lines for visualization: {counting_lines}")
        if counting_lines:
            logger.info(f"[VisualizationService] Will draw {len(counting_lines)} counting line(s) on output video")

        frame_count = 0
        queues = [tracking_queue, OCR_queue, vehicle_count_queue]
        
        while not shutdown_event.is_set():
            # Drain all queues into the buffer
            for q in queues:
                try:
                    while True:
                        msg = q.get_nowait()
                        if msg is None:  # Sentinel value
                            shutdown_event.set()
                            break
                        visualizer._buffer_message(msg)
                except Empty:
                    continue
            if shutdown_event.is_set():
                break

            # Process any ready frames
            while True:
                ready_frame_data = visualizer._get_ready_frame()
                if ready_frame_data is None:
                    break  # No more frames are ready right now

                display_frame = visualizer.process_buffered_frame(ready_frame_data)
                
                if enable_gui:
                    cv2.imshow("Traffic Monitor", display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        shutdown_event.set()
                        break
                
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames. Buffer size: {len(visualizer.buffer)}")

            time.sleep(0.001)  # Prevent busy-waiting
    
    except Exception as e:
        logger.error(f"Visualizer process encountered critical error: {e}")
        logger.exception("Full exception traceback")
    
    finally:
        logger.info("Cleaning up visualizer process")
        try:
            if visualizer: # Check if visualizer was successfully initialized
                visualizer.release()
            if enable_gui:
                cv2.destroyAllWindows()
            logger.debug("OpenCV windows destroyed")
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")
        
        logger.info(f"Visualizer process {process_name} shutting down")
