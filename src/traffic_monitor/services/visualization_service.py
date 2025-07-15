import multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
from pathlib import Path
from queue import Empty
from typing import Dict
from loguru import logger

import cv2
import numpy as np
import time
from collections import deque
import os

from ..utils.custom_types import TrackedVehicleMessage, VehicleCountMessage, OCRResultMessage, TrackedObject
from ..utils.utils import relative_to_absolute_coords
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

        # Store counting lines for visualization (now in relative coordinates)
        self.counting_lines_relative = config.get("counting_lines", [])
        self.counting_line_color = self._parse_color(config.get("counting_line_color", [0, 255, 255]))  # Yellow by default
        self.counting_line_thickness = config.get("counting_line_thickness", 3)
        
        # key: track_id  -> {"text": str, "confidence": float}
        self.latest_ocr_results: dict[int, dict] = {}
        self.latest_vehicle_count: VehicleCountMessage | Dict = {}
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
            return tuple(color_value)
        elif isinstance(color_value, str):
            # Handle string format like "(255, 0, 0)"
            try:
                if color_value.startswith('(') and color_value.endswith(')'):
                    color_str = color_value.strip('()')
                    return tuple(int(x.strip()) for x in color_str.split(','))
                else:
                    logger.warning(f"[Visualizer] Invalid color format: {color_value}, using default")
                    return (255, 255, 255)
            except Exception as e:
                logger.warning(f"[Visualizer] Error parsing color '{color_value}': {e}, using default")
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

    def _draw_vehicle_info(self, image: np.ndarray, vehicle: TrackedObject):
        x1, y1, x2, y2 = vehicle["bbox_xyxy"]
        class_name = vehicle["class_name"]
        track_id = vehicle["track_id"]

        color = self.colors.get(class_name, self.default_color)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.font_thickness)

        label = f"{class_name} {track_id}"
        
        # Always show the most confident plate we have recorded for this track
        if track_id in self.latest_ocr_results:
            label += f" {self.latest_ocr_results[track_id]['text']}"
        
        (text_width, text_height), baseline = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
        cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1 - baseline), color, cv2.FILLED)
        cv2.putText(image, label, (x1, y1 - baseline), self.font, self.font_scale, (0, 0, 0), self.font_thickness)
        
    def _draw_stats(self, image: np.ndarray):
        # Calculate FPS - require at least 10 frames to avoid early inflation
        if len(self.fps_calculator) >= 10:
            fps = len(self.fps_calculator) / (self.fps_calculator[-1] - self.fps_calculator[0])
            fps_text = f"FPS: {fps:.1f}"
        else:
            fps_text = f"FPS: Initializing... ({len(self.fps_calculator)}/10)"
        
        cv2.putText(image, fps_text, (10, 30), self.font, self.font_scale, (255, 255, 255), self.font_thickness)

        # Draw vehicle counts
        #if self.latest_vehicle_count:
        total = self.latest_vehicle_count.get("total_count", 0)
        if total > 0 and hash(self.frame_count) % 100 == 0:
            logger.debug(f"[Visualizer] Drawing stats with total_count={total}")
        by_class = self.latest_vehicle_count.get("class_counts", {})

        # Draw total count
        count_text = f"Total: {total}"
        cv2.putText(image, count_text, (10, 70), self.font, self.font_scale, (255, 255, 255), self.font_thickness)

        # Draw class counts
        index = 0
        for class_name, count in by_class.items():
            class_text = f"{class_name}: {count}"
            cv2.putText(image, class_text, (10, 100 + (index * 20)), self.font, self.font_scale, (255, 255, 255), self.font_thickness)
            index += 1

    def _draw_counting_lines(self, image: np.ndarray, frame_width: int, frame_height: int):
        """Draw counting lines on the frame using relative coordinates."""
        if not self.counting_lines_relative:
            return
        
        # Convert all relative lines to absolute coordinates at once
        counting_lines_absolute = relative_to_absolute_coords(
            self.counting_lines_relative, frame_width, frame_height
        )
        
        for i, absolute_line in enumerate(counting_lines_absolute):
            if len(absolute_line) >= 2:
                start_abs = (absolute_line[0][0], absolute_line[0][1])
                end_abs = (absolute_line[1][0], absolute_line[1][1])
                
                # Draw the line
                cv2.line(image, start_abs, end_abs, self.counting_line_color, self.counting_line_thickness)
                
    def process_frame(self, frame_msg: TrackedVehicleMessage) -> np.ndarray:
        jpeg_bytes = frame_msg["frame_data_jpeg"]
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        current_time = time.time()
        
        self.fps_calculator.append(current_time)

        # Initialize video timing on first frame
        if self.video_start_time is None:
            self.video_start_time = current_time
            self.last_frame_timestamp = frame_msg["timestamp"]

        if self.save_to_file and self.video_writer is None:
            self._init_video_writer(frame_msg["frame_width"], frame_msg["frame_height"], frame_msg["og_fps"])

        for vehicle in frame_msg["tracked_objects"]:
            self._draw_vehicle_info(frame, vehicle)
        
        self._draw_stats(frame)
        self._draw_counting_lines(frame, frame_msg["frame_width"], frame_msg["frame_height"])

        # Write every processed frame to maintain proper video timing
        if self.video_writer:
            self.video_writer.write(frame)
            self.frame_count += 1
            
            # Log frame writing progress less frequently
            if self.frame_count % 500 == 0:  # Every 500 frames instead of 100
                elapsed_time = current_time - self.video_start_time
                expected_frames = elapsed_time * frame_msg["og_fps"]
                frame_ratio = self.frame_count / expected_frames if expected_frames > 0 else 1.0
                logger.debug(f"Written {self.frame_count} frames. Frame ratio: {frame_ratio:.2f}")

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

        frame_count = 0
        while not shutdown_event.is_set():
            try:
                # Use non-blocking get to maintain real-time behavior
                tracking_msg: TrackedVehicleMessage = tracking_queue.get_nowait()
                logger.trace(f"Received tracking message for frame: {tracking_msg.get('frame_id', 'unknown')}")

                # NEW: Drain OCR and vehicle count queues on every iteration to keep overlays current
                try:
                    while True:
                        ocr_msg: OCRResultMessage = OCR_queue.get_nowait()
                        if ocr_msg:
                            track_id_from_ocr = ocr_msg["vehicle_id"]
                            new_conf = ocr_msg["ocr_confidence"]
                            existing = visualizer.latest_ocr_results.get(track_id_from_ocr)
                            if existing is None or new_conf > existing.get("confidence", 0):
                                visualizer.latest_ocr_results[track_id_from_ocr] = {
                                    "text": ocr_msg["lp_text"],
                                    "confidence": new_conf
                                }
                                logger.debug(f"[Visualizer] Updated plate for track {track_id_from_ocr}: {ocr_msg['lp_text']} (conf={new_conf:.3f})")
                            else:
                                logger.trace(f"[Visualizer] Ignored lower-confidence plate for track {track_id_from_ocr}: {ocr_msg['lp_text']} (conf={new_conf:.3f})")
                except Empty:
                    pass

                try:
                    while True:
                        count_msg: VehicleCountMessage = vehicle_count_queue.get_nowait()
                        if count_msg:
                            visualizer.latest_vehicle_count = count_msg
                            logger.debug(f"[Visualizer] Updated latest vehicle count: total={count_msg['total_count']} class_counts={count_msg['class_counts']}")
                except Empty:
                    pass
                
            except Empty:
                # No tracking message available, continue processing other queues and check again
                logger.trace("No tracking message received, continuing")
                
                # Process any remaining OCR and count messages for this frame
                try:
                    # Process any available OCR messages
                    while True:
                        try:
                            ocr_msg: OCRResultMessage = OCR_queue.get_nowait()
                            if ocr_msg:
                                track_id_from_ocr = ocr_msg["vehicle_id"]
                                new_conf = ocr_msg["ocr_confidence"]
                                existing = visualizer.latest_ocr_results.get(track_id_from_ocr)
                                if existing is None or new_conf > existing.get("confidence", 0):
                                    visualizer.latest_ocr_results[track_id_from_ocr] = {
                                        "text": ocr_msg["lp_text"],
                                        "confidence": new_conf
                                    }
                        except Empty:
                            break
                    
                    # Process any available vehicle count messages
                    while True:
                        try:
                            count_msg: VehicleCountMessage = vehicle_count_queue.get_nowait()
                            if count_msg:
                                visualizer.latest_vehicle_count = count_msg
                        except Empty:
                            break
                except Exception as e:
                    logger.error(f"Error processing OCR/count messages: {e}")
                
                if shutdown_event.is_set():
                    break
                time.sleep(0.001)  # Very short sleep to prevent busy waiting
                continue
            
            try:
                # Process and display the frame
                logger.trace("Processing frame for display")
                display_frame = visualizer.process_frame(tracking_msg)

                if enable_gui:
                    cv2.imshow("Traffic Monitor", display_frame)

                frame_count += 1
                if frame_count % 100 == 0:  # Log every 100 frames instead of 30
                    logger.info(f"Processed {frame_count} frames so far")

                # Check for quit signal
                if enable_gui:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("Quit signal received (q key)")
                        shutdown_event.set()
                        break
                    elif key != 255:  # Any other key pressed
                        logger.trace(f"Key pressed: {key}")
                    
            except Exception as frame_error:
                logger.error(f"Error processing frame: {frame_error}")
                continue
    
    except Exception as e:
        logger.error(f"Visualizer process encountered critical error: {e}")
        logger.exception("Full exception traceback")
    
    finally:
        logger.info("Cleaning up visualizer process")
        try:
            visualizer.release()
            if enable_gui:
                cv2.destroyAllWindows()
            logger.debug("OpenCV windows destroyed")
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")
        
        logger.info(f"Visualizer process {process_name} shutting down")

