import multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
from queue import Empty
from typing import Any, Dict
from loguru import logger

import cv2
import numpy as np
import time
from collections import deque

from ..utils.custom_types import VehicleTrackingMessage, VehicleCountMessage, OCRResultMessage, TrackedObject
from ..utils.logging_config import setup_logging

class Visualizer:
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
         
        # Counting line information will now come from VehicleCountMessage
        self.counting_lines_absolute = None
        self.line_color = [0, 0, 255]  # Default, will be updated from VehicleCountMessage
        self.line_thickness = 2  # Default, will be updated from VehicleCountMessage
        
        self.latest_ocr_results = {}
        self.latest_vehicle_count: VehicleCountMessage | Dict = {}
        self.fps_calculator = deque(maxlen=60)
        
        logger.info(f"[Visualizer] Visualizer initialized with font: {self.font}, font scale: {self.font_scale}, font thickness: {self.font_thickness}")
        logger.debug(f"[Visualizer] Parsed colors: {self.colors}")
        logger.info(f"[Visualizer] Counting lines will be received from VehicleCounter process")
    
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
    

    
    def _draw_vehicle_info(self, image: np.ndarray, vehicle: TrackedObject):
        x1, y1, x2, y2 = vehicle["bbox_xyxy"]
        class_name = vehicle["class_name"]
        track_id = vehicle["track_id"]

        color = self.colors.get(class_name, self.default_color)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.font_thickness)

        label = f"{class_name} {track_id}"
        
        if track_id in self.latest_ocr_results:
            ocr_data = self.latest_ocr_results[track_id]
            if time.time() - ocr_data["timestamp"] < self.ocr_duration:
                label += f" {ocr_data['text']}"
        
        (text_width, text_height), baseline = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
        cv2.rectangle(image, (x1, y1 - text_height - baseline), (x1 + text_width, y1 - baseline), color, cv2.FILLED)
        cv2.putText(image, label, (x1, y1 - baseline), self.font, self.font_scale, (0, 0, 0), self.font_thickness)
        
    def _draw_stats(self, image: np.ndarray):
        # Calculate FPS
        if len(self.fps_calculator) > 1:
            fps = len(self.fps_calculator) / (self.fps_calculator[-1] - self.fps_calculator[0])
            fps_text = f"FPS: {fps:.1f}"
        else:
            fps_text = "FPS: N/A"
        cv2.putText(image, fps_text, (10, 30), self.font, self.font_scale, (255, 255, 255), self.font_thickness)
         
        # Draw counting lines (received from VehicleCounter)
        if self.counting_lines_absolute and len(self.counting_lines_absolute) > 0:
            # counting_lines_absolute is now a list of lines: [[[x1,y1],[x2,y2]], ...]
            for line_coords in self.counting_lines_absolute:
                if len(line_coords) >= 2:
                    pt1 = tuple(line_coords[0])  # [x1, y1] -> (x1, y1)
                    pt2 = tuple(line_coords[1])  # [x2, y2] -> (x2, y2)
                    color = tuple(self.line_color) if self.line_color else (0, 0, 255)
                    thickness = self.line_thickness if self.line_thickness else 2
                    cv2.line(image, pt1, pt2, color, thickness)

        # Draw vehicle counts
        if self.latest_vehicle_count:
            total = self.latest_vehicle_count.get("total_count", 0)
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
    
    def process_frame(self, frame_msg: VehicleTrackingMessage) -> np.ndarray:
        jpeg_bytes = frame_msg["frame_data_jpeg"]
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        self.fps_calculator.append(time.time())

        for vehicle in frame_msg["tracked_objects"]:
            self._draw_vehicle_info(frame, vehicle)
        
        self._draw_stats(frame)
        return frame
    
    def update_counting_lines(self, count_message: VehicleCountMessage):
        """Update counting line information from VehicleCountMessage."""
        if count_message.get("counting_lines_absolute"):
            self.counting_lines_absolute = count_message["counting_lines_absolute"]
            logger.debug(f"[Visualizer] Updated counting lines from VehicleCounter: {self.counting_lines_absolute}")
        
        if count_message.get("line_display_color"):
            self.line_color = count_message["line_display_color"]
            logger.debug(f"[Visualizer] Updated line color: {self.line_color}")
            
        if count_message.get("line_thickness"):
            self.line_thickness = count_message["line_thickness"]
            logger.debug(f"[Visualizer] Updated line thickness: {self.line_thickness}")

def visualize_process(config: dict, tracking_queue: Queue, OCR_queue: Queue, vehicle_count_queue: Queue, shutdown_event: Event):
    # Setup logging for this process
    try:
        setup_logging()
        logger.info(f"[Visualizer] Logger initialized for visualizer process.")
    except Exception as e:
        print(f"[Visualizer] Failed to setup logging: {e}")
    
    process_name = mp.current_process().name
    logger.info(f"[Visualizer] Visualizer process {process_name} started.")
    
    try:
        # Test OpenCV GUI capabilities
        logger.info(f"[Visualizer] Testing OpenCV GUI capabilities...")
        
        # Check if we can create a window
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        logger.info(f"[Visualizer] Created test image.")
        
        try:
            cv2.imshow("Traffic Monitor", test_img)
            logger.info(f"[Visualizer] Successfully created OpenCV window.")
            cv2.waitKey(1)  # Process window events
        except Exception as window_error:
            logger.error(f"[Visualizer] Failed to create OpenCV window: {window_error}")
            logger.error(f"[Visualizer] This might indicate a display/GUI environment issue.")
            return
        
        logger.info(f"[Visualizer] Initializing visualizer...")
        visualizer = Visualizer(config)
        logger.info(f"[Visualizer] Visualizer initialized successfully.")

        frame_count = 0
        while not shutdown_event.is_set():
            try:
                tracking_msg: VehicleTrackingMessage = tracking_queue.get(timeout=1)
                logger.debug(f"[Visualizer] Received tracking message for frame: {tracking_msg.get('frame_id', 'unknown')}")
            except Empty:
                logger.debug(f"[Visualizer] No tracking message received, continuing...")
                if shutdown_event.is_set():
                    break
                continue
            
            if tracking_msg is None:
                logger.debug(f"[Visualizer] Received None tracking message, continuing...")
                continue
            
            try:
                # Process OCR messages
                ocr_messages_processed = 0
                while not OCR_queue.empty():
                    try:
                        ocr_msg: OCRResultMessage = OCR_queue.get_nowait()
                        if ocr_msg:
                            track_id_from_ocr = ocr_msg["vehicle_id"]
                            visualizer.latest_ocr_results[track_id_from_ocr] = {
                                "text": ocr_msg["lp_text"],
                                "timestamp": time.time()
                            }
                            ocr_messages_processed += 1
                    except Empty:
                        break
                
                if ocr_messages_processed > 0:
                    logger.debug(f"[Visualizer] Processed {ocr_messages_processed} OCR messages.")
                
                # Process vehicle count messages
                count_messages_processed = 0
                while not vehicle_count_queue.empty():
                    try:
                        count_msg: VehicleCountMessage = vehicle_count_queue.get_nowait()
                        if count_msg:
                            visualizer.latest_vehicle_count = count_msg
                            # Update counting line information from VehicleCounter
                            visualizer.update_counting_lines(count_msg)
                            count_messages_processed += 1
                    except Empty:
                        break
                
                if count_messages_processed > 0:
                    logger.debug(f"[Visualizer] Processed {count_messages_processed} count messages.")

                # Process and display the frame
                logger.debug(f"[Visualizer] Processing frame for display...")
                display_frame = visualizer.process_frame(tracking_msg)
                
                logger.debug(f"[Visualizer] Displaying frame...")
                cv2.imshow("Traffic Monitor", display_frame)
                
                frame_count += 1
                if frame_count % 30 == 0:  # Log every 30 frames
                    logger.info(f"[Visualizer] Displayed {frame_count} frames so far.")

                # Check for quit signal
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info(f"[Visualizer] Visualizer process {process_name} received quit signal (q key).")
                    shutdown_event.set()
                    break
                elif key != 255:  # Any other key pressed
                    logger.debug(f"[Visualizer] Key pressed: {key}")
                    
            except Exception as frame_error:
                logger.error(f"[Visualizer] Error processing frame: {frame_error}")
                continue
    
    except Exception as e:
        logger.error(f"[Visualizer] Visualizer process {process_name} encountered a critical error: {e}")
        logger.exception(f"[Visualizer] Full exception traceback:")
    
    finally:
        logger.info(f"[Visualizer] Cleaning up visualizer process...")
        try:
            cv2.destroyAllWindows()
            logger.info(f"[Visualizer] OpenCV windows destroyed.")
        except Exception as cleanup_error:
            logger.error(f"[Visualizer] Error during cleanup: {cleanup_error}")
        
        logger.info(f"[Visualizer] Visualizer process {process_name} shutting down.")

