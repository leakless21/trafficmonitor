import multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
from typing import Any, Dict
from loguru import logger

import cv2
import numpy as np
import time
from collections import deque

from ..utils.custom_types import TrackedVehicleMessage, VehicleCountMessage, OCRResultMessage

class Visualizer:
    def __init__(self, config: dict):
        self.font = config.get("font", cv2.FONT_HERSHEY_SIMPLEX)
        self.font_scale = config.get("font_scale", 0.6)
        self.font_thickness = config.get("font_thickness", 2)
        self.ocr_duration = config.get("ocr_duration", 3.0)

        self.colors = config.get("class_colors", {})
        self.default_color = config.get("default_color", (255, 255, 255))

        self.latest_ocr_results = {}
        self.latest_vehicle_counts = {}
        self.fps_calculator = deque(maxlen=60)
        logger.info(f"[Visualizer] Visualizer initialized with font: {self.font}, font scale: {self.font_scale}, font thickness: {self.font_thickness}")
    
    def _draw_vehicle_info(self, image: np.ndarray, vehicle: Dict[str, Any]):
        x1, y1, x2, y2 = vehicle["bbox_xyxy"]
        class_name = vehicle["class_name"]
        track_id = vehicle["track_id"]

        color = self.colors.get(class_name, self.default_color)

        cv2.rectangle(image, x1, y1, x2, y2, color, self.font_thickness)

        label = f"{class_name} {track_id}"
        
        if track_id in self.latest_ocr_results:
            ocr_data = self.latest_ocr_results[track_id]
            if time.time() - ocr_data["timestamp"] < self.ocr_duration:
                label += f" {ocr_data['lp_text']}"
        
        (text_width, text_height), baseline = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
        cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 + text_height), color, cv2.FILLED)
        cv2.putText(image, label, (x1, y1 - baseline), self.font, self.font_scale, (0, 0, 0), self.font_thickness)
       
    def _draw_stats(self, image: np.ndarray):
        # Calculate FPS
        if len(self.fps_calculator) > 1:
            fps = len(self.fps_calculator) / (self.fps_calculator[-1] - self.fps_calculator[0])
            fps_text = f"FPS: {fps:.1f}"
        else:
            fps_text = "FPS: N/A"
        
        cv2.putText(image, fps_text, (10, 30), self.font, self.font_scale, (0, 0, 0), self.font_thickness)

        # Draw vehicle counts
        if self.latest_vehicle_counts:
            total = self.latest_vehicle_counts.get("total_count", 0)
            by_class = self.latest_vehicle_counts.get("class_counts", {})

            # Draw total count
            count_text = f"Total: {total}"
            cv2.putText(image, count_text, (10, 70), self.font, self.font_scale, (0, 0, 0), self.font_thickness)

            # Draw class counts
            index = 0
            for class_name, count in by_class.items():
                class_text = f"{class_name}: {count}"
                cv2.putText(image, class_text, (10, 100 + (index * 20)), self.font, self.font_scale, (0, 0, 0), self.font_thickness)
                index += 1
    
    def process_frame(self, frame_msg: TrackedVehicleMessage) -> np.ndarray:
        jpeg_bytes = frame_msg["frame_data"]
        frame = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
        self.fps_calculator.append(time.time())

        for vehicle in frame_msg["tracked_objects"]:
            self._draw_vehicle_info(frame, vehicle)
        
        self._draw_stats(frame)
        return frame
    
def visualize_process(config: dict, tracking_queue: Queue, OCR_queue: Queue, vehicle_count_queue: Queue, shutdown_event: Event):
    process_name = mp.current_process().name
    logger.info(f"[Visualizer] Visualizer process {process_name} started.")

    try:
        visualizer = Visualizer(config)

        while not shutdown_event.is_set():
            try:
                ocr_msg: OCRResultMessage = OCR_queue.get_nowait()
                if ocr_msg:
                    visualizer.latest_ocr_results[ocr_msg["vehicle_id"]] = {
                        "text": ocr_msg["lp_text"],
                        "timestamp": time.time()
                    }
            except Empty:
                pass