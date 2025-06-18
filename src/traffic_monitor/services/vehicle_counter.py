import multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
from queue import Empty, Full
from typing import Dict, Any, Tuple
from loguru import logger
import time
from shapely.geometry import LineString, Point

from ..utils.custom_types import TrackedVehicleMessage, VehicleCountMessage
from ..utils.logging_config import setup_logging

class Counter:
    def __init__(self, counting_lines_coords: list):
        self.counting_lines = [LineString(line) for line in counting_lines_coords]
        self.vehicle_last_positions = {}
        self.counted_track_ids = set()
        self.counts = {}
        logger.info(f"[Counter] Counter initialized with {len(self.counting_lines)} counting line(s).")

    def _get_bbox_center(self, bbox: list) -> Point:
        x1, y1, x2, y2 = bbox
        return Point((x1 + x2) / 2, y2)
    
    def update(self, tracked_objects: list) -> VehicleCountMessage | None:
        count_changed = False
        current_frame_track_ids = {obj["track_id"] for obj in tracked_objects}

        for obj in tracked_objects:
            track_id = obj["track_id"]
            current_position = self._get_bbox_center(obj["bbox_xyxy"])

            if track_id in self.vehicle_last_positions:
                last_position = self.vehicle_last_positions[track_id]
                movement_line = LineString([last_position, current_position])

                # Check intersection with any of the counting lines
                for i, counting_line in enumerate(self.counting_lines):
                    if counting_line.intersects(movement_line) and track_id not in self.counted_track_ids:
                        class_name = obj["class_name"]
                        logger.info(f"[Counter] {class_name.capitalize()} (ID: {track_id}) crossed counting line {i+1}")
                        self.counted_track_ids.add(track_id)

                        self.counts["total"] = self.counts.get("total", 0) + 1
                        self.counts[class_name] = self.counts.get(class_name, 0) + 1
                        count_changed = True
                        break # Only count once per vehicle per frame, even if it crosses multiple lines
                
            self.vehicle_last_positions[track_id] = current_position

        lost_track_ids = set(self.vehicle_last_positions.keys()) - current_frame_track_ids
        for track_id in lost_track_ids:
            del self.vehicle_last_positions[track_id]
            if track_id in self.counted_track_ids:
                self.counted_track_ids.remove(track_id)
            
        if count_changed:
            return VehicleCountMessage(
                camera_id="camera_id",
                timestamp=time.time(),
                total_count=self.counts["total"],
                count_by_class={k: v for k, v in self.counts.items() if k != "total"}
            )
        return None
    
def vehicle_counter_process(config: dict, input_queue: Queue, output_queue: Queue, shutdown_event: Event):
    setup_logging(config.get("loguru"))  # Initialize logging for this process
    process_name = mp.current_process().name
    logger.info(f"[VehicleCounter] Process {process_name} started")
    try:
        counting_line_coords_list = config.get("counting_lines", [])
        if not counting_line_coords_list:
            logger.error("[VehicleCounter] No counting lines configured")
            return
        
        counter = Counter(counting_line_coords_list)
        while not shutdown_event.is_set():
            try:
                message: TrackedVehicleMessage = input_queue.get(timeout=1)
                logger.debug(f"[VehicleCounter] Received message: {message.get('frame_id')}")
            except Empty:
                continue
            
            if message is None:
                logger.warning("[VehicleCounter] Received None message, shutting down")
                break
            
            tracked_objects = message["tracked_objects"]
            if not tracked_objects:
                logger.debug("[VehicleCounter] No tracked objects in message.")
                continue
            
            # Enhanced logging with class-specific tracking information
            class_counts = {}
            for obj in tracked_objects:
                class_name = obj["class_name"]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            class_summary = ", ".join([f"{count} {class_name}{'s' if count > 1 else ''}" for class_name, count in class_counts.items()])
            logger.debug(f"[VehicleCounter] Processing {len(tracked_objects)} tracked objects: {class_summary}")
            count_update_message = counter.update(tracked_objects)
            if count_update_message:
                total_count = count_update_message["total_count"]
                count_by_class = count_update_message["count_by_class"]
                logger.info(f"[VehicleCounter] Total count: {total_count}, Count by class: {count_by_class}")
                try:
                    output_queue.put(count_update_message, timeout=1)
                except Full:
                    logger.warning("[VehicleCounter] Output queue is full, dropping message")
            else:
                logger.debug("[VehicleCounter] No count update")
    except Exception as e:
        logger.exception(f"[VehicleCounter] Process {process_name} crashed: {e}")
    finally:
        logger.info(f"[VehicleCounter] Process {process_name} shutting down")
    



