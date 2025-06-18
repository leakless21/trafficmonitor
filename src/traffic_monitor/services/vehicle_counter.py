import multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
from queue import Empty, Full
from typing import Dict, Any, Tuple, List, Union, Optional
from loguru import logger
import time
from shapely.geometry import LineString, Point

from ..utils.custom_types import VehicleTrackingMessage, VehicleCountMessage
from ..utils.logging_config import setup_logging

class Counter:
    def __init__(self, counting_lines_coords: List[List[List[Union[int, float]]]], display_color: Optional[List[int]] = None, line_thickness: int = 2):
        self.counting_lines_coords = counting_lines_coords
        self.relative_lines: List[List[List[float]]] | None = None
        self.counting_lines_absolute: List[LineString] | None = None
        self.counting_lines_absolute_coords: List[List[List[int]]] | None = None  # For visualizer
        self.vehicle_last_positions = {}
        self.counted_track_ids = set()
        self.counts = {}
        self.display_color = display_color or [0, 0, 255]  # Default red
        self.line_thickness = line_thickness
        logger.info(f"[Counter] Counter initialized with {len(self.counting_lines_coords)} raw counting line(s).")
    
    def _normalize_counting_lines(self, counting_lines_coords: List[List[List[Union[int, float]]]], original_frame_height: int, original_frame_width: int):
        # Handle empty counting lines
        if not counting_lines_coords:
            logger.warning(f"[Counter] No counting lines provided. Vehicle counting will be disabled.")
            self.relative_lines = []
            self.counting_lines_absolute = []
            return
            
        # Check if the first line's first point has float coordinates
        first_line = counting_lines_coords[0]
        first_point = first_line[0]
        
        logger.info(f"[Counter] Normalizing counting lines. Original resolution: {original_frame_width}x{original_frame_height}")
        logger.info(f"[Counter] Raw counting lines: {counting_lines_coords}")
        
        if isinstance(first_point[0], float) and 0.0 <= first_point[0] <= 1.0:
            # Already in relative coordinates (0.0 to 1.0 range)
            logger.info(f"[Counter] Counting lines detected as relative coordinates")
            self.relative_lines = [[[float(coord) for coord in point] for point in line] for line in counting_lines_coords]
        else:
            # Assume absolute coordinates - normalize them to relative coordinates
            logger.info(f"[Counter] Counting lines detected as absolute coordinates - normalizing to relative")
            self.relative_lines = []
            for line_coords in counting_lines_coords:
                relative_line = [
                    [float(line_coords[0][0]) / original_frame_width, float(line_coords[0][1]) / original_frame_height],
                    [float(line_coords[1][0]) / original_frame_width, float(line_coords[1][1]) / original_frame_height]
                ]
                self.relative_lines.append(relative_line)
                logger.debug(f"[Counter] Normalized line {line_coords} to {relative_line}")
        
        if self.relative_lines:
            self.counting_lines_absolute = []
            for i, relative_line in enumerate(self.relative_lines):
                # Convert relative coordinates back to absolute using current frame dimensions
                pt1_abs = (int(relative_line[0][0] * original_frame_width), int(relative_line[0][1] * original_frame_height))
                pt2_abs = (int(relative_line[1][0] * original_frame_width), int(relative_line[1][1] * original_frame_height))
                line_string = LineString([pt1_abs, pt2_abs])
                self.counting_lines_absolute.append(line_string)
                logger.debug(f"[Counter] Created absolute line {i+1}: {pt1_abs} -> {pt2_abs}")
        
        logger.info(f"[Counter] Successfully normalized {len(self.relative_lines)} counting lines")
        logger.info(f"[Counter] Relative coordinates: {self.relative_lines}")

    
    def _get_bbox_center(self, bbox: list) -> Point:
        x1, y1, x2, y2 = bbox
        return Point((x1 + x2) / 2, y2)
    
    def update(self, tracked_objects: list, original_frame_height: int, original_frame_width: int) -> VehicleCountMessage | None:
        # Check if we need to initialize or re-normalize counting lines
        if self.relative_lines is None:
            self._normalize_counting_lines(self.counting_lines_coords, original_frame_height, original_frame_width)
        
        # Re-create absolute lines if frame dimensions changed or if they don't exist
        if not self.counting_lines_absolute and self.relative_lines:
            self.counting_lines_absolute = []
            for i, relative_line in enumerate(self.relative_lines):
                # Convert relative coordinates to absolute using current frame dimensions
                pt1_abs = (int(relative_line[0][0] * original_frame_width), int(relative_line[0][1] * original_frame_height))
                pt2_abs = (int(relative_line[1][0] * original_frame_width), int(relative_line[1][1] * original_frame_height))
                line_string = LineString([pt1_abs, pt2_abs])
                self.counting_lines_absolute.append(line_string)
                logger.debug(f"[Counter] Updated absolute line {i+1} for {original_frame_width}x{original_frame_height}: {pt1_abs} -> {pt2_abs}")
        
        if not self.counting_lines_absolute:
            logger.error(f"[Counter] Counting lines not normalized. Skipping update.")
            return None

        count_changed = False
        current_frame_track_ids = {obj["track_id"] for obj in tracked_objects}

        for obj in tracked_objects:
            track_id = obj["track_id"]
            current_position = self._get_bbox_center(obj["bbox_xyxy"])

            if track_id in self.vehicle_last_positions:
                last_position = self.vehicle_last_positions[track_id]
                movement_line = LineString([last_position, current_position])

                # Check intersection with any of the counting lines
                for i, counting_line in enumerate(self.counting_lines_absolute):
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
            # Prepare counting line coordinates for visualizer
            counting_lines_coords_for_visualizer = []
            if self.counting_lines_absolute:
                for line_string in self.counting_lines_absolute:
                    coords = list(line_string.coords)
                    # Convert from tuple coordinates to list format [[x1,y1],[x2,y2]]
                    line_coords = [[int(coords[0][0]), int(coords[0][1])], [int(coords[1][0]), int(coords[1][1])]]
                    counting_lines_coords_for_visualizer.append(line_coords)
            
            return VehicleCountMessage(
                camera_id="camera_id",
                timestamp=time.time(),
                total_count=self.counts["total"],
                class_counts={k: v for k, v in self.counts.items() if k != "total"},
                counting_lines_absolute=counting_lines_coords_for_visualizer,
                line_display_color=self.display_color,
                line_thickness=self.line_thickness
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
        
        # Get display properties from config
        display_color = config.get("display_color", [0, 0, 255])
        line_thickness = config.get("line_thickness", 2)
        
        counter = Counter(counting_line_coords_list, display_color, line_thickness)
        while not shutdown_event.is_set():
            try:
                message: VehicleTrackingMessage = input_queue.get(timeout=1)
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
            count_update_message = counter.update(tracked_objects, message["original_frame_height"], message["original_frame_width"])
            if count_update_message:
                total_count = count_update_message["total_count"]
                class_counts = count_update_message["class_counts"]
                logger.info(f"[VehicleCounter] Total count: {total_count}, Count by class: {class_counts}")
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
    



