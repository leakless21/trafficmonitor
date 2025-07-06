import multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.queues import Queue
from queue import Empty, Full
from typing import Dict, Any, Tuple
from loguru import logger
import time
from shapely.geometry import LineString, Point

from ..utils.custom_types import TrackedVehicleMessage, VehicleCountMessage
from ..utils.utils import relative_to_absolute_coords
from ..utils.logging_config import setup_logging
from ..utils.minidb import configure_database, write_vehicle_count
from ..utils.queue_utils import safe_put, log_queue_stats

class VehicleCountingService:
    def __init__(self, counting_lines_config: list):
        # Handle case where config is a list of lines vs a single line
        if not counting_lines_config:
            logger.error("[Counter] Empty counting lines configuration")
            self.line_config_raw = []
        elif len(counting_lines_config[0]) > 0 and isinstance(counting_lines_config[0][0], list):
            # Config is a list of lines: [[[x1,y1], [x2,y2]], [[x3,y3], [x4,y4]]]
            # For now, use the first line only
            self.line_config_raw = counting_lines_config[0]
            logger.info(f"[Counter] Using first counting line from {len(counting_lines_config)} configured lines.")
        else:
            # Config is a single line: [[x1,y1], [x2,y2]]
            self.line_config_raw = counting_lines_config
            
        self.relative_coords: list[list[float]] | None = None
        self.absolute_coords: LineString | None = None
        self.vehicle_last_positions = {}
        self.counted_track_ids = set()
        self.counts = {}
        logger.info(f"[Counter] Counter initialized with counting line in relative coordinates.")

    def _init_and_normalize_line(self, line_config_raw: list, og_frame_height: int, og_frame_width: int, frame_width: int, frame_height: int):
        if isinstance(line_config_raw[0][0], float):
            self.relative_coords = line_config_raw
            logger.info(f"[Counter] Line config is already in relative coordinates.")
        elif isinstance(line_config_raw[0][0], int):
            self.relative_coords = [
                [line_config_raw[0][0] / og_frame_width, line_config_raw[0][1] / og_frame_height],
                [line_config_raw[1][0] / og_frame_width, line_config_raw[1][1] / og_frame_height]
            ]
            logger.info(f"[Counter] Line config is in absolute coordinates. Converting to relative coordinates.")
        else:
            logger.error(f"[Counter] Line config is in an unknown format.")
            self.relative_coords = []
            return None
        
        if self.relative_coords:
            pt1_abs = (self.relative_coords[0][0] * frame_width, self.relative_coords[0][1] * frame_height)
            pt2_abs = (self.relative_coords[1][0] * frame_width, self.relative_coords[1][1] * frame_height)
            self.absolute_coords = LineString([pt1_abs, pt2_abs])
            logger.info(f"[Counter] Line config is in relative coordinates. Converting to absolute coordinates.")
        return self.absolute_coords
    
    def _get_bbox_center(self, bbox: list) -> Point:
        x1, y1, x2, y2 = bbox
        return Point((x1 + x2) / 2, y2)
    
    def update(self, tracked_objects: list, frame_width: int, frame_height: int, og_width: int, og_height: int) -> VehicleCountMessage | None:
        count_changed = False
        current_frame_track_ids = {obj["track_id"] for obj in tracked_objects}
        if self.relative_coords is None:
            self._init_and_normalize_line(self.line_config_raw, og_height, og_width, frame_width, frame_height)
        if not self.absolute_coords:
            logger.error("[Counter] Failed to initialize line config")
            return None

        for obj in tracked_objects:
            track_id = obj["track_id"]
            current_position = self._get_bbox_center(obj["bbox_xyxy"])

            if track_id in self.vehicle_last_positions:
                last_position = self.vehicle_last_positions[track_id]
                movement_line = LineString([last_position, current_position])
                # Check intersection with counting line (absolute_coords is a single LineString)
                if self.absolute_coords.intersects(movement_line) and track_id not in self.counted_track_ids:
                    class_name = obj["class_name"]
                    logger.info(f"[Counter] {class_name.capitalize()} (ID: {track_id}) crossed counting line")
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
                class_counts={k: v for k, v in self.counts.items() if k != "total"}
            )
        return None
    
def vehicle_counting_process(config: dict, input_queue: Queue, output_queue: Queue, shutdown_event: Event):
    setup_logging(config.get("loguru"))  # Initialize logging for this process
    # Configure database for this subprocess
    configure_database(config)
    process_name = mp.current_process().name
    offline_mode = config.get("offline_mode", False)
    service_name = config.get("service_name", "VehicleCountingService")
    logger.info(f"[VehicleCountingService] Process {process_name} started")
    try:
        counting_line_coords_list = config.get("counting_lines", [])
        if not counting_line_coords_list:
            logger.error("[VehicleCounter] No counting lines configured")
            return
        
        counter = VehicleCountingService(counting_line_coords_list)
        while not shutdown_event.is_set():
            try:
                message: TrackedVehicleMessage = input_queue.get(timeout=1)
                logger.debug(f"[VehicleCounter] Received message: {message.get('frame_id')}")
            except Empty:
                continue
            
            if message is None:
                logger.warning("[VehicleCounter] Received None message, shutting down")
                break

            current_width = message["frame_width"]
            current_height = message["frame_height"]
            og_width = message["og_frame_width"]
            og_height = message["og_frame_height"]
            
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
            count_update_message = counter.update(tracked_objects, current_width, current_height, og_width, og_height)
            if count_update_message:
                total_count = count_update_message["total_count"]
                class_counts = count_update_message["class_counts"]
                logger.info(f"[VehicleCounter] Total count: {total_count}, Count by class: {class_counts}")
                
                # Use mode-aware queue operation
                success = safe_put(output_queue, count_update_message, offline_mode, service_name)
                if success:
                    # Persist to database
                    try:
                        write_vehicle_count(
                            camera_id=count_update_message["camera_id"],
                            total_count=count_update_message["total_count"],
                            class_counts=count_update_message["class_counts"],
                            ts=int(count_update_message["timestamp"] * 1000)
                        )
                    except Exception as db_exc:
                        logger.exception(f"[VehicleCounter] Failed to write vehicle count to database: {db_exc}")
                else:
                    logger.warning(f"[{service_name}] Failed to put count message")
            else:
                logger.debug("[VehicleCounter] No count update")
    except Exception as e:
        logger.exception(f"[VehicleCounter] Process {process_name} crashed: {e}")
    finally:
        logger.info(f"[VehicleCounter] Process {process_name} shutting down")
    



