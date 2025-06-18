from re import I
from typing import Dict, List, TypedDict, Optional # Optional is useful for fields that might not always be there
import numpy as np # For type hinting numpy arrays if needed

# Example message structure for frames from FrameGrabber
# (Matches your plan for FrameGrabber output)
class FrameMessage(TypedDict):
    frame_id: str
    camera_id: str
    timestamp: float
    frame_data_jpeg: bytes # JPEG binary
    frame_height: int
    frame_width: int
    original_frame_height: int
    original_frame_width: int

# Example for Vehicle Detections (Matches your plan for VehicleDetector output)
class Detection(TypedDict):
    bbox_xyxy: List[int] # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str

class VehicleDetectionMessage(FrameMessage): # Inherits fields from FrameMessage
    detections: List[Detection]

class TrackedObject(TypedDict):
    bbox_xyxy: List[int] # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    track_id: int

class VehicleTrackingMessage(FrameMessage):
    tracked_objects: List[TrackedObject]

class PlateDetectionMessage(FrameMessage):
    #Passthrough
    frame_id: str
    camera_id: str
    timestamp: float
    frame_data_jpeg: bytes
    vehicle_id: int
    vehicle_class: str
    plate_bbox_original: List[int]
    plate_confidence: float

class OCRResultMessage(TypedDict):
    frame_id: str
    camera_id: str
    timestamp: float
    vehicle_id: int
    lp_text: str
    ocr_confidence: float

class VehicleCountMessage(TypedDict):
    camera_id: str
    timestamp: float
    total_count: int
    class_counts: Dict[str, int]
    # Counting line coordinates for visualizer (scaled to current frame resolution)
    counting_lines_absolute: Optional[List[List[List[int]]]]  # [[[x1,y1],[x2,y2]], ...]
    line_display_color: Optional[List[int]]  # BGR color for visualization
    line_thickness: Optional[int]


