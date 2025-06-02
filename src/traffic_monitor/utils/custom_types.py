from typing import List, TypedDict, Optional # Optional is useful for fields that might not always be there
import numpy as np # For type hinting numpy arrays if needed

# Example message structure for frames from FrameGrabber
# (Matches your plan for FrameGrabber output)
class FrameMessage(TypedDict):
    frame_id: str
    camera_id: str
    timestamp: float
    frame_data_jpeg: str # Base64 encoded JPEG string
    frame_height: int
    frame_width: int

# Example for Vehicle Detections (Matches your plan for VehicleDetector output)
class Detection(TypedDict):
    bbox_xyxy: List[int] # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str

class VehicleDetectionMessage(FrameMessage): # Inherits fields from FrameMessage
    detections: List[Detection]

# Add more TypedDicts for other message types as you define them:
# - TrackedObject
# - TrackedVehicleMessage
# - PlateDetectionMessage
# - OCRResultMessage
# - VehicleCountMessage