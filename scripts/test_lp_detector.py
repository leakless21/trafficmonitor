import ultralytics
import cv2
import numpy as np
from loguru import logger
import os
import sys
import yaml # Added import for yaml

# Configure logging for the test script
logger.remove()
logger.add(f"data/log/test_lp_detector.log", rotation="10 MB", level="DEBUG")
logger.add(sys.stdout, level="DEBUG")

class LPDetector:
    """
    Encapsulates the license plate detection model and its configuration.
    Handles loading the model, setting confidence thresholds, and processing detection results.
    """
    def __init__(self, model_path: str, conf_threshold: float):
        """
        Initializes the LPDetector with the specified model and confidence threshold.
        """
        logger.info(f"[TestLPDetector] Attempting to load model from {model_path}...")
        if not os.path.exists(model_path):
            logger.error(f"[TestLPDetector] Model file not found at: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            self.model = ultralytics.YOLO(model_path)
            self.conf_threshold = conf_threshold
            logger.info(f"[TestLPDetector] Model loaded successfully from {model_path}")
        except Exception as e:
            logger.exception(f"[TestLPDetector] Failed to load model from {model_path}: {e}")
            raise # Re-raise the exception to propagate the error
    
    def find_plates(self, frame: np.ndarray) -> tuple[list[int], float] | None:
        """
        Finds license plates in the given frame.
        """
        logger.info("[TestLPDetector] Running inference...")
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=True) # Set verbose to True for more output
        if not results or not results[0].boxes:
            logger.info("[TestLPDetector] No plates found.")
            return None
        best_plate = results[0].boxes[0]
        bbox = best_plate.xyxy[0].tolist()
        confidence = float(best_plate.conf)
        logger.info(f"[TestLPDetector] Found plate with bbox: {bbox}, confidence: {confidence}")
        return (bbox, confidence)

def test_detector_functionality():
    logger.info("Starting LPDetector standalone test...")
    
    # Load configuration from settings.yaml
    config_path = "src/traffic_monitor/config/settings.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found at: {config_path}")
        return
    except yaml.YAMLError as e:
        logger.error(f"Error reading configuration file: {e}")
        return

    model_path = config.get("lp_detector", {}).get("model_path")
    conf_threshold = config.get("lp_detector", {}).get("conf_threshold")
    video_source = config.get("frame_grabber", {}).get("video_source")

    if not all([model_path, conf_threshold, video_source]):
        logger.error("Missing required configuration parameters in settings.yaml for LPDetector or FrameGrabber.")
        return

    logger.info(f"Using model: {model_path}, confidence: {conf_threshold}, video: {video_source}")

    cap = None
    try:
        lp_detector = LPDetector(model_path, conf_threshold)

        # Load first frame from video source
        if not os.path.exists(video_source):
            logger.error(f"Video source file not found at: {video_source}")
            return
        
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            logger.error(f"Failed to open video source: {video_source}")
            return
        
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.error("Failed to read first frame from video.")
            return

        logger.info(f"Successfully read first frame from {video_source}. Frame shape: {frame.shape}")
        plate_results = lp_detector.find_plates(frame)

        if plate_results:
            logger.info(f"Test successful: Plate detected with bbox {plate_results[0]} and confidence {plate_results[1]}")
        else:
            logger.info("Test completed: No plate detected in the first frame of the video.")

    except FileNotFoundError as fnfe:
        logger.error(f"Test failed: {fnfe}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during the test: {e}")
    finally:
        if cap:
            cap.release()
            logger.info("Video capture released.")

if __name__ == "__main__":
    test_detector_functionality() 