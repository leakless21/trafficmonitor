import time
import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from loguru import logger
from queue import Empty
from multiprocessing.queues import Queue
from multiprocessing.synchronize import Event

from ..utils.logging_config import setup_logging
from ..utils.custom_types import VehicleCountMessage, OCRResultMessage


@dataclass
class ProcessingMetrics:
    """Container for video processing metrics."""
    # Video information
    video_source: str = ""
    video_duration_seconds: float = 0.0
    total_frames: int = 0
    processed_frames: int = 0
    
    # Timing metrics
    processing_start_time: float = 0.0
    processing_end_time: float = 0.0
    processing_duration_seconds: float = 0.0
    
    # Performance metrics
    average_fps: float = 0.0
    frames_per_second_actual: float = 0.0
    
    # Detection and tracking metrics
    total_detections: int = 0
    total_tracked_vehicles: int = 0
    unique_vehicle_ids: set = None
    
    # Vehicle counting metrics
    final_vehicle_counts: Dict[str, int] = None
    total_vehicles_counted: int = 0
    
    # License plate metrics
    plates_detected: int = 0
    plates_recognized: int = 0
    ocr_success_rate: float = 0.0
    
    # Quality metrics
    frame_drops: int = 0
    processing_errors: int = 0
    
    # Configuration and model information
    configuration_used: Dict[str, Any] = None
    models_used: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.unique_vehicle_ids is None:
            self.unique_vehicle_ids = set()
        if self.final_vehicle_counts is None:
            self.final_vehicle_counts = {}
        if self.configuration_used is None:
            self.configuration_used = {}
        if self.models_used is None:
            self.models_used = {}


class SummaryService:
    """Service for collecting and reporting video processing metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = ProcessingMetrics()
        self.metrics.video_source = config.get("video_source", "Unknown")
        self.metrics.processing_start_time = time.time()
        
        # Store configuration and model information
        self._extract_configuration_info(config)
        self._extract_model_info(config)
        
        # Counters for tracking
        self.frame_count = 0
        self.detection_count = 0
        self.tracking_count = 0
        self.error_count = 0
        
        # Storage for aggregated data
        self.class_detection_counts = defaultdict(int)
        self.vehicle_ids_seen = set()
        self.ocr_results = {}
        
        # Output configuration
        self.output_dir = Path(config.get("summary_output_dir", "data/reports"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[SummaryService] Initialized for video: {self.metrics.video_source}")
    
    def _extract_configuration_info(self, config: Dict[str, Any]):
        """Extract relevant configuration information for the report."""
        self.metrics.configuration_used = {
            "video_processing": {
                "video_source": config.get("video_source", "Unknown"),
                "resize_resolution": config.get("resize_resolution", "Not specified"),
                "process_every_n_frame": config.get("process_every_n_frame", 1),
                "offline_mode": config.get("offline_mode", False)
            },
            "detection": {
                "confidence_threshold": config.get("conf_threshold", "Not specified"),
                "device": config.get("device", "Not specified"),
                "class_mapping": config.get("class_mapping", {})
            },
            "tracking": {
                "tracker_type": config.get("tracker_type", "Not specified"),
                "use_half_precision": config.get("half", False),
                "device": config.get("device", "Not specified")
            },
            "license_plate": {
                "lp_confidence_threshold": config.get("conf_threshold", "Not specified"),
                "ocr_backend": config.get("backend", "Not specified"),
                "ocr_language": config.get("lang", "Not specified"),
                "ocr_use_gpu": config.get("use_gpu", False),
                "hub_model_name": config.get("hub_model_name", "Not specified")
            },
            "counting": {
                "counting_lines": config.get("counting_lines", []),
                "number_of_counting_lines": len(config.get("counting_lines", []))
            },
            "visualization": {
                "save_to_file": config.get("save_to_file", False),
                "save_path": config.get("save_path", "Not specified"),
                "output_fourcc": config.get("output_fourcc", "Not specified")
            }
        }
    
    def _extract_model_info(self, config: Dict[str, Any]):
        """Extract model information for the report."""
        from pathlib import Path
        import os
        
        self.metrics.models_used = {
            "vehicle_detection": {
                "model_path": config.get("model_path", "Not specified"),
                "model_exists": False,
                "model_size_mb": 0,
                "model_type": "YOLO"
            },
            "license_plate_detection": {
                "model_path": config.get("lp_model_path", "Not specified"),
                "model_exists": False,
                "model_size_mb": 0,
                "model_type": "YOLO"
            },
            "tracking": {
                "tracker_type": config.get("tracker_type", "Not specified"),
                "reid_model_path": config.get("reid_model_path", None),
                "reid_model_exists": False,
                "reid_model_size_mb": 0
            },
            "ocr": {
                "backend": config.get("backend", "Not specified"),
                "hub_model_name": config.get("hub_model_name", "Not specified"),
                "language": config.get("lang", "Not specified")
            }
        }
        
        # Check if model files exist and get their sizes
        vehicle_model_path = config.get("model_path")
        if vehicle_model_path and os.path.exists(vehicle_model_path):
            self.metrics.models_used["vehicle_detection"]["model_exists"] = True
            self.metrics.models_used["vehicle_detection"]["model_size_mb"] = round(
                os.path.getsize(vehicle_model_path) / (1024 * 1024), 2
            )
        
        lp_model_path = config.get("lp_model_path")
        if lp_model_path and os.path.exists(lp_model_path):
            self.metrics.models_used["license_plate_detection"]["model_exists"] = True
            self.metrics.models_used["license_plate_detection"]["model_size_mb"] = round(
                os.path.getsize(lp_model_path) / (1024 * 1024), 2
            )
        
        reid_model_path = config.get("reid_model_path")
        if reid_model_path and os.path.exists(reid_model_path):
            self.metrics.models_used["tracking"]["reid_model_exists"] = True
            self.metrics.models_used["tracking"]["reid_model_size_mb"] = round(
                os.path.getsize(reid_model_path) / (1024 * 1024), 2
            )
    
    def record_frame_processed(self, frame_id: str, timestamp: float):
        """Record that a frame has been processed."""
        self.frame_count += 1
        self.metrics.processed_frames = self.frame_count
        
        # Update timing metrics
        if self.frame_count == 1:
            self.first_frame_time = timestamp
        self.last_frame_time = timestamp
    
    def record_detections(self, detections: list, frame_id: str):
        """Record vehicle detections for a frame."""
        self.detection_count += len(detections)
        self.metrics.total_detections = self.detection_count
        
        # Count by class
        for detection in detections:
            class_name = detection.get("class_name", "unknown")
            self.class_detection_counts[class_name] += 1
    
    def record_tracking(self, tracked_objects: list, frame_id: str):
        """Record vehicle tracking results for a frame."""
        self.tracking_count += len(tracked_objects)
        self.metrics.total_tracked_vehicles = self.tracking_count
        
        # Track unique vehicle IDs
        for obj in tracked_objects:
            track_id = obj.get("track_id")
            if track_id:
                self.vehicle_ids_seen.add(track_id)
        
        self.metrics.unique_vehicle_ids = self.vehicle_ids_seen.copy()
    
    def record_vehicle_count(self, count_message: VehicleCountMessage):
        """Record vehicle count updates."""
        self.metrics.final_vehicle_counts = dict(count_message.get("class_counts", {}))
        self.metrics.total_vehicles_counted = count_message.get("total_count", 0)
    
    def record_ocr_result(self, ocr_message: OCRResultMessage):
        """Record OCR results."""
        vehicle_id = ocr_message.get("vehicle_id")
        lp_text = ocr_message.get("lp_text", "")
        confidence = ocr_message.get("ocr_confidence", 0.0)
        
        if vehicle_id:
            self.ocr_results[vehicle_id] = {
                "text": lp_text,
                "confidence": confidence,
                "timestamp": time.time()
            }
            
            # Update metrics
            self.metrics.plates_detected += 1
            if lp_text and lp_text.strip():
                self.metrics.plates_recognized += 1
    
    def record_error(self, error_type: str, error_message: str):
        """Record processing errors."""
        self.error_count += 1
        self.metrics.processing_errors = self.error_count
        logger.warning(f"[SummaryService] Recorded error: {error_type} - {error_message}")
    
    def finalize_metrics(self):
        """Finalize all metrics when processing is complete."""
        self.metrics.processing_end_time = time.time()
        self.metrics.processing_duration_seconds = (
            self.metrics.processing_end_time - self.metrics.processing_start_time
        )
        
        # Calculate performance metrics
        if self.metrics.processing_duration_seconds > 0:
            self.metrics.frames_per_second_actual = (
                self.metrics.processed_frames / self.metrics.processing_duration_seconds
            )
        
        # Calculate OCR success rate
        if self.metrics.plates_detected > 0:
            self.metrics.ocr_success_rate = (
                self.metrics.plates_recognized / self.metrics.plates_detected * 100
            )
        
        logger.info(f"[SummaryService] Metrics finalized - processed {self.metrics.processed_frames} frames in {self.metrics.processing_duration_seconds:.2f}s")
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a comprehensive summary report."""
        self.finalize_metrics()
        
        # Convert metrics to dictionary for JSON serialization
        metrics_dict = asdict(self.metrics)
        
        # Handle set serialization
        metrics_dict["unique_vehicle_ids"] = list(self.metrics.unique_vehicle_ids)
        
        # Add additional analysis
        report = {
            "summary": {
                "video_source": self.metrics.video_source,
                "processing_completed_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.metrics.processing_end_time)),
                "total_processing_time": f"{self.metrics.processing_duration_seconds:.2f} seconds",
                "frames_processed": self.metrics.processed_frames,
                "processing_fps": f"{self.metrics.frames_per_second_actual:.2f} FPS"
            },
            "configuration_used": self.metrics.configuration_used,
            "models_used": self.metrics.models_used,
            "vehicle_analysis": {
                "total_detections": self.metrics.total_detections,
                "unique_vehicles_tracked": len(self.metrics.unique_vehicle_ids),
                "total_vehicles_counted": self.metrics.total_vehicles_counted,
                "vehicle_counts_by_class": self.metrics.final_vehicle_counts,
                "detection_counts_by_class": dict(self.class_detection_counts)
            },
            "license_plate_analysis": {
                "plates_detected": self.metrics.plates_detected,
                "plates_successfully_recognized": self.metrics.plates_recognized,
                "ocr_success_rate": f"{self.metrics.ocr_success_rate:.1f}%",
                "recognized_plates": {
                    vid: data["text"] for vid, data in self.ocr_results.items() 
                    if data["text"] and data["text"].strip()
                }
            },
            "performance_metrics": {
                "processing_duration_seconds": self.metrics.processing_duration_seconds,
                "average_processing_fps": self.metrics.frames_per_second_actual,
                "total_errors": self.metrics.processing_errors,
                "error_rate": f"{(self.metrics.processing_errors / max(1, self.metrics.processed_frames)) * 100:.2f}%"
            },
            "detailed_metrics": metrics_dict
        }
        
        return report
    
    def save_summary_report(self, report: Optional[Dict[str, Any]] = None) -> Path:
        """Save the summary report to a JSON file."""
        if report is None:
            report = self.generate_summary_report()
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_name = Path(self.metrics.video_source).stem if self.metrics.video_source else "unknown"
        filename = f"processing_summary_{video_name}_{timestamp}.json"
        
        report_path = self.output_dir / filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"[SummaryService] Summary report saved to: {report_path}")
        return report_path
    
    def print_summary(self, report: Optional[Dict[str, Any]] = None):
        """Print a formatted summary to the console."""
        if report is None:
            report = self.generate_summary_report()
        
        print("\n" + "="*80)
        print("VIDEO PROCESSING SUMMARY REPORT")
        print("="*80)
        
        summary = report["summary"]
        print(f"Video Source: {summary['video_source']}")
        print(f"Processing Completed: {summary['processing_completed_at']}")
        print(f"Total Processing Time: {summary['total_processing_time']}")
        print(f"Frames Processed: {summary['frames_processed']}")
        print(f"Processing Speed: {summary['processing_fps']}")
        
        print("\n" + "-"*40)
        print("CONFIGURATION & MODELS")
        print("-"*40)
        
        # Display configuration summary
        config = report["configuration_used"]
        models = report["models_used"]
        
        print(f"Video Processing:")
        print(f"  Resize Resolution: {config['video_processing']['resize_resolution']}")
        print(f"  Frame Skip: Every {config['video_processing']['process_every_n_frame']} frame(s)")
        print(f"  Mode: {'Offline' if config['video_processing']['offline_mode'] else 'Real-time'}")
        
        print(f"Detection Model:")
        vehicle_model = models["vehicle_detection"]
        print(f"  Type: {vehicle_model['model_type']}")
        print(f"  Path: {vehicle_model['model_path']}")
        print(f"  Exists: {'✅' if vehicle_model['model_exists'] else '❌'}")
        if vehicle_model['model_exists']:
            print(f"  Size: {vehicle_model['model_size_mb']} MB")
        print(f"  Confidence Threshold: {config['detection']['confidence_threshold']}")
        print(f"  Device: {config['detection']['device']}")
        
        print(f"Tracking:")
        print(f"  Tracker Type: {config['tracking']['tracker_type']}")
        print(f"  Half Precision: {config['tracking']['use_half_precision']}")
        
        print(f"License Plate:")
        lp_model = models["license_plate_detection"]
        print(f"  Detection Model: {lp_model['model_path']}")
        print(f"  Model Exists: {'✅' if lp_model['model_exists'] else '❌'}")
        if lp_model['model_exists']:
            print(f"  Model Size: {lp_model['model_size_mb']} MB")
        print(f"  OCR Backend: {config['license_plate']['ocr_backend']}")
        print(f"  OCR Model: {config['license_plate']['hub_model_name']}")
        print(f"  Language: {config['license_plate']['ocr_language']}")
        
        print(f"Counting:")
        print(f"  Number of Counting Lines: {config['counting']['number_of_counting_lines']}")
        
        print("\n" + "-"*40)
        print("VEHICLE ANALYSIS")
        print("-"*40)
        
        vehicle_analysis = report["vehicle_analysis"]
        print(f"Total Detections: {vehicle_analysis['total_detections']}")
        print(f"Unique Vehicles Tracked: {vehicle_analysis['unique_vehicles_tracked']}")
        print(f"Total Vehicles Counted: {vehicle_analysis['total_vehicles_counted']}")
        
        if vehicle_analysis['vehicle_counts_by_class']:
            print("\nVehicle Counts by Class:")
            for class_name, count in vehicle_analysis['vehicle_counts_by_class'].items():
                print(f"  {class_name}: {count}")
        
        print("\n" + "-"*40)
        print("LICENSE PLATE ANALYSIS")
        print("-"*40)
        
        plate_analysis = report["license_plate_analysis"]
        print(f"Plates Detected: {plate_analysis['plates_detected']}")
        print(f"Plates Successfully Recognized: {plate_analysis['plates_successfully_recognized']}")
        print(f"OCR Success Rate: {plate_analysis['ocr_success_rate']}")
        
        if plate_analysis['recognized_plates']:
            print("\nRecognized License Plates:")
            for vehicle_id, plate_text in plate_analysis['recognized_plates'].items():
                print(f"  Vehicle {vehicle_id}: {plate_text}")
        
        print("\n" + "-"*40)
        print("PERFORMANCE METRICS")
        print("-"*40)
        
        performance = report["performance_metrics"]
        print(f"Processing Duration: {performance['processing_duration_seconds']:.2f} seconds")
        print(f"Average Processing FPS: {performance['average_processing_fps']:.2f}")
        print(f"Total Errors: {performance['total_errors']}")
        print(f"Error Rate: {performance['error_rate']}")
        
        print("\n" + "="*80)


def summary_service_process(
    config: Dict[str, Any],
    vehicle_tracking_queue: Queue,
    vehicle_count_queue: Queue,
    ocr_queue: Queue,
    shutdown_event: Event
):
    """Main process function for the summary service."""
    setup_logging(config.get("loguru"))
    process_name = mp.current_process().name
    logger.info(f"[SummaryService] Process {process_name} started")
    
    try:
        summary_service = SummaryService(config)
        
        # Main processing loop
        while not shutdown_event.is_set():
            messages_processed = 0
            
            # Process vehicle tracking messages
            try:
                while True:
                    try:
                        tracking_msg = vehicle_tracking_queue.get_nowait()
                        if tracking_msg is None:
                            logger.info("[SummaryService] Received shutdown signal from tracking queue")
                            break
                        
                        # Record frame processing
                        summary_service.record_frame_processed(
                            tracking_msg.get("frame_id", ""),
                            tracking_msg.get("timestamp", time.time())
                        )
                        
                        # Record tracking data
                        tracked_objects = tracking_msg.get("tracked_objects", [])
                        summary_service.record_tracking(tracked_objects, tracking_msg.get("frame_id", ""))
                        
                        messages_processed += 1
                        
                    except Empty:
                        break
            except Exception as e:
                summary_service.record_error("tracking_processing", str(e))
            
            # Process vehicle count messages
            try:
                while True:
                    try:
                        count_msg = vehicle_count_queue.get_nowait()
                        if count_msg is None:
                            logger.info("[SummaryService] Received shutdown signal from count queue")
                            break
                        
                        summary_service.record_vehicle_count(count_msg)
                        messages_processed += 1
                        
                    except Empty:
                        break
            except Exception as e:
                summary_service.record_error("count_processing", str(e))
            
            # Process OCR messages
            try:
                while True:
                    try:
                        ocr_msg = ocr_queue.get_nowait()
                        if ocr_msg is None:
                            logger.info("[SummaryService] Received shutdown signal from OCR queue")
                            break
                        
                        summary_service.record_ocr_result(ocr_msg)
                        messages_processed += 1
                        
                    except Empty:
                        break
            except Exception as e:
                summary_service.record_error("ocr_processing", str(e))
            
            # If no messages were processed, sleep briefly to prevent busy waiting
            if messages_processed == 0:
                time.sleep(0.1)
        
        # Generate and save final summary report
        logger.info("[SummaryService] Processing complete, generating final report...")
        report = summary_service.generate_summary_report()
        report_path = summary_service.save_summary_report(report)
        
        # Print summary to console
        summary_service.print_summary(report)
        
        logger.info(f"[SummaryService] Final report saved to: {report_path}")
        
    except Exception as e:
        logger.exception(f"[SummaryService] Process {process_name} crashed: {e}")
        raise
    finally:
        logger.info(f"[SummaryService] Process {process_name} shutting down") 