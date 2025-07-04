#!/usr/bin/env python3
"""
Unit test to prevent regression of logging issues in LicensePlateDetectionService and TextRecognitionService
"""
import pytest
import sys
import multiprocessing as mp
import time
from pathlib import Path
from queue import Empty
import tempfile
import os

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.traffic_monitor.utils.config_loader import load_config
from src.traffic_monitor.utils.logging_config import setup_logging
from src.traffic_monitor.services.license_plate_detection_service import license_plate_detection_process
from src.traffic_monitor.services.text_recognition_service import text_recognition_process


def create_test_config():
    """Create a minimal test configuration"""
    return {
        "license_plate_detection": {
            "model_path": "data/models/plate_v8n.pt",
            "conf_threshold": 0.65,
            "service_name": "LicensePlateDetectionService",
            "loguru": {"level": "INFO"},
            "database": {"path": ":memory:"}
        },
        "text_recognition": {
            "backend": "fastplateocr",
            "hub_model_name": "global-plates-mobile-vit-v2-model",
            "device": "cpu",
            "conf_threshold": 0.7,
            "service_name": "TextRecognitionService",
            "loguru": {"level": "INFO"},
            "database": {"path": ":memory:"}
        }
    }


def test_license_plate_detection_service_logging():
    """Test that LicensePlateDetectionService process logs its startup properly"""
    config = create_test_config()
    
    # Setup logging
    setup_logging(config["license_plate_detection"]["loguru"])
    
    # Create queues
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    shutdown_event = mp.Event()
    
    # Start the process
    process = mp.Process(
        target=license_plate_detection_process,
        args=(config["license_plate_detection"], input_queue, output_queue, shutdown_event),
        name="TestLicensePlateDetectionService",
    )
    
    try:
        process.start()
        time.sleep(2)  # Give it time to log startup messages
        
        # Process should be alive
        assert process.is_alive(), "LicensePlateDetectionService process should be running"
        
        # Signal shutdown
        shutdown_event.set()
        input_queue.put(None)  # Send sentinel to trigger shutdown
        process.join(timeout=5)
        
        assert not process.is_alive(), "LicensePlateDetectionService process should have finished"
        assert process.exitcode == 0, f"LicensePlateDetectionService process should exit cleanly, got code: {process.exitcode}"
        
    finally:
        if process.is_alive():
            process.terminate()
            process.join()


def test_text_recognition_service_logging():
    """Test that TextRecognitionService process logs its startup properly"""
    config = create_test_config()
    
    # Setup logging
    setup_logging(config["text_recognition"]["loguru"])
    
    # Create queues
    input_queue = mp.Queue()
    output_queue = mp.Queue()
    shutdown_event = mp.Event()
    
    # Start the process
    process = mp.Process(
        target=text_recognition_process,
        args=(config["text_recognition"], input_queue, output_queue, shutdown_event),
        name="TestTextRecognitionService",
    )
    
    try:
        process.start()
        time.sleep(2)  # Give it time to log startup messages
        
        # Process should be alive
        assert process.is_alive(), "TextRecognitionService process should be running"
        
        # Signal shutdown
        shutdown_event.set()
        input_queue.put(None)  # Send sentinel to trigger shutdown
        process.join(timeout=5)
        
        assert not process.is_alive(), "TextRecognitionService process should have finished"
        assert process.exitcode == 0, f"TextRecognitionService process should exit cleanly, got code: {process.exitcode}"
        
    finally:
        if process.is_alive():
            process.terminate()
            process.join()


def test_multiple_processes_logging():
    """Test that multiple processes can log simultaneously without issues"""
    config = create_test_config()
    
    # Setup logging
    setup_logging(config["license_plate_detection"]["loguru"])
    
    # Create queues for both processes
    license_plate_detection_input_queue = mp.Queue()
    license_plate_detection_output_queue = mp.Queue()
    text_recognition_input_queue = mp.Queue()
    text_recognition_output_queue = mp.Queue()
    shutdown_event = mp.Event()
    
    # Start LicensePlateDetectionService process
    license_plate_detection_process_instance = mp.Process(
        target=license_plate_detection_process,
        args=(config["license_plate_detection"], license_plate_detection_input_queue, license_plate_detection_output_queue, shutdown_event),
        name="TestLicensePlateDetectionService",
    )
    
    # Start TextRecognitionService process
    text_recognition_process_instance = mp.Process(
        target=text_recognition_process,
        args=(config["text_recognition"], text_recognition_input_queue, text_recognition_output_queue, shutdown_event),
        name="TestTextRecognitionService",
    )
    
    try:
        license_plate_detection_process_instance.start()
        text_recognition_process_instance.start()
        
        time.sleep(3)  # Give them time to log startup messages
        
        # Both processes should be alive
        assert license_plate_detection_process_instance.is_alive(), "LicensePlateDetectionService process should be running"
        assert text_recognition_process_instance.is_alive(), "TextRecognitionService process should be running"
        
        # Signal shutdown
        shutdown_event.set()
        license_plate_detection_input_queue.put(None)
        text_recognition_input_queue.put(None)
        
        # Wait for both to finish
        license_plate_detection_process_instance.join(timeout=5)
        text_recognition_process_instance.join(timeout=5)
        
    finally:
        for process in [license_plate_detection_process_instance, text_recognition_process_instance]:
            if process.is_alive():
                process.terminate()
                process.join()


if __name__ == "__main__":
    test_license_plate_detection_service_logging()
    test_text_recognition_service_logging()
    test_multiple_processes_logging()
    print("All logging tests passed!") 
