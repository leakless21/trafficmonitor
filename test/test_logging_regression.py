#!/usr/bin/env python3
"""
Unit test to prevent regression of logging issues in LPDetector and OCRReader
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

from traffic_monitor.services.lp_detector import lp_detector_process
from traffic_monitor.services.ocr_reader import ocr_reader_process

class TestLoggingRegression:
    """Test class to ensure logging works correctly in multiprocessing services"""
    
    def test_lp_detector_process_logs_startup(self):
        """Test that LPDetector process logs its startup properly"""
        config = {
            "lp_detector": {
                "model_path": "data/models/plate_v8n.pt",
                "conf_threshold": 0.6
            }
        }
        
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        shutdown_event = mp.Event()
        
        # Use a temporary log file to capture process logs
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as temp_log:
            temp_log_path = temp_log.name
        
        try:
            process = mp.Process(
                target=lp_detector_process,
                name="TestLPDetector",
                args=(config, input_queue, output_queue, shutdown_event)
            )
            
            process.start()
            
            # Wait for initialization
            time.sleep(2)
            
            # Process should be alive
            assert process.is_alive(), "LPDetector process should be running"
            
            # Gracefully shutdown
            shutdown_event.set()
            process.join(timeout=5)
            
            # Process should have finished cleanly
            assert not process.is_alive(), "LPDetector process should have finished"
            assert process.exitcode == 0, f"LPDetector process should exit cleanly, got code: {process.exitcode}"
            
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
            # Clean up temp file
            if os.path.exists(temp_log_path):
                os.unlink(temp_log_path)
    
    def test_ocr_reader_process_logs_startup(self):
        """Test that OCRReader process logs its startup properly"""
        config = {
            "hub_model_name": "global-plates-mobile-vit-v2-model",
            "device": "auto",
            "conf_threshold": 0.5
        }
        
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        shutdown_event = mp.Event()
        
        try:
            process = mp.Process(
                target=ocr_reader_process,
                name="TestOCRReader",
                args=(config, input_queue, output_queue, shutdown_event)
            )
            
            process.start()
            
            # Wait for initialization
            time.sleep(2)
            
            # Process should be alive
            assert process.is_alive(), "OCRReader process should be running"
            
            # Gracefully shutdown
            shutdown_event.set()
            process.join(timeout=5)
            
            # Process should have finished cleanly
            assert not process.is_alive(), "OCRReader process should have finished"
            assert process.exitcode == 0, f"OCRReader process should exit cleanly, got code: {process.exitcode}"
            
        finally:
            if process.is_alive():
                process.terminate()
                process.join()
    
    def test_both_processes_can_start_simultaneously(self):
        """Test that both processes can start and log without interfering with each other"""
        lp_config = {
            "lp_detector": {
                "model_path": "data/models/plate_v8n.pt",
                "conf_threshold": 0.6
            }
        }
        
        ocr_config = {
            "hub_model_name": "global-plates-mobile-vit-v2-model",
            "device": "auto",
            "conf_threshold": 0.5
        }
        
        # Create queues
        lp_input_queue = mp.Queue()
        lp_output_queue = mp.Queue()
        ocr_input_queue = mp.Queue()
        ocr_output_queue = mp.Queue()
        shutdown_event = mp.Event()
        
        processes = []
        
        try:
            # Start LPDetector process
            lp_process = mp.Process(
                target=lp_detector_process,
                name="TestLPDetector",
                args=(lp_config, lp_input_queue, lp_output_queue, shutdown_event)
            )
            
            # Start OCRReader process
            ocr_process = mp.Process(
                target=ocr_reader_process,
                name="TestOCRReader",
                args=(ocr_config, ocr_input_queue, ocr_output_queue, shutdown_event)
            )
            
            lp_process.start()
            ocr_process.start()
            
            processes = [lp_process, ocr_process]
            
            # Wait for both to initialize
            time.sleep(3)
            
            # Both processes should be alive
            assert lp_process.is_alive(), "LPDetector process should be running"
            assert ocr_process.is_alive(), "OCRReader process should be running"
            
            # Gracefully shutdown both
            shutdown_event.set()
            
            for process in processes:
                process.join(timeout=5)
                assert not process.is_alive(), f"Process {process.name} should have finished"
                assert process.exitcode == 0, f"Process {process.name} should exit cleanly, got code: {process.exitcode}"
            
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    pytest.main([__file__, "-v"]) 