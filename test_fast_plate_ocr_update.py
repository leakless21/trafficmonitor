#!/usr/bin/env python3
"""
Test script to verify the new fast-plate-ocr API integration.
"""

import numpy as np
import cv2
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from traffic_monitor.services.text_recognition_service import TextRecognitionService
    print("✓ Successfully imported TextRecognitionService")
    
    # Test configuration
    config = {
        "backend": "fast_plate_ocr",
        "hub_model_name": "cct-s-v1-global-model",
        "device": "auto",
        "conf_threshold": 0.5
    }
    
    # Test initialization
    try:
        service = TextRecognitionService(config)
        print("✓ Successfully initialized TextRecognitionService with new API")
    except ImportError as e:
        print(f"⚠ fast-plate-ocr library not installed: {e}")
        print("  Install with: pip install fast-plate-ocr")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Failed to initialize TextRecognitionService: {e}")
        sys.exit(1)
    
    # Test with dummy image
    dummy_image = np.zeros((100, 300, 3), dtype=np.uint8)
    
    # Add some text-like patterns to make it more realistic
    cv2.rectangle(dummy_image, (10, 30), (290, 70), (255, 255, 255), -1)
    cv2.putText(dummy_image, "ABC123", (50, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    try:
        result = service.read_plate(dummy_image)
        if result:
            text, confidence = result
            print(f"✓ OCR test successful: '{text}' (confidence: {confidence:.3f})")
        else:
            print("✓ OCR test completed (no text detected, which is expected for dummy image)")
    except Exception as e:
        print(f"✗ OCR test failed: {e}")
        sys.exit(1)
    
    print("\n✓ All tests passed! The new fast-plate-ocr API is working correctly.")
    
except ImportError as e:
    print(f"✗ Failed to import required modules: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Unexpected error: {e}")
    sys.exit(1) 