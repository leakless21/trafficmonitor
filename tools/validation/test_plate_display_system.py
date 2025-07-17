#!/usr/bin/env python3
"""
Test Plate Display System - Comprehensive validation with synthetic data.

This tool creates a controlled test to verify that the Event Fusion Service
properly merges and displays plate data by injecting known plate information.
"""

import sys
import time
import json
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List
import uuid

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from traffic_monitor.services.event_fusion_service import EventFusionService
from traffic_monitor.utils.custom_types import (
    TrackedVehicleMessage, PlateDetectionMessage, OCRResultMessage
)


def create_test_tracking_message(frame_id: str, vehicle_id: int, vehicle_class: str = "car") -> TrackedVehicleMessage:
    """Create a test tracking message."""
    return {
        "frame_id": frame_id,
        "camera_id": "test_camera",
        "timestamp": time.time(),
        "frame_data_jpeg": b"fake_jpeg_data_for_testing",
        "frame_height": 720,
        "frame_width": 1280,
        "og_frame_height": 1080,
        "og_frame_width": 1920,
        "og_fps": 30.0,
        "tracked_objects": [
            {
                "bbox_xyxy": [100 + vehicle_id * 50, 100, 200 + vehicle_id * 50, 200],
                "confidence": 0.9,
                "class_id": 3,
                "class_name": vehicle_class,
                "track_id": vehicle_id
            }
        ]
    }


def create_test_plate_detection_message(frame_id: str, vehicle_id: int, plate_bbox: List[int]) -> PlateDetectionMessage:
    """Create a test plate detection message."""
    return {
        "frame_id": frame_id,
        "camera_id": "test_camera",
        "timestamp": time.time(),
        "frame_data_jpeg": b"fake_jpeg_data_for_testing",
        "frame_height": 720,
        "frame_width": 1280,
        "og_frame_height": 1080,
        "og_frame_width": 1920,
        "og_fps": 30.0,
        "vehicle_id": vehicle_id,
        "vehicle_class": "car",
        "plate_bbox_original": plate_bbox,
        "plate_confidence": 0.85
    }


def create_test_ocr_message(frame_id: str, vehicle_id: int, plate_text: str) -> OCRResultMessage:
    """Create a test OCR message."""
    return {
        "frame_id": frame_id,
        "camera_id": "test_camera",
        "timestamp": time.time(),
        "vehicle_id": vehicle_id,
        "lp_text": plate_text,
        "ocr_confidence": 0.92
    }


def test_complete_plate_pipeline():
    """Test the complete plate detection and display pipeline."""
    print("🧪 Testing Complete Plate Display Pipeline")
    print("=" * 60)
    
    # Test configuration
    test_config = {
        "ttl_sec": 1.0,
        "max_buffer_size": 100,
        "offline_mode": True,  # Use offline mode for complete data
        "service_name": "PlateDisplayTestService"
    }
    
    # Create fusion service
    fusion_service = EventFusionService(test_config)
    fusion_service.start_cleanup_thread()
    
    # Test data: 3 vehicles with different plate scenarios
    test_scenarios = [
        {
            "frame_id": "frame_001",
            "vehicle_id": 101,
            "plate_text": "ABC123",
            "plate_bbox": [150, 120, 250, 150],
            "description": "Complete data (tracking + plate + OCR)"
        },
        {
            "frame_id": "frame_002", 
            "vehicle_id": 102,
            "plate_text": "XYZ789",
            "plate_bbox": [200, 130, 300, 160],
            "description": "Out-of-order data (OCR before tracking)"
        },
        {
            "frame_id": "frame_003",
            "vehicle_id": 103,
            "plate_text": None,  # No OCR data
            "plate_bbox": [250, 140, 350, 170],
            "description": "Plate detected but no OCR"
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n🔬 Test {i}: {scenario['description']}")
        
        frame_id = scenario["frame_id"]
        vehicle_id = scenario["vehicle_id"]
        plate_text = scenario["plate_text"]
        plate_bbox = scenario["plate_bbox"]
        
        # Create messages
        tracking_msg = create_test_tracking_message(frame_id, vehicle_id)
        plate_msg = create_test_plate_detection_message(frame_id, vehicle_id, plate_bbox)
        
        if scenario["description"] == "Out-of-order data (OCR before tracking)":
            # Send OCR first, then tracking
            if plate_text:
                ocr_msg = create_test_ocr_message(frame_id, vehicle_id, plate_text)
                result_ocr = fusion_service.process_ocr_message(ocr_msg)
                print(f"   📤 Sent OCR message first: {result_ocr}")
            
            result_tracking = fusion_service.process_tracking_message(tracking_msg)
            print(f"   📤 Sent tracking message: {result_tracking}")
            
            result_plate = fusion_service.process_plate_detection_message(plate_msg)
            print(f"   📤 Sent plate detection message: {result_plate}")
        else:
            # Normal order: tracking, plate detection, OCR
            result_tracking = fusion_service.process_tracking_message(tracking_msg)
            print(f"   📤 Sent tracking message: {result_tracking}")
            
            result_plate = fusion_service.process_plate_detection_message(plate_msg)
            print(f"   📤 Sent plate detection message: {result_plate}")
            
            if plate_text:
                ocr_msg = create_test_ocr_message(frame_id, vehicle_id, plate_text)
                result_ocr = fusion_service.process_ocr_message(ocr_msg)
                print(f"   📤 Sent OCR message: {result_ocr}")
        
        # Wait a bit for processing
        time.sleep(0.1)
        
        # Check current state
        key = (frame_id, vehicle_id)
        if key in fusion_service.state:
            obj = fusion_service.state[key]
            print(f"   📊 State: plate_detected={obj.get('plate_detected')}, "
                  f"plate_text_read={obj.get('plate_text_read')}")
            print(f"   📊 Plate text: '{obj.get('plate_text')}'")
        else:
            print(f"   ⚠️  Object not found in state")
        
        results.append({
            "scenario": scenario["description"],
            "frame_id": frame_id,
            "vehicle_id": vehicle_id,
            "expected_plate": plate_text,
            "state_found": key in fusion_service.state
        })
    
    # Flush all objects
    print(f"\n🔄 Flushing all objects...")
    enriched_messages = fusion_service.flush_ready_objects()
    
    print(f"✅ Generated {len(enriched_messages)} enriched messages")
    
    # Validate enriched messages
    validation_results = []
    
    for msg in enriched_messages:
        frame_id = msg["frame_id"]
        print(f"\n📋 Validating frame {frame_id}:")
        
        for obj in msg["tracked_objects"]:
            vehicle_id = obj["track_id"]
            plate_detected = obj.get("plate_detected", False)
            plate_text_read = obj.get("plate_text_read", False)
            plate_text = obj.get("plate_text")
            
            print(f"   🚗 Vehicle {vehicle_id}:")
            print(f"      - Plate detected: {plate_detected}")
            print(f"      - Plate text read: {plate_text_read}")
            print(f"      - Plate text: '{plate_text}'")
            
            # Find expected result
            expected = next((s for s in test_scenarios if s["vehicle_id"] == vehicle_id), None)
            
            if expected:
                expected_text = expected["plate_text"]
                
                validation_result = {
                    "vehicle_id": vehicle_id,
                    "expected_plate": expected_text,
                    "actual_plate": plate_text,
                    "plate_detected": plate_detected,
                    "plate_text_read": plate_text_read,
                    "validation_passed": True,
                    "issues": []
                }
                
                # Validate plate detection
                if expected["plate_bbox"]:  # Should have plate detection
                    if not plate_detected:
                        validation_result["validation_passed"] = False
                        validation_result["issues"].append("Plate should be detected but wasn't")
                else:
                    if plate_detected:
                        validation_result["validation_passed"] = False
                        validation_result["issues"].append("Plate detected but shouldn't be")
                
                # Validate OCR
                if expected_text:  # Should have OCR text
                    if not plate_text_read:
                        validation_result["validation_passed"] = False
                        validation_result["issues"].append("OCR text should be read but wasn't")
                    elif plate_text != expected_text:
                        validation_result["validation_passed"] = False
                        validation_result["issues"].append(f"OCR text mismatch: expected '{expected_text}', got '{plate_text}'")
                else:
                    if plate_text_read and plate_text:
                        validation_result["validation_passed"] = False
                        validation_result["issues"].append("OCR text found but none expected")
                
                validation_results.append(validation_result)
                
                if validation_result["validation_passed"]:
                    print(f"      ✅ Validation PASSED")
                else:
                    print(f"      ❌ Validation FAILED: {', '.join(validation_result['issues'])}")
    
    # Final summary
    print(f"\n" + "=" * 60)
    print("📊 FINAL VALIDATION SUMMARY:")
    
    total_tests = len(validation_results)
    passed_tests = sum(1 for r in validation_results if r["validation_passed"])
    
    print(f"   Total tests: {total_tests}")
    print(f"   Passed: {passed_tests}")
    print(f"   Failed: {total_tests - passed_tests}")
    print(f"   Success rate: {passed_tests / total_tests * 100:.1f}%")
    
    overall_success = passed_tests == total_tests
    
    if overall_success:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   ✅ Event Fusion Service correctly merges plate data")
        print(f"   ✅ Enriched messages contain proper plate information")
        print(f"   ✅ Out-of-order message handling works")
        print(f"   ✅ Visualization will display plates correctly")
    else:
        print(f"\n❌ SOME TESTS FAILED!")
        for result in validation_results:
            if not result["validation_passed"]:
                print(f"   - Vehicle {result['vehicle_id']}: {', '.join(result['issues'])}")
    
    # Cleanup
    fusion_service.shutdown()
    
    # Save detailed results
    test_results = {
        "timestamp": time.time(),
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "success_rate": passed_tests / total_tests * 100,
        "overall_success": overall_success,
        "detailed_results": validation_results,
        "test_scenarios": test_scenarios
    }
    
    output_file = Path("plate_display_test_results.json")
    with open(output_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {output_file}")
    
    return overall_success


def main():
    """Main function."""
    print("🔬 Plate Display System Validation")
    print("This test validates the Event Fusion Service with synthetic data")
    print("to ensure plates are properly merged and displayed.")
    print()
    
    success = test_complete_plate_pipeline()
    
    if success:
        print(f"\n✅ SYSTEM VALIDATION PASSED")
        print(f"The Event Fusion Service correctly handles plate data!")
    else:
        print(f"\n❌ SYSTEM VALIDATION FAILED")
        print(f"Check the detailed results for specific issues.")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()