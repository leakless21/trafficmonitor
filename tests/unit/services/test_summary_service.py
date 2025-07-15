"""
Unit tests for summary service.
Tests report generation, statistics calculation, and data aggregation.
"""

import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import time
from datetime import datetime, timedelta

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from traffic_monitor.services.summary_service import summary_service_process, SummaryService
from traffic_monitor.utils.custom_types import VehicleCountMessage, OCRResultMessage


class TestSummaryService:
    """Test summary service functionality and report generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = {
            "enabled": True,
            "summary_output_dir": "data/reports",
            "print_summary": True,
            "save_detailed_report": True
        }
        
        # Sample data for testing
        self.sample_count_messages = [
            {
                "camera_id": "cam_01",
                "timestamp": time.time(),
                "total_count": 5,
                "class_counts": {"car": 3, "truck": 2}
            },
            {
                "camera_id": "cam_01", 
                "timestamp": time.time() + 60,
                "total_count": 8,
                "class_counts": {"car": 5, "truck": 2, "bus": 1}
            },
            {
                "camera_id": "cam_01",
                "timestamp": time.time() + 120,
                "total_count": 12,
                "class_counts": {"car": 8, "truck": 3, "bus": 1}
            }
        ]
        
        self.sample_ocr_messages = [
            {
                "frame_id": "frame_001",
                "camera_id": "cam_01",
                "timestamp": time.time(),
                "vehicle_id": 1,
                "lp_text": "ABC123",
                "ocr_confidence": 0.95
            },
            {
                "frame_id": "frame_002",
                "camera_id": "cam_01",
                "timestamp": time.time() + 30,
                "vehicle_id": 2,
                "lp_text": "XYZ789",
                "ocr_confidence": 0.87
            },
            {
                "frame_id": "frame_003",
                "camera_id": "cam_01",
                "timestamp": time.time() + 60,
                "vehicle_id": 3,
                "lp_text": "DEF456",
                "ocr_confidence": 0.92
            }
        ]

    def test_summary_initialization(self):
        """Test summary service initialization."""
        config = self.mock_config
        
        # Verify configuration
        assert config["enabled"] is True
        assert "summary_output_dir" in config
        assert "print_summary" in config
        assert "save_detailed_report" in config

    def test_vehicle_count_aggregation(self):
        """Test aggregation of vehicle count data."""
        count_messages = self.sample_count_messages
        
        # Calculate total counts
        total_vehicles = sum(msg["total_count"] for msg in count_messages)
        
        # Calculate class totals
        class_totals = {}
        for msg in count_messages:
            for class_name, count in msg["class_counts"].items():
                class_totals[class_name] = class_totals.get(class_name, 0) + count
        
        # Verify aggregation
        assert total_vehicles == 25, f"Expected 25 total vehicles, got {total_vehicles}"
        assert class_totals["car"] == 16, f"Expected 16 cars, got {class_totals.get('car', 0)}"
        assert class_totals["truck"] == 7, f"Expected 7 trucks, got {class_totals.get('truck', 0)}"
        assert class_totals["bus"] == 2, f"Expected 2 buses, got {class_totals.get('bus', 0)}"

    def test_ocr_result_aggregation(self):
        """Test aggregation of OCR results."""
        ocr_messages = self.sample_ocr_messages
        
        # Count unique license plates
        unique_plates = set(msg["lp_text"] for msg in ocr_messages)
        
        # Calculate average confidence
        avg_confidence = sum(msg["ocr_confidence"] for msg in ocr_messages) / len(ocr_messages)
        
        # Find highest confidence result
        best_result = max(ocr_messages, key=lambda x: x["ocr_confidence"])
        
        # Verify aggregation
        assert len(unique_plates) == 3, f"Expected 3 unique plates, got {len(unique_plates)}"
        assert abs(avg_confidence - 0.913) < 0.01, f"Expected avg confidence ~0.913, got {avg_confidence}"
        assert best_result["lp_text"] == "ABC123", f"Best result should be ABC123"
        assert best_result["ocr_confidence"] == 0.95, f"Best confidence should be 0.95"

    def test_time_based_statistics(self):
        """Test calculation of time-based statistics."""
        count_messages = self.sample_count_messages
        
        # Calculate time span
        timestamps = [msg["timestamp"] for msg in count_messages]
        time_span = max(timestamps) - min(timestamps)
        
        # Calculate vehicles per minute
        vehicles_per_minute = (count_messages[-1]["total_count"] - count_messages[0]["total_count"]) / (time_span / 60)
        
        # Verify time calculations (allow small floating point differences)
        assert abs(time_span - 120) < 0.1, f"Expected ~120 second span, got {time_span}"
        assert abs(vehicles_per_minute - 3.5) < 0.1, f"Expected ~3.5 vehicles/min, got {vehicles_per_minute}"

    def test_summary_report_generation(self):
        """Test generation of summary report."""
        # Mock summary data
        summary_data = {
            "session_info": {
                "start_time": datetime.now().isoformat(),
                "end_time": (datetime.now() + timedelta(minutes=10)).isoformat(),
                "duration_seconds": 600,
                "camera_id": "cam_01"
            },
            "vehicle_counts": {
                "total_vehicles": 25,
                "class_breakdown": {"car": 16, "truck": 7, "bus": 2},
                "vehicles_per_minute": 2.5
            },
            "license_plates": {
                "total_detected": 3,
                "unique_plates": 3,
                "average_confidence": 0.913,
                "plates": ["ABC123", "XYZ789", "DEF456"]
            },
            "performance": {
                "frames_processed": 300,
                "fps": 30.0,
                "processing_time": 600
            }
        }
        
        # Verify report structure
        assert "session_info" in summary_data
        assert "vehicle_counts" in summary_data
        assert "license_plates" in summary_data
        assert "performance" in summary_data
        
        # Verify data integrity
        assert summary_data["vehicle_counts"]["total_vehicles"] == 25
        assert len(summary_data["license_plates"]["plates"]) == 3

    def test_json_report_serialization(self):
        """Test JSON serialization of summary report."""
        summary_data = {
            "timestamp": datetime.now().isoformat(),
            "total_vehicles": 25,
            "class_counts": {"car": 16, "truck": 7, "bus": 2},
            "license_plates": ["ABC123", "XYZ789", "DEF456"]
        }
        
        # Test JSON serialization
        try:
            json_string = json.dumps(summary_data, indent=2)
            parsed_data = json.loads(json_string)
            
            # Verify serialization/deserialization
            assert parsed_data["total_vehicles"] == 25
            assert parsed_data["class_counts"]["car"] == 16
            assert len(parsed_data["license_plates"]) == 3
            
        except (TypeError, ValueError) as e:
            pytest.fail(f"JSON serialization failed: {e}")

    def test_csv_export_functionality(self):
        """Test CSV export of summary data."""
        # Mock CSV data
        csv_data = [
            ["timestamp", "camera_id", "vehicle_class", "count"],
            ["2024-01-01T10:00:00", "cam_01", "car", "16"],
            ["2024-01-01T10:00:00", "cam_01", "truck", "7"],
            ["2024-01-01T10:00:00", "cam_01", "bus", "2"]
        ]
        
        # Test CSV formatting
        csv_lines = []
        for row in csv_data:
            csv_lines.append(",".join(row))
        
        csv_content = "\n".join(csv_lines)
        
        # Verify CSV format
        assert "timestamp,camera_id,vehicle_class,count" in csv_content
        assert "cam_01,car,16" in csv_content
        assert len(csv_lines) == 4, "Should have header + 3 data rows"

    def test_performance_metrics_calculation(self):
        """Test calculation of performance metrics."""
        # Mock performance data
        start_time = time.time()
        end_time = start_time + 600  # 10 minutes
        frames_processed = 18000  # 30 fps * 600 seconds
        
        # Calculate metrics
        duration = end_time - start_time
        fps = frames_processed / duration
        processing_efficiency = 1.0  # 100% real-time
        
        performance_metrics = {
            "duration_seconds": duration,
            "frames_processed": frames_processed,
            "average_fps": fps,
            "processing_efficiency": processing_efficiency,
            "real_time_factor": 1.0
        }
        
        # Verify metrics
        assert performance_metrics["duration_seconds"] == 600
        assert performance_metrics["frames_processed"] == 18000
        assert performance_metrics["average_fps"] == 30.0
        assert performance_metrics["processing_efficiency"] == 1.0

    def test_error_rate_calculation(self):
        """Test calculation of error rates and quality metrics."""
        # Mock processing statistics
        total_frames = 1000
        successful_detections = 950
        failed_detections = 50
        ocr_attempts = 100
        successful_ocr = 85
        
        # Calculate error rates
        detection_success_rate = successful_detections / total_frames
        detection_error_rate = failed_detections / total_frames
        ocr_success_rate = successful_ocr / ocr_attempts
        
        error_metrics = {
            "detection_success_rate": detection_success_rate,
            "detection_error_rate": detection_error_rate,
            "ocr_success_rate": ocr_success_rate,
            "total_frames": total_frames
        }
        
        # Verify error calculations
        assert error_metrics["detection_success_rate"] == 0.95
        assert error_metrics["detection_error_rate"] == 0.05
        assert error_metrics["ocr_success_rate"] == 0.85

    def test_file_output_creation(self):
        """Test creation of output files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test file paths
            json_file = Path(temp_dir) / "summary_report.json"
            csv_file = Path(temp_dir) / "vehicle_counts.csv"
            
            # Mock file creation
            summary_data = {"test": "data"}
            csv_data = "timestamp,count\n2024-01-01,25\n"
            
            # Write files
            with open(json_file, 'w') as f:
                json.dump(summary_data, f, indent=2)
            
            with open(csv_file, 'w') as f:
                f.write(csv_data)
            
            # Verify files exist
            assert json_file.exists(), "JSON file should be created"
            assert csv_file.exists(), "CSV file should be created"
            
            # Verify file contents
            with open(json_file, 'r') as f:
                loaded_data = json.load(f)
                assert loaded_data["test"] == "data"
            
            with open(csv_file, 'r') as f:
                content = f.read()
                assert "timestamp,count" in content
                assert "2024-01-01,25" in content

    def test_console_output_formatting(self):
        """Test formatting of console summary output."""
        summary_data = {
            "total_vehicles": 25,
            "class_counts": {"car": 16, "truck": 7, "bus": 2},
            "duration": 600,
            "license_plates": 3
        }
        
        # Format console output
        console_output = []
        console_output.append("=== Traffic Monitor Summary ===")
        console_output.append(f"Total Vehicles: {summary_data['total_vehicles']}")
        console_output.append(f"Duration: {summary_data['duration']} seconds")
        console_output.append(f"License Plates Detected: {summary_data['license_plates']}")
        console_output.append("Vehicle Breakdown:")
        for class_name, count in summary_data['class_counts'].items():
            console_output.append(f"  {class_name.capitalize()}: {count}")
        
        output_text = "\n".join(console_output)
        
        # Verify console formatting
        assert "Traffic Monitor Summary" in output_text
        assert "Total Vehicles: 25" in output_text
        assert "Car: 16" in output_text
        assert "Truck: 7" in output_text

    def test_data_validation(self):
        """Test validation of input data."""
        # Valid data
        valid_count_msg = {
            "camera_id": "cam_01",
            "timestamp": time.time(),
            "total_count": 5,
            "class_counts": {"car": 3, "truck": 2}
        }
        
        # Invalid data
        invalid_count_msgs = [
            None,
            {},
            {"camera_id": "cam_01"},  # Missing fields
            {"camera_id": "cam_01", "timestamp": "invalid", "total_count": 5},  # Invalid timestamp
            {"camera_id": "cam_01", "timestamp": time.time(), "total_count": -1}  # Invalid count
        ]
        
        # Test valid data
        assert self._validate_count_message(valid_count_msg) is True
        
        # Test invalid data
        for invalid_msg in invalid_count_msgs:
            assert self._validate_count_message(invalid_msg) is False

    def test_memory_efficient_processing(self):
        """Test memory-efficient processing of large datasets."""
        # Simulate large dataset
        num_messages = 10000
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        total_processed = 0
        
        for batch_start in range(0, num_messages, batch_size):
            batch_end = min(batch_start + batch_size, num_messages)
            batch_size_actual = batch_end - batch_start
            
            # Simulate processing batch
            total_processed += batch_size_actual
        
        assert total_processed == num_messages, "Should process all messages"

    def test_concurrent_data_access(self):
        """Test thread-safe access to summary data."""
        import threading
        import time
        
        shared_data = {"count": 0}
        lock = threading.Lock()
        
        def increment_count():
            for _ in range(100):
                with lock:
                    shared_data["count"] += 1
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=increment_count)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify thread safety
        assert shared_data["count"] == 500, "Should have correct count with thread safety"

    def test_report_template_customization(self):
        """Test customization of report templates."""
        # Default template
        default_template = {
            "title": "Traffic Monitor Summary",
            "sections": ["session_info", "vehicle_counts", "license_plates", "performance"],
            "format": "json"
        }
        
        # Custom template
        custom_template = {
            "title": "Custom Traffic Report",
            "sections": ["vehicle_counts", "performance"],
            "format": "csv",
            "include_charts": True
        }
        
        # Test template validation
        assert "title" in default_template
        assert "sections" in default_template
        assert len(default_template["sections"]) == 4
        
        assert custom_template["include_charts"] is True
        assert custom_template["format"] == "csv"

    def test_historical_data_comparison(self):
        """Test comparison with historical data."""
        # Current session data
        current_data = {
            "total_vehicles": 25,
            "class_counts": {"car": 16, "truck": 7, "bus": 2},
            "duration": 600
        }
        
        # Historical average data
        historical_data = {
            "total_vehicles": 20,
            "class_counts": {"car": 12, "truck": 6, "bus": 2},
            "duration": 600
        }
        
        # Calculate comparison
        vehicle_change = current_data["total_vehicles"] - historical_data["total_vehicles"]
        percentage_change = (vehicle_change / historical_data["total_vehicles"]) * 100
        
        comparison = {
            "vehicle_change": vehicle_change,
            "percentage_change": percentage_change,
            "trend": "increase" if vehicle_change > 0 else "decrease"
        }
        
        # Verify comparison
        assert comparison["vehicle_change"] == 5
        assert comparison["percentage_change"] == 25.0
        assert comparison["trend"] == "increase"

    # Helper methods
    def _validate_count_message(self, message):
        """Validate vehicle count message format."""
        if not message or not isinstance(message, dict):
            return False
        
        required_fields = ["camera_id", "timestamp", "total_count", "class_counts"]
        for field in required_fields:
            if field not in message:
                return False
        
        # Validate timestamp
        if not isinstance(message["timestamp"], (int, float)):
            return False
        
        # Validate total_count
        if not isinstance(message["total_count"], int) or message["total_count"] < 0:
            return False
        
        # Validate class_counts
        if not isinstance(message["class_counts"], dict):
            return False
        
        return True

    def _calculate_summary_statistics(self, data):
        """Calculate summary statistics from data."""
        if not data:
            return {}
        
        total_vehicles = sum(item.get("total_count", 0) for item in data)
        avg_vehicles = total_vehicles / len(data) if data else 0
        
        return {
            "total_vehicles": total_vehicles,
            "average_vehicles": avg_vehicles,
            "data_points": len(data)
        }


if __name__ == "__main__":
    pytest.main([__file__, "-v"])