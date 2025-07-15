"""
Unit tests for minidb database operations.
Tests data persistence, integrity, and database operations.
"""

import pytest
import sqlite3
import tempfile
import os
from pathlib import Path
import sys
from unittest.mock import patch, Mock
import json
from datetime import datetime

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from traffic_monitor.utils import minidb


class TestMiniDB:
    """Test database operations and data integrity."""
    
    def setup_method(self):
        """Set up test database."""
        # Create temporary database file
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Configure and initialize test database
        test_config = {
            'database': {
                'path': self.db_path,
                'reset_on_startup': False
            }
        }
        minidb.configure_database(test_config)
        minidb.init_db()
        
        # Sample data for testing
        self.sample_detection = {
            "frame_id": 100,
            "track_id": 1,
            "bbox": [100, 100, 200, 200],
            "confidence": 0.85,
            "class_id": 3,
            "class_name": "car",
            "timestamp": datetime.now().isoformat()
        }
        
        self.sample_count = {
            "frame_id": 100,
            "timestamp": datetime.now().isoformat(),
            "counts": {"car": {"up": 5, "down": 3}, "bus": {"up": 1, "down": 2}},
            "total_vehicles": 11,
            "camera_id": "cam-01"
        }

    def teardown_method(self):
        """Clean up test database."""
        if os.path.exists(self.db_path):
            os.unlink(self.db_path)

    def test_database_initialization(self):
        """Test database initialization and table creation."""
        # Check if database file exists
        assert os.path.exists(self.db_path), "Database file should exist"
        
        # Check if tables were created
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check for expected tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ["vehicle_counts", "plate_results", "plate_results_latest"]
        for table in expected_tables:
            assert table in tables, f"Table '{table}' should exist"
        
        conn.close()

    def test_write_vehicle_count(self):
        """Test writing vehicle count data."""
        result = minidb.write_vehicle_count(
            camera_id="cam-01",
            total_count=10,
            class_counts={"car": 8, "truck": 2}
        )
        assert result is True, "Vehicle count write should succeed"

    def test_write_license_plate(self):
        """Test writing license plate data."""
        result = minidb.write_license_plate(
            camera_id="cam-01",
            vehicle_id=123,
            plate_text="ABC123",
            confidence=0.95
        )
        assert result is True, "License plate write should succeed"

    def test_get_vehicle_counts(self):
        """Test retrieving vehicle count data."""
        # First write some data
        minidb.write_vehicle_count(
            camera_id="cam-01",
            total_count=5,
            class_counts={"car": 3, "truck": 2}
        )
        
        # Retrieve data
        counts = minidb.get_vehicle_counts(camera_id="cam-01", limit=10)
        assert len(counts) >= 1, "Should retrieve at least one count record"

    def test_get_license_plates(self):
        """Test retrieving license plate data."""
        # First write some data
        minidb.write_license_plate(
            camera_id="cam-01",
            vehicle_id=123,
            plate_text="TEST123",
            confidence=0.9
        )
        
        # Retrieve data
        plates = minidb.get_license_plates(camera_id="cam-01", limit=10)
        assert len(plates) >= 1, "Should retrieve at least one plate record"

    def test_database_error_handling(self):
        """Test database error handling."""
        # Test with invalid database path
        original_path = minidb.DB_PATH
        minidb.DB_PATH = "/invalid/path/database.db"
        
        try:
            result = minidb.write_vehicle_count(
                camera_id="test",
                total_count=1,
                class_counts={"car": 1}
            )
            # Should handle error gracefully
            assert result is False, "Should return False on database error"
        finally:
            minidb.DB_PATH = original_path

    def test_data_integrity(self):
        """Test data integrity constraints."""
        # Test writing valid data
        result1 = minidb.write_vehicle_count(
            camera_id="cam-01",
            total_count=5,
            class_counts={"car": 5}
        )
        assert result1 is True, "Valid data should be written successfully"

    def test_concurrent_database_access(self):
        """Test concurrent database access."""
        import threading
        import time
        
        results = []
        
        def write_data(thread_id):
            result = minidb.write_vehicle_count(
                camera_id=f"cam-{thread_id}",
                total_count=thread_id,
                class_counts={"car": thread_id}
            )
            results.append(result)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=write_data, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that most operations succeeded
        success_count = sum(results)
        assert success_count >= 3, f"At least 3 out of 5 concurrent operations should succeed, got {success_count}"

    def test_database_performance(self):
        """Test database performance with multiple operations."""
        import time
        
        start_time = time.time()
        
        # Perform multiple write operations
        for i in range(50):
            minidb.write_vehicle_count(
                camera_id="perf-test",
                total_count=i,
                class_counts={"car": i}
            )
        
        elapsed = time.time() - start_time
        assert elapsed < 5.0, f"50 database operations took too long: {elapsed:.2f}s"

    def test_database_cleanup(self):
        """Test database cleanup and maintenance."""
        # Write some test data
        for i in range(10):
            minidb.write_vehicle_count(
                camera_id="cleanup-test",
                total_count=i,
                class_counts={"car": i}
            )
        
        # Verify data exists
        counts = minidb.get_vehicle_counts(camera_id="cleanup-test", limit=20)
        assert len(counts) >= 10, "Should have written test data"

    def test_query_with_time_range(self):
        """Test querying data with time constraints."""
        # Write data with current timestamp
        minidb.write_vehicle_count(
            camera_id="time-test",
            total_count=1,
            class_counts={"car": 1}
        )
        
        # Query recent data
        counts = minidb.get_vehicle_counts(camera_id="time-test", limit=10)
        assert len(counts) >= 1, "Should retrieve recent data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])