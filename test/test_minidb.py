"""
Unit tests for the minidb SQLite helper module.
"""
import pytest
import tempfile
import sqlite3
from pathlib import Path
from unittest.mock import patch

from src.traffic_monitor.utils import minidb


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        original_path = minidb.DB_PATH
        minidb.DB_PATH = Path(tmp.name)
        yield Path(tmp.name)
        # Cleanup
        minidb.DB_PATH = original_path
        Path(tmp.name).unlink(missing_ok=True)


@pytest.fixture
def memory_db():
    """Use in-memory database for testing."""
    original_path = minidb.DB_PATH
    original_config = minidb._CONFIG
    original_connect = minidb._connect
    
    # Create a shared in-memory connection for the test
    shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
    
    def mock_connect():
        return shared_conn
    
    # Configure for testing
    minidb.DB_PATH = ":memory:"
    minidb._CONFIG = {}  # Empty config for tests
    minidb._connect = mock_connect
    
    yield shared_conn
    
    # Cleanup
    shared_conn.close()
    minidb.DB_PATH = original_path
    minidb._CONFIG = original_config
    minidb._connect = original_connect


def test_init_db_creates_tables(memory_db):
    """Test that init_db creates the required tables and indices."""
    minidb.init_db()
    
    with minidb._connect() as con:
        cur = con.cursor()
        
        # Check plate_results table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='plate_results';")
        assert cur.fetchone() is not None
        
        # Check plate_results_latest table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='plate_results_latest';")
        assert cur.fetchone() is not None
        
        # Check vehicle_counts table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vehicle_counts';")
        assert cur.fetchone() is not None
        
        # Check indices exist
        cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_plate_cam_time';")
        assert cur.fetchone() is not None
        
        cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_counts_cam_time';")
        assert cur.fetchone() is not None
        
        # Check new vehicle_class index exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_plate_vehicle_class';")
        assert cur.fetchone() is not None
        
        # Check latest table vehicle_class index exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_latest_vehicle_class';")
        assert cur.fetchone() is not None
        
        # Verify plate_results table has vehicle_class column
        cur.execute("PRAGMA table_info(plate_results);")
        columns = [row[1] for row in cur.fetchall()]
        assert 'vehicle_class' in columns
        
        # Verify plate_results_latest table structure
        cur.execute("PRAGMA table_info(plate_results_latest);")
        latest_columns = [row[1] for row in cur.fetchall()]
        expected_latest_columns = ['camera_id', 'vehicle_id', 'vehicle_class', 'lp_text', 'ocr_conf', 'first_seen', 'last_updated']
        for col in expected_latest_columns:
            assert col in latest_columns


def test_write_plate_result(memory_db):
    """Test writing plate result data to both history and latest tables."""
    minidb.init_db()
    
    # Write a plate result
    minidb.write_plate_result(
        camera_id="cam-01",
        vehicle_id=123,
        vehicle_class="car",
        lp_text="ABC123",
        ocr_conf=0.95,
        ts=1234567890000
    )
    
    # Verify data was written to history table
    with minidb._connect() as con:
        cur = con.cursor()
        cur.execute("SELECT * FROM plate_results WHERE camera_id='cam-01';")
        row = cur.fetchone()
        
        assert row is not None
        assert row[2] == "cam-01"  # camera_id
        assert row[3] == 123       # vehicle_id
        assert row[4] == "car"     # vehicle_class
        assert row[5] == "ABC123"  # lp_text
        assert row[6] == 0.95      # ocr_conf
        assert row[1] == 1234567890000  # ts
        
        # Verify data was written to latest table
        cur.execute("SELECT * FROM plate_results_latest WHERE camera_id='cam-01' AND vehicle_id=123;")
        latest_row = cur.fetchone()
        
        assert latest_row is not None
        assert latest_row[0] == "cam-01"  # camera_id
        assert latest_row[1] == 123       # vehicle_id
        assert latest_row[2] == "car"     # vehicle_class
        assert latest_row[3] == "ABC123"  # lp_text
        assert latest_row[4] == 0.95      # ocr_conf
        assert latest_row[5] == 1234567890000  # first_seen
        assert latest_row[6] == 1234567890000  # last_updated


def test_write_plate_result_auto_timestamp(memory_db):
    """Test writing plate result with automatic timestamp."""
    minidb.init_db()
    
    # Write without timestamp
    minidb.write_plate_result(
        camera_id="cam-02",
        vehicle_class="truck",
        lp_text="XYZ789",
        ocr_conf=0.88
    )
    
    # Verify data was written with timestamp
    with minidb._connect() as con:
        cur = con.cursor()
        cur.execute("SELECT ts, vehicle_class FROM plate_results WHERE camera_id='cam-02';")
        row = cur.fetchone()
        
        assert row is not None
        assert row[0] > 0  # Should have a timestamp
        assert row[1] == "truck"  # vehicle_class


def test_write_vehicle_count(memory_db):
    """Test writing vehicle count data."""
    minidb.init_db()
    
    # Write vehicle count
    minidb.write_vehicle_count(
        camera_id="cam-01",
        total_count=42,
        class_counts={"car": 30, "truck": 12},
        ts=1234567890000
    )
    
    # Verify data was written
    with minidb._connect() as con:
        cur = con.cursor()
        cur.execute("SELECT * FROM vehicle_counts WHERE camera_id='cam-01';")
        row = cur.fetchone()
        
        assert row is not None
        assert row[2] == "cam-01"  # camera_id
        assert row[3] == 42        # total_count
        assert '{"car":30,"truck":12}' in row[4]  # class_counts JSON
        assert row[1] == 1234567890000  # ts


def test_write_vehicle_count_auto_timestamp(memory_db):
    """Test writing vehicle count with automatic timestamp."""
    minidb.init_db()
    
    # Write without timestamp
    minidb.write_vehicle_count(
        camera_id="cam-03",
        total_count=5,
        class_counts={"bicycle": 5}
    )
    
    # Verify data was written with timestamp
    with minidb._connect() as con:
        cur = con.cursor()
        cur.execute("SELECT ts FROM vehicle_counts WHERE camera_id='cam-03';")
        row = cur.fetchone()
        
        assert row is not None
        assert row[0] > 0  # Should have a timestamp


def test_database_retry_mechanism(memory_db):
    """Test that the retry decorator is applied correctly."""
    minidb.init_db()
    
    # Simply test that the write functions work (retry decorator is applied)
    # The retry mechanism itself is hard to test without actual database locks
    minidb.write_plate_result(
        camera_id="test",
        vehicle_class="bus",
        lp_text="TEST123",
        ocr_conf=0.9
    )
    
    # Verify it was written
    with minidb._connect() as con:
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) FROM plate_results WHERE camera_id='test';")
        count = cur.fetchone()[0]
        assert count == 1


def test_multiple_writes(memory_db):
    """Test writing multiple records to ensure no conflicts."""
    minidb.init_db()
    
    # Vehicle classes for testing
    vehicle_classes = ["car", "truck", "bus", "motorcycle", "bicycle"]
    
    # Write multiple plate results
    for i in range(10):
        minidb.write_plate_result(
            camera_id=f"cam-{i % 3}",
            vehicle_id=i,
            vehicle_class=vehicle_classes[i % len(vehicle_classes)],
            lp_text=f"PLATE{i:03d}",
            ocr_conf=0.8 + (i * 0.01)
        )
    
    # Write multiple vehicle counts
    for i in range(10):
        minidb.write_vehicle_count(
            camera_id=f"cam-{i % 3}",
            total_count=i * 5,
            class_counts={"car": i * 3, "truck": i * 2}
        )
    
    # Verify all records were written
    with minidb._connect() as con:
        cur = con.cursor()
        
        cur.execute("SELECT COUNT(*) FROM plate_results;")
        assert cur.fetchone()[0] == 10
        
        cur.execute("SELECT COUNT(*) FROM vehicle_counts;")
        assert cur.fetchone()[0] == 10
        
        # Test vehicle class filtering
        cur.execute("SELECT COUNT(*) FROM plate_results WHERE vehicle_class='car';")
        car_count = cur.fetchone()[0]
        assert car_count >= 1  # Should have at least one car record


def test_database_configuration():
    """Test database configuration from settings."""
    import tempfile
    original_path = minidb.DB_PATH
    original_config = minidb._CONFIG
    
    try:
        # Test configuration with custom path
        config = {
            'database': {
                'path': 'custom/path/test.db',
                'pragmas': {
                    'journal_mode': 'DELETE',
                    'synchronous': 'FULL'
                }
            }
        }
        
        minidb.configure_database(config)
        
        # Check that path was set correctly (should be absolute)
        assert minidb.DB_PATH.name == 'test.db'
        assert 'custom' in str(minidb.DB_PATH) and 'path' in str(minidb.DB_PATH)
        
        # Check that config was stored
        assert minidb._CONFIG['path'] == 'custom/path/test.db'
        assert minidb._CONFIG['pragmas']['journal_mode'] == 'DELETE'
        
    finally:
        # Restore original values
        minidb.DB_PATH = original_path
        minidb._CONFIG = original_config


def test_database_reset_on_startup():
    """Test database reset_on_startup functionality."""
    import tempfile
    import os
    
    original_path = minidb.DB_PATH
    original_config = minidb._CONFIG
    
    try:
        # Create a temporary database file
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
            temp_db_path = tmp.name
        
        # Configuration without reset
        config_no_reset = {
            'database': {
                'path': temp_db_path,
                'reset_on_startup': False
            }
        }
        
        # Initialize database without reset
        minidb.configure_database(config_no_reset)
        minidb.init_db()
        
        # Write some data
        minidb.write_plate_result(
            camera_id="test",
            vehicle_id=1,
            vehicle_class="car",
            lp_text="ORIGINAL",
            ocr_conf=0.9
        )
        
        # Verify file exists and has data
        db_path = Path(temp_db_path)
        assert db_path.exists()
        original_size = db_path.stat().st_size
        assert original_size > 0
        
        # Configuration with reset
        config_with_reset = {
            'database': {
                'path': temp_db_path,
                'reset_on_startup': True
            }
        }
        
        # Force cleanup before reset
        import gc
        import time
        gc.collect()
        time.sleep(0.1)
        
        # Initialize database with reset
        minidb.configure_database(config_with_reset)
        minidb.init_db()
        
        # Verify database was reset (should be a fresh file)
        assert db_path.exists()
        # Fresh database should have same minimal size
        new_size = db_path.stat().st_size
        assert new_size > 0  # Should still have schema
        
        # Write new data to verify it works
        minidb.write_plate_result(
            camera_id="test",
            vehicle_id=2,
            vehicle_class="truck",
            lp_text="RESET",
            ocr_conf=0.95
        )
        
    finally:
        # Cleanup
        try:
            if Path(temp_db_path).exists():
                Path(temp_db_path).unlink()
        except:
            pass  # Ignore cleanup errors
        
        # Restore original values
        minidb.DB_PATH = original_path
        minidb._CONFIG = original_config 


def test_plate_result_confidence_based_updates(memory_db):
    """Test that latest table only updates when confidence improves."""
    minidb.init_db()
    
    # Write initial plate result with medium confidence
    minidb.write_plate_result(
        camera_id="cam-test",
        vehicle_id=42,
        vehicle_class="car",
        lp_text="INITIAL",
        ocr_conf=0.75,
        ts=1000
    )
    
    # Write a worse confidence result - should NOT update latest table
    minidb.write_plate_result(
        camera_id="cam-test",
        vehicle_id=42,
        vehicle_class="car",
        lp_text="WORSE",
        ocr_conf=0.60,
        ts=2000
    )
    
    # Write a better confidence result - SHOULD update latest table
    minidb.write_plate_result(
        camera_id="cam-test",
        vehicle_id=42,
        vehicle_class="truck",  # Also change vehicle class
        lp_text="BETTER",
        ocr_conf=0.90,
        ts=3000
    )
    
    with minidb._connect() as con:
        cur = con.cursor()
        
        # Verify history table has all 3 records
        cur.execute("SELECT COUNT(*) FROM plate_results WHERE camera_id='cam-test' AND vehicle_id=42;")
        assert cur.fetchone()[0] == 3
        
        # Verify latest table has only the best result
        cur.execute("SELECT lp_text, ocr_conf, vehicle_class, first_seen, last_updated FROM plate_results_latest WHERE camera_id='cam-test' AND vehicle_id=42;")
        latest = cur.fetchone()
        
        assert latest is not None
        assert latest[0] == "BETTER"  # lp_text - should be the high confidence one
        assert latest[1] == 0.90      # ocr_conf - should be the highest
        assert latest[2] == "truck"   # vehicle_class - should be updated
        assert latest[3] == 1000      # first_seen - should be original timestamp
        assert latest[4] == 3000      # last_updated - should be latest update timestamp


def test_get_latest_plate_result_functions(memory_db):
    """Test the helper functions for retrieving latest plate results."""
    minidb.init_db()
    
    # Add some test data
    minidb.write_plate_result(camera_id="cam-01", vehicle_id=1, vehicle_class="car", lp_text="ABC123", ocr_conf=0.95)
    minidb.write_plate_result(camera_id="cam-01", vehicle_id=2, vehicle_class="truck", lp_text="DEF456", ocr_conf=0.85)
    minidb.write_plate_result(camera_id="cam-02", vehicle_id=3, vehicle_class="bus", lp_text="GHI789", ocr_conf=0.92)
    
    # Test get_latest_plate_result for specific vehicle
    result = minidb.get_latest_plate_result("cam-01", 1)
    assert result is not None
    assert result['lp_text'] == "ABC123"
    assert result['vehicle_class'] == "car"
    assert result['ocr_conf'] == 0.95
    
    # Test get_latest_plate_result for non-existent vehicle
    result = minidb.get_latest_plate_result("cam-01", 999)
    assert result is None
    
    # Test get_all_latest_plates without filter
    all_results = minidb.get_all_latest_plates()
    assert len(all_results) == 3
    
    # Test get_all_latest_plates with camera filter
    cam01_results = minidb.get_all_latest_plates("cam-01")
    assert len(cam01_results) == 2
    assert all(r['camera_id'] == "cam-01" for r in cam01_results) 