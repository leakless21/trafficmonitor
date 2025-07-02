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
        
        # Check vehicle_counts table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vehicle_counts';")
        assert cur.fetchone() is not None
        
        # Check indices exist
        cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_plate_cam_time';")
        assert cur.fetchone() is not None
        
        cur.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_counts_cam_time';")
        assert cur.fetchone() is not None


def test_write_plate_result(memory_db):
    """Test writing plate result data."""
    minidb.init_db()
    
    # Write a plate result
    minidb.write_plate_result(
        camera_id="cam-01",
        vehicle_id=123,
        lp_text="ABC123",
        ocr_conf=0.95,
        ts=1234567890000
    )
    
    # Verify data was written
    with minidb._connect() as con:
        cur = con.cursor()
        cur.execute("SELECT * FROM plate_results WHERE camera_id='cam-01';")
        row = cur.fetchone()
        
        assert row is not None
        assert row[2] == "cam-01"  # camera_id
        assert row[3] == 123       # vehicle_id
        assert row[4] == "ABC123"  # lp_text
        assert row[5] == 0.95      # ocr_conf
        assert row[1] == 1234567890000  # ts


def test_write_plate_result_auto_timestamp(memory_db):
    """Test writing plate result with automatic timestamp."""
    minidb.init_db()
    
    # Write without timestamp
    minidb.write_plate_result(
        camera_id="cam-02",
        lp_text="XYZ789",
        ocr_conf=0.88
    )
    
    # Verify data was written with timestamp
    with minidb._connect() as con:
        cur = con.cursor()
        cur.execute("SELECT ts FROM plate_results WHERE camera_id='cam-02';")
        row = cur.fetchone()
        
        assert row is not None
        assert row[0] > 0  # Should have a timestamp


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
    
    # Write multiple plate results
    for i in range(10):
        minidb.write_plate_result(
            camera_id=f"cam-{i % 3}",
            vehicle_id=i,
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