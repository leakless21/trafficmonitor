"""
Light-weight SQLite helper for Traffic Monitor.
Creates a single file database for storing plate results and vehicle counts.

Features:
- Zero external dependencies (uses built-in sqlite3)
- Automatic retry on database locks
- WAL mode for better concurrency
- Proper indexing for common queries

Usage:
    >>> from traffic_monitor.utils.minidb import init_db, write_vehicle_count
    >>> init_db()  # one-time at program start
    >>> write_vehicle_count(
    ...     camera_id="cam-01",
    ...     total_count=42,
    ...     class_counts={"car": 30, "truck": 12},
    ... )
"""
from __future__ import annotations

import sqlite3
import json
import time
from pathlib import Path
from typing import Dict, Callable
from loguru import logger
import os # Import os for file deletion

# Database file location - will be set from config or default
DB_PATH = None
_CONFIG = None


def _with_retry(fn: Callable) -> Callable:
    """Decorator: retry with exponential backoff on database locks."""
    MAX_TRIES = 5
    BACKOFF = 0.05  # seconds

    def wrapper(*args, **kwargs):
        delay = BACKOFF
        for attempt in range(MAX_TRIES):
            try:
                return fn(*args, **kwargs)
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                if attempt < MAX_TRIES - 1:
                    logger.warning(f"Database locked, retrying in {delay:.3f}s (attempt {attempt + 1}/{MAX_TRIES})")
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"Database remained locked after {MAX_TRIES} attempts")
                    raise RuntimeError("SQLite database remained locked after multiple retries")
        
    wrapper.__name__ = fn.__name__
    return wrapper


def configure_database(config: dict) -> None:
    """Configure database settings from application config."""
    global DB_PATH, _CONFIG
    _CONFIG = config.get('database', {})
    
    # Set database path from config
    db_path = _CONFIG.get('path') # No default here, rely on init_db or configure to set it
    if db_path is None:
        # Default to project root if not specified in config
        DB_PATH = Path(__file__).resolve().parent.parent.parent.parent / "traffic_monitor.db"
    elif not Path(db_path).is_absolute():
        # Relative path - resolve from project root
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        DB_PATH = project_root / db_path
    else:
        DB_PATH = Path(db_path)
    
    logger.debug(f"Database configured at: {DB_PATH}")


def _connect() -> sqlite3.Connection:
    """Create SQLite connection with optimized settings."""
    # Ensure DB_PATH is set before connecting
    if DB_PATH is None:
        raise ValueError("Database path not configured. Call configure_database() first.")
    
    # Ensure directory exists (skip for in-memory databases)
    if DB_PATH != ":memory:" and isinstance(DB_PATH, Path):
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    return sqlite3.connect(
        DB_PATH,
        timeout=5.0,
        isolation_level=None,  # autocommit
        detect_types=sqlite3.PARSE_DECLTYPES,
    )


def init_db() -> None:
    """Create tables and indices if they don't exist yet."""
    # Ensure DB_PATH is set
    if DB_PATH is None:
        raise ValueError("Database path not configured. Call configure_database() first.")
    
    logger.info(f"Initializing database at {DB_PATH}")
    
    # Check for reset_on_startup option
    if _CONFIG and _CONFIG.get('reset_on_startup', False):
        if DB_PATH.exists():
            logger.warning(f"Resetting database at {DB_PATH}")
            try:
                # More aggressive approach for Windows file handle cleanup
                import gc
                import time
                
                # Force garbage collection to cleanup any lingering connections
                gc.collect()
                time.sleep(0.2)
                
                # Try to connect and immediately close to flush any pending operations
                try:
                    temp_conn = sqlite3.connect(DB_PATH)
                    temp_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")  # Flush WAL
                    temp_conn.close()
                    del temp_conn
                except:
                    pass  # Ignore any errors during cleanup
                
                # Another garbage collection and delay
                gc.collect()
                time.sleep(0.3)
                
                # Try to delete files
                DB_PATH.unlink() # Delete the main database file
                logger.info(f"Database file deleted successfully: {DB_PATH}")
                
                # Also delete WAL and SHM files if they exist
                wal_path = DB_PATH.with_suffix(".db-wal")
                shm_path = DB_PATH.with_suffix(".db-shm")
                if wal_path.exists():
                    wal_path.unlink()
                    logger.debug(f"WAL file deleted: {wal_path}")
                if shm_path.exists():
                    shm_path.unlink()
                    logger.debug(f"SHM file deleted: {shm_path}")
                    
            except OSError as e:
                logger.error(f"Error deleting database files at {DB_PATH}: {e}")
                logger.warning(f"Continuing with initialization despite deletion failure")
                # If deletion fails, continue with initialization anyway
        else:
            logger.info(f"Database file not found at {DB_PATH}, no reset needed.")

    with _connect() as con:
        cur = con.cursor()
        
        # Apply pragma settings from config
        if _CONFIG:
            pragmas = _CONFIG.get('pragmas', {})
            for pragma, value in pragmas.items():
                cur.execute(f"PRAGMA {pragma}={value};")
                logger.debug(f"Applied PRAGMA {pragma}={value}")
        else:
            # Default: Enable WAL mode for better concurrency
            cur.execute("PRAGMA journal_mode=WAL;")
        
        # Create plate_results table (history - all OCR attempts)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS plate_results (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           INTEGER DEFAULT (strftime('%s','now')*1000),
                camera_id    TEXT,
                vehicle_id   INTEGER,
                vehicle_class TEXT,
                lp_text      TEXT,
                ocr_conf     REAL
            );
        """)
        
        # Create plate_results_latest table (one authoritative result per vehicle)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS plate_results_latest (
                camera_id    TEXT,
                vehicle_id   INTEGER,
                vehicle_class TEXT,
                lp_text      TEXT,
                ocr_conf     REAL,
                first_seen   INTEGER DEFAULT (strftime('%s','now')*1000),
                last_updated INTEGER DEFAULT (strftime('%s','now')*1000),
                PRIMARY KEY (camera_id, vehicle_id)
            );
        """)

        # Create vehicle_counts table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_counts (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           INTEGER DEFAULT (strftime('%s','now')*1000),
                camera_id    TEXT,
                total_count  INTEGER,
                class_counts TEXT
            );
        """)
        
        # Create indices for common queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_plate_cam_time
            ON plate_results(camera_id, ts);
        """)
        
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_counts_cam_time
            ON vehicle_counts(camera_id, ts);
        """)
        
        # Create index for vehicle class queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_plate_vehicle_class
            ON plate_results(vehicle_class, ts);
        """)
        
        # Create index for latest results by vehicle class
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_latest_vehicle_class
            ON plate_results_latest(vehicle_class);
        """)
        
        logger.debug("Database schema initialized successfully")


@_with_retry
def write_plate_result(
    *,
    camera_id: str,
    vehicle_id: int | None = None,
    vehicle_class: str | None = None,
    lp_text: str,
    ocr_conf: float,
    ts: int | None = None,
) -> None:
    """
    Insert a license-plate OCR result into both history and latest tables.
    
    The history table (plate_results) keeps all OCR attempts for audit trail.
    The latest table (plate_results_latest) maintains one authoritative result 
    per (camera_id, vehicle_id) pair using "best confidence wins" strategy.
    """
    if ts is None:
        ts = int(time.time() * 1000)
    
    with _connect() as con:
        # Always insert into history table
        con.execute("""
            INSERT INTO plate_results
            (ts, camera_id, vehicle_id, vehicle_class, lp_text, ocr_conf)
            VALUES (?, ?, ?, ?, ?, ?);
        """, (ts, camera_id, vehicle_id, vehicle_class, lp_text, ocr_conf))
        
        # Upsert into latest table - only update if confidence is better
        con.execute("""
            INSERT INTO plate_results_latest 
            (camera_id, vehicle_id, vehicle_class, lp_text, ocr_conf, first_seen, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(camera_id, vehicle_id) DO UPDATE SET
                vehicle_class = excluded.vehicle_class,
                lp_text = excluded.lp_text,
                ocr_conf = excluded.ocr_conf,
                last_updated = excluded.last_updated
            WHERE excluded.ocr_conf > plate_results_latest.ocr_conf;
        """, (camera_id, vehicle_id, vehicle_class, lp_text, ocr_conf, ts, ts))
        
    logger.trace(f"Stored plate result: {lp_text} (conf={ocr_conf:.3f}) for {vehicle_class} vehicle {vehicle_id}")


@_with_retry
def write_vehicle_count(
    *,
    camera_id: str,
    total_count: int,
    class_counts: Dict[str, int],
    ts: int | None = None,
) -> None:
    """Insert aggregate counts for a frame or time-window."""
    if ts is None:
        ts = int(time.time() * 1000)
    
    with _connect() as con:
        con.execute("""
            INSERT INTO vehicle_counts
            (ts, camera_id, total_count, class_counts)
            VALUES (?, ?, ?, ?);
        """, (
            ts,
            camera_id,
            total_count,
            json.dumps(class_counts, separators=(",", ":")),
        ))
        
    logger.trace(f"Stored vehicle count: {total_count} total, {class_counts} by class")


@_with_retry
def get_latest_plate_result(camera_id: str, vehicle_id: int) -> Dict | None:
    """
    Get the latest (highest confidence) plate result for a specific vehicle.
    
    Returns:
        Dict with keys: camera_id, vehicle_id, vehicle_class, lp_text, ocr_conf, 
                       first_seen, last_updated
        None if no result found
    """
    with _connect() as con:
        cur = con.cursor()
        cur.execute("""
            SELECT camera_id, vehicle_id, vehicle_class, lp_text, ocr_conf, 
                   first_seen, last_updated
            FROM plate_results_latest 
            WHERE camera_id = ? AND vehicle_id = ?;
        """, (camera_id, vehicle_id))
        
        row = cur.fetchone()
        if row:
            return {
                'camera_id': row[0],
                'vehicle_id': row[1], 
                'vehicle_class': row[2],
                'lp_text': row[3],
                'ocr_conf': row[4],
                'first_seen': row[5],
                'last_updated': row[6]
            }
        return None


@_with_retry
def get_all_latest_plates(camera_id: str | None = None) -> list[Dict]:
    """
    Get all latest plate results, optionally filtered by camera.
    
    Args:
        camera_id: Optional camera filter
        
    Returns:
        List of dicts with latest plate results
    """
    with _connect() as con:
        cur = con.cursor()
        
        if camera_id:
            cur.execute("""
                SELECT camera_id, vehicle_id, vehicle_class, lp_text, ocr_conf,
                       first_seen, last_updated
                FROM plate_results_latest 
                WHERE camera_id = ?
                ORDER BY last_updated DESC;
            """, (camera_id,))
        else:
            cur.execute("""
                SELECT camera_id, vehicle_id, vehicle_class, lp_text, ocr_conf,
                       first_seen, last_updated
                FROM plate_results_latest 
                ORDER BY last_updated DESC;
            """)
        
        results = []
        for row in cur.fetchall():
            results.append({
                'camera_id': row[0],
                'vehicle_id': row[1],
                'vehicle_class': row[2], 
                'lp_text': row[3],
                'ocr_conf': row[4],
                'first_seen': row[5],
                'last_updated': row[6]
            })
        
        return results 