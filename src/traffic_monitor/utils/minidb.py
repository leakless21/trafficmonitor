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

# Database file location - will be set from config or default
DB_PATH = Path(__file__).resolve().parent.parent.parent.parent / "traffic_monitor.db"
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
    db_path = _CONFIG.get('path', 'traffic_monitor.db')
    if not Path(db_path).is_absolute():
        # Relative path - resolve from project root
        project_root = Path(__file__).resolve().parent.parent.parent.parent
        DB_PATH = project_root / db_path
    else:
        DB_PATH = Path(db_path)
    
    logger.debug(f"Database configured at: {DB_PATH}")


def _connect() -> sqlite3.Connection:
    """Create SQLite connection with optimized settings."""
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
    logger.info(f"Initializing database at {DB_PATH}")
    
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
        
        # Create plate_results table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS plate_results (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                ts           INTEGER DEFAULT (strftime('%s','now')*1000),
                camera_id    TEXT,
                vehicle_id   INTEGER,
                lp_text      TEXT,
                ocr_conf     REAL
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
        
        logger.debug("Database schema initialized successfully")


@_with_retry
def write_plate_result(
    *,
    camera_id: str,
    vehicle_id: int | None = None,
    lp_text: str,
    ocr_conf: float,
    ts: int | None = None,
) -> None:
    """Insert a single license-plate OCR result."""
    if ts is None:
        ts = int(time.time() * 1000)
    
    with _connect() as con:
        con.execute("""
            INSERT INTO plate_results
            (ts, camera_id, vehicle_id, lp_text, ocr_conf)
            VALUES (?, ?, ?, ?, ?);
        """, (ts, camera_id, vehicle_id, lp_text, ocr_conf))
        
    logger.trace(f"Stored plate result: {lp_text} (conf={ocr_conf:.3f}) for vehicle {vehicle_id}")


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