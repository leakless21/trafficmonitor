# Persistence Component Documentation

## Overview

The Persistence component provides lightweight SQLite-based data storage for the Traffic Monitor system. It captures and stores critical events such as license plate detections and vehicle count summaries for later analysis and reporting.

## Responsibilities

- **Data Storage**: Persist plate recognition results and vehicle counting events
- **Schema Management**: Create and maintain SQLite database schema with proper indexing
- **Concurrency Handling**: Manage database locks and concurrent access from multiple processes
- **Performance Optimization**: Use WAL mode and indexing for efficient operations

## Architecture

### Database Schema

#### plate_results table

```sql
CREATE TABLE plate_results (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           INTEGER DEFAULT (strftime('%s','now')*1000),  -- Unix timestamp in milliseconds
    camera_id    TEXT,                                         -- Source camera identifier
    vehicle_id   INTEGER,                                      -- Tracking ID from vehicle tracker
    lp_text      TEXT,                                         -- Recognized license plate text
    ocr_conf     REAL                                          -- OCR confidence score (0.0-1.0)
);
```

#### vehicle_counts table

```sql
CREATE TABLE vehicle_counts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           INTEGER DEFAULT (strftime('%s','now')*1000),  -- Unix timestamp in milliseconds
    camera_id    TEXT,                                         -- Source camera identifier
    total_count  INTEGER,                                      -- Total vehicles counted
    class_counts TEXT                                          -- JSON object with class-specific counts
);
```

### Indexing Strategy

- `idx_plate_cam_time`: Index on (camera_id, ts) for plate_results table
- `idx_counts_cam_time`: Index on (camera_id, ts) for vehicle_counts table

These indices optimize common query patterns filtering by camera and time range.

## Key Features

### Zero Dependencies

- Uses Python's built-in `sqlite3` module
- No external database servers required
- Single-file database for easy deployment

### Concurrency Support

- WAL (Write-Ahead Logging) mode for better concurrent access
- Automatic retry with exponential backoff on database locks
- Process-safe operations for multiprocessing environment

### Error Handling

- Graceful handling of database lock conditions
- Comprehensive logging of retry attempts and failures
- Automatic database and directory creation

## Classes and Functions

### Core Module: `src/traffic_monitor/utils/minidb.py`

#### Functions

**`configure_database(config: dict) -> None`**

- Configures database path and settings from application config
- Sets database path (relative paths resolved from project root)
- Stores pragma settings for database optimization
- Called before `init_db()` at application startup

**`init_db() -> None`**

- Initializes database schema and indices
- Applies pragma settings from configuration
- Called once at application startup after configuration

**`write_plate_result(**kwargs) -> None`\*\*

- Stores license plate detection results
- Parameters: camera_id, vehicle_id, lp_text, ocr_conf, ts (optional)
- Automatic timestamp generation if not provided

**`write_vehicle_count(**kwargs) -> None`\*\*

- Stores vehicle counting summaries
- Parameters: camera_id, total_count, class_counts, ts (optional)
- JSON serialization of class_counts dictionary

#### Internal Functions

**`_connect() -> sqlite3.Connection`**

- Creates optimized SQLite connection
- Handles directory creation and connection settings

**`_with_retry(fn) -> Callable`**

- Decorator for automatic retry on database locks
- Exponential backoff with logging

## Usage Examples

### Configuration

The database can be configured via `settings.yaml`:

```yaml
# Database Configuration
database:
  # Path to SQLite database file (relative to project root)
  path: "data/db/traffic_monitor.db"

  # SQLite optimization settings
  pragmas:
    journal_mode: "WAL" # Write-Ahead Logging for better concurrency
    synchronous: "NORMAL" # Balance between safety and performance
    cache_size: -64000 # 64MB cache (negative = KB)
    temp_store: "MEMORY" # Store temporary tables in memory
```

### Basic Usage

```python
from traffic_monitor.utils.minidb import configure_database, init_db, write_plate_result, write_vehicle_count

# Configure database from settings (once at startup)
configure_database(config)
init_db()

# Store plate detection
write_plate_result(
    camera_id="cam-01",
    vehicle_id=123,
    lp_text="ABC123",
    ocr_conf=0.95
)

# Store vehicle count
write_vehicle_count(
    camera_id="cam-01",
    total_count=42,
    class_counts={"car": 30, "truck": 10, "bus": 2}
)
```

### Integration Points

#### Main Supervisor

```python
# In main_supervisor.py after setup_logging()
from .utils.minidb import configure_database, init_db
configure_database(config)
init_db()
```

#### License Plate Detector

```python
# In lp_detector.py when OCR result is ready
from traffic_monitor.utils.minidb import write_plate_result
write_plate_result(
    camera_id=camera_id,
    vehicle_id=track_id,
    lp_text=plate_text,
    ocr_conf=confidence
)
```

#### Vehicle Counter

```python
# In vehicle_counter.py when count is updated
from traffic_monitor.utils.minidb import write_vehicle_count
write_vehicle_count(
    camera_id=camera_id,
    total_count=total_vehicles,
    class_counts=class_count_dict
)
```

## Query Examples

### Retrieve Recent Plate Detections

```sql
SELECT ts, camera_id, lp_text, ocr_conf
FROM plate_results
WHERE camera_id = 'cam-01'
  AND ts > (strftime('%s','now')-3600)*1000  -- Last hour
ORDER BY ts DESC;
```

### Get Vehicle Count Summary

```sql
SELECT camera_id, COUNT(*) as events, AVG(total_count) as avg_count
FROM vehicle_counts
WHERE ts > (strftime('%s','now')-86400)*1000  -- Last 24 hours
GROUP BY camera_id;
```

### Extract Class-Specific Counts

```sql
SELECT ts, camera_id, total_count,
       json_extract(class_counts, '$.car') as cars,
       json_extract(class_counts, '$.truck') as trucks
FROM vehicle_counts
ORDER BY ts DESC
LIMIT 100;
```

## Testing

### Test Coverage

- Schema creation and initialization
- Data insertion with and without timestamps
- Automatic retry mechanism for database locks
- Multiple concurrent writes
- Error handling and edge cases

### Test File: `test/test_minidb.py`

- Uses in-memory database for isolation
- Comprehensive coverage of all public functions
- Mock testing for retry scenarios

## Performance Considerations

### Optimization Features

- WAL mode enables concurrent readers with writers
- Proper indexing for time-based queries
- Minimal connection overhead with context managers
- Automatic timestamp generation reduces application logic

### Scalability

- Suitable for thousands of inserts per second
- Single-file database simplifies backup and deployment
- Can be easily migrated to client-server databases if needed

## Future Enhancements

### Potential Improvements

- Configuration-based database path from settings.yaml
- Batch insert operations for high-throughput scenarios
- Database migration utilities for schema updates
- REST API for external data access
- Automated cleanup of old records

### Analytics Integration

- Export utilities for common formats (CSV, JSON)
- Integration with visualization tools
- Aggregation views for reporting
- Real-time dashboard data feeds

## File Locations

### Source Files

- `src/traffic_monitor/utils/minidb.py` - Main persistence module

### Test Files

- `test/test_minidb.py` - Unit tests for persistence functionality

### Database Files

- `traffic_monitor.db` - SQLite database file (created at runtime)
- `traffic_monitor.db-wal` - WAL file (created automatically)
- `traffic_monitor.db-shm` - Shared memory file (created automatically)
