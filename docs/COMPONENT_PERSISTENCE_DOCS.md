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

#### `plate_results` Table (History)

Stores ALL license plate detection attempts for complete audit trail.

| Column          | Type                | Description                                                     |
| --------------- | ------------------- | --------------------------------------------------------------- |
| `id`            | INTEGER PRIMARY KEY | Auto-incrementing unique identifier                             |
| `ts`            | INTEGER             | Unix timestamp in milliseconds (auto-generated if not provided) |
| `camera_id`     | TEXT                | Camera identifier                                               |
| `vehicle_id`    | INTEGER             | Vehicle tracking ID from the tracker                            |
| `vehicle_class` | TEXT                | Vehicle class (e.g., "car", "truck", "bus", "motorcycle")       |
| `lp_text`       | TEXT                | Extracted license plate text                                    |
| `ocr_conf`      | REAL                | OCR confidence score (0.0 to 1.0)                               |

**Purpose**: Complete forensic record of all OCR attempts for debugging and analysis.

#### `plate_results_latest` Table (Authoritative)

Stores ONE authoritative result per vehicle using "best confidence wins" strategy.

| Column          | Type    | Description                                      |
| --------------- | ------- | ------------------------------------------------ |
| `camera_id`     | TEXT    | Camera identifier                                |
| `vehicle_id`    | INTEGER | Vehicle tracking ID from the tracker             |
| `vehicle_class` | TEXT    | Vehicle class (updated with latest detection)    |
| `lp_text`       | TEXT    | Best confidence license plate text               |
| `ocr_conf`      | REAL    | Highest OCR confidence score achieved            |
| `first_seen`    | INTEGER | Unix timestamp of first detection (milliseconds) |
| `last_updated`  | INTEGER | Unix timestamp of last update (milliseconds)     |

**Primary Key**: `(camera_id, vehicle_id)` - ensures exactly one result per vehicle.

**Purpose**: Fast, duplicate-free queries for real-time applications and dashboards.

#### `vehicle_counts` Table

Stores aggregated vehicle count data per frame or time window.

| Column         | Type                | Description                                                     |
| -------------- | ------------------- | --------------------------------------------------------------- |
| `id`           | INTEGER PRIMARY KEY | Auto-incrementing unique identifier                             |
| `ts`           | INTEGER             | Unix timestamp in milliseconds (auto-generated if not provided) |
| `camera_id`    | TEXT                | Camera identifier                                               |
| `total_count`  | INTEGER             | Total vehicle count                                             |
| `class_counts` | TEXT                | JSON object with counts per vehicle class                       |

**Indices**:

- `idx_plate_cam_time` on `plate_results(camera_id, ts)` for history queries
- `idx_counts_cam_time` on `vehicle_counts(camera_id, ts)` for time-range queries
- `idx_plate_vehicle_class` on `plate_results(vehicle_class, ts)` for class-based filtering
- `idx_latest_vehicle_class` on `plate_results_latest(vehicle_class)` for current state queries

## Confidence-Based Update Logic

The system implements a **"best confidence wins"** strategy:

1. **ALL** OCR attempts are stored in `plate_results` (history table)
2. **ONLY** the highest confidence result is stored/updated in `plate_results_latest`
3. Lower confidence readings are **ignored** for updates but preserved in history
4. Vehicle class and other fields are updated along with improved confidence readings

### Update Behavior Example

```
Vehicle ID 42 OCR attempts:
1. "ABC123" (conf: 0.75) → Latest table: "ABC123" (0.75)
2. "ABG123" (conf: 0.65) → Latest table: "ABC123" (0.75) [unchanged - lower confidence]
3. "ABC123" (conf: 0.95) → Latest table: "ABC123" (0.95) [updated - higher confidence]
4. "ABC12X" (conf: 0.80) → Latest table: "ABC123" (0.95) [unchanged - lower confidence]
```

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
- Stores `reset_on_startup` flag
- Called before `init_db()` at application startup

**`init_db() -> None`**

- **Resets database if `reset_on_startup` is true**: Deletes the database file and associated WAL/SHM files.
- Initializes database schema and indices
- Applies pragma settings from configuration
- Called once at application startup after configuration

**`write_plate_result(camera_id, vehicle_id, vehicle_class, lp_text, ocr_conf, ts=None)`**

- Stores license plate detection results in BOTH history and latest tables
- History table: ALL OCR attempts preserved for audit trail
- Latest table: UPSERT with "best confidence wins" logic
- Parameters: camera_id, vehicle_id, vehicle_class, lp_text, ocr_conf, ts (optional)
- Automatic timestamp generation if not provided
- **Confidence Logic**: Only updates latest table if new confidence > existing confidence

**`get_latest_plate_result(camera_id, vehicle_id)`**

- Retrieves the authoritative (highest confidence) result for a specific vehicle
- Returns dict with all fields or None if not found
- Fast lookup using PRIMARY KEY (camera_id, vehicle_id)

**`get_all_latest_plates(camera_id=None)`**

- Retrieves all authoritative plate results, optionally filtered by camera
- Returns list of dicts with latest results only
- Optimized for dashboard and real-time display use cases

**`write_vehicle_count(camera_id, total_count, class_counts, ts=None)`**

- Stores aggregated vehicle count data
- Parameters: camera_id, total_count, class_counts (dict), ts (optional)
- class_counts stored as JSON for flexible querying

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

  # Option to reset the database file on every application startup
  reset_on_startup: false

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
    vehicle_class="car",
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
    vehicle_class=vehicle_class,
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

#### With Vehicle Tracker

```python
# In tracking service - store result with confidence evaluation
def store_detection_result(camera_id, track_id, vehicle_class, plate_text, confidence):
    write_plate_result(
        camera_id=camera_id,
        vehicle_id=track_id,
        vehicle_class=vehicle_class,
        lp_text=plate_text,
        ocr_conf=confidence
    )

    # Get current best result for this vehicle
    latest = get_latest_plate_result(camera_id, track_id)
    if latest and latest['ocr_conf'] >= MIN_CONFIDENCE_THRESHOLD:
        # Use authoritative result for display/alerts
        display_plate = latest['lp_text']
```

#### With Analytics/Dashboard

```python
# Get current vehicle inventory
current_vehicles = get_all_latest_plates("cam-01")

# Show only high-confidence results
reliable_detections = [
    v for v in current_vehicles
    if v['ocr_conf'] >= 0.90
]
```

## Query Examples

### Current State Queries (Latest Table)

```sql
-- Dashboard: Current vehicles with best plates
SELECT vehicle_id, lp_text, ocr_conf, vehicle_class,
       datetime(first_seen/1000, 'unixepoch') as first_seen,
       datetime(last_updated/1000, 'unixepoch') as last_updated
FROM plate_results_latest
WHERE camera_id = 'cam-001'
ORDER BY last_updated DESC;

-- High confidence detections only
SELECT * FROM plate_results_latest WHERE ocr_conf >= 0.90;

-- Count by vehicle class (current state)
SELECT vehicle_class, COUNT(*) as vehicle_count, AVG(ocr_conf) as avg_confidence
FROM plate_results_latest
GROUP BY vehicle_class;
```

### Historical Analysis (History Table)

```sql
-- Audit trail for specific vehicle
SELECT lp_text, ocr_conf, datetime(ts/1000, 'unixepoch') as timestamp
FROM plate_results
WHERE camera_id = 'cam-001' AND vehicle_id = 42
ORDER BY ts;

-- OCR performance analysis
SELECT
    vehicle_id,
    COUNT(*) as total_attempts,
    MIN(ocr_conf) as worst_conf,
    MAX(ocr_conf) as best_conf,
    AVG(ocr_conf) as avg_conf
FROM plate_results
WHERE camera_id = 'cam-001'
GROUP BY vehicle_id
HAVING COUNT(*) > 1;

-- Time-based detection patterns
SELECT
    DATE(datetime(ts/1000, 'unixepoch')) as date,
    vehicle_class,
    COUNT(*) as detections
FROM plate_results
GROUP BY date, vehicle_class
ORDER BY date DESC;
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
