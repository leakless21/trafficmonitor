# Vehicle Tracker Component Documentation

## Overview

The Vehicle Tracker component is responsible for tracking vehicles across frames using BoxMOT (Multi-Object Tracking) library integration. It maintains consistent track IDs for vehicles as they move through the scene and provides counting functionality when vehicles cross predefined counting lines.

## Core Components

### Multi-Object Tracking

**Purpose**: Track detected vehicles across multiple frames, maintaining consistent identifiers.

**Location**: `src/traffic_monitor/services/vehicle_tracker.py`

**Key Features**:

- **Tracker Integration**: Uses BoxMOT library with configurable tracker algorithms (ByteTrack, OC-SORT, DeepOC-SORT, etc.)
- **Track ID Persistence**: Maintains consistent track IDs across frames, handling temporary occlusions
- **Multi-Class Support**: Tracks different vehicle types (cars, trucks, buses, motorcycles, etc.)
- **Real-time Processing**: Optimized for real-time video processing with configurable performance settings

**Configuration**: Tracker settings are defined in `src/traffic_monitor/config/settings.yaml` under the `vehicle_tracker` section:

```yaml
vehicle_tracker:
  tracker_type: "ocsort" # Tracking algorithm
  half: true # Use half precision for performance
  device: "cpu" # Processing device
  reid_model_path: None # Optional re-identification model
  evolve_param_dict: None # Custom tracker parameters
  per_class: None # Per-class tracking settings
```

## Vehicle Counting

### Line-based Counting Logic

**Purpose**: Count vehicles that cross predefined counting lines using relative coordinates.

**Location**: `src/traffic_monitor/services/vehicle_counter.py`

**Key Features**:

- **Relative Coordinates**: Uses relative coordinates (0.0-1.0) for resolution-independent line placement
- **Line Intersection Detection**: Uses Shapely geometry library for precise line-path intersection calculations
- **Multi-Line Support**: Supports multiple counting lines simultaneously
- **Class-Specific Counting**: Tracks counts by vehicle class (cars, trucks, buses, etc.)
- **Duplicate Prevention**: Prevents double-counting of the same vehicle crossing multiple lines

**Configuration**: Counting line coordinates are defined in `src/traffic_monitor/config/settings.yaml` under the `vehicle_counter` section. The format uses relative coordinates (0.0 to 1.0) where:

- 0.0 = left edge (x) or top edge (y)
- 1.0 = right edge (x) or bottom edge (y)

Each line is represented by two points: `[[x1, y1], [x2, y2]]`

Example:

```yaml
vehicle_counter:
  counting_lines:
    - # Horizontal line at 69.4% height spanning full width
      - [0.0, 0.694]
      - [1.0, 0.694]
    - # Vertical line at center, from 40% to 70% height
      - [0.5, 0.4]
      - [0.5, 0.7]
```

**Benefits of Relative Coordinates**:

- **Resolution Independent**: Same configuration works for any video resolution
- **Intuitive Configuration**: Easy to understand percentages vs pixel values
- **Automatic Scaling**: Lines maintain proportional placement across different cameras
- **No Reconfiguration**: Works with 480p, 720p, 1080p, 4K without changes

## Counting Line Visualization

**Purpose**: Display counting lines on the video feed to provide visual feedback about where vehicle counting occurs.

**Location**: `src/traffic_monitor/services/visualizer.py`

**Configuration**: Counting line visualization is configured in the `visualizer` section of `settings.yaml`:

```yaml
visualizer:
  # Counting line visualization settings
  counting_lines: # Copy from vehicle_counter for visualization
    - # Horizontal line at 69.4% height spanning full width
      - [0.0, 0.694]
      - [1.0, 0.694]
  counting_line_color: [0, 255, 255] # Yellow in BGR format
  counting_line_thickness: 3
```

**Features**:

- **Relative Coordinate Support**: Automatically converts relative coordinates to absolute pixels for drawing
- **Line Drawing**: Each counting line is drawn as a colored line across the video frame
- **Line Labels**: Each line is labeled with "Count Line X" where X is the line number
- **Customizable Appearance**: Line color and thickness can be configured
- **Multiple Lines**: Supports multiple counting lines simultaneously
- **Real-time Display**: Lines are drawn on every frame and scale automatically to frame dimensions

**Key Methods**:

- `_draw_counting_lines(image, frame_width, frame_height)`: Draws all configured counting lines on the frame with automatic coordinate conversion
- Color parsing supports BGR list format `[B, G, R]` and string format `"(B, G, R)"`

**Integration**: The visualizer automatically displays counting lines if they are configured, providing immediate visual feedback about the counting zones that adapt to any video resolution.

## Coordinate System Migration

The system has been updated to use relative coordinates (0.0-1.0) instead of absolute pixel coordinates for improved flexibility and maintainability.

**Migration Benefits**:

- **Camera Independence**: Same configuration works across different camera resolutions
- **Future Proof**: No need to reconfigure when upgrading cameras or changing resolutions
- **Easier Setup**: Intuitive percentage-based positioning instead of pixel counting
- **Consistent Behavior**: Lines maintain the same relative position regardless of video size

**Backward Compatibility**: Existing absolute coordinates can be converted to relative coordinates using:

- `relative_x = absolute_x / frame_width`
- `relative_y = absolute_y / frame_height`

## Related Classes and Files

- **Counter Class**: `src/traffic_monitor/services/vehicle_counter.py` - Core counting logic with relative coordinate support
- **Visualizer Class**: `src/traffic_monitor/services/visualizer.py` - Line visualization with automatic coordinate conversion
- **Coordinate Utilities**: `src/traffic_monitor/utils/utils.py` - Helper functions for coordinate conversion
- **Configuration**: `src/traffic_monitor/config/settings.yaml` - Relative coordinate line definitions
- **Tests**: `test/services/test_vehicle_counter.py`, `test/services/test_visualizer_counting_lines.py` - Comprehensive testing of relative coordinate functionality
