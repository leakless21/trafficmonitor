# Gap Analysis

## Issues Found and Fixed

### ✅ COMPLETED: Counting Lines Not Working with Resolution Changes

**Issue Description**:
Vehicle counter and visualizer counting lines did not work correctly when the frame resolution differed from the original video resolution. The counting lines were defined in absolute coordinates for the original resolution but were not properly scaled when frames were resized.

**Root Cause**:

1. Counting lines were configured in absolute coordinates (e.g., `[0, 750], [1920, 750]` for 1920x1080 video)
2. When frame grabber resized frames to different resolution (e.g., 1280x720), the counting lines remained at original coordinates
3. Vehicle counter's normalization logic had edge cases that didn't handle coordinate conversion properly
4. Visualizer used counting line coordinates directly without scaling
5. Empty counting lines list caused IndexError

**Files Modified**:

- `src/traffic_monitor/services/vehicle_counter.py`
- `src/traffic_monitor/services/visualizer.py`
- `src/traffic_monitor/utils/custom_types.py`
- `src/traffic_monitor/main_supervisor.py`
- `src/traffic_monitor/config/settings.yaml` (updated with examples)

**Solution Implemented**:

1. **Vehicle Counter**: Enhanced `_normalize_counting_lines()` method to properly detect and convert absolute coordinates to relative coordinates, then scale them back to current frame dimensions
2. **Visualizer**: Added `_normalize_counting_line()` method and updated `process_frame()` to handle dynamic scaling of counting lines based on current and original frame dimensions
3. **Added comprehensive logging** to track the normalization and scaling process
4. **Added edge case handling** for empty counting lines lists
5. **Created comprehensive test suite** to verify the fix works correctly

**Tests Created**:

- `test/test_resolution_scaling.py` with 4 test cases covering:
  - Vehicle counter with absolute coordinates
  - Vehicle counter with relative coordinates
  - Visualizer receiving counting line information from VehicleCounter
  - Edge cases (empty counting lines)

**Verification**:

- ✅ All tests pass
- ✅ Coordinate normalization works for both absolute and relative coordinates
- ✅ Resolution scaling works correctly (1920x1080 → 1280x720)
- ✅ Edge cases handled gracefully
- ✅ Comprehensive logging added for debugging

**Status**: ✅ **COMPLETED** - Fix implemented, tested, and verified

---

### ✅ COMPLETED: Eliminated Configuration Duplication for Counting Lines

**Issue Description**:
User had to define counting lines twice in `settings.yaml` - once for vehicle_counter and once for visualizer, leading to configuration duplication and potential inconsistency.

**Root Cause**:
Poor architecture where both vehicle counter and visualizer had separate counting line configurations without coordination.

**Solution Implemented**:
**New Architecture**: **Vehicle Counter → Visualizer** data flow

1. **Central Configuration**: Define counting lines once in `counting_lines` section
2. **Single Source of Truth**: Vehicle Counter normalizes and manages all counting line geometry
3. **Direct Communication**: Vehicle Counter sends counting line info to Visualizer via `VehicleCountMessage`
4. **Guaranteed Consistency**: What you see is exactly what's being used for counting

**Files Modified**:

- `src/traffic_monitor/utils/custom_types.py` - Extended `VehicleCountMessage` with counting line info
- `src/traffic_monitor/services/vehicle_counter.py` - Now includes counting line coordinates in output
- `src/traffic_monitor/services/visualizer.py` - Receives counting lines from VehicleCounter instead of config
- `src/traffic_monitor/main_supervisor.py` - Passes centralized config to VehicleCounter
- `src/traffic_monitor/config/settings.yaml` - Centralized counting line configuration

**Benefits**:

- ✅ **No more duplication** - Define counting lines once
- ✅ **Guaranteed consistency** - Visualizer shows exactly what Counter uses
- ✅ **Better architecture** - Clear data flow and separation of concerns
- ✅ **Easier maintenance** - Single place to configure counting lines

**Configuration Example**:

```yaml
# Before: Had to define twice
vehicle_counter:
  counting_lines: [[[0, 750], [1920, 750]]]
visualizer:
  counting_line: [[0, 750], [1920, 750]]

# After: Define once
counting_lines:
  lines: [[[0, 750], [1920, 750]]]
  display_color: [0, 0, 255]
  line_thickness: 2
```

**Status**: ✅ **COMPLETED** - Cleaner architecture implemented and tested

---

### ✅ FIXED: VehicleCounter Configuration Bug After Architecture Change

**Issue Description**:
After implementing the new architecture, VehicleCounter process failed to start with error "No counting lines configured" because the centralized configuration passing was accidentally removed.

**Root Cause**:
The main supervisor wasn't passing the centralized counting lines configuration to the VehicleCounter process, causing it to receive an empty configuration.

**Log Evidence**:

```
2025-06-19 02:24:11.635 | ERROR | VehicleCounter | [VehicleCounter] No counting lines configured
VehicleCounter config: {'enabled': True, 'service_name': 'VehicleCounter'}
```

**Solution Implemented**:
Restored the centralized configuration passing in `main_supervisor.py`:

```python
# Pass centralized counting lines configuration to vehicle counter
central_counting_lines = config.get("counting_lines", {})
vc_config["counting_lines"] = central_counting_lines.get("lines", [])
vc_config["display_color"] = central_counting_lines.get("display_color", [0, 0, 255])
vc_config["line_thickness"] = central_counting_lines.get("line_thickness", 2)
```

**Verification**:

- ✅ VehicleCounter now receives proper configuration
- ✅ Counter initializes successfully with counting lines
- ✅ All existing tests still pass
- ✅ Integration test confirms configuration loading works

**Status**: ✅ **FIXED** - VehicleCounter starts correctly with centralized configuration

---

## Current Outstanding Issues

_No current outstanding issues_

## Testing Completed

- [x] Test counting lines with different resolutions (original vs resized)
- [x] Test with relative coordinate configuration (floats 0.0-1.0)
- [x] Test with absolute coordinate configuration (integers)
- [x] Verify counting accuracy across resolution changes
- [x] Test edge cases (empty counting lines)
- [x] Verify visualizer displays counting lines in correct position
- [x] Test new architecture with VehicleCounter → Visualizer data flow
- [x] Verify configuration duplication elimination
- [x] Verify VehicleCounter configuration loading after architecture changes
