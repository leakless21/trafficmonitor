# Plate-OCR & Visualisation Data-Fusion Plan (Option B)

> Goal: every licence-plate recognised by the OCR pipeline **must** appear on the corresponding vehicle in the video frame, with zero additional latency and no message loss.

---

## 1  Current Flow (simplified)

```
FrameCapture → Detection → Tracking ─┬→ LicencePlateDetection → OCR → +--------------+
                                     │                                   │
                                     └──────── VehicleCounting ───────────┤ Visualization
                                                                           +--------------+
```

* `Tracking` produces `TrackedVehicleMessage` (TVM) for each frame and sends it directly to Visualisation.
* OCR produces a separate `OCRResultMessage` which arrives **later** via its own queue.
* Visualisation maintains a `latest_ocr_results` dict and merges the late data in best-effort fashion — plates may be missed if the vehicle quits the frame quickly or messages drop.

---

## 2  Data-Fusion Architecture

```
FrameCapture
   ↓
Detection
   ↓
Tracking   ──┐
            │   (same TVM, enriched)
LicencePlateDetection
            │
OCR (TextRecognition)
            │
┌────────────────────┐
│  Event Fusion Svc  │  ← NEW
└────────────────────┘
            ↓
Visualization (single input queue)
```

Key idea: **every stage appends its result to the *same* message object** before it reaches Visualisation:

* `TrackedVehicleMessage` will be upgraded to include optional fields:
  ```yaml
  tracked_objects:
    - track_id: 42
      class_name: car
      bbox_xyxy: [x1,y1,x2,y2]
      plate_bbox_xyxy: [..]        # added by LicencePlateDetection
      plate_text: "30F12345"       # added by OCR
      plate_confidence: 0.91
  ```

* A lightweight *Event Fusion Service* sits between producers and Visualisation.  It maintains a dictionary keyed by `(frame_id, track_id)` and merges partial updates into a single TVM.
* When a complete (or sufficiently updated) message is ready, it is forwarded to Visualisation.

Benefits
- Zero extra queues • No sync latency • Single source of truth • Less bookkeeping in Visualiser

---

## 3  Implementation Steps

### 3.1  Define the Upgraded Message Schema
1. Create `utils/custom_types.py → EnrichedTrackedVehicleMessage` (or extend existing TVM) with optional plate fields.
2. Update all type hints & docs.

### 3.1.1  Progress Flags (fast-path optimisation)
Add two boolean fields that every downstream stage can read without extra image processing:
```yaml
plate_detected:    bool  # Licence-plate detector found a plausible bbox on this vehicle
plate_text_read:   bool  # OCR successfully returned text above conf threshold
```

* Default values: `false` (set by Tracking).
* `LicencePlateDetection` sets `plate_detected = true` when it appends `plate_bbox_xyxy`.
* `TextRecognition` sets `plate_text_read = true` when it appends `plate_text`.
* Fusion flips them to final state when flushing.

Benefits
1. **Early short-circuit** – If `plate_detected = false` after detection stage, OCR service can skip entirely for that object.
2. **Simpler analytics** – Counting missing plates is one aggregation over `plate_text_read`.
3. **UI clarity** – Visualiser can colour-code bbox: red = no plate, yellow = plate found but unread, green = read.

### 3.2  Build **Event Fusion Service**
1. Inputs: `vehicle_tracking_output_queue`, `license_plate_detection_output_queue`, `text_recognition_output_queue`, `vehicle_counting_output_queue`.
2. Internal state:
   ```python
   state: dict[tuple[int,int], dict]  # (frame_id, track_id) → partial obj
   frame_buffer: dict[int, list]      # frame_id → list of obj
   ttl_sec = 1.0                      # configurable flush timeout
   ```
3. On each incoming message:
   - Merge fields into `state[(fid, tid)]`.
   - Move object into `frame_buffer[fid]`.
4. Flush strategy:
   - When *all* expected producers have contributed **or** TTL expires, assemble the complete frame message and push to `visualization_input_queue`.
   - Clean up old state.
5. Provide metrics: merges/sec, flush cause, dropped.

### 3.3  Remove Direct Lines to Visualiser
- Change `main_supervisor.py` wiring so **only Event Fusion Service feeds Visualisation**.
- Delete redundant *vis* copies of OCR/Count queues.

### 3.4  Adapt VisualizationService
- Stop reading OCR & count queues.
- Expect plate data inside `tracked_objects` directly.
- Retain fallback (if plate not present) for regression phase.

### 3.5  Queue Configuration
- All producers keep their current queue sizes.
- Fusion → Visualisation queue may stay small (3-5) for real-time, unlimited for offline.

### 3.6  Testing
1. **Unit tests** for Fusion logic:
   - Merge scenarios, TTL expiry, partial data.
2. **Integration test** (pytest + cv2):
   - Feed synthetic pipeline, assert plate text appears within ≤2 frames.
   - Capture metrics (100 plates in, 100 plates drawn).

### 3.7  Migration & Roll-out
1. Feature-flag the new path: `config.visualizer.use_data_fusion: true`.
2. Deploy on dev instance, check latency & CPU.
3. Remove old path after validation.

---

## 4  Timeline (1 FTE)
| Week | Task |
|------|------|
| 1 | Schema definition, Fusion service skeleton, supervisor wiring |
| 2 | Full merge logic, unit tests, basic visualiser changes |
| 3 | Integration tests, logging/metrics, doc updates |
| 4 | Field testing, performance tuning, remove legacy queues |

---

## 5  Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Fusion adds latency | Medium | Keep TTL ≤1 s, monitor queue sizes |
| Partial data never flushes (bug) | Missing overlays | TTL flush + watchdog log |
| Schema mismatch across services | Crash | Version field in message, CI tests |

---

## 6  Deliverables
- `event_fusion_service.py` + unit tests
- Updated message schemas & utils
- Updated supervisor wiring
- Simplified `VisualizationService`
- Documentation (this file + README changes)
- CI green ✅ 

---

## 7  Edge Cases & Corner-Case Handling

| # | Scenario | Consequence | Planned Handling |
|---|-----------|-------------|------------------|
| 1 | **Out-of-order arrival** – OCR/LP messages land before their base `Tracking` object | Fusion cannot find merge target → data dropped | Keep a *pending* dict keyed by `(frame_id, track_id)` for every producer; if target missing, create a stub and wait up to `ttl_sec` before flushing |
| 2 | **Missing producer** – A frame never receives an OCR or LP update (e.g.
  small plate, detection miss) | Object waits indefinitely, memory leak | Flush partial object when `ttl_sec` expires; mark `plate_text = null` so visualiser can decide how to draw |
| 3 | **Vehicle exits quickly** – Track disappears before OCR ready | Plate would be lost | Visualiser keeps last text for 2 s; Fusion forces flush on `track_lost` event if tracker emits one |
| 4 | **Tracker ID reuse after reset** | Plates may attach to wrong vehicle | Embed `tracker_session_id` (monotonic counter) in messages; Fusion drops state when new session starts |
| 5 | **Duplicate / conflicting OCR results** | Wrong text shown | Fusion stores *best-confidence so far*; keep full list for audit; expose via metrics |
| 6 | **Multiple plates per vehicle (truck + trailer)** | Need to show all texts | `plate_text` becomes list; Visualiser concatenates or draws multiple labels |
| 7 | **Gaps in `frame_id` sequence** (dropped frames) | Late flush of buffer | Fusion drops frame-buffer entry when `frame_id` gap > `max_gap` (configurable) |
| 8 | **Very high FPS / burst traffic** | Memory bloat & latency | Fusion back-pressure: if buffer size > `N`, start discarding oldest *completed* frames and warn |
| 9 | **Producer crash & restart** | No new messages → stale buffer | Supervisor restarts producer; Fusion drops state older than 5 × `ttl_sec`; health metrics trigger alert |
|10 | **Edge crops / invalid bboxes** from LP detector | OCR fails / crash | Validate bbox inside image shape; if invalid, discard gracefully |
|11 | **Unicode / weird plate chars** | Text render error | Strip non-alnum in Fusion; fallback font supports wide glyphs |
|12 | **Clock skew between processes** | TTL mis-calculations | Use *relative* monotonic time (`time.monotonic()`) inside Fusion |
|13 | **No plate detected / OCR returns empty or low-confidence** | No text label, potential confusion in UI | After TTL expiry, Fusion sets `plate_text = "UNKNOWN"` and Visualiser draws a grey label; counting & analytics flag as `plate_missing=true` |

> Edge-case logic will be unit-tested with synthetic messages that explicitly trigger each scenario. 