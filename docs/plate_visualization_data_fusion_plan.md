# Plate-OCR & Visualisation Data-Fusion Plan (Option B)

> Goal: every licence-plate recognised by the OCR pipeline **must** appear on the corresponding vehicle in the video frame, with zero additional latency and no message loss.
Use UV for venv
## Executive Summary

This plan addresses the critical synchronization issue where OCR results arrive separately from vehicle tracking data, causing missed plate visualizations. The proposed Event Fusion Service centralizes data merging with comprehensive error handling, performance monitoring, and graceful degradation capabilities.

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

#### 3.2.1 Core Architecture
1. **Inputs**: `vehicle_tracking_output_queue`, `license_plate_detection_output_queue`, `text_recognition_output_queue`, `vehicle_counting_output_queue`.
2. **Internal state with memory management**:
   ```python
   state: dict[tuple[int,int], dict]  # (frame_id, track_id) → partial obj
   frame_buffer: dict[int, list]      # frame_id → list of obj
   pending_updates: dict[tuple[int,int], list]  # for out-of-order messages
   
   # Configuration
   ttl_sec = 1.0                      # configurable flush timeout
   max_buffer_size = 1000             # prevent memory bloat
   max_state_age_sec = 5.0            # aggressive cleanup threshold
   max_frame_gap = 10                 # drop frames with large gaps
   ```

#### 3.2.2 Enhanced Processing Logic
3. **Message Processing Pipeline**:
   - **Validation**: Check message integrity, bbox bounds, frame_id sequence
   - **Deduplication**: Handle duplicate messages from producer restarts
   - **Merge**: Update `state[(fid, tid)]` with new fields
   - **Buffer Management**: Move to `frame_buffer[fid]` when ready
   - **Backpressure**: Drop oldest completed frames if buffer exceeds limits

4. **Intelligent Flush Strategy**:
   - **Complete**: All expected producers contributed
   - **TTL Expiry**: Timeout reached, flush partial data
   - **Track Lost**: Vehicle exited frame (if tracker provides this event)
   - **Memory Pressure**: Buffer size exceeded, force flush oldest
   - **Graceful Degradation**: Continue with partial data rather than blocking

#### 3.2.3 Performance & Monitoring
5. **Comprehensive Metrics**:
   ```python
   # Throughput metrics
   messages_processed_per_sec: float
   frames_flushed_per_sec: float
   
   # Quality metrics  
   complete_merges_ratio: float       # all producers contributed
   partial_flushes_ratio: float       # TTL/pressure triggered
   dropped_messages_count: int        # validation failures
   
   # Performance metrics
   avg_merge_latency_ms: float
   buffer_size_current: int
   state_dict_size_current: int
   memory_usage_mb: float
   
   # Error metrics
   out_of_order_messages: int
   producer_timeouts: int
   validation_failures: int
   ```

#### 3.2.4 Circuit Breaker Pattern
6. **Producer Health Monitoring**:
   ```python
   producer_health = {
       'tracking': CircuitBreaker(failure_threshold=5, timeout=30),
       'plate_detection': CircuitBreaker(failure_threshold=3, timeout=20),
       'ocr': CircuitBreaker(failure_threshold=3, timeout=20),
       'counting': CircuitBreaker(failure_threshold=5, timeout=30)
   }
   ```
   - Monitor message arrival rates per producer
   - Open circuit on sustained failures
   - Provide degraded service (skip failed producer data)
   - Auto-recovery with exponential backoff

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

### 3.6  Enhanced Testing Strategy

#### 3.6.1 Unit Tests
1. **Core Fusion Logic**:
   - Message merging scenarios (complete, partial, out-of-order)
   - TTL expiry handling with various timing scenarios
   - Memory management (buffer overflow, cleanup)
   - Circuit breaker state transitions
   - Edge case handling (all 13 scenarios from Section 7)

2. **Performance Tests**:
   ```python
   def test_high_throughput_scenario():
       # 1000 vehicles/frame, 30 FPS for 10 seconds
       # Assert: latency < 50ms, memory < 500MB
   
   def test_memory_pressure():
       # Simulate slow consumer, verify backpressure
       # Assert: buffer size stays under limits
   
   def test_producer_failure_recovery():
       # Kill OCR producer, verify graceful degradation
       # Assert: tracking continues, plates marked as missing
   ```

#### 3.6.2 Integration Tests
3. **End-to-End Pipeline**:
   - **Synthetic Data Test**: Feed known plate images, verify 100% accuracy
   - **Real Video Test**: Process sample videos, measure plate detection rate
   - **Latency Test**: Measure frame-to-visualization delay (target: <100ms)
   - **Stress Test**: High FPS scenarios (60+ FPS) with memory monitoring

4. **Failure Injection Tests**:
   - Network partitions between services
   - Producer crashes and restarts
   - Message corruption scenarios
   - Queue overflow conditions

#### 3.6.3 Performance Benchmarking
5. **Baseline Measurements**:
   ```python
   # Current system metrics to beat/maintain
   baseline_metrics = {
       'frame_processing_latency_ms': 'TBD',
       'memory_usage_mb': 'TBD', 
       'cpu_usage_percent': 'TBD',
       'plate_detection_accuracy': 'TBD'
   }
   ```

6. **Regression Testing**:
   - Automated performance tests in CI/CD
   - Alert on >10% performance degradation
   - Memory leak detection over 24h runs

### 3.7  Phased Migration & Roll-out

#### 3.7.1 Implementation Phases
1. **Phase 1: Minimal Viable Product (Week 1-2)**
   - Core fusion service with basic merging
   - Feature flag: `config.fusion.enabled: false` (default)
   - Parallel deployment (old + new paths running)
   - Basic monitoring and logging

2. **Phase 2: Enhanced Features (Week 3)**
   - Circuit breaker implementation
   - Comprehensive edge case handling
   - Performance optimizations
   - Advanced metrics collection

3. **Phase 3: Production Readiness (Week 4)**
   - Load testing and performance tuning
   - Documentation and runbooks
   - Monitoring dashboards
   - Gradual traffic migration

---

## 4  Enhanced Timeline (1 FTE)

### 4.1 Detailed Weekly Breakdown
| Week | Phase | Key Deliverables | Success Criteria |
|------|-------|------------------|------------------|
| **1** | **Foundation** | • Enhanced message schema<br>• Fusion service skeleton<br>• Basic unit tests<br>• Supervisor wiring | • Schema validates all message types<br>• Service starts without errors<br>• 80% unit test coverage |
| **2** | **Core Logic** | • Complete merge logic<br>• Memory management<br>• TTL handling<br>• Basic visualizer changes | • Handles all merge scenarios<br>• Memory usage stable<br>• Visualizer displays fused data |
| **3** | **Resilience** | • Circuit breaker implementation<br>• Edge case handling<br>• Comprehensive monitoring<br>• Integration tests | • Graceful failure handling<br>• All 13 edge cases covered<br>• Performance within targets |
| **4** | **Production** | • Load testing<br>• Performance optimization<br>• Documentation<br>• Canary deployment | • Handles production load<br>• <100ms latency maintained<br>• Zero data loss verified |

### 4.2 Risk-Adjusted Timeline
**Conservative Estimate**: 5-6 weeks (includes 25% buffer for unforeseen issues)

**Critical Path Dependencies**:
1. Message schema changes (affects all downstream services)
2. Fusion service core logic (blocks integration testing)
3. Performance validation (blocks production deployment)

### 4.3 Parallel Work Opportunities
- **Week 1-2**: Documentation can be written in parallel with development
- **Week 2-3**: Integration test setup while core logic is being finalized
- **Week 3-4**: Performance tuning can overlap with edge case implementation

---

## 5  Risks & Mitigations
| Risk | Impact | Mitigation |
|------|--------|-----------|
| Fusion adds latency | Medium | Keep TTL ≤1 s, monitor queue sizes |
| Partial data never flushes (bug) | Missing overlays | TTL flush + watchdog log |
| Schema mismatch across services | Crash | Version field in message, CI tests |

---

## 6  Enhanced Deliverables

### 6.1 Core Implementation
- **`event_fusion_service.py`** + comprehensive unit tests (>90% coverage)
- **Enhanced message schemas** (`EnrichedTrackedVehicleMessage`) with backward compatibility
- **Updated supervisor wiring** with feature flag support
- **Simplified `VisualizationService`** with fallback mechanisms
- **Circuit breaker utilities** for producer health monitoring

### 6.2 Infrastructure & Monitoring
- **Performance monitoring dashboard** (Grafana/Prometheus integration)
- **Automated deployment scripts** with rollback capabilities
- **Load testing suite** for performance validation
- **Health check endpoints** for all services
- **Alerting rules** for SLA violations and system health

### 6.3 Documentation & Operations
- **Updated technical documentation** (this file + README changes)
- **Operational runbooks** for troubleshooting and maintenance
- **Performance benchmarking reports** (before/after comparison)
- **Migration guide** for production deployment
- **CI/CD pipeline enhancements** with automated testing
- **Training materials** for operations team

### 6.4 Quality Assurance
- **Comprehensive test suite** (unit, integration, performance, chaos)
- **Code review checklist** specific to fusion service
- **Performance regression tests** in CI pipeline
- **Security review** of message handling and data flow
- **Disaster recovery procedures** and testing 

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

---

## 8  Success Metrics & KPIs

### 8.1 Performance Metrics
| Metric | Current Baseline | Target | Measurement Method |
|--------|------------------|--------|-------------------|
| **Frame Processing Latency** | TBD | <100ms (p95) | End-to-end timing from frame capture to visualization |
| **Plate Detection Accuracy** | TBD | >95% | Ratio of detected plates to actual plates in test videos |
| **Memory Usage** | TBD | <500MB steady state | Process memory monitoring over 24h |
| **CPU Utilization** | TBD | <70% average | System resource monitoring |
| **Message Loss Rate** | TBD | <0.1% | Ratio of lost messages to total messages |

### 8.2 Quality Metrics
| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Plate Visualization Completeness** | >99% | Plates detected by OCR that appear in visualization |
| **Data Synchronization Accuracy** | 100% | Correct plate text matched to correct vehicle |
| **System Availability** | >99.9% | Uptime monitoring of fusion service |
| **Error Recovery Time** | <30 seconds | Time to recover from producer failures |

### 8.3 Business Impact Metrics
| Metric | Target | Business Value |
|--------|--------|----------------|
| **Operational Efficiency** | 25% reduction in manual verification | Reduced human oversight needed |
| **Data Quality** | 90% reduction in missed plates | Improved analytics and reporting |
| **System Reliability** | 50% reduction in sync-related issues | Better user experience |
| **Maintenance Overhead** | 30% reduction in troubleshooting time | Lower operational costs |

---

## 9  Conclusion & Next Steps

This enhanced plan provides a robust foundation for implementing the Event Fusion Service with comprehensive risk mitigation, performance monitoring, and quality assurance. The phased approach ensures minimal disruption while delivering measurable improvements to the traffic monitoring system.

### 9.1 Immediate Actions Required
1. **Stakeholder approval** of the enhanced plan and timeline
2. **Resource allocation** confirmation for the 4-6 week implementation
3. **Baseline performance measurement** of the current system
4. **Development environment setup** for parallel implementation

### 9.2 Long-term Considerations
- **Scalability planning** for higher traffic volumes
- **Integration opportunities** with other system components
- **Performance optimization** based on production data
- **Feature enhancements** based on user feedback

The success of this implementation will establish a pattern for future data fusion requirements and demonstrate the value of systematic approach to complex system integration challenges. 