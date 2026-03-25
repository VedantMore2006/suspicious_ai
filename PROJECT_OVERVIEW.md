# Suspicious Behavior Detection System

## Executive Summary

A real-time behavioral intelligence system that analyzes live video streams to detect potentially risky behavioral patterns before incidents escalate. Rather than treating surveillance as simple object detection, this prototype evaluates **motion dynamics, proximity patterns, and interaction sequences** to generate contextual, interpretable alerts.

The system demonstrates that behavioral analysis is fundamentally different from object classification—behavior unfolds over time and requires tracking state, velocity, acceleration, and interaction context.

---

## Project Intent & Motivation

### The Problem
Most computer vision systems focus on **"what is happening in a frame"**—detecting objects or recognizing static attributes. But **suspicious behavior is temporal**. It emerges through motion patterns, sustained proximity, velocity changes, and interaction dynamics that evolve over multiple frames.

### The Approach
Instead of a black-box behavior classifier, this prototype uses **interpretable motion signals** to reason about risk:
- **Proximity**: Are people unusually close?
- **Velocity**: Are movements rapid or erratic?
- **Acceleration**: Are velocities changing suddenly?
- **Duration**: How long has the suspicious pattern persisted?
- **Interaction**: Are multiple people involved in close contact?

This makes the system **explainable and debuggable**—operators understand *why* an alert was triggered, not just *that* an alert fired.

---

## Technology Stack & Reasoning

### Core Computer Vision: **Ultralytics YOLOv8 (nano)**

**Why YOLOv8?**
- **Real-time capable**: Nano variant runs at acceptable FPS on CPU/GPU
- **Built-in tracking**: YOLOv8 includes ByteTrack multi-object tracking out of the box
- **COCO pretrained**: Recognizes persons, bags, phones—critical for behavior analysis
- **Active development**: Modern architecture with good inference speed/accuracy tradeoff
- **Model fusion**: Ultralytics provides `.fuse()` to optimize inference

**Trade-offs**:
- Nano model trades accuracy for speed; detection confidence set to 0.2 (permissive) to catch edge cases
- Fixed input size (416×416) processed on 640×480 frames

**Detection targets** (COCO class indices):
- `PERSON (0)`: Core actor for behavior analysis
- `BACKPACK (24)`, `HANDBAG (26)`: For abandoned object detection
- `CELL_PHONE (67)`: For suspicious phone behavior detection

---

### Video Processing: **OpenCV (cv2)**

**Why OpenCV?**
- Industry standard for real-time video I/O and frame manipulation
- Efficient matrix operations for image processing
- Wraps lower-level optimized libraries (Intel IPP, etc.)
- Cross-platform compatibility

**Usage in this system**:
- Frame capture from camera
- Drawing bounding boxes, text, alert banners on frames
- Window management with configurable modes (normal, fullscreen, resizable)
- Display scaling (UI layer separate from detection pipeline)

---

### Numerical Computing: **NumPy**

**Why NumPy?**
- Fast vectorized operations on detection results
- Efficient frame buffer operations
- Coordinate transformations for geometry calculations

**Usage**:
- Creating extended display canvas (camera + side panel)
- Matrix operations for bounding box calculations
- Batch processing of tracked object data

---

### Key Dependencies & Their Purposes

#### **Pygame** (Audio Management)
- Plays alert sound (`assets/alert.mp3`) when behavioral thresholds exceeded
- Provides non-blocking audio playback in the main UI loop
- Configurable volume control

#### **Matplotlib** (Performance Analysis)
- Generates FPS performance plots after session ends
- Creates CSV logs of frame-level performance metrics
- Enables post-hoc analysis of system stability

#### **ByteTrack** (Object Tracking)
- Integrated via YOLOv8; handles ID assignment across frames
- Maintains `object["id"]` field critical for temporal behavior analysis
- Enables long-term motion history and state tracking

---

## System Architecture

### Data Flow Pipeline

```
Camera Frame → YOLOv8 Detection+Tracking → Tracked Objects
                                              ↓
                ┌─────────────────────────────┼─────────────────────────────┐
                ↓                             ↓                             ↓
        LoiteringDetector          AbandonedObjectDetector      ConflictDetector
        (movement speed)           (proximity + duration)        (velocity + proximity)
                ↓                             ↓                             ↓
        suspicious_ids             suspicious_bags              conflict_alert
                │                             │                             │
                └─────────────────────────────┼─────────────────────────────┘
                                              ↓
                                    ThreatScorer
                                 (per-person scores)
                                              ↓
                                    instant_scores
                                   session_scores
                                              ↓
                        Alert State Manager → AudioManager
                                              ↓
                                         Display & Log
```

### Modular Behavior Detectors

Each detector encapsulates one behavioral pattern and returns suspicious indicators independently:

#### 1. **LoiteringDetector** (`behavior/loitering.py`)
**What it detects**: People remaining stationary in the scene
- Tracks per-person position history
- Measures movement distance between frames
- If movement < `LOITER_MOVEMENT_THRESHOLD` (30px) for > `LOITER_TIME` (3s) → suspicious

**State maintained**: Last position, last movement time for each tracked person

**Reasoning**: "Someone standing around" can indicate surveillance, loitering before theft, or waiting for accomplice

---

#### 2. **AbandonedObjectDetector** (`behavior/abandoned_object.py`)
**What it detects**: Bags/backpacks left unattended without a nearby person
- Tracks detection of backpacks and handbags
- Measures distance to nearest person
- If distance > `ABANDON_DISTANCE` (130px) for > `ABANDON_TIME` (3s) → suspicious

**State maintained**: Per-bag tracking, last seen time, abandoned flag

**Grace period logic**: Maintains 0.7s hysteresis to filter flicker (person temporarily walks away)

**Reasoning**: Unattended baggage is security risk; could contain weapons or explosives

---

#### 3. **ConflictDetector** (`behavior/conflict_detection.py`)
**What it detects**: Physical altercations or hostile interactions
- Analyzes pairs of persons
- Computes:
  - **Proximity**: Center-to-center distance < 200px
  - **Velocity**: Relative approach speed
  - **Acceleration**: Rapid velocity changes
  - **Area overlap**: Bounding box overlap indicating close contact

**Confirmation logic**: Requires 2+ consecutive frames of conflict signals to trigger (noise filtering)

**Reasoning**: Violence typically involves sustained proximity + rapid directional changes + physical contact

---

#### 4. **PhoneBehaviorDetector** (`behavior/phone_behavior.py`)
**What it detects**: Suspicious phone usage (potential recording/eavesdropping)
- Classifies phone position by vertical zone:
  - **ACTIVE** (< 30% of person height): Face-level → likely recording
  - **HOLDING** (30-60%): Torso-level → normal holding
  - **POCKET** (> 60%): Waist-level → pocket use
- Measures phone raise speed
- If rapid upward movement (> 80px/sec) to face zone → suspicious

**Status**: `ENABLE_PHONE_BEHAVIOR = False` (disabled by default; tuning needed)

---

#### 5. **ThreatScorer** (`behavior/scoring.py`)
**What it does**: Aggregates multiple detectors into per-person risk scores
- **Instant score**: Frame-level threat (loitering +2, conflict +3, abandoned bag +2)
- **Session score**: Cumulative across entire session (only adds non-zero events)
- Risk levels: `LOW` (0-2), `SUSPICIOUS` (3-4), `HIGH` (≥5)

**Reasoning**: Compound risk—person loitering near abandoned bag is higher risk than either alone

---

### Real-time Display & Alerting

**Main UI Layout** (when `DISPLAY_SCALE = 1.6`):
```
┌────────────────────────────────────┬──────────────────┐
│                                    │   Side Panel     │
│        Camera Feed (640×480)       │  • Active Threats│
│        with Bounding Boxes         │  • Threat Scores │
│        + FPS Counter               │  • Event Timeline│
│                                    │                  │
└────────────────────────────────────┴──────────────────┘
```

**Alert State Machine**:
- Priority: `CONFLICT` > `ABANDONED_BAG` > `LOITERING` > `NONE`
- State changes trigger audio alert + visual banner
- Cooldown of 5 seconds between repeated alerts (configurable)
- Banner displays for 3 seconds per `ALERT_BANNER_DURATION`

**EventLogger** (`utils/event_logger.py`):
- Maintains timeline of all events with timestamps
- Applies per-event-type cooldown to prevent alert spam
- Displayed in real-time on side panel

---

## Capabilities Explained

### 1. Loitering Detection
```python
move_distance = distance(prev_pos, current_pos)
if move_distance < 30px for > 3 seconds:
    → SUSPICIOUS
```
**Real-world application**: Monitor security-sensitive areas (banks, airports, government buildings)

---

### 2. Abandoned Object Detection
```python
nearest_person_distance = min(distances to all persons)
if nearest_person_distance > 130px for > 3 seconds:
    → SUSPICIOUS (with 0.7s grace period for false negatives)
```
**Real-world application**: Automatically flag unattended luggage in public spaces

---

### 3. Motion-based Conflict Detection
```python
for each person pair (i, j):
    if proximity < 200px AND
       relative_velocity > 40px/s AND
       acceleration > 40 AND
       area_overlap > 20%:
        → potential_conflict
        
if potential_conflict for > 2 frames:
    → CONFLICT ALERT
```
**Real-world application**: Detect altercations in real-time, dispatch security immediately

---

### 4. Per-Person Threat Scoring
```
score = 0
if person in loitering_ids: score += 2
if conflict_active: score += 3
if abandoned_bags_nearby: score += 2

threat_level = "HIGH" if score >= 5 else "SUSPICIOUS" if score >= 3 else "LOW"
```
**Real-world application**: Prioritize which individuals operators should focus on

---

### 5. Real-time Dashboard
- **Active Threats Panel**: Live list of flagged individuals + scores
- **Event Timeline**: Colored log of all triggered events
- **FPS Monitoring**: Performance metrics for system health
- **Session Persistence**: Accumulates scores across entire session

---

## Technical Configuration

### Input/Output Settings (`config.py`)

```python
# Video input
CAMERA_SOURCE = 0                    # Webcam index
FRAME_WIDTH, FRAME_HEIGHT = 640, 480 # Detection resolution
DISPLAY_SCALE = 1.6                  # UI zoom (doesn't affect detection)

# Detection
CONFIDENCE = 0.2                      # Permissive threshold (catch edge cases)
IOU_THRESHOLD = 0.5                   # NMS overlap threshold
IMG_SIZE = 416                        # YOLOv8 input size

# Behavior Thresholds
LOITER_TIME = 3                       # seconds of stationary before alert
LOITER_MOVEMENT_THRESHOLD = 30        # pixels, movement threshold
ABANDON_TIME = 3                      # seconds unattended before alert
ABANDON_DISTANCE = 130                # pixels, proximity threshold
GRACE_PERIOD = 0.7                    # hysteresis for flicker filtering
```

### Why These Numbers?
- **LOITER_TIME = 3s**: Balances catching suspicious loitering without flagging normal waiting
- **LOITER_MOVEMENT_THRESHOLD = 30px**: ~5-10% of frame width; accounts for pose variation
- **ABANDON_DISTANCE = 130px**: Close enough that an owner *should* be in frame
- **GRACE_PERIOD = 0.7s**: Filters momentary occlusions and tracking jitter

---

## Current Limitations & Known Issues

### 1. **False Positives from Cooperative Interactions**
- Handshakes, crowding, hugging trigger conflict detection
- **Root cause**: Conflict detector based only on proximity + velocity, not actual contact intent
- **Mitigation**: Session scoring helps—true conflicts sustained longer

### 2. **Tracking ID Resets**
- When person leaves frame and re-enters, gets new ID
- Session threat score **not** preserved across re-entry
- **Impact**: Can't track persistent threats across exits

### 3. **Environmental Sensitivity**
- **Poor lighting**: YOLOv8 confidence drops, detections missed
- **Heavy occlusion**: Tracking IDs lost; can't measure movement
- **Crowded scenes**: ByteTrack ID swapping between similar-sized people

### 4. **Limited Detection Classes**
- Only 4 classes supported (person, 2 bag types, phone)
- Can't detect weapons, suspicious containers, or other threat indicators
- Phone detection unreliable if held partially out of frame

### 5. **Stateless System**
- Behavioral history resets on restart
- No persistent threat database
- Session scores lost on program exit

---

## Why This Approach?

### Interpretability Over Black Boxes
Traditional deep learning behavior classifiers (trained on hours of video) are powerful but opaque:
- "Model predicts unusual behavior" — but why?
- Hard to debug false positives
- Difficult to adapt to new environments
- Expensive to collect and label training data

**This system prioritizes transparency**:
- Each signal (proximity, velocity) is human-understandable
- Operators can see *exactly* which measurements triggered an alert
- Thresholds are tunable per-deployment without retraining
- Works out-of-the-box on any camera with YOLOv8 weights

### Real-time Constraints
- Single-threaded Python main loop
- Frame processing must complete in < 33ms (30 FPS target)
- Can't afford expensive GPU models on edge devices
- Must output decision in same frame (no buffering latency)

### Modularity
Each behavior detector is independent:
- Easy to enable/disable (`ENABLE_CONFLICT_DETECTION`, etc.)
- Easy to test in isolation
- Easy to add new behaviors (extend scorer logic)
- Easy to weight differently in final threat calculation

---

## Future Directions

### 1. **Pose-Based Interaction Modeling**
- Use pose estimation (MediaPipe, OpenPose) to detect actual contact
- Distinguish fighting from friendly greeting geometrically
- Detect prone persons (injury/overdose)

### 2. **Person Re-identification (ReID)**
- Embed person appearance features
- Maintain identity across frame exits/entries
- Persistent threat tracking across sessions

### 3. **Multi-Camera Behavioral Analysis**
- Correlate events across multiple viewpoints
- Track individuals through space using camera handoff
- Detect organized activity (coordinated theft, etc.)

### 4. **Adaptive Threshold Learning**
- Collect per-location baseline statistics
- Measure typical movement patterns, proximity distributions
- Adapt thresholds to environment automatically

### 5. **Temporal Sequence Models**
- LSTM or Transformer on motion history
- Learn suspicious **sequences** (approach → linger → leave, etc.)
- Better distinguish anomalies from normal behavior clusters

---

## Performance & Metrics

### Frame Processing
- **Resolution**: 640×480 input
- **Target FPS**: 30 (detection) + 60 (display scaling)
- **Model inference**: ~20-30ms per frame (YOLOv8 nano)
- **Tracking overhead**: ~5-10ms per frame (ByteTrack)
- **Behavior analysis**: <5ms (simple geometric calculations)

### Outputs Tracked
- **FPS History**: Deque of 300 samples; saved to CSV at session end
- **Performance Plot**: Matplotlib visualization of FPS over time
- **Event Log**: Timestamped event timeline for post-session review

---

## Running the System

```bash
# Activate environment
conda activate project_env

# Start detector
python main.py

# Press 'q' to quit
# Outputs saved to ./saves/
```

**Output artifacts**:
- `detected_person_*.jpg`: Frames with bounding boxes (if `SAVE_FRAMES=True`)
- `fps_data.csv`: Raw FPS measurements
- `fps_plot.png`: Performance visualization

---

## Summary: Why This Matters

This project demonstrates that **behavioral surveillance doesn't require magic**. It requires:
1. **Accurate tracking** (YOLOv8 + ByteTrack)
2. **Interpretable features** (proximity, velocity, duration)
3. **Temporal reasoning** (state machines, hysteresis, confirmation logic)
4. **Real-time constraints** (efficient code, no buffering)
5. **Explainability** (humans understand *why* alerts fire)

The gap between "object detection" and "behavior understanding" is bridged by **tracking time-series signals and combining multiple weak indicators into a confidence score**. This prototype proves that approach works in real-time with modest compute.

---

**Keywords**: Computer Vision • Real-time Processing • Multi-object Tracking • Behavioral Analysis • Interpretable AI • Video Surveillance • YOLOv8 • OpenCV • ByteTrack
