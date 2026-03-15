# Suspicious AI

A real-time suspicious behavior detection system built with YOLOv8 + OpenCV.

This project tracks people and objects from a live camera/video feed and raises alerts for potentially risky situations:
- Loitering person detection
- Abandoned bag/object detection
- Possible physical conflict detection
- Per-person threat scoring and session risk accumulation
- Visual alert banner, side-panel threat dashboard, event timeline, and alarm sound

## What This Project Does

At runtime, the app:
1. Captures frames from a camera source.
2. Runs YOLOv8 tracking (`ByteTrack`) on selected classes (person, backpack, handbag, phone).
3. Sends tracked objects into behavior detectors.
4. Computes instant and cumulative threat scores.
5. Draws bounding boxes, threat labels, and a right-side analysis panel.
6. Plays/stops alarm audio based on alert state transitions.
7. Logs events and keeps a short in-memory timeline.
8. Optionally saves detected frames and FPS analytics into `saves/`.

## Project Structure

```text
suspicious_ai/
├── main.py                        # Runtime loop + orchestration + visualization
├── config.py                      # All thresholds, class IDs, audio/display toggles
├── requirements.txt               # Python dependencies
├── yolov8n.pt                     # YOLO model weights
├── extract_project.py             # Utility to export project code/structure
├── project_extraction.txt         # Generated extraction report
├── assets/
│   └── alert.mp3                  # Alarm sound asset
├── detection/
│   ├── __init__.py
│   └── detector.py                # YOLO wrapper + tracking call
├── behavior/
│   ├── __init__.py
│   ├── loitering.py               # Stationary-person detection logic
│   ├── abandoned_object.py        # Bag away-from-person timer logic
│   ├── conflict_detection.py      # Pairwise motion/area change conflict logic
│   ├── scoring.py                 # Threat scoring + session accumulation
│   ├── phone_behavior.py          # Optional phone state/misuse detector (currently not wired in main loop)
│   └── phone_usage.py             # Placeholder (empty)
├── utils/
│   ├── __init__.py
│   ├── drawing.py                 # OpenCV window mode setup
│   ├── event_logger.py            # Cooldown-based alert logger + timeline
│   ├── audio.py                   # Alarm playback manager (pygame)
│   ├── fps_tracker.py             # FPS history, CSV export, chart export
│   ├── geometry.py                # Center + distance helpers
│   └── fps.py                     # Placeholder (empty)
└── data/                          # Optional data folder
```

## Core Runtime Flow

Main entrypoint: `main.py`

### 1) Detection and Tracking
`detection/detector.py` uses:
- `YOLO(model_path)` with `yolov8n.pt`
- `model.track(..., persist=True, tracker="bytetrack.yaml")`

Tracking IDs are reused frame-to-frame, enabling per-person and per-object behavior history.

### 2) Behavior Analysis
Each frame builds `tracked_objects` in this format:

```python
{
    "id": int,
    "class": int,
    "bbox": (x1, y1, x2, y2)
}
```

These detectors run on the same tracked list:
- `LoiteringDetector.update(...)`
- `AbandonedObjectDetector.update(...)`
- `ConflictDetector.update(...)`

### 3) Threat Scoring
`behavior/scoring.py` computes per-person frame score (`instant_scores`) and cumulative `session_scores`:
- `+1` if person is loitering
- `+2` if abandoned object condition is active
- `+4` if conflict condition is active

Level mapping:
- `0-2` -> `NORMAL`
- `3-4` -> `SUSPICIOUS`
- `5+` -> `HIGH`

### 4) Alerts and Alarm
State machine priority in `main.py`:
- `CONFLICT` has highest priority
- `ABANDONED` next
- otherwise `NONE`

On state changes:
- alarm starts/stops via `AudioManager`
- event is logged via `EventLogger` with cooldown
- top red alert banner displays for configured duration

### 5) Visualization Panel
The output window shows:
- camera feed with boxes and labels
- right-side panel titled `THREAT ANALYSIS`
- per-ID threat level + session total
- recent event timeline entries
- FPS and average FPS (optional)

### 6) Frame + Performance Exports
If `SAVE_FRAMES=True`, snapshots with detections are stored in `saves/`.

At shutdown, `FPSTracker.finalize()` writes:
- `saves/fps_data.csv`
- `saves/fps_plot.png`

## Configuration Guide (`config.py`)

You can tune most system behavior without changing code.

### Input and Display
- `CAMERA_SOURCE`: camera index or video path
- `WINDOW_MODE`: `normal`, `resizable`, `maximized`, `fullscreen`
- `DISPLAY_SCALE`: display-only scaling factor
- `SHOW_FPS`: toggles FPS overlay

### Detection
- `CONFIDENCE`, `IOU_THRESHOLD`, `IMG_SIZE`
- `DETECTION_CLASSES`: COCO classes to track

### Class IDs Used
- `PERSON = 0`
- `BACKPACK = 24`
- `HANDBAG = 26`
- `CELL_PHONE = 67`

### Behavior Thresholds
- Loitering: `LOITER_TIME`, `LOITER_MOVEMENT_THRESHOLD`
- Abandoned object: `ABANDON_TIME`, `ABANDON_DISTANCE`, `GRACE_PERIOD`
- Conflict: proximity, velocity, acceleration, area-change, confirm frames

### Alerts and Audio
- `SHOW_ALERT_BANNER`, `ALERT_BANNER_DURATION`
- `ALERT_COOLDOWN`
- `ALERT_SOUND_PATH`, `AUDIO_VOLUME`, `ENABLE_BEEP`

### Saving
- `SAVE_FRAMES`, `SAVE_CONFIDENCE`, `MOVEMENT_THRESHOLD`

## Setup

### 1) Create and Activate Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install Dependencies

```bash
pip install -r requirements.txt
pip install pygame
```

`pygame` is required by `utils/audio.py` for alarm playback.

### 3) Run

```bash
python main.py
```

Press `q` in the OpenCV window to exit.

## Requirements

Current `requirements.txt` includes:
- `opencv-python`
- `ultralytics`
- `matplotlib`
- `numpy`

Additional runtime dependency used by code:
- `pygame`

## Outputs

During/after execution, these outputs can appear:
- On-screen detections and threat panel
- Console alerts like `[ALERT] ...`
- Saved detection frames in `saves/`
- FPS CSV and FPS plot in `saves/`

## Notes

- `behavior/phone_behavior.py` exists and is configurable but is not currently called from `main.py`.
- `behavior/phone_usage.py` and `utils/fps.py` are placeholders (empty).
- For better model accuracy/performance tradeoffs, you can swap `yolov8n.pt` with another YOLOv8 variant.

## Limitations

### Behavioral Detection
- **Loitering does not trigger audio** — it only changes bounding box color and updates the threat score. The alarm only fires for `CONFLICT` and `ABANDONED` states.
- **Conflict detection has no pose awareness** — it is based purely on proximity, bounding-box velocity, and area change. Two people standing close together, hugging, or shaking hands can trigger false positives.
- **Loitering timer resets on re-entry** — if a person leaves the frame and comes back, their stationary timer starts fresh; repeat loitering across separate visits is not tracked.
- **Abandoned object distance is pixel-based** — `ABANDON_DISTANCE` is measured in screen pixels, so its effective real-world range changes with camera zoom, angle, or resolution.
- **Phone behavior is unfinished** — `PhoneBehaviorDetector` is implemented but not wired into the main loop or the scoring system yet.

### Tracking and Detection
- **Single camera only** — the system reads one `CAMERA_SOURCE` at a time; there is no multi-feed or multi-camera support.
- **Limited detection classes** — only `PERSON`, `BACKPACK`, `HANDBAG`, and `CELL_PHONE` (COCO IDs) are tracked. Other suspicious items (e.g. suitcases, weapons) are ignored by default.
- **Low-light / occlusion sensitivity** — YOLOv8n is a lightweight model. Detection quality degrades in poor lighting, heavy occlusion, or at long distances.
- **No re-identification across sessions** — tracking IDs are local to a single run. The same person will get a new ID if the process restarts.

### Data and Storage
- **All state is in-memory** — session scores, event timeline, and behavior history are lost when the process exits. There is no database or log file written at runtime.
- **`ENABLE_BEEP` is defined twice in `config.py`** — the second definition (`True`) silently overrides the first (`False`). This can cause confusion when trying to disable audio.
- **No persistent FPS log during runtime** — FPS data is only written to `saves/fps_data.csv` and `saves/fps_plot.png` at clean shutdown; a crash discards all FPS history.

### Infrastructure
- **No authentication or access control** — the camera feed and saves directory are not protected; deployment in a shared or networked environment requires additional hardening.
- **No GPU auto-detection** — inference runs on whichever device PyTorch defaults to. On CPU-only machines, real-time performance may drop significantly.
- **Display requires a screen** — the system uses `cv2.imshow`, which needs a display/X11 session. Headless or server deployments require additional configuration.

## Future Improvements (Suggested)

- Integrate `PhoneBehaviorDetector` into the main loop.
- Add persistent event storage (JSON/SQLite) and replay tools.
- Add unit tests for detector logic (loitering/abandon/conflict/scoring).
- Add CLI flags for runtime overrides instead of editing `config.py` directly.
