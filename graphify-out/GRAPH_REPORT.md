# Graph Report - .  (2026-04-25)

## Corpus Check
- Corpus is ~22,597 words - fits in a single context window. You may not need a graph.

## Summary
- 112 nodes · 139 edges · 10 communities detected
- Extraction: 71% EXTRACTED · 29% INFERRED · 0% AMBIGUOUS · INFERRED: 40 edges (avg confidence: 0.68)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_AbandonedLoiteringLogging|Abandoned/Loitering/Logging]]
- [[_COMMUNITY_FPS Tracking|FPS Tracking]]
- [[_COMMUNITY_Conflict Detection|Conflict Detection]]
- [[_COMMUNITY_Audio & Threat Scoring|Audio & Threat Scoring]]
- [[_COMMUNITY_YOLO Detection & Inference|YOLO Detection & Inference]]
- [[_COMMUNITY_Project Exporter Script|Project Exporter Script]]
- [[_COMMUNITY_Phone Behavior|Phone Behavior]]
- [[_COMMUNITY_Drawing Utils|Drawing Utils]]
- [[_COMMUNITY_Geometry & Math|Geometry & Math]]
- [[_COMMUNITY_Export Utils|Export Utils]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 20 edges
2. `FPSTracker` - 11 edges
3. `ConflictDetector` - 10 edges
4. `Clear time/history-dependent detector state after a seek jump.` - 9 edges
5. `Runs in a background thread.     Captures frames, runs detection every DETECT_EV` - 9 edges
6. `ThreatScorer` - 8 edges
7. `Detector` - 8 edges
8. `AudioManager` - 8 edges
9. `inference_worker()` - 7 edges
10. `AbandonedObjectDetector` - 7 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `Detector`  [INFERRED]
  /home/vedant/suspicious_ai/main.py → /home/vedant/suspicious_ai/detection/detector.py
- `main()` --calls--> `LoiteringDetector`  [INFERRED]
  /home/vedant/suspicious_ai/main.py → /home/vedant/suspicious_ai/behavior/loitering.py
- `main()` --calls--> `AbandonedObjectDetector`  [INFERRED]
  /home/vedant/suspicious_ai/main.py → /home/vedant/suspicious_ai/behavior/abandoned_object.py
- `main()` --calls--> `ConflictDetector`  [INFERRED]
  /home/vedant/suspicious_ai/main.py → /home/vedant/suspicious_ai/behavior/conflict_detection.py
- `main()` --calls--> `EventLogger`  [INFERRED]
  /home/vedant/suspicious_ai/main.py → /home/vedant/suspicious_ai/utils/event_logger.py

## Communities

### Community 0 - "Abandoned/Loitering/Logging"
Cohesion: 0.15
Nodes (5): AbandonedObjectDetector, EventLogger, LoiteringDetector, Clear time/history-dependent detector state after a seek jump., Runs in a background thread.     Captures frames, runs detection every DETECT_EV

### Community 1 - "FPS Tracking"
Cohesion: 0.16
Nodes (6): FPSTracker, Add a new FPS measurement, Save FPS data to CSV file, Generate and save FPS plot, Save CSV and plot at the end of session, Get current average FPS

### Community 2 - "Conflict Detection"
Cohesion: 0.24
Nodes (6): ConflictDetector, _kp(), _pose_signals(), Return keypoint (x, y) if smoothed confidence is above threshold, else None., Apply per-keypoint EMA to positions and confidences for a person.         Return, Analyse smoothed keypoints for two people and return:       conflict_boost (bool

### Community 3 - "Audio & Threat Scoring"
Cohesion: 0.19
Nodes (3): AudioManager, main(), ThreatScorer

### Community 4 - "YOLO Detection & Inference"
Cohesion: 0.23
Nodes (5): Detector, Extract tracked objects + keypoints (when pose model) from a results object., inference_worker(), parse_args(), _reset_temporal_state()

### Community 5 - "Project Exporter Script"
Cohesion: 0.47
Nodes (4): extract_python_files(), get_project_structure(), Generate a text representation of the project structure., Extract all Python files and their content.

### Community 6 - "Phone Behavior"
Cohesion: 0.4
Nodes (1): PhoneBehaviorDetector

### Community 7 - "Drawing Utils"
Cohesion: 0.4
Nodes (4): draw_keypoints(), Setup window based on WINDOW_MODE configuration., Draw skeleton, joint dots, and conflict-analysis overlays on `frame`.      obj, setup_window()

### Community 8 - "Geometry & Math"
Cohesion: 0.53
Nodes (2): distance(), get_center()

### Community 9 - "Export Utils"
Cohesion: 0.67
Nodes (1): One-time script to export yolov8n.pt → yolov8n.onnx at 320px input size. Run onc

## Knowledge Gaps
- **13 isolated node(s):** `Generate a text representation of the project structure.`, `Extract all Python files and their content.`, `Return keypoint (x, y) if smoothed confidence is above threshold, else None.`, `Analyse smoothed keypoints for two people and return:       conflict_boost (bool`, `Apply per-keypoint EMA to positions and confidences for a person.         Return` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Phone Behavior`** (6 nodes): `phone_behavior.py`, `phone_behavior.py`, `PhoneBehaviorDetector`, `.get_vertical_zone()`, `.__init__()`, `.update()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Geometry & Math`** (6 nodes): `.update()`, `distance()`, `get_center()`, `geometry.py`, `.update()`, `geometry.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Export Utils`** (3 nodes): `export_model.py`, `One-time script to export yolov8n.pt → yolov8n.onnx at 320px input size. Run onc`, `export_model.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Audio & Threat Scoring` to `Abandoned/Loitering/Logging`, `FPS Tracking`, `Conflict Detection`, `YOLO Detection & Inference`, `Drawing Utils`?**
  _High betweenness centrality (0.260) - this node is a cross-community bridge._
- **Why does `ConflictDetector` connect `Conflict Detection` to `Abandoned/Loitering/Logging`, `Audio & Threat Scoring`?**
  _High betweenness centrality (0.135) - this node is a cross-community bridge._
- **Why does `Runs in a background thread.     Captures frames, runs detection every DETECT_EV` connect `Abandoned/Loitering/Logging` to `FPS Tracking`, `Conflict Detection`, `Audio & Threat Scoring`, `YOLO Detection & Inference`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Are the 17 inferred relationships involving `main()` (e.g. with `Detector` and `LoiteringDetector`) actually correct?**
  _`main()` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `FPSTracker` (e.g. with `Clear time/history-dependent detector state after a seek jump.` and `Runs in a background thread.     Captures frames, runs detection every DETECT_EV`) actually correct?**
  _`FPSTracker` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `ConflictDetector` (e.g. with `Clear time/history-dependent detector state after a seek jump.` and `Runs in a background thread.     Captures frames, runs detection every DETECT_EV`) actually correct?**
  _`ConflictDetector` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `Clear time/history-dependent detector state after a seek jump.` (e.g. with `Detector` and `FPSTracker`) actually correct?**
  _`Clear time/history-dependent detector state after a seek jump.` has 8 INFERRED edges - model-reasoned connections that need verification._