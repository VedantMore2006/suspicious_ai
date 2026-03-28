# config.py

# Camera configuration
CAMERA_SOURCE = "/home/vedant/suspicious_ai/assets/test.mp4"
SHOW_FPS = True

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# ===============================
# DISPLAY CONFIGURATION
# ===============================

WINDOW_NAME = "Suspicious Behavior Detector"

# Modes: "normal", "resizable", "maximized", "fullscreen"
WINDOW_MODE = "normal"

# Used only if mode == "resizable"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

# Scale factor applied ONLY to display (not detection)
DISPLAY_SCALE = 1.6  # 1.0 = original, 1.5 = 150%, etc.

# Video control step for seeking in playback mode.
SEEK_STEP_SECONDS = 5

# Drawing appearance
BOX_THICKNESS = 2
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# Auto-save configuration
SAVE_FRAMES = False
SAVE_CONFIDENCE = 0.5
MOVEMENT_THRESHOLD = 30  # pixels

MODEL_PATH = "yolov8n-pose.pt"  # pose model — gives keypoints for accurate conflict detection
                                 # export with: YOLO("yolov8n-pose.pt").export(format="onnx", imgsz=320)
                                 # then switch to "yolov8n-pose.onnx" for faster CPU inference

CONFIDENCE = 0.2
IOU_THRESHOLD = 0.5
IMG_SIZE = 320  # reduced from 416 for CPU performance (~40% faster inference)

# Frame skipping: run full detection every N frames; intermediate frames reuse last result.
# N=1 = no skipping. N=2 roughly doubles display FPS on CPU (10 FPS detect → ~17 FPS display).
DETECT_EVERY_N = 2

# Classes (COCO indices)
PERSON = 0
BACKPACK = 24
HANDBAG = 26
#SUITCASE = 28
#CELL_PHONE = 67

DETECTION_CLASSES = [PERSON, BACKPACK, HANDBAG]

# Behavior thresholds
LOITER_TIME = 3
LOITER_MOVEMENT_THRESHOLD = 30

ABANDON_TIME = 3
ABANDON_DISTANCE = 130
GRACE_PERIOD = 0.7
PHONE_TIME = 5

# ===============================
# PHONE BEHAVIOR CONFIG
# ===============================

PHONE_FACE_ZONE = 0.3
PHONE_TORSO_ZONE = 0.6

PHONE_RAISE_SPEED_THRESHOLD = 80  # pixels/sec
PHONE_MISUSE_CONFIRM_FRAMES = 3

ENABLE_PHONE_BEHAVIOR = False

SUSPICION_THRESHOLD = 3

ENABLE_CONSOLE_LOG = True
ALERT_COOLDOWN = 5  # seconds between repeated alerts

SHOW_ALERT_BANNER = True
ALERT_BANNER_DURATION = 3  # seconds

# ===============================
# AUDIO CONFIGURATION
# ===============================

ENABLE_BEEP = True
ALERT_SOUND_PATH = "/home/vedant/suspicious_ai/assets/alert.mp3"
AUDIO_VOLUME = 0.8

# ===============================
# ADVANCED CONFLICT DETECTION
# ===============================

ENABLE_CONFLICT_DETECTION = True

PROXIMITY_DISTANCE = 200  # pixels
DISTANCE_VELOCITY_THRESHOLD = 40  # pixels/second
ACCELERATION_THRESHOLD = 40
AREA_CHANGE_THRESHOLD = 0.20

# Conflict confirmation — require sustained signal, not just N frames
CONFLICT_CONFIRM_FRAMES = 5      # consecutive frames with raw signal before confirming
CONFLICT_MIN_DURATION = 1.5      # seconds of sustained signal required (prevents flash false positives)
CONFLICT_CALM_FRAMES = 5         # calm frames before resetting the confirm counter

# Calm contact suppression — handshake/hug: close together + low velocity sustained
CALM_VELOCITY_THRESHOLD = 15     # px/s — below this while close = social contact, not conflict
CALM_CONTACT_TIME = 1.5          # seconds of calm proximity → suppress conflict for this pair

# Keypoint-based conflict analysis (pose model only)
KP_CONF_MIN = 0.3        # minimum keypoint confidence to use a point
STRIKE_DISTANCE = 80     # px — wrist within this distance of opponent nose → strike zone
HIP_TOLERANCE = 70       # px — wrist within this of own hip height → handshake-like pose

# Relative wrist velocity threshold (pixels/second, body-motion corrected)
# A wrist moving faster than this relative to the person's own hips → strike signal
RELATIVE_WRIST_VEL_THRESHOLD = 60   # px/s — lower = more sensitive, raise if too many false positives

# Keypoint visualisation
SHOW_KEYPOINTS = True    # draw skeleton + strike-zone overlay (useful for tuning, disable in prod)