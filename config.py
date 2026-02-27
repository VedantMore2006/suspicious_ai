# config.py

# Camera configuration
CAMERA_SOURCE = 2
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

# Drawing appearance
BOX_THICKNESS = 2
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# Auto-save configuration
SAVE_FRAMES = False
SAVE_CONFIDENCE = 0.5
MOVEMENT_THRESHOLD = 30  # pixels

CONFIDENCE = 0.2
IOU_THRESHOLD = 0.5
IMG_SIZE = 416
# IMG_SIZE = 640
# IMG_SIZE = 256
# IMG_SIZE = 320

# Classes (COCO indices)
PERSON = 0
BACKPACK = 24
HANDBAG = 26
SUITCASE = 28
CELL_PHONE = 67

DETECTION_CLASSES = [PERSON, BACKPACK, HANDBAG, SUITCASE, CELL_PHONE]

# Behavior thresholds
LOITER_TIME = 3
LOITER_MOVEMENT_THRESHOLD = 30

ABANDON_TIME = 10
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

ENABLE_PHONE_BEHAVIOR = True

SUSPICION_THRESHOLD = 3

ENABLE_BEEP = True
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
DISTANCE_VELOCITY_THRESHOLD = 30  # pixels/second
ACCELERATION_THRESHOLD = 30
AREA_CHANGE_THRESHOLD = 0.20  # 25%
CONFLICT_CONFIRM_FRAMES = 2