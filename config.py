# config.py

# Camera configuration
CAMERA_SOURCE = 0
SHOW_FPS = True

# ===============================
# DISPLAY CONFIGURATION
# ===============================

WINDOW_NAME = "Suspicious Behavior Detector"

# Modes: "normal", "resizable", "maximized", "fullscreen"
WINDOW_MODE = "fullscreen"

# Used only if mode == "resizable"
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800

# Scale factor applied ONLY to display (not detection)
DISPLAY_SCALE = 1.5  # 1.0 = original, 1.5 = 150%, etc.

# Drawing appearance
BOX_THICKNESS = 1
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# Auto-save configuration
SAVE_CONFIDENCE = 0.5
MOVEMENT_THRESHOLD = 30  # pixels

CONFIDENCE = 0.2
IOU_THRESHOLD = 0.5
IMG_SIZE = 256

# Classes (COCO indices)
PERSON = 0
BACKPACK = 24
HANDBAG = 26
SUITCASE = 28
CELL_PHONE = 67

DETECTION_CLASSES = [PERSON, BACKPACK, HANDBAG, SUITCASE, CELL_PHONE]

# Behavior thresholds
LOITER_TIME = 8
LOITER_MOVEMENT_THRESHOLD = 15

ABANDON_TIME = 5
ABANDON_DISTANCE = 120

PHONE_TIME = 5

SUSPICION_THRESHOLD = 3