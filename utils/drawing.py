import cv2
import math
import config

# COCO 17-point skeleton connections (index pairs)
_SKELETON = [
    (0, 1), (0, 2),           # nose → eyes
    (1, 3), (2, 4),           # eyes → ears
    (5, 6),                   # shoulders
    (5, 7), (7, 9),           # left arm
    (6, 8), (8, 10),          # right arm
    (5, 11), (6, 12),         # torso sides
    (11, 12),                 # hips
    (11, 13), (13, 15),       # left leg
    (12, 14), (14, 16),       # right leg
]

# Per-keypoint colours (BGR) — arms bright, legs muted, face neutral
_KP_COLORS = {
    0:  (200, 200, 200),   # nose
    1:  (200, 200, 200),   # left eye
    2:  (200, 200, 200),   # right eye
    3:  (200, 200, 200),   # left ear
    4:  (200, 200, 200),   # right ear
    5:  (0, 255, 128),     # left shoulder
    6:  (0, 255, 128),     # right shoulder
    7:  (0, 200, 255),     # left elbow
    8:  (0, 200, 255),     # right elbow
    9:  (0, 60, 255),      # left wrist   ← highlighted
    10: (0, 60, 255),      # right wrist  ← highlighted
    11: (128, 0, 255),     # left hip
    12: (128, 0, 255),     # right hip
    13: (200, 128, 0),     # left knee
    14: (200, 128, 0),     # right knee
    15: (200, 200, 0),     # left ankle
    16: (200, 200, 0),     # right ankle
}


def draw_keypoints(frame, obj, all_persons=None):
    """
    Draw skeleton, joint dots, and conflict-analysis overlays on `frame`.

    obj         — tracked object dict with 'keypoints' (17,2) and 'kp_conf' (17,)
    all_persons — list of other person dicts; used to colour wrists red when
                  they are inside another person's strike zone.
    """
    kps  = obj.get("keypoints")
    kpcs = obj.get("kp_conf")
    if kps is None or kpcs is None:
        return

    threshold = config.KP_CONF_MIN

    def pt(idx):
        """Return integer (x, y) if confidence is sufficient, else None."""
        if kpcs[idx] >= threshold:
            return (int(kps[idx][0]), int(kps[idx][1]))
        return None

    # ── 1. Skeleton lines ──────────────────────────────────────────────
    for a, b in _SKELETON:
        pa, pb = pt(a), pt(b)
        if pa and pb:
            cv2.line(frame, pa, pb, (80, 80, 80), 1, cv2.LINE_AA)

    # ── 2. Collect nose positions of all OTHER persons for strike-zone check ──
    other_noses = []
    if all_persons:
        for other in all_persons:
            if other is obj:
                continue
            okps  = other.get("keypoints")
            okpcs = other.get("kp_conf")
            if okps is not None and okpcs is not None and okpcs[0] >= threshold:
                other_noses.append((float(okps[0][0]), float(okps[0][1])))

    # ── 3. Strike-zone circle around THIS person's nose ──────────────
    nose = pt(0)
    if nose:
        cv2.circle(frame, nose, config.STRIKE_DISTANCE,
                   (0, 140, 255), 1, cv2.LINE_AA)   # orange ring

    # ── 4. Keypoint dots ──────────────────────────────────────────────
    for idx in range(17):
        p = pt(idx)
        if not p:
            continue

        color = _KP_COLORS.get(idx, (180, 180, 180))

        # Wrists: check if inside any opponent's strike zone → paint red
        if idx in (9, 10) and other_noses:
            px, py = float(kps[idx][0]), float(kps[idx][1])
            in_strike = any(
                math.hypot(px - nx, py - ny) < config.STRIKE_DISTANCE
                for nx, ny in other_noses
            )
            if in_strike:
                color = (0, 0, 255)   # bright red — strike zone active
                # Draw a pulsing ring around the wrist for visibility
                cv2.circle(frame, p, 10, (0, 0, 255), 2, cv2.LINE_AA)

        radius = 5 if idx in (9, 10) else 3   # wrists slightly bigger
        cv2.circle(frame, p, radius, color, -1, cv2.LINE_AA)


def setup_window():
    """Setup window based on WINDOW_MODE configuration."""
    mode = config.WINDOW_MODE
    name = config.WINDOW_NAME

    if mode == "normal":
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)

    elif mode == "resizable":
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(name, config.WINDOW_WIDTH, config.WINDOW_HEIGHT)

    elif mode == "maximized":
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    elif mode == "fullscreen":
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    else:
        raise ValueError(f"Invalid WINDOW_MODE: {mode}")
