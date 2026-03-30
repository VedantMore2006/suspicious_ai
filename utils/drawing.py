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

# Which category each bone belongs to — drives color-coding in debug mode
# "arm" = conflict-relevant, "face"/"torso"/"leg" = neutral
_BONE_CATEGORY = {
    (0, 1): "face",  (0, 2): "face",
    (1, 3): "face",  (2, 4): "face",
    (5, 6): "torso",
    (5, 7): "arm",   (7, 9): "arm",
    (6, 8): "arm",   (8, 10): "arm",
    (5, 11): "torso", (6, 12): "torso",
    (11, 12): "torso",
    (11, 13): "leg",  (13, 15): "leg",
    (12, 14): "leg",  (14, 16): "leg",
}

# Bone colors by category (BGR) — normal (inactive) state
_BONE_COLORS_NORMAL = {
    "face":  (90, 90, 90),
    "arm":   (180, 180, 60),   # yellow-grey arms
    "torso": (80, 80, 80),
    "leg":   (60, 60, 60),
}
# Arm bones turn orange when arm-raised / strike-zone active
_ARM_ACTIVE_COLOR  = (0, 140, 255)   # orange
_ARM_SUPPRESS_COLOR = (0, 200, 80)   # green when suppressed (handshake)

# Per-keypoint dot colours (BGR) — default state
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
    9:  (0, 60, 255),      # left wrist   ← primary conflict signal
    10: (0, 60, 255),      # right wrist  ← primary conflict signal
    11: (128, 0, 255),     # left hip
    12: (128, 0, 255),     # right hip
    13: (200, 128, 0),     # left knee
    14: (200, 128, 0),     # right knee
    15: (200, 200, 0),     # left ankle
    16: (200, 200, 0),     # right ankle
}

# Human-readable name for each COCO keypoint index
_KP_NAMES = {
    0: "nose",
    1: "L-eye", 2: "R-eye",
    3: "L-ear", 4: "R-ear",
    5: "L-sho", 6: "R-sho",
    7: "L-elb", 8: "R-elb",
    9: "L-wri", 10: "R-wri",
    11: "L-hip", 12: "R-hip",
    13: "L-kne", 14: "R-kne",
    15: "L-ank", 16: "R-ank",
}


def draw_keypoints(frame, obj, all_persons=None):
    """
    Draw skeleton, joint dots, and conflict-analysis overlays on `frame`.

    obj         — tracked object dict with keypoints (17,2) and kp_conf (17,)
    all_persons — list of all person dicts; used to detect wrists inside opponent strike zones
    """
    # Prefer smoothed keypoints (written by ConflictDetector) for accuracy
    kps  = obj.get("_smooth_kp") if obj.get("_smooth_kp") is not None else obj.get("keypoints")
    kpcs = obj.get("_smooth_conf") if obj.get("_smooth_conf") is not None else obj.get("kp_conf")
    if kps is None or kpcs is None:
        return

    threshold = config.KP_CONF_MIN
    debug = getattr(config, "DEBUG_KEYPOINTS", False)

    def pt(idx):
        """Return integer (x, y) if confidence is sufficient, else None."""
        if kpcs[idx] >= threshold:
            return (int(kps[idx][0]), int(kps[idx][1]))
        return None

    # ── Active signal state for this person ──
    active_signals = obj.get("_signals", [])
    rel_wrist_vel  = obj.get("_rel_wrist_vel", 0.0)
    arm_raised     = any("ARM RAISED"   in s for s in active_signals)
    strike_self    = any("STRIKE ZONE"  in s for s in active_signals)
    suppressed     = any("SUPPRESS"     in s for s in active_signals)

    # ── Collect nose positions of all OTHER persons for strike-zone check ──
    other_noses = []
    if all_persons:
        for other in all_persons:
            if other is obj:
                continue
            okps  = other.get("_smooth_kp") if other.get("_smooth_kp") is not None else other.get("keypoints")
            okpcs = other.get("_smooth_conf") if other.get("_smooth_conf") is not None else other.get("kp_conf")
            if okps is not None and okpcs is not None and okpcs[0] >= threshold:
                other_noses.append((float(okps[0][0]), float(okps[0][1])))

    # Determine per-wrist strike-zone state
    wrist_in_strike = {}
    for wrist_idx in (9, 10):
        if kpcs[wrist_idx] < threshold:
            wrist_in_strike[wrist_idx] = False
            continue
        px, py = float(kps[wrist_idx][0]), float(kps[wrist_idx][1])
        wrist_in_strike[wrist_idx] = any(
            math.hypot(px - nx, py - ny) < config.STRIKE_DISTANCE
            for nx, ny in other_noses
        )

    any_wrist_strike = any(wrist_in_strike.values())

    # ── 1. Skeleton lines ─────────────────────────────────────────────────
    for a, b in _SKELETON:
        pa, pb = pt(a), pt(b)
        if not (pa and pb):
            continue
        cat = _BONE_CATEGORY.get((a, b), "torso")
        if cat == "arm":
            if suppressed:
                color = _ARM_SUPPRESS_COLOR
            elif arm_raised or any_wrist_strike:
                color = _ARM_ACTIVE_COLOR
            else:
                color = _BONE_COLORS_NORMAL["arm"]
        else:
            color = _BONE_COLORS_NORMAL[cat]
        thickness = 2 if cat == "arm" else 1
        cv2.line(frame, pa, pb, color, thickness, cv2.LINE_AA)

    # ── 2. Strike-zone circle around THIS person's nose ───────────────────
    nose = pt(0)
    if nose:
        cv2.circle(frame, nose, config.STRIKE_DISTANCE,
                   (0, 140, 255), 1, cv2.LINE_AA)   # orange ring

    # ── 3. Keypoint dots ──────────────────────────────────────────────────
    for idx in range(17):
        p = pt(idx)
        if not p:
            continue

        color = _KP_COLORS.get(idx, (180, 180, 180))

        # Wrists: red + ring when inside opponent strike zone
        if idx in (9, 10) and wrist_in_strike.get(idx, False):
            color = (0, 0, 255)
            cv2.circle(frame, p, 12, (0, 0, 255), 2, cv2.LINE_AA)

        radius = 5 if idx in (9, 10) else 3
        cv2.circle(frame, p, radius, color, -1, cv2.LINE_AA)

        # ── DEBUG: index number + short name next to each joint ───────────
        if debug:
            label = f"{idx}:{_KP_NAMES[idx]}"
            # Offset label slightly to avoid overlapping the dot
            lx = p[0] + 7
            ly = p[1] - 4
            # Thin black shadow first for readability
            cv2.putText(frame, label, (lx + 1, ly + 1),
                        cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, label, (lx, ly),
                        cv2.FONT_HERSHEY_PLAIN, 0.75, (220, 220, 220), 1, cv2.LINE_AA)

    # ── 4. DEBUG: active signal labels above the person ───────────────────
    if debug and active_signals:
        # Find topmost visible keypoint to place labels above the person
        all_pts = [pt(i) for i in range(17)]
        visible = [p for p in all_pts if p is not None]
        if visible:
            top_y = min(p[1] for p in visible) - 10
            top_x = min(p[0] for p in visible)

            for line_idx, sig in enumerate(active_signals):
                # Color label by signal type
                if "STRIKE" in sig:
                    sig_color = (0, 0, 255)       # red
                elif "ARM RAISED" in sig:
                    sig_color = (0, 140, 255)     # orange
                elif "SUPPRESS" in sig:
                    sig_color = (0, 200, 80)      # green
                elif "rel-vel" in sig:
                    sig_color = (0, 220, 220)     # yellow
                else:
                    sig_color = (200, 200, 200)

                ly = top_y - line_idx * 16
                cv2.putText(frame, sig, (top_x + 1, ly + 1),
                            cv2.FONT_HERSHEY_PLAIN, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, sig, (top_x, ly),
                            cv2.FONT_HERSHEY_PLAIN, 0.85, sig_color, 1, cv2.LINE_AA)

    # ── 5. DEBUG: rel-wrist-vel bar (even without active signals) ─────────
    if debug and rel_wrist_vel is not None:
        nose_or_top = pt(0)
        if nose_or_top:
            bar_x = nose_or_top[0]
            bar_y = nose_or_top[1] - config.STRIKE_DISTANCE - 20
            vel_label = f"relV:{rel_wrist_vel:.0f}"
            bar_color = (0, 0, 255) if rel_wrist_vel > config.RELATIVE_WRIST_VEL_THRESHOLD else (120, 120, 120)
            cv2.putText(frame, vel_label, (bar_x - 20, bar_y),
                        cv2.FONT_HERSHEY_PLAIN, 0.8, bar_color, 1, cv2.LINE_AA)


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
