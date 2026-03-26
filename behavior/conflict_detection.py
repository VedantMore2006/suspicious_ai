import config
import time
import math


# COCO keypoint indices
_NOSE        = 0
_L_SHOULDER  = 5
_R_SHOULDER  = 6
_L_ELBOW     = 7
_R_ELBOW     = 8
_L_WRIST     = 9
_R_WRIST     = 10
_L_HIP       = 11
_R_HIP       = 12


def _kp(kps, confs, idx):
    """Return keypoint (x, y) if confidence is above threshold, else None."""
    if kps is None or confs is None:
        return None
    if confs[idx] >= config.KP_CONF_MIN:
        return (float(kps[idx][0]), float(kps[idx][1]))
    return None


def _pose_signals(personA, personB):
    """
    Analyse keypoints for two people and return:
      conflict_boost (bool) — pose clearly indicates aggression
      suppress       (bool) — pose clearly indicates friendly contact

    Falls back to (False, False) when keypoints unavailable.
    """
    kpA, kcA = personA.get("keypoints"), personA.get("kp_conf")
    kpB, kcB = personB.get("keypoints"), personB.get("kp_conf")

    if kpA is None or kpB is None:
        return False, False

    # --- Extract key points for A ---
    wL_A = _kp(kpA, kcA, _L_WRIST)
    wR_A = _kp(kpA, kcA, _R_WRIST)
    sL_A = _kp(kpA, kcA, _L_SHOULDER)
    sR_A = _kp(kpA, kcA, _R_SHOULDER)
    hL_A = _kp(kpA, kcA, _L_HIP)
    hR_A = _kp(kpA, kcA, _R_HIP)
    wrists_A    = [w for w in [wL_A, wR_A] if w]
    shoulders_A = [s for s in [sL_A, sR_A] if s]
    hips_A      = [h for h in [hL_A, hR_A] if h]

    # --- Extract key points for B ---
    wL_B = _kp(kpB, kcB, _L_WRIST)
    wR_B = _kp(kpB, kcB, _R_WRIST)
    sL_B = _kp(kpB, kcB, _L_SHOULDER)
    sR_B = _kp(kpB, kcB, _R_SHOULDER)
    hL_B = _kp(kpB, kcB, _L_HIP)
    hR_B = _kp(kpB, kcB, _R_HIP)
    nose_B      = _kp(kpB, kcB, _NOSE)
    nose_A      = _kp(kpA, kcA, _NOSE)
    wrists_B    = [w for w in [wL_B, wR_B] if w]
    hips_B      = [h for h in [hL_B, hR_B] if h]

    conflict_boost = False
    suppress       = False

    # ── Conflict signal 1: wrist of A near nose of B (strike zone) ──
    if nose_B and wrists_A:
        for w in wrists_A:
            if math.hypot(w[0] - nose_B[0], w[1] - nose_B[1]) < config.STRIKE_DISTANCE:
                conflict_boost = True

    # ── Conflict signal 1 (reciprocal): wrist of B near nose of A ──
    if nose_A and wrists_B:
        for w in wrists_B:
            if math.hypot(w[0] - nose_A[0], w[1] - nose_A[1]) < config.STRIKE_DISTANCE:
                conflict_boost = True

    # ── Conflict signal 2: wrist raised well above own shoulder (overhead strike / choke) ──
    if shoulders_A and wrists_A:
        avg_sh_y = sum(s[1] for s in shoulders_A) / len(shoulders_A)
        # In image coordinates y increases downward, so wrist ABOVE shoulder → wrist_y < shoulder_y
        if any(w[1] < avg_sh_y - 20 for w in wrists_A):
            conflict_boost = True

    if (sL_B or sR_B) and wrists_B:
        shoulders_B = [s for s in [sL_B, sR_B] if s]
        avg_sh_y = sum(s[1] for s in shoulders_B) / len(shoulders_B)
        if any(w[1] < avg_sh_y - 20 for w in wrists_B):
            conflict_boost = True

    # ── Friendly signal: handshake — wrists near own hip level ──
    # A handshake means at least one person's wrists are at waist height
    if hips_A and wrists_A:
        avg_hip_y = sum(h[1] for h in hips_A) / len(hips_A)
        if all(abs(w[1] - avg_hip_y) < config.HIP_TOLERANCE for w in wrists_A):
            suppress = True

    if hips_B and wrists_B:
        avg_hip_y = sum(h[1] for h in hips_B) / len(hips_B)
        if all(abs(w[1] - avg_hip_y) < config.HIP_TOLERANCE for w in wrists_B):
            suppress = True

    # A conflict boost always overrides a suppress signal
    if conflict_boost:
        suppress = False

    return conflict_boost, suppress


class ConflictDetector:
    def __init__(self):
        # Per-pair state keyed by tuple(sorted((idA, idB)))
        self.history = {}

    def compute_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def compute_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def update(self, tracked_objects, video_timestamp=None):
        if not config.ENABLE_CONFLICT_DETECTION:
            return False

        persons = [obj for obj in tracked_objects if obj["class"] == config.PERSON]
        if len(persons) < 2:
            return False

        # Use video timestamp when available so velocity is accurate
        # regardless of CPU processing speed (important for recorded video)
        current_time = video_timestamp if video_timestamp is not None else time.time()

        any_confirmed = False
        active_pairs  = set()

        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                idA = persons[i]["id"]
                idB = persons[j]["id"]

                centerA = self.compute_center(persons[i]["bbox"])
                centerB = self.compute_center(persons[j]["bbox"])
                areaA   = self.compute_area(persons[i]["bbox"])
                areaB   = self.compute_area(persons[j]["bbox"])

                distance = math.hypot(centerA[0] - centerB[0], centerA[1] - centerB[1])

                if distance > config.PROXIMITY_DISTANCE:
                    continue

                pair_key = tuple(sorted((idA, idB)))
                active_pairs.add(pair_key)

                # ── Initialise per-pair state ──
                if pair_key not in self.history:
                    self.history[pair_key] = {
                        "prev_distance":  distance,
                        "prev_velocity":  0.0,
                        "prev_areaA":     areaA,
                        "prev_areaB":     areaB,
                        "prev_time":      current_time,
                        "confirm_count":  0,
                        "calm_count":     0,
                        "conflict_start": None,
                        "close_since":    current_time,
                        "calm_contact":   False,
                    }
                    continue

                prev = self.history[pair_key]
                dt = current_time - prev["prev_time"]
                if dt <= 0:
                    continue

                velocity     = (distance - prev["prev_distance"]) / dt
                acceleration = (velocity  - prev["prev_velocity"]) / dt

                area_changeA = abs(areaA - prev["prev_areaA"]) / (prev["prev_areaA"] + 1e-5)
                area_changeB = abs(areaB - prev["prev_areaB"]) / (prev["prev_areaB"] + 1e-5)

                # ── Pose-based signals ──
                conflict_boost, pose_suppress = _pose_signals(persons[i], persons[j])

                # ── Raw bbox conflict signal ──
                bbox_conflict = (
                    (
                        abs(velocity)     > config.DISTANCE_VELOCITY_THRESHOLD and
                        abs(acceleration) > config.ACCELERATION_THRESHOLD
                    ) or (
                        area_changeA > config.AREA_CHANGE_THRESHOLD or
                        area_changeB > config.AREA_CHANGE_THRESHOLD
                    )
                )

                # Keypoints boost: wrist near face counts as conflict even if
                # the bbox velocity hasn't crossed the threshold yet
                raw_conflict = bbox_conflict or conflict_boost

                # ── Calm contact suppression ──
                time_close = current_time - prev["close_since"]
                if (
                    abs(velocity) < config.CALM_VELOCITY_THRESHOLD and
                    time_close > config.CALM_CONTACT_TIME
                ):
                    prev["calm_contact"] = True

                if pose_suppress:
                    prev["calm_contact"] = True

                # High aggression (2× velocity threshold) resets calm contact label
                if conflict_boost or abs(velocity) > config.DISTANCE_VELOCITY_THRESHOLD * 2:
                    prev["calm_contact"] = False

                # ── Per-pair confirmation ──
                if raw_conflict and not prev["calm_contact"]:
                    prev["calm_count"] = 0
                    if prev["confirm_count"] == 0:
                        prev["conflict_start"] = current_time
                    prev["confirm_count"] += 1
                else:
                    prev["calm_count"] += 1
                    if prev["calm_count"] >= config.CONFLICT_CALM_FRAMES:
                        prev["confirm_count"] = 0
                        prev["conflict_start"] = None

                duration_ok = (
                    prev["conflict_start"] is not None and
                    (current_time - prev["conflict_start"]) >= config.CONFLICT_MIN_DURATION
                )
                if prev["confirm_count"] >= config.CONFLICT_CONFIRM_FRAMES and duration_ok:
                    any_confirmed = True

                # ── Update history ──
                prev["prev_distance"] = distance
                prev["prev_velocity"] = velocity
                prev["prev_areaA"]    = areaA
                prev["prev_areaB"]    = areaB
                prev["prev_time"]     = current_time

        # Clean up pairs that left proximity
        for k in [k for k in self.history if k not in active_pairs]:
            del self.history[k]

        return any_confirmed
