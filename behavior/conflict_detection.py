import config
import time
import math
import numpy as np


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

_KP_SMOOTH_ALPHA      = 0.4   # EMA weight for keypoint positions
_KP_CONF_SMOOTH_ALPHA = 0.35  # EMA weight for keypoint confidences (slightly slower — stability)


def _kp(kps, confs, idx):
    """Return keypoint (x, y) if smoothed confidence is above threshold, else None."""
    if kps is None or confs is None:
        return None
    if confs[idx] >= config.KP_CONF_MIN:
        return (float(kps[idx][0]), float(kps[idx][1]))
    return None


def _pose_signals(personA, personB, rel_wrist_vel_A=None, rel_wrist_vel_B=None):
    """
    Analyse smoothed keypoints for two people and return:
      conflict_boost (bool) — pose clearly indicates aggression
      suppress       (bool) — pose clearly indicates friendly contact
      fight_score    (float) — continuous aggression signal for session scoring

    rel_wrist_vel_A/B: relative wrist speed (wrist vel minus hip vel) for each person.
    Falls back gracefully when keypoints or relative velocity are unavailable.
    """
    kpA, kcA = personA.get("_smooth_kp"), personA.get("_smooth_conf")
    kpB, kcB = personB.get("_smooth_kp"), personB.get("_smooth_conf")

    # Fall back to raw keypoints if smoothed not yet available
    if kpA is None:
        kpA, kcA = personA.get("keypoints"), personA.get("kp_conf")
    if kpB is None:
        kpB, kcB = personB.get("keypoints"), personB.get("kp_conf")

    if kpA is None or kpB is None:
        return False, False, 0.0

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
    shoulders_B = [s for s in [sL_B, sR_B] if s]
    hips_B      = [h for h in [hL_B, hR_B] if h]

    conflict_boost = False
    suppress       = False
    fight_score    = 0.0

    # ── Conflict signal 1: wrist of A near nose of B (strike zone) ──
    if nose_B and wrists_A:
        for w in wrists_A:
            if math.hypot(w[0] - nose_B[0], w[1] - nose_B[1]) < config.STRIKE_DISTANCE:
                conflict_boost = True
                fight_score += 3.0

    # ── Conflict signal 1 (reciprocal): wrist of B near nose of A ──
    if nose_A and wrists_B:
        for w in wrists_B:
            if math.hypot(w[0] - nose_A[0], w[1] - nose_A[1]) < config.STRIKE_DISTANCE:
                conflict_boost = True
                fight_score += 3.0

    # ── Conflict signal 2: wrist raised well above own shoulder ──
    if shoulders_A and wrists_A:
        avg_sh_y = sum(s[1] for s in shoulders_A) / len(shoulders_A)
        if any(w[1] < avg_sh_y - 20 for w in wrists_A):
            conflict_boost = True
            fight_score += 1.5

    if shoulders_B and wrists_B:
        avg_sh_y = sum(s[1] for s in shoulders_B) / len(shoulders_B)
        if any(w[1] < avg_sh_y - 20 for w in wrists_B):
            conflict_boost = True
            fight_score += 1.5

    # ── Conflict signal 3: high relative wrist velocity (body-motion corrected) ──
    # Uses smoothed relative velocity to avoid false spikes from walking/running
    rel_vel_thresh = config.RELATIVE_WRIST_VEL_THRESHOLD
    if rel_wrist_vel_A is not None and rel_wrist_vel_A > rel_vel_thresh:
        fight_score += min(rel_wrist_vel_A / rel_vel_thresh, 2.0)  # cap contribution
        if rel_wrist_vel_A > rel_vel_thresh * 1.5:
            conflict_boost = True
    if rel_wrist_vel_B is not None and rel_wrist_vel_B > rel_vel_thresh:
        fight_score += min(rel_wrist_vel_B / rel_vel_thresh, 2.0)
        if rel_wrist_vel_B > rel_vel_thresh * 1.5:
            conflict_boost = True

    # ── Friendly signal: handshake — wrists near own hip level ──
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
        fight_score = max(fight_score, 2.0)  # floor when boost is active

    if suppress:
        fight_score = 0.0

    return conflict_boost, suppress, fight_score


class ConflictDetector:
    def __init__(self):
        # Per-pair state keyed by tuple(sorted((idA, idB)))
        self.history = {}
        # Per-person keypoint smoothing state keyed by person ID
        self.person_kp = {}

    def compute_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def compute_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def _smooth_person(self, pid, kp_xy, kp_conf, current_time):
        """
        Apply per-keypoint EMA to positions and confidences for a person.
        Returns (smooth_kp, smooth_conf, rel_wrist_vel) where rel_wrist_vel
        is the max wrist speed minus hip speed (body-motion corrected).
        """
        if kp_xy is None or kp_conf is None:
            return None, None, None

        if pid not in self.person_kp:
            # First observation — initialise with raw values
            self.person_kp[pid] = {
                "smooth_kp":   kp_xy.copy(),
                "smooth_conf": kp_conf.copy(),
                "prev_wrist":  None,   # average wrist position
                "prev_hip":    None,   # average hip position
                "prev_time":   current_time,
                "rel_wrist_vel": 0.0,
            }
            return kp_xy, kp_conf, 0.0

        state = self.person_kp[pid]
        dt = current_time - state["prev_time"]

        # ── EMA on positions (only where current conf is adequate) ──
        new_smooth_kp   = state["smooth_kp"].copy()
        new_smooth_conf = state["smooth_conf"].copy()

        for idx in range(17):
            # Confidence EMA — smooths flickering confidence scores
            new_smooth_conf[idx] = (
                _KP_CONF_SMOOTH_ALPHA * kp_conf[idx] +
                (1 - _KP_CONF_SMOOTH_ALPHA) * state["smooth_conf"][idx]
            )
            # Position EMA — only blend when raw detection has decent confidence
            if kp_conf[idx] >= config.KP_CONF_MIN * 0.7:
                new_smooth_kp[idx] = (
                    _KP_SMOOTH_ALPHA * kp_xy[idx] +
                    (1 - _KP_SMOOTH_ALPHA) * state["smooth_kp"][idx]
                )
            # else: keep previous smooth position (occlusion/low-conf frame)

        # ── Relative wrist velocity (wrist speed minus hip speed) ──
        rel_wrist_vel = state["rel_wrist_vel"]  # carry previous if can't compute
        if dt > 0:
            # Average wrist position
            wrist_indices = [_L_WRIST, _R_WRIST]
            hip_indices   = [_L_HIP,   _R_HIP]

            valid_wrists = [new_smooth_kp[i] for i in wrist_indices
                            if new_smooth_conf[i] >= config.KP_CONF_MIN]
            valid_hips   = [new_smooth_kp[i] for i in hip_indices
                            if new_smooth_conf[i] >= config.KP_CONF_MIN]

            if valid_wrists and valid_hips and state["prev_wrist"] is not None:
                curr_wrist = np.mean(valid_wrists, axis=0)
                curr_hip   = np.mean(valid_hips,   axis=0)
                prev_wrist = state["prev_wrist"]
                prev_hip   = state["prev_hip"]

                wrist_speed = math.hypot(*(curr_wrist - prev_wrist)) / dt
                hip_speed   = math.hypot(*(curr_hip   - prev_hip))   / dt

                # Relative speed: how fast the wrist moves relative to the body
                raw_rel = max(wrist_speed - hip_speed, 0.0)

                # EMA smooth the relative velocity to prevent 1-frame spikes
                rel_wrist_vel = (
                    0.45 * raw_rel +
                    0.55 * state["rel_wrist_vel"]
                )

                state["prev_wrist"] = curr_wrist
                state["prev_hip"]   = curr_hip
            elif valid_wrists and valid_hips:
                # First frame with valid wrists and hips — initialise positions
                state["prev_wrist"] = np.mean(valid_wrists, axis=0)
                state["prev_hip"]   = np.mean(valid_hips,   axis=0)

        state["smooth_kp"]     = new_smooth_kp
        state["smooth_conf"]   = new_smooth_conf
        state["prev_time"]     = current_time
        state["rel_wrist_vel"] = rel_wrist_vel

        return new_smooth_kp, new_smooth_conf, rel_wrist_vel

    def update(self, tracked_objects, video_timestamp=None):
        if not config.ENABLE_CONFLICT_DETECTION:
            return False, {}

        persons = [obj for obj in tracked_objects if obj["class"] == config.PERSON]
        if len(persons) < 2:
            # Still smooth single-person keypoints so state is ready when a second appears
            current_time = video_timestamp if video_timestamp is not None else time.time()
            for p in persons:
                sk, sc, _ = self._smooth_person(
                    p["id"], p.get("keypoints"), p.get("kp_conf"), current_time)
                p["_smooth_kp"]   = sk
                p["_smooth_conf"] = sc
            # Clean up person state for IDs no longer tracked
            active_ids = {p["id"] for p in persons}
            for k in [k for k in self.person_kp if k not in active_ids]:
                del self.person_kp[k]
            return False, {}

        current_time = video_timestamp if video_timestamp is not None else time.time()

        # ── Smooth keypoints for all persons ──
        rel_wrist_vels = {}
        for p in persons:
            sk, sc, rv = self._smooth_person(
                p["id"], p.get("keypoints"), p.get("kp_conf"), current_time)
            p["_smooth_kp"]   = sk
            p["_smooth_conf"] = sc
            rel_wrist_vels[p["id"]] = rv

        any_confirmed  = False
        active_pairs   = set()
        pair_scores    = {}   # pair_key → cumulative fight score for this update

        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                idA = persons[i]["id"]
                idB = persons[j]["id"]

                centerA = self.compute_center(persons[i]["bbox"])
                centerB = self.compute_center(persons[j]["bbox"])
                areaA   = self.compute_area(persons[i]["bbox"])
                areaB   = self.compute_area(persons[j]["bbox"])

                dist = math.hypot(centerA[0] - centerB[0], centerA[1] - centerB[1])

                if dist > config.PROXIMITY_DISTANCE:
                    continue

                pair_key = tuple(sorted((idA, idB)))
                active_pairs.add(pair_key)

                # ── Initialise per-pair state ──
                if pair_key not in self.history:
                    self.history[pair_key] = {
                        "prev_distance":  dist,
                        "prev_velocity":  0.0,
                        "prev_areaA":     areaA,
                        "prev_areaB":     areaB,
                        "prev_time":      current_time,
                        "confirm_count":  0,
                        "calm_count":     0,
                        "conflict_start": None,
                        "close_since":    current_time,
                        "calm_contact":   False,
                        "fight_session":  0.0,   # cumulative fight score for this pair
                    }
                    continue

                prev = self.history[pair_key]
                dt = current_time - prev["prev_time"]
                if dt <= 0:
                    continue

                velocity     = (dist - prev["prev_distance"]) / dt
                acceleration = (velocity - prev["prev_velocity"]) / dt

                area_changeA = abs(areaA - prev["prev_areaA"]) / (prev["prev_areaA"] + 1e-5)
                area_changeB = abs(areaB - prev["prev_areaB"]) / (prev["prev_areaB"] + 1e-5)

                # ── Pose-based signals (using smoothed keypoints + relative velocity) ──
                conflict_boost, pose_suppress, fight_score = _pose_signals(
                    persons[i], persons[j],
                    rel_wrist_vel_A=rel_wrist_vels.get(idA),
                    rel_wrist_vel_B=rel_wrist_vels.get(idB),
                )

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

                # ── Per-pair fight session score ──
                # Accumulates when both persons are in proximity with active signals.
                # Decays slowly when no conflict signal present.
                if raw_conflict and not prev["calm_contact"]:
                    prev["fight_session"] += fight_score * dt
                else:
                    prev["fight_session"] = max(0.0, prev["fight_session"] - 0.5 * dt)

                pair_scores[pair_key] = prev["fight_session"]

                # ── Update history ──
                prev["prev_distance"] = dist
                prev["prev_velocity"] = velocity
                prev["prev_areaA"]    = areaA
                prev["prev_areaB"]    = areaB
                prev["prev_time"]     = current_time

        # Clean up pairs that left proximity
        for k in [k for k in self.history if k not in active_pairs]:
            del self.history[k]

        # Clean up person state for IDs no longer tracked
        active_person_ids = {p["id"] for p in persons}
        for k in [k for k in self.person_kp if k not in active_person_ids]:
            del self.person_kp[k]

        return any_confirmed, pair_scores
