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


def _pose_signals(
    personA,
    personB,
    rel_wrist_vel_A=None,
    rel_wrist_vel_B=None,
    raw_rel_wrist_vel_A=None,
    raw_rel_wrist_vel_B=None,
    windup_speed_A=0.0,
    windup_speed_B=0.0,
):
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
        return False, False, 0.0, [], [], False

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
    fast_track     = False
    fight_score    = 0.0

    # debug_signals: per-person event labels for the drawing layer.
    # Format: (person_ref, label_string) so drawing.py can attach to correct person.
    signals_A = []
    signals_B = []

    # Opponent torso centers for light punch/shove detection.
    x1A, y1A, x2A, y2A = personA["bbox"]
    x1B, y1B, x2B, y2B = personB["bbox"]
    torso_A = ((x1A + x2A) / 2.0, (y1A + y2A) / 2.0)
    torso_B = ((x1B + x2B) / 2.0, (y1B + y2B) / 2.0)
    torso_r_A = max(15.0, min(x2A - x1A, y2A - y1A) * config.TORSO_STRIKE_RADIUS_RATIO)
    torso_r_B = max(15.0, min(x2B - x1B, y2B - y1B) * config.TORSO_STRIKE_RADIUS_RATIO)
    head_radius_A = max(15.0, (y2A - y1A) * 0.12)
    head_radius_B = max(15.0, (y2B - y1B) * 0.12)

    # ── Conflict signal 1: wrist of A near nose of B (strike zone) ──
    if nose_B and wrists_A:
        for w_idx, w in zip([_L_WRIST, _R_WRIST], [wL_A, wR_A]):
            if w and math.hypot(w[0] - nose_B[0], w[1] - nose_B[1]) < head_radius_B:
                conflict_boost = True
                fight_score += 3.0
                signals_A.append(f"KP{w_idx}→nose[B]: STRIKE ZONE")

    # ── Conflict signal 1 (reciprocal): wrist of B near nose of A ──
    if nose_A and wrists_B:
        for w_idx, w in zip([_L_WRIST, _R_WRIST], [wL_B, wR_B]):
            if w and math.hypot(w[0] - nose_A[0], w[1] - nose_A[1]) < head_radius_A:
                conflict_boost = True
                fight_score += 3.0
                signals_B.append(f"KP{w_idx}→nose[A]: STRIKE ZONE")

    # ── Conflict signal 1b: wrist of A/B entering opponent torso strike zone ──
    if wrists_A:
        for w_idx, w in zip([_L_WRIST, _R_WRIST], [wL_A, wR_A]):
            if w and math.hypot(w[0] - torso_B[0], w[1] - torso_B[1]) < torso_r_B:
                conflict_boost = True
                fight_score += 1.8
                signals_A.append(f"KP{w_idx}→torso[B]: BODY HIT")

    if wrists_B:
        for w_idx, w in zip([_L_WRIST, _R_WRIST], [wL_B, wR_B]):
            if w and math.hypot(w[0] - torso_A[0], w[1] - torso_A[1]) < torso_r_A:
                conflict_boost = True
                fight_score += 1.8
                signals_B.append(f"KP{w_idx}→torso[A]: BODY HIT")

    # ── Conflict signal 2: wrist raised well above own shoulder ──
    if shoulders_A and wrists_A:
        avg_sh_y = sum(s[1] for s in shoulders_A) / len(shoulders_A)
        for w_idx, w in zip([_L_WRIST, _R_WRIST], [wL_A, wR_A]):
            if w and w[1] < avg_sh_y - 20:
                conflict_boost = True
                fight_score += 1.5
                signals_A.append(f"KP{w_idx}<KP5/6: ARM RAISED")

    if shoulders_B and wrists_B:
        avg_sh_y = sum(s[1] for s in shoulders_B) / len(shoulders_B)
        for w_idx, w in zip([_L_WRIST, _R_WRIST], [wL_B, wR_B]):
            if w and w[1] < avg_sh_y - 20:
                conflict_boost = True
                fight_score += 1.5
                signals_B.append(f"KP{w_idx}<KP5/6: ARM RAISED")

    # ── Conflict signal 3: high relative wrist velocity (body-motion corrected) ──
    rel_vel_thresh = config.RELATIVE_WRIST_VEL_THRESHOLD
    if rel_wrist_vel_A is not None and rel_wrist_vel_A > rel_vel_thresh:
        fight_score += min(rel_wrist_vel_A / rel_vel_thresh, 2.0)
        if rel_wrist_vel_A > rel_vel_thresh * 1.5:
            conflict_boost = True
        signals_A.append(f"KP9/10 rel-vel: {rel_wrist_vel_A:.0f}px/s")
    if rel_wrist_vel_B is not None and rel_wrist_vel_B > rel_vel_thresh:
        fight_score += min(rel_wrist_vel_B / rel_vel_thresh, 2.0)
        if rel_wrist_vel_B > rel_vel_thresh * 1.5:
            conflict_boost = True
        signals_B.append(f"KP9/10 rel-vel: {rel_wrist_vel_B:.0f}px/s")

    # ── Fast-track trigger: raw (unsmoothed) velocity spike catches quick punches ──
    raw_fast_thresh = rel_vel_thresh * config.FAST_TRACK_RAW_MULTIPLIER
    if raw_rel_wrist_vel_A is not None and raw_rel_wrist_vel_A > raw_fast_thresh:
        conflict_boost = True
        fast_track = True
        fight_score += config.FAST_TRACK_IMPACT_SCORE
        signals_A.append(f"FAST raw-rel: {raw_rel_wrist_vel_A:.0f}px/s")
    if raw_rel_wrist_vel_B is not None and raw_rel_wrist_vel_B > raw_fast_thresh:
        conflict_boost = True
        fast_track = True
        fight_score += config.FAST_TRACK_IMPACT_SCORE
        signals_B.append(f"FAST raw-rel: {raw_rel_wrist_vel_B:.0f}px/s")

    # ── Anticipation signal: elbow retracts toward shoulder (wind-up) ──
    if windup_speed_A > config.WINDUP_SHRINK_SPEED_THRESHOLD:
        conflict_boost = True
        fight_score += 1.2
        signals_A.append(f"WIND-UP: {windup_speed_A:.0f}px/s")
    if windup_speed_B > config.WINDUP_SHRINK_SPEED_THRESHOLD:
        conflict_boost = True
        fight_score += 1.2
        signals_B.append(f"WIND-UP: {windup_speed_B:.0f}px/s")

    # ── Friendly signal: handshake — wrists near own hip level ──
    if hips_A and wrists_A:
        avg_hip_y = sum(h[1] for h in hips_A) / len(hips_A)
        if all(abs(w[1] - avg_hip_y) < config.HIP_TOLERANCE for w in wrists_A):
            suppress = True
            signals_A.append("KP9/10≈KP11/12: HANDSHAKE SUPPRESS")

    if hips_B and wrists_B:
        avg_hip_y = sum(h[1] for h in hips_B) / len(hips_B)
        if all(abs(w[1] - avg_hip_y) < config.HIP_TOLERANCE for w in wrists_B):
            suppress = True
            signals_B.append("KP9/10≈KP11/12: HANDSHAKE SUPPRESS")

    # ── Symmetry heuristic: mirrored/similar movement is often non-violent contact ──
    if rel_wrist_vel_A is not None and rel_wrist_vel_B is not None:
        hi = max(rel_wrist_vel_A, rel_wrist_vel_B, 1e-6)
        if hi > 20.0:
            lo = min(rel_wrist_vel_A, rel_wrist_vel_B)
            symmetry = lo / hi
            if symmetry >= config.SYMMETRY_SCORE_THRESHOLD and not fast_track and not conflict_boost:
                suppress = True
                fight_score = max(0.0, fight_score - 0.8)
                signals_A.append(f"SYNC: {symmetry:.2f}")
                signals_B.append(f"SYNC: {symmetry:.2f}")

    # A conflict boost always overrides a suppress signal
    if conflict_boost:
        suppress = False
        fight_score = max(fight_score, 2.0)

    if suppress:
        fight_score = 0.0

    return conflict_boost, suppress, fight_score, signals_A, signals_B, fast_track


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
        Returns (smooth_kp, smooth_conf, rel_wrist_vel, raw_rel_wrist_vel, windup_speed)
        where rel_wrist_vel is smoothed body-motion-corrected wrist speed,
        raw_rel_wrist_vel is the instantaneous unsmoothed value, and
        windup_speed is elbow-to-shoulder shrink speed used for anticipation.
        """
        if kp_xy is None or kp_conf is None:
            return None, None, None, None, 0.0

        if pid not in self.person_kp:
            # First observation — initialise with raw values
            self.person_kp[pid] = {
                "smooth_kp":   kp_xy.copy(),
                "smooth_conf": kp_conf.copy(),
                "prev_wrist":  None,   # average wrist position
                "prev_hip":    None,   # average hip position
                "prev_time":   current_time,
                "last_seen_active": current_time,
                "rel_wrist_vel": 0.0,
                "raw_rel_wrist_vel": 0.0,
                "prev_arm_dist": None,
                "windup_speed": 0.0,
            }
            return kp_xy, kp_conf, 0.0, 0.0, 0.0

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
        raw_rel_wrist_vel = state["raw_rel_wrist_vel"]
        windup_speed = state["windup_speed"]
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
                raw_rel_wrist_vel = raw_rel

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

            # Elbow-to-shoulder shrink speed (wind-up anticipation)
            arm_pairs = [(_L_SHOULDER, _L_ELBOW), (_R_SHOULDER, _R_ELBOW)]
            arm_distances = []
            for sh_idx, el_idx in arm_pairs:
                if (
                    new_smooth_conf[sh_idx] >= config.KP_CONF_MIN and
                    new_smooth_conf[el_idx] >= config.KP_CONF_MIN
                ):
                    sh = new_smooth_kp[sh_idx]
                    el = new_smooth_kp[el_idx]
                    arm_distances.append(math.hypot(float(el[0] - sh[0]), float(el[1] - sh[1])))

            if arm_distances:
                curr_arm_dist = float(np.mean(arm_distances))
                if state["prev_arm_dist"] is not None:
                    # Positive value means elbow is pulling inward toward shoulder.
                    windup_speed = max((state["prev_arm_dist"] - curr_arm_dist) / dt, 0.0)
                state["prev_arm_dist"] = curr_arm_dist

        state["smooth_kp"]     = new_smooth_kp
        state["smooth_conf"]   = new_smooth_conf
        state["prev_time"]     = current_time
        state["last_seen_active"] = current_time
        state["rel_wrist_vel"] = rel_wrist_vel
        state["raw_rel_wrist_vel"] = raw_rel_wrist_vel
        state["windup_speed"] = windup_speed

        return new_smooth_kp, new_smooth_conf, rel_wrist_vel, raw_rel_wrist_vel, windup_speed

    def update(self, tracked_objects, video_timestamp=None):
        if not config.ENABLE_CONFLICT_DETECTION:
            return False, {}

        persons = [obj for obj in tracked_objects if obj["class"] == config.PERSON]
        if len(persons) < 2:
            # Still smooth single-person keypoints so state is ready when a second appears
            current_time = video_timestamp if video_timestamp is not None else time.time()
            for p in persons:
                sk, sc, _, _, _ = self._smooth_person(
                    p["id"], p.get("keypoints"), p.get("kp_conf"), current_time)
                p["_smooth_kp"]   = sk
                p["_smooth_conf"] = sc

        current_time = video_timestamp if video_timestamp is not None else time.time()

        # ── Smooth keypoints for all persons ──
        rel_wrist_vels = {}
        raw_rel_wrist_vels = {}
        windup_speeds = {}
        for p in persons:
            sk, sc, rv, raw_rv, windup_speed = self._smooth_person(
                p["id"], p.get("keypoints"), p.get("kp_conf"), current_time)
            p["_smooth_kp"]      = sk
            p["_smooth_conf"]    = sc
            p["_rel_wrist_vel"]  = rv if rv is not None else 0.0
            p["_signals"]        = []   # reset signal list each frame
            rel_wrist_vels[p["id"]] = rv
            raw_rel_wrist_vels[p["id"]] = raw_rv
            windup_speeds[p["id"]] = windup_speed

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
                        "last_seen_active": current_time,
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
                conflict_boost, pose_suppress, fight_score, sig_A, sig_B, fast_track = _pose_signals(
                    persons[i], persons[j],
                    rel_wrist_vel_A=rel_wrist_vels.get(idA),
                    rel_wrist_vel_B=rel_wrist_vels.get(idB),
                    raw_rel_wrist_vel_A=raw_rel_wrist_vels.get(idA),
                    raw_rel_wrist_vel_B=raw_rel_wrist_vels.get(idB),
                    windup_speed_A=windup_speeds.get(idA, 0.0),
                    windup_speed_B=windup_speeds.get(idB, 0.0),
                )
                # Write signal labels back into person dicts for drawing layer
                persons[i].setdefault("_signals", [])
                persons[j].setdefault("_signals", [])
                persons[i]["_signals"].extend(sig_A)
                persons[j]["_signals"].extend(sig_B)

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
                is_separating = velocity > 15.0  # Tweak: slightly higher threshold for a clean separation
                is_struggling = area_changeA > 0.25 or area_changeB > 0.25

                # Capture the state before we modify it.
                was_calm = prev["calm_contact"]

                if (
                    abs(velocity) < config.CALM_VELOCITY_THRESHOLD and
                    time_close > config.CALM_CONTACT_TIME
                    and not is_struggling
                ):
                    prev["calm_contact"] = True

                if pose_suppress and not is_struggling:
                    prev["calm_contact"] = True

                # The fix: only allow pose-based conflict to break calm state if they are not stepping back.
                if (conflict_boost and not is_separating) or (abs(velocity) > config.DISTANCE_VELOCITY_THRESHOLD * 2 and not is_separating):
                    prev["calm_contact"] = False

                # Grace period: if they were calm and are currently separating, enforce the calm state.
                if was_calm and is_separating:
                    prev["calm_contact"] = True
                    fight_score = 0.0  # Zero out temporary threat spikes from arm retraction.

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
                    if fast_track:
                        prev["fight_session"] += config.FAST_TRACK_IMPACT_SCORE
                else:
                    prev["fight_session"] = max(0.0, prev["fight_session"] - 0.5 * dt)

                pair_scores[pair_key] = prev["fight_session"]

                # Proactive score gate: trigger as soon as pair score crosses threshold.
                if prev["fight_session"] >= config.FIGHT_SESSION_TRIGGER:
                    any_confirmed = True

                # ── Update history ──
                prev["prev_distance"] = dist
                prev["prev_velocity"] = velocity
                prev["prev_areaA"]    = areaA
                prev["prev_areaB"]    = areaB
                prev["prev_time"]     = current_time
                prev["last_seen_active"] = current_time

        # Clean up pairs that left proximity after a short grace period.
        for k in list(self.history.keys()):
            if k not in active_pairs:
                if current_time - self.history[k].get("last_seen_active", self.history[k]["prev_time"]) > 1.5:
                    del self.history[k]

        # Clean up person state after a short grace period so flickering IDs can recover.
        active_person_ids = {p["id"] for p in persons}
        for k in list(self.person_kp.keys()):
            if k not in active_person_ids:
                if current_time - self.person_kp[k].get("last_seen_active", self.person_kp[k]["prev_time"]) > 1.5:
                    del self.person_kp[k]

        return any_confirmed, pair_scores
