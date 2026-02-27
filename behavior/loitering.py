import time
import math
import config


def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class LoiteringDetector:
    def __init__(self):
        self.person_state = {}

    def update(self, tracked_objects):
        current_time = time.time()
        suspicious_ids = []

        for obj in tracked_objects:
            obj_id = obj["id"]
            cls = obj["class"]
            bbox = obj["bbox"]

            if cls != config.PERSON:
                continue

            cx, cy = get_center(bbox)

            if obj_id not in self.person_state:
                self.person_state[obj_id] = {
                    "first_seen": current_time,
                    "last_position": (cx, cy),
                    "last_move_time": current_time,
                    "loiter_flag": False
                }
                continue

            state = self.person_state[obj_id]
            prev_pos = state["last_position"]

            move_dist = distance(prev_pos, (cx, cy))

            if move_dist > config.LOITER_MOVEMENT_THRESHOLD:
                state["last_move_time"] = current_time
                state["last_position"] = (cx, cy)

            stationary_time = current_time - state["last_move_time"]

            if stationary_time > config.LOITER_TIME:
                state["loiter_flag"] = True
                suspicious_ids.append(obj_id)

        active_ids = [obj["id"] for obj in tracked_objects]
        for saved_id in list(self.person_state.keys()):
            if saved_id not in active_ids:
                del self.person_state[saved_id]

        return suspicious_ids