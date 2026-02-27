import config
import time
import math


class PhoneBehaviorDetector:
    def __init__(self):
        self.prev_phone_positions = {}
        self.confirm_counter = {}

    def get_vertical_zone(self, phone_center_y, person_bbox):
        x1, y1, x2, y2 = person_bbox
        height = y2 - y1

        relative_y = phone_center_y - y1
        ratio = relative_y / height

        if ratio < config.PHONE_FACE_ZONE:
            return "ACTIVE"
        elif ratio < config.PHONE_TORSO_ZONE:
            return "HOLDING"
        else:
            return "POCKET"

    def update(self, tracked_objects):

        if not config.ENABLE_PHONE_BEHAVIOR:
            return {}

        persons = [o for o in tracked_objects if o["class"] == config.PERSON]
        phones = [o for o in tracked_objects if o["class"] == config.CELL_PHONE]

        results = {}
        current_time = time.time()

        for person in persons:
            pid = person["id"]
            px1, py1, px2, py2 = person["bbox"]
            person_center_x = (px1 + px2) / 2

            nearest_phone = None
            min_dist = float("inf")

            for phone in phones:
                fx1, fy1, fx2, fy2 = phone["bbox"]
                phone_center = ((fx1 + fx2) / 2, (fy1 + fy2) / 2)

                # simple horizontal proximity check
                if px1 <= phone_center[0] <= px2:
                    dist = abs(phone_center[0] - person_center_x)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_phone = phone

            if nearest_phone is None:
                continue

            fx1, fy1, fx2, fy2 = nearest_phone["bbox"]
            phone_center = ((fx1 + fx2) / 2, (fy1 + fy2) / 2)

            state = self.get_vertical_zone(phone_center[1], person["bbox"])

            misuse = False

            # Detect rapid phone raise (possible recording)
            if pid in self.prev_phone_positions:
                prev = self.prev_phone_positions[pid]
                dt = current_time - prev["time"]

                if dt > 0:
                    velocity_y = (prev["pos"][1] - phone_center[1]) / dt

                    if velocity_y > config.PHONE_RAISE_SPEED_THRESHOLD:
                        misuse = True

            self.prev_phone_positions[pid] = {
                "pos": phone_center,
                "time": current_time
            }

            # Confirmation logic
            if misuse:
                if pid not in self.confirm_counter:
                    self.confirm_counter[pid] = 0
                self.confirm_counter[pid] += 1
            else:
                self.confirm_counter[pid] = 0

            if self.confirm_counter.get(pid, 0) >= config.PHONE_MISUSE_CONFIRM_FRAMES:
                misuse = True
            else:
                misuse = False

            results[pid] = {
                "state": state,
                "misuse": misuse
            }

        return results
