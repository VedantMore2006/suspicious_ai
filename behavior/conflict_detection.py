import config
import time
import math


class ConflictDetector:
    def __init__(self):
        self.history = {}
        self.confirm_counter = 0

    def compute_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def compute_area(self, bbox):
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)

    def update(self, tracked_objects):
        if not config.ENABLE_CONFLICT_DETECTION:
            return False

        persons = [obj for obj in tracked_objects if obj["class"] == config.PERSON]

        if len(persons) < 2:
            self.confirm_counter = 0
            return False

        current_time = time.time()
        conflict = False

        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):

                idA = persons[i]["id"]
                idB = persons[j]["id"]

                centerA = self.compute_center(persons[i]["bbox"])
                centerB = self.compute_center(persons[j]["bbox"])

                areaA = self.compute_area(persons[i]["bbox"])
                areaB = self.compute_area(persons[j]["bbox"])

                distance = math.hypot(centerA[0] - centerB[0], centerA[1] - centerB[1])

                if distance > config.PROXIMITY_DISTANCE:
                    continue

                pair_key = tuple(sorted((idA, idB)))

                if pair_key not in self.history:
                    self.history[pair_key] = {
                        "prev_distance": distance,
                        "prev_velocity": 0,
                        "prev_areaA": areaA,
                        "prev_areaB": areaB,
                        "prev_time": current_time
                    }
                    continue

                prev = self.history[pair_key]

                dt = current_time - prev["prev_time"]
                if dt == 0:
                    continue

                velocity = (distance - prev["prev_distance"]) / dt
                acceleration = (velocity - prev["prev_velocity"]) / dt

                area_changeA = abs(areaA - prev["prev_areaA"]) / (prev["prev_areaA"] + 1e-5)
                area_changeB = abs(areaB - prev["prev_areaB"]) / (prev["prev_areaB"] + 1e-5)

                if (
                    abs(velocity) > config.DISTANCE_VELOCITY_THRESHOLD
                    and abs(acceleration) > config.ACCELERATION_THRESHOLD
                ) or (
                    area_changeA > config.AREA_CHANGE_THRESHOLD
                    or area_changeB > config.AREA_CHANGE_THRESHOLD
                ):
                    conflict = True

                self.history[pair_key] = {
                    "prev_distance": distance,
                    "prev_velocity": velocity,
                    "prev_areaA": areaA,
                    "prev_areaB": areaB,
                    "prev_time": current_time
                }

        if conflict:
            self.confirm_counter += 1
        else:
            self.confirm_counter = 0

        return self.confirm_counter >= config.CONFLICT_CONFIRM_FRAMES
