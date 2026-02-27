import time
import config
from utils.geometry import get_center, distance


class AbandonedObjectDetector:
    def __init__(self):
        self.bag_state = {}

    def update(self, tracked_objects):
        current_time = time.time()
        suspicious_bags = []

        persons = []
        bags = []

        for obj in tracked_objects:
            if obj["class"] == config.PERSON:
                persons.append(obj)
            elif obj["class"] in [config.BACKPACK, config.HANDBAG, config.SUITCASE]:
                bags.append(obj)

        active_bag_ids = []

        for bag in bags:
            bag_id = bag["id"]
            active_bag_ids.append(bag_id)

            bag_center = get_center(bag["bbox"])
            nearest_person_dist = float("inf")

            for person in persons:
                person_center = get_center(person["bbox"])
                dist = distance(bag_center, person_center)
                if dist < nearest_person_dist:
                    nearest_person_dist = dist

            if bag_id not in self.bag_state:
                self.bag_state[bag_id] = {
                    "last_seen": current_time,
                    "last_near_time": current_time,
                    "abandoned": False
                }

            state = self.bag_state[bag_id]
            state["last_seen"] = current_time

            # If someone is near bag
            if nearest_person_dist < config.ABANDON_DISTANCE:
                state["last_near_time"] = current_time
                state["abandoned"] = False

            else:
                time_away = current_time - state["last_near_time"]

                if time_away > config.ABANDON_TIME:
                    state["abandoned"] = True

            if state["abandoned"]:
                suspicious_bags.append(bag_id)

        # Grace period for flicker (IMPORTANT)
        GRACE_PERIOD = 0.7  # seconds

        for saved_id in list(self.bag_state.keys()):
            state = self.bag_state[saved_id]

            # If bag not seen recently
            if current_time - state["last_seen"] > GRACE_PERIOD:
                del self.bag_state[saved_id]

        return suspicious_bags