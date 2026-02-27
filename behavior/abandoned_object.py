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

        for bag in bags:
            bag_id = bag["id"]
            bag_center = get_center(bag["bbox"])

            nearest_person_dist = float("inf")

            for person in persons:
                person_center = get_center(person["bbox"])
                dist = distance(bag_center, person_center)
                if dist < nearest_person_dist:
                    nearest_person_dist = dist

            if bag_id not in self.bag_state:
                self.bag_state[bag_id] = {
                    "first_seen": current_time,
                    "last_near_time": current_time,
                    "abandoned": False
                }

            state = self.bag_state[bag_id]

            if nearest_person_dist < config.ABANDON_DISTANCE:
                state["last_near_time"] = current_time
            else:
                time_away = current_time - state["last_near_time"]
                if time_away > config.ABANDON_TIME:
                    state["abandoned"] = True
                    suspicious_bags.append(bag_id)

        # Cleanup removed bags
        active_bag_ids = [bag["id"] for bag in bags]
        for saved_id in list(self.bag_state.keys()):
            if saved_id not in active_bag_ids:
                del self.bag_state[saved_id]

        return suspicious_bags