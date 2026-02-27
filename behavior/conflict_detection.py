import time
import config
from utils.geometry import get_center


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = boxAArea + boxBArea - interArea

    if union == 0:
        return 0

    return interArea / union


class ConflictDetector:
    def __init__(self):
        self.prev_positions = {}
        self.confirm_counter = 0

    def update(self, tracked_objects):
        if not config.ENABLE_CONFLICT_DETECTION:
            return False

        persons = [obj for obj in tracked_objects if obj["class"] == config.PERSON]

        if len(persons) < 2:
            self.confirm_counter = 0
            return False

        conflict_detected = False

        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                boxA = persons[i]["bbox"]
                boxB = persons[j]["bbox"]

                iou = compute_iou(boxA, boxB)

                if iou > config.PERSON_IOU_THRESHOLD:

                    idA = persons[i]["id"]
                    idB = persons[j]["id"]

                    centerA = get_center(boxA)
                    centerB = get_center(boxB)

                    motionA = 0
                    motionB = 0

                    if idA in self.prev_positions:
                        prevA = self.prev_positions[idA]
                        motionA = ((centerA[0]-prevA[0])**2 + (centerA[1]-prevA[1])**2)**0.5

                    if idB in self.prev_positions:
                        prevB = self.prev_positions[idB]
                        motionB = ((centerB[0]-prevB[0])**2 + (centerB[1]-prevB[1])**2)**0.5

                    if motionA > config.PERSON_MOTION_THRESHOLD or motionB > config.PERSON_MOTION_THRESHOLD:
                        conflict_detected = True

        # Update positions
        for person in persons:
            self.prev_positions[person["id"]] = get_center(person["bbox"])

        if conflict_detected:
            self.confirm_counter += 1
        else:
            self.confirm_counter = 0

        if self.confirm_counter >= config.CONFLICT_CONFIRM_FRAMES:
            return True

        return False
