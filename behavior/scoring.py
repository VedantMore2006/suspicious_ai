import config


class ThreatScorer:
    def __init__(self):
        self.scores = {}

    def update(self, tracked_objects, loiter_ids, abandoned_bags, conflict_flag, phone_results):
        persons = [o for o in tracked_objects if o["class"] == config.PERSON]

        for person in persons:
            pid = person["id"]
            score = 0

            if pid in loiter_ids:
                score += 1

            if conflict_flag:
                score += 4

            if pid in phone_results and phone_results[pid]["misuse"]:
                score += 2

            # optional: proximity to abandoned bag
            if abandoned_bags:
                score += 2

            self.scores[pid] = score

        return self.scores

    def get_level(self, score):
        if score >= 5:
            return "HIGH"
        elif score >= 3:
            return "SUSPICIOUS"
        else:
            return "NORMAL"