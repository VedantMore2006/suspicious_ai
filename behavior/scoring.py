import config


class ThreatScorer:
    def __init__(self):
        self.instant_scores = {}
        self.session_scores = {}

    def update(self, tracked_objects, loiter_ids, abandoned_bags, conflict_flag):
        persons = [o for o in tracked_objects if o["class"] == config.PERSON]

        current_ids = set()

        for person in persons:
            pid = person["id"]
            current_ids.add(pid)

            score = 0

            if pid in loiter_ids:
                score += 1

            if conflict_flag:
                score += 4

            if abandoned_bags:
                score += 2

            self.instant_scores[pid] = score

            if pid not in self.session_scores:
                self.session_scores[pid] = 0

            # Only add if non-zero event
            if score > 0:
                self.session_scores[pid] += score

        # Remove IDs no longer present
        for saved_id in list(self.instant_scores.keys()):
            if saved_id not in current_ids:
                del self.instant_scores[saved_id]

        return self.instant_scores, self.session_scores

    def get_level(self, score):
        if score >= 5:
            return "HIGH"
        elif score >= 3:
            return "SUSPICIOUS"
        else:
            return "NORMAL"