import config


class ThreatScorer:
    def __init__(self):
        self.instant_scores = {}
        self.session_scores = {}

    def update(self, tracked_objects, loiter_ids, abandoned_bags, conflict_flag, pair_scores=None):
        persons = [o for o in tracked_objects if o["class"] == config.PERSON]
        pair_scores = pair_scores or {}

        # Build per-person fight score from pair_scores:
        # a person's fight contribution = max fight_session across all pairs they appear in
        person_fight_score = {}
        for (idA, idB), fscore in pair_scores.items():
            person_fight_score[idA] = max(person_fight_score.get(idA, 0.0), fscore)
            person_fight_score[idB] = max(person_fight_score.get(idB, 0.0), fscore)

        current_ids = set()

        for person in persons:
            pid = person["id"]
            current_ids.add(pid)

            score = 0

            if pid in loiter_ids:
                score += 1

            if conflict_flag:
                score += 4

            # Per-pair fight session contributes a scaled bonus (capped at +3)
            # This rewards sustained aggression without double-counting the conflict flag
            fscore = person_fight_score.get(pid, 0.0)
            if fscore > 0:
                score += min(int(fscore / 5), 3)

            if abandoned_bags:
                score += 2

            self.instant_scores[pid] = score

            if pid not in self.session_scores:
                self.session_scores[pid] = 0

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