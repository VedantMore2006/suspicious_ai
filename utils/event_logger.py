import time


class EventLogger:
    def __init__(self):
        self.last_events = {}

    def log(self, event_type, message, cooldown=5):
        current_time = time.time()

        if event_type not in self.last_events:
            self.last_events[event_type] = 0

        if current_time - self.last_events[event_type] > cooldown:
            print(f"[ALERT] {message}")
            self.last_events[event_type] = current_time
