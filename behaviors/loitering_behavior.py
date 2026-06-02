import time


class LoiteringDetector:
    def __init__(self, threshold, disappear_reset=60, max_visible_gap=1.0):
        self.threshold = threshold
        self.disappear_reset = disappear_reset
        self.max_visible_gap = max_visible_gap
        self.person_data = {}

    def update(self, person_id):
        return self.update_with_timestamp(person_id, time.time())

    def update_with_timestamp(self, person_id, timestamp_seconds):
        now = timestamp_seconds
        data = self.person_data.get(person_id)

        if data is None:
            self.person_data[person_id] = {
                "last_seen": now,
                "accum": 0.0,
                "alerted": False,
            }
            return False

        time_since_last = max(0.0, now - data["last_seen"])
        if time_since_last > self.disappear_reset:
            self.person_data[person_id] = {
                "last_seen": now,
                "accum": 0.0,
                "alerted": False,
            }
            return False

        if time_since_last <= self.max_visible_gap:
            data["accum"] += time_since_last
        data["last_seen"] = now

        if data["accum"] >= self.threshold and not data["alerted"]:
            data["alerted"] = True
            return True

        return False

    def get_duration(self, person_id):
        data = self.person_data.get(person_id)
        if data is None:
            return 0.0
        return data["accum"]
