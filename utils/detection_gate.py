from collections import deque


class DetectionGate:

    def __init__(self, min_confidence=0.6, window_size=10, required_hits=6, cooldown_frames=30):

        self.min_confidence = min_confidence
        self.window_size = window_size
        self.required_hits = required_hits
        self.cooldown_frames = cooldown_frames
        self.histories = {}
        self.cooldowns = {}
        self.last_qualified = {}

    def update(self, detections):

        triggered = []
        qualified = {}

        for label in list(self.cooldowns.keys()):
            if self.cooldowns[label] > 0:
                self.cooldowns[label] -= 1

        for det in detections:

            label = det["class_name"]
            confidence = det["confidence"]

            if confidence < self.min_confidence:
                continue

            prev = qualified.get(label)
            if prev is None or confidence > prev["confidence"]:
                qualified[label] = det
                self.last_qualified[label] = det

        for label in set(self.histories) | set(qualified):
            history = self.histories.setdefault(label, deque(maxlen=self.window_size))
            history.append(1 if label in qualified else 0)

            if sum(history) >= self.required_hits and self.cooldowns.get(label, 0) == 0:
                det = qualified.get(label) or self.last_qualified.get(label)
                if det is None:
                    continue
                triggered.append(det)
                self.cooldowns[label] = self.cooldown_frames
                history.clear()

        return triggered
