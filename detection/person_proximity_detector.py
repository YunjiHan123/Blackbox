import time

from settings import DEVICE

from core.event_types import EVENT_FACE_NEAR


class PersonProximityDetector:

    def __init__(
        self,
        model,
        area_threshold=0.4,
        required_time_seconds=2.0,
        min_confidence=0.35,
        cooldown_seconds=3.0,
    ):

        self.model = model
        self.class_names = self.model.names
        self.area_threshold = area_threshold
        self.required_time_seconds = required_time_seconds
        self.min_confidence = min_confidence
        self.cooldown_seconds = cooldown_seconds
        self.start_time = None
        self.last_trigger_time = 0.0

    def analyze(self, frame, timestamp=None):

        results = self.model(frame, device=DEVICE, verbose=False)
        frame_area = frame.shape[0] * frame.shape[1]
        best_candidate = None

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.class_names.get(class_id, "unknown")

                if class_name != "person":
                    continue

                confidence = float(box.conf[0])
                if confidence < self.min_confidence:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                area_ratio = ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / float(frame_area)

                if area_ratio < self.area_threshold:
                    continue

                candidate = {
                    "class_name": EVENT_FACE_NEAR,
                    "confidence": confidence,
                    "bbox": bbox,
                    "area_ratio": area_ratio,
                }

                if best_candidate is None or area_ratio > best_candidate["area_ratio"]:
                    best_candidate = candidate

        if best_candidate is None:
            self.start_time = None
            return None

        now = time.time() if timestamp is None else timestamp
        if self.start_time is None:
            self.start_time = now

        elapsed = now - self.start_time
        remaining = max(0.0, self.required_time_seconds - elapsed)
        triggered = elapsed >= self.required_time_seconds

        should_capture = False
        if triggered and now - self.last_trigger_time >= self.cooldown_seconds:
            self.last_trigger_time = now
            should_capture = True

        best_candidate.update({
            "elapsed": elapsed,
            "remaining": remaining,
            "triggered": triggered,
            "should_capture": should_capture,
        })
        return best_candidate
