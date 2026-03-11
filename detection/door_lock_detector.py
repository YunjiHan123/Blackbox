import time
from collections import deque

from core.event_types import EVENT_DOOR_LOCK_MANIPULATION, EVENT_LABELS


class DoorLockDetector:

    def __init__(
        self,
        roi_x_ratio=0.30,
        roi_y_ratio=0.65,
        roi_width_ratio=0.15,
        roi_height_ratio=0.23,
        wrist_confidence_threshold=0.5,
        required_seconds=2.0,
        movement_history_size=20,
        decay_frames=2,
        movement_delta_threshold=40,
        cooldown_seconds=3.0,
    ):

        self.roi_x_ratio = roi_x_ratio
        self.roi_y_ratio = roi_y_ratio
        self.roi_width_ratio = roi_width_ratio
        self.roi_height_ratio = roi_height_ratio
        self.wrist_confidence_threshold = wrist_confidence_threshold
        self.required_seconds = required_seconds
        self.movement_history = deque(maxlen=movement_history_size)
        self.decay_frames = decay_frames
        self.movement_delta_threshold = movement_delta_threshold
        self.cooldown_seconds = cooldown_seconds
        self.wrist_in_zone_frames = 0
        self.last_trigger_time = 0.0
        self.zone_entered_at = None

    def _compute_roi(self, frame):

        height, width = frame.shape[:2]
        roi_x1 = int(width * self.roi_x_ratio)
        roi_y1 = int(height * self.roi_y_ratio)
        roi_width = int(width * self.roi_width_ratio)
        roi_height = int(height * self.roi_height_ratio)
        return [roi_x1, roi_y1, roi_x1 + roi_width, roi_y1 + roi_height]

    def analyze(self, frame, poses, timestamp=None):

        roi = self._compute_roi(frame)
        target_wrist = None

        for pose in poses:
            for wrist_index in (9, 10):
                x, y, confidence = pose["keypoints"][wrist_index]
                if confidence < self.wrist_confidence_threshold:
                    continue

                wrist_point = (int(x), int(y))
                if roi[0] < wrist_point[0] < roi[2] and roi[1] < wrist_point[1] < roi[3]:
                    target_wrist = wrist_point
                    break

            if target_wrist is not None:
                break

        event = {
            "roi": roi,
            "target_wrist": target_wrist,
            "frames_in_zone": self.wrist_in_zone_frames,
            "elapsed_seconds": 0.0,
            "countdown_seconds": self.required_seconds,
            "triggered": False,
            "should_capture": False,
            "event_name": None,
            "subtitle": None,
            "color": (0, 0, 0),
        }

        if target_wrist is None:
            self.wrist_in_zone_frames = max(0, self.wrist_in_zone_frames - self.decay_frames)
            if self.wrist_in_zone_frames == 0:
                self.movement_history.clear()
                self.zone_entered_at = None
            event["frames_in_zone"] = self.wrist_in_zone_frames
            return event

        now = time.time() if timestamp is None else timestamp
        if self.zone_entered_at is None:
            self.zone_entered_at = now

        self.wrist_in_zone_frames += 1
        self.movement_history.append(target_wrist)
        elapsed_seconds = now - self.zone_entered_at
        event["target_wrist"] = target_wrist
        event["frames_in_zone"] = self.wrist_in_zone_frames
        event["elapsed_seconds"] = elapsed_seconds
        event["countdown_seconds"] = max(0.0, self.required_seconds - elapsed_seconds)
        event["color"] = (0, 165, 255)

        if elapsed_seconds < self.required_seconds:
            return event

        xs = [point[0] for point in self.movement_history]
        ys = [point[1] for point in self.movement_history]
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)

        event_name = EVENT_DOOR_LOCK_MANIPULATION
        subtitle = EVENT_LABELS[EVENT_DOOR_LOCK_MANIPULATION]

        color = (0, 165, 255)

        event["triggered"] = True
        event["event_name"] = event_name
        event["subtitle"] = subtitle
        event["color"] = color
        event["movement_dx"] = dx
        event["movement_dy"] = dy

        if now - self.last_trigger_time >= self.cooldown_seconds:
            self.last_trigger_time = now
            event["should_capture"] = True

        return event
