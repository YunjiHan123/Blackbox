import time
from collections import deque


class DoorLockDetector:

    def __init__(
        self,
        roi_x_ratio=0.30,
        roi_y_ratio=0.65,
        roi_width_ratio=0.15,
        roi_height_ratio=0.23,
        wrist_confidence_threshold=0.5,
        required_frames=60,
        movement_history_size=20,
        decay_frames=2,
        movement_delta_threshold=40,
        cooldown_seconds=3.0,
        fps_hint=30.0,
    ):

        self.roi_x_ratio = roi_x_ratio
        self.roi_y_ratio = roi_y_ratio
        self.roi_width_ratio = roi_width_ratio
        self.roi_height_ratio = roi_height_ratio
        self.wrist_confidence_threshold = wrist_confidence_threshold
        self.required_frames = required_frames
        self.movement_history = deque(maxlen=movement_history_size)
        self.decay_frames = decay_frames
        self.movement_delta_threshold = movement_delta_threshold
        self.cooldown_seconds = cooldown_seconds
        self.fps_hint = fps_hint
        self.wrist_in_zone_frames = 0
        self.last_trigger_time = 0.0

    def _compute_roi(self, frame):

        height, width = frame.shape[:2]
        roi_x1 = int(width * self.roi_x_ratio)
        roi_y1 = int(height * self.roi_y_ratio)
        roi_width = int(width * self.roi_width_ratio)
        roi_height = int(height * self.roi_height_ratio)
        return [roi_x1, roi_y1, roi_x1 + roi_width, roi_y1 + roi_height]

    def analyze(self, frame, poses):

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
            "countdown_seconds": max(0.0, (self.required_frames - self.wrist_in_zone_frames) / self.fps_hint),
            "triggered": False,
            "should_capture": False,
            "event_name": None,
            "subtitle": None,
            "color": (0, 255, 0),
        }

        if target_wrist is None:
            self.wrist_in_zone_frames = max(0, self.wrist_in_zone_frames - self.decay_frames)
            if self.wrist_in_zone_frames == 0:
                self.movement_history.clear()
            event["frames_in_zone"] = self.wrist_in_zone_frames
            event["countdown_seconds"] = max(0.0, (self.required_frames - self.wrist_in_zone_frames) / self.fps_hint)
            return event

        self.wrist_in_zone_frames += 1
        self.movement_history.append(target_wrist)
        event["target_wrist"] = target_wrist
        event["frames_in_zone"] = self.wrist_in_zone_frames
        event["countdown_seconds"] = max(0.0, (self.required_frames - self.wrist_in_zone_frames) / self.fps_hint)
        event["color"] = (0, 0, 255)

        if self.wrist_in_zone_frames <= self.required_frames:
            return event

        xs = [point[0] for point in self.movement_history]
        ys = [point[1] for point in self.movement_history]
        dx = max(xs) - min(xs)
        dy = max(ys) - min(ys)

        if dx > self.movement_delta_threshold or dy > self.movement_delta_threshold:
            event_name = "shaking_hand_detected"
            subtitle = "Hand movement inside lock zone detected"
            color = (0, 165, 255)
        else:
            event_name = "door_lock_try_detected"
            subtitle = "A hand stayed inside the lock zone"
            color = (0, 0, 255)

        event["triggered"] = True
        event["event_name"] = event_name
        event["subtitle"] = subtitle
        event["color"] = color
        event["movement_dx"] = dx
        event["movement_dy"] = dy

        now = time.time()
        if now - self.last_trigger_time >= self.cooldown_seconds:
            self.last_trigger_time = now
            event["should_capture"] = True

        return event
