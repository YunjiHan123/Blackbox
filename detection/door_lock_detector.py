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
        contact_roi_width_ratio=0.45,
        contact_roi_height_ratio=0.45,
        wrist_confidence_threshold=0.5,
        required_seconds=2.0,
        miss_grace_seconds=1.0,
        movement_history_size=20,
        decay_frames=2,
        movement_delta_threshold=40,
        min_person_height_ratio=0.28,
        min_shoulder_width_ratio=0.10,
        hand_extension_ratio=0.45,
        wrist_distance_shoulder_ratio=0.75,
        min_wrist_distance_pixels=24,
        cooldown_seconds=3.0,
    ):

        self.roi_x_ratio = roi_x_ratio
        self.roi_y_ratio = roi_y_ratio
        self.roi_width_ratio = roi_width_ratio
        self.roi_height_ratio = roi_height_ratio
        self.contact_roi_width_ratio = contact_roi_width_ratio
        self.contact_roi_height_ratio = contact_roi_height_ratio
        self.wrist_confidence_threshold = wrist_confidence_threshold
        self.required_seconds = required_seconds
        self.miss_grace_seconds = miss_grace_seconds
        self.movement_history = deque(maxlen=movement_history_size)
        self.decay_frames = decay_frames
        self.movement_delta_threshold = movement_delta_threshold
        self.min_person_height_ratio = min_person_height_ratio
        self.min_shoulder_width_ratio = min_shoulder_width_ratio
        self.hand_extension_ratio = hand_extension_ratio
        self.wrist_distance_shoulder_ratio = wrist_distance_shoulder_ratio
        self.min_wrist_distance_pixels = min_wrist_distance_pixels
        self.cooldown_seconds = cooldown_seconds
        self.wrist_in_zone_frames = 0
        self.last_trigger_time = 0.0
        self.zone_entered_at = None
        self.accumulated_seconds = 0.0
        self.missed_since = None

    def _compute_roi(self, frame):

        height, width = frame.shape[:2]
        roi_x1 = int(width * self.roi_x_ratio)
        roi_y1 = int(height * self.roi_y_ratio)
        roi_width = int(width * self.roi_width_ratio)
        roi_height = int(height * self.roi_height_ratio)
        return [roi_x1, roi_y1, roi_x1 + roi_width, roi_y1 + roi_height]

    def _compute_contact_roi(self, roi):

        x1, y1, x2, y2 = roi
        width = x2 - x1
        height = y2 - y1
        contact_width = max(1, int(width * self.contact_roi_width_ratio))
        contact_height = max(1, int(height * self.contact_roi_height_ratio))
        center_x = x1 + width // 2
        center_y = y1 + height // 2
        contact_x1 = center_x - contact_width // 2
        contact_y1 = center_y - contact_height // 2
        return [
            contact_x1,
            contact_y1,
            contact_x1 + contact_width,
            contact_y1 + contact_height,
        ]

    def _compute_contact_band(self, frame, contact_roi):

        return [
            contact_roi[0],
            contact_roi[1],
            contact_roi[2],
            frame.shape[0] - 1,
        ]

    def _extend_roi_to_bottom(self, frame, roi):

        return [
            roi[0],
            roi[1],
            roi[2],
            frame.shape[0] - 1,
        ]

    def _point_in_roi(self, point, roi):

        return roi[0] < point[0] < roi[2] and roi[1] < point[1] < roi[3]

    def _extract_body_scale(self, pose, frame_shape):

        frame_height, frame_width = frame_shape[:2]
        bbox = pose["bbox"]
        bbox_width = max(1, bbox[2] - bbox[0])
        bbox_height = max(1, bbox[3] - bbox[1])

        left_shoulder = pose["keypoints"][5]
        right_shoulder = pose["keypoints"][6]
        shoulder_width = 0.0

        if (
            left_shoulder[2] >= self.wrist_confidence_threshold
            and right_shoulder[2] >= self.wrist_confidence_threshold
        ):
            shoulder_width = abs(left_shoulder[0] - right_shoulder[0])

        body_scale = shoulder_width if shoulder_width > 0 else bbox_width * 0.35
        height_close_enough = bbox_height >= frame_height * self.min_person_height_ratio
        shoulder_close_enough = shoulder_width >= frame_width * self.min_shoulder_width_ratio
        is_close_enough = height_close_enough and (
            shoulder_close_enough if shoulder_width > 0 else bbox_width >= frame_width * self.min_shoulder_width_ratio
        )

        return {
            "bbox_height": bbox_height,
            "shoulder_width": shoulder_width,
            "body_scale": max(body_scale, 1.0),
            "is_close_enough": is_close_enough,
        }

    def _project_hand_point(self, pose, wrist_index, wrist_point):

        elbow_index = 7 if wrist_index == 9 else 8
        elbow_x, elbow_y, elbow_confidence = pose["keypoints"][elbow_index]
        if elbow_confidence < self.wrist_confidence_threshold:
            return wrist_point

        projected_x = wrist_point[0] + int((wrist_point[0] - elbow_x) * self.hand_extension_ratio)
        projected_y = wrist_point[1] + int((wrist_point[1] - elbow_y) * self.hand_extension_ratio)
        return (projected_x, projected_y)

    def analyze(self, frame, poses, timestamp=None):

        now = time.time() if timestamp is None else timestamp
        roi = self._compute_roi(frame)
        active_roi = self._extend_roi_to_bottom(frame, roi)
        contact_roi = self._compute_contact_roi(roi)
        contact_band = self._compute_contact_band(frame, contact_roi)
        target_wrist = None
        target_pose_metrics = None

        for pose in poses:
            pose_metrics = self._extract_body_scale(pose, frame.shape)
            if not pose_metrics["is_close_enough"]:
                continue

            for wrist_index in (9, 10):
                x, y, confidence = pose["keypoints"][wrist_index]
                if confidence < self.wrist_confidence_threshold:
                    continue

                wrist_point = (int(x), int(y))
                hand_point = self._project_hand_point(pose, wrist_index, wrist_point)

                if not self._point_in_roi(wrist_point, active_roi) and not self._point_in_roi(hand_point, active_roi):
                    continue

                if not self._point_in_roi(wrist_point, contact_band) and not self._point_in_roi(hand_point, contact_band):
                    continue

                target_wrist = hand_point if self._point_in_roi(hand_point, contact_band) else wrist_point
                target_pose_metrics = pose_metrics
                break

            if target_wrist is not None:
                break

        event = {
            "roi": roi,
            "active_roi": active_roi,
            "contact_roi": contact_roi,
            "contact_band": contact_band,
            "target_wrist": target_wrist,
            "frames_in_zone": self.wrist_in_zone_frames,
            "elapsed_seconds": 0.0,
            "countdown_seconds": self.required_seconds,
            "triggered": False,
            "should_capture": False,
            "event_name": None,
            "subtitle": None,
            "color": (0, 0, 0),
            "movement_dx": 0,
            "movement_dy": 0,
            "movement_detected": False,
            "person_is_close": target_pose_metrics is not None,
        }

        if target_wrist is None:
            if self.missed_since is None:
                if self.zone_entered_at is not None:
                    self.accumulated_seconds += max(0.0, now - self.zone_entered_at)
                    self.zone_entered_at = None
                self.missed_since = now

            missed_duration = now - self.missed_since
            event["elapsed_seconds"] = self.accumulated_seconds
            event["countdown_seconds"] = max(0.0, self.required_seconds - event["elapsed_seconds"])

            if missed_duration < self.miss_grace_seconds:
                event["frames_in_zone"] = self.wrist_in_zone_frames
                return event

            self.wrist_in_zone_frames = 0
            self.movement_history.clear()
            self.zone_entered_at = None
            self.accumulated_seconds = 0.0
            self.missed_since = None
            event["frames_in_zone"] = self.wrist_in_zone_frames
            event["elapsed_seconds"] = 0.0
            event["countdown_seconds"] = self.required_seconds
            return event

        self.missed_since = None
        if self.zone_entered_at is None:
            self.zone_entered_at = now

        self.wrist_in_zone_frames += 1
        self.movement_history.append(target_wrist)
        elapsed_seconds = self.accumulated_seconds + max(0.0, now - self.zone_entered_at)
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
        movement_detected = max(dx, dy) >= self.movement_delta_threshold

        event_name = EVENT_DOOR_LOCK_MANIPULATION
        subtitle = EVENT_LABELS[EVENT_DOOR_LOCK_MANIPULATION]

        color = (0, 165, 255)

        event["movement_dx"] = dx
        event["movement_dy"] = dy
        event["movement_detected"] = movement_detected

        if not movement_detected:
            return event

        event["triggered"] = True
        event["event_name"] = event_name
        event["subtitle"] = subtitle
        event["color"] = color

        if now - self.last_trigger_time >= self.cooldown_seconds:
            self.last_trigger_time = now
            event["should_capture"] = True

        return event
