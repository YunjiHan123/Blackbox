import time

from core.event_types import EVENT_FACE_NEAR


class PersonProximityDetector:

    def __init__(
        self,
        model,
        area_threshold=0.4,
        required_time_seconds=2.0,
        min_confidence=0.35,
        cooldown_seconds=3.0,
        head_area_threshold=0.18,
        head_keypoint_confidence=0.35,
        min_visible_face_points=3,
    ):

        self.model = model
        self.area_threshold = area_threshold
        self.required_time_seconds = required_time_seconds
        self.min_confidence = min_confidence
        self.cooldown_seconds = cooldown_seconds
        self.head_area_threshold = head_area_threshold
        self.head_keypoint_confidence = head_keypoint_confidence
        self.min_visible_face_points = min_visible_face_points
        self.start_time = None
        self.last_trigger_time = 0.0

    def analyze(self, frame, poses=None, timestamp=None):

        frame_height, frame_width = frame.shape[:2]
        frame_area = frame_height * frame_width
        best_candidate = None

        for pose in poses or []:
            confidence = float(pose["confidence"])
            if confidence < self.min_confidence:
                continue

            head_bbox = self._extract_head_bbox(
                pose["bbox"],
                pose["keypoints"],
                frame_width,
                frame_height,
            )
            if head_bbox is None:
                continue

            head_area_ratio = self._bbox_area_ratio(head_bbox, frame_area)
            if head_area_ratio < self.head_area_threshold:
                continue

            candidate = {
                "class_name": EVENT_FACE_NEAR,
                "confidence": confidence,
                "bbox": head_bbox,
                "area_ratio": head_area_ratio,
                "person_bbox": pose["bbox"],
            }

            if best_candidate is None or head_area_ratio > best_candidate["area_ratio"]:
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

    def _extract_head_bbox(self, person_bbox, keypoints, frame_width, frame_height):
        face_points = [
            point for point in keypoints[:5]
            if float(point[2]) >= self.head_keypoint_confidence
        ]
        if len(face_points) < self.min_visible_face_points:
            return None

        xs = [float(point[0]) for point in face_points]
        ys = [float(point[1]) for point in face_points]
        face_x1 = min(xs)
        face_x2 = max(xs)
        face_y1 = min(ys)
        face_y2 = max(ys)

        person_width = max(1, person_bbox[2] - person_bbox[0])
        person_height = max(1, person_bbox[3] - person_bbox[1])
        face_width = max(1.0, face_x2 - face_x1)
        face_height = max(1.0, face_y2 - face_y1)

        expand_x = max(face_width * 0.45, person_width * 0.05)
        expand_top = max(face_height * 0.70, person_height * 0.04)
        expand_bottom = max(face_height * 1.10, person_height * 0.06)

        x1 = max(0, int(face_x1 - expand_x))
        x2 = min(frame_width, int(face_x2 + expand_x))
        y1 = max(0, int(face_y1 - expand_top))
        y2 = min(frame_height, int(face_y2 + expand_bottom))

        if x2 <= x1 or y2 <= y1:
            return None

        return [x1, y1, x2, y2]

    def _bbox_area_ratio(self, bbox, frame_area):
        width = max(0, bbox[2] - bbox[0])
        height = max(0, bbox[3] - bbox[1])
        return (width * height) / float(frame_area)
