import time

import cv2
import numpy as np


class CameraBlockDetector:

    def __init__(
        self,
        change_ratio_threshold=0.35,
        min_brightness=40.0,
        min_pixel_std=15.0,
        capture_interval_seconds=1.0,
    ):

        self.change_ratio_threshold = change_ratio_threshold
        self.min_brightness = min_brightness
        self.min_pixel_std = min_pixel_std
        self.capture_interval_seconds = capture_interval_seconds
        self.prev_gray = None
        self.last_capture_time = 0.0

    def analyze(self, frame, timestamp=None):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        event = None

        if self.prev_gray is not None:
            diff = cv2.absdiff(self.prev_gray, gray)
            _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)

            changed_pixels = int(np.sum(thresh > 0))
            change_ratio = changed_pixels / float(thresh.size)
            brightness = float(np.mean(gray))
            pixel_std = float(np.std(gray))

            reasons = []

            if change_ratio > self.change_ratio_threshold:
                reasons.append("sudden_change")
            if brightness < self.min_brightness:
                reasons.append("dark_frame")
            if pixel_std < self.min_pixel_std:
                reasons.append("low_texture")

            if reasons:
                now = time.time() if timestamp is None else timestamp
                should_capture = now - self.last_capture_time > self.capture_interval_seconds

                if should_capture:
                    self.last_capture_time = now

                event = {
                    "class_name": "camera_blocking",
                    "confidence": 1.0,
                    "reasons": reasons,
                    "metrics": {
                        "change_ratio": change_ratio,
                        "brightness": brightness,
                        "pixel_std": pixel_std,
                    },
                    "should_capture": should_capture,
                }

        self.prev_gray = gray
        return event
