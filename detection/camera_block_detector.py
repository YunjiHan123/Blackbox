import time

import cv2
import numpy as np

from core.event_types import EVENT_CAMERA_BLOCK


class CameraBlockDetector:

    def __init__(
        self,
        change_ratio_threshold=0.35,
        min_brightness=40.0,
        min_pixel_std=15.0,
        min_texture=50.0,
        min_edge_ratio=0.02,
        hist_threshold=0.4,
        flow_threshold=2.0,
        capture_interval_seconds=1.0,
    ):

        self.change_ratio_threshold = change_ratio_threshold
        self.min_brightness = min_brightness
        self.min_pixel_std = min_pixel_std
        self.min_texture = min_texture
        self.min_edge_ratio = min_edge_ratio
        self.hist_threshold = hist_threshold
        self.flow_threshold = flow_threshold
        self.capture_interval_seconds = capture_interval_seconds
        self.prev_gray = None
        self.last_capture_time = 0.0
        self.prev_hist = None
        self.prev_gray_small = None

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

            lap = cv2.Laplacian(gray, cv2.CV_64F)
            texture_var = lap.var()

            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / edges.size

            hist = cv2.calcHist([frame],[0,1,2],None,[8,8,8],[0,256,0,256,0,256])
            hist = cv2.normalize(hist, hist).flatten()

            hist_change = 0

            if self.prev_hist is not None:
                hist_change = cv2.compareHist(
                    self.prev_hist,
                    hist,
                    cv2.HISTCMP_BHATTACHARYYA
                )

            self.prev_hist = hist

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation = np.mean(hsv[:,:,1])

            gray_small = cv2.resize(gray, (160,120))

            flow_mag = 0

            if self.prev_gray_small is not None:
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_gray_small,
                    gray_small,
                    None,
                    0.5,
                    3,
                    15,
                    3,
                    5,
                    1.2,
                    0
                )

                mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
                flow_mag = np.mean(mag)

            self.prev_gray_small = gray_small

            score = 0
            reasons = []

            if change_ratio > self.change_ratio_threshold:
                score += 1
                reasons.append("sudden_change")

            if brightness < self.min_brightness:
                score += 1
                reasons.append("dark_frame")

            if texture_var < self.min_texture:
                score += 1
                reasons.append("low_texture")

            if edge_ratio < self.min_edge_ratio:
                score += 1
                reasons.append("low_edges")

            if hist_change > self.hist_threshold:
                score += 1
                reasons.append("color_distribution_change")

            if flow_mag > self.flow_threshold:
                score += 1
                reasons.append("camera_motion")

            if score >= 2:

                now = time.time() if timestamp is None else timestamp

                should_capture = (
                    now - self.last_capture_time
                    > self.capture_interval_seconds
                )

                if should_capture:
                    self.last_capture_time = now

                event = {
                    "class_name": EVENT_CAMERA_BLOCK,
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
