import time

import cv2

from utils.visualization import draw_alert_banner


class AlertState:

    def __init__(self, hold_seconds, banner_color, text_color):

        self.hold_seconds = hold_seconds
        self.banner_color = banner_color
        self.text_color = text_color
        self.alert_until = 0.0
        self.alert_frame = None
        self.title = ""
        self.subtitle = ""

    def activate(self, frame, title, subtitle):

        self.alert_until = time.time() + self.hold_seconds
        self.alert_frame = frame.copy()
        self.title = title
        self.subtitle = subtitle
        draw_alert_banner(
            self.alert_frame,
            title,
            subtitle,
            self.banner_color,
            self.text_color,
        )

    def is_active(self):

        return self.alert_frame is not None and time.time() < self.alert_until

    def clear_if_expired(self):

        if self.alert_frame is not None and time.time() >= self.alert_until:
            self.alert_frame = None
            self.title = ""
            self.subtitle = ""

    def get_flash_strength(self):

        if not self.is_active():
            return 0.0

        remaining = max(0.0, self.alert_until - time.time())
        progress = 1.0 - (remaining / self.hold_seconds)
        pulse = 0.55 + 0.45 * abs(1.0 - ((progress * 2.6) % 2.0))
        fade = min(1.0, remaining / max(self.hold_seconds * 0.45, 0.001))
        return pulse * fade

    def draw_inline_warning(self, frame, message):

        cv2.putText(
            frame,
            message,
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
        )
