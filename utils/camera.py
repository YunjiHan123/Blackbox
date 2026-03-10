import cv2


class CameraStream:

    def __init__(self, camera_index, frame_width, frame_height, rotation=None):

        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.rotation = rotation
        self.cap = None

    def open(self):

        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        if not cap.isOpened():
            cap.release()
            return None

        self.cap = cap
        return cap

    def read(self):

        if self.cap is None:
            return False, None

        ret, frame = self.cap.read()

        if not ret or frame is None:
            return False, None

        if self.rotation is not None:
            frame = cv2.rotate(frame, self.rotation)

        return True, frame

    def release(self):

        if self.cap is not None:
            self.cap.release()
            self.cap = None


def resize_to_window(frame, window_name):

    _, _, window_width, window_height = cv2.getWindowImageRect(window_name)

    if window_width <= 0 or window_height <= 0:
        return frame

    height, width = frame.shape[:2]
    scale = min(window_width / width, window_height / height)

    if scale <= 0:
        return frame

    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))

    return cv2.resize(frame, (resized_width, resized_height))
