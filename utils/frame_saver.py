
import os
import cv2
from datetime import datetime


class FrameSaver:

    def __init__(self):

        self.output_dir = "outputs"
        self.saved_paths = []

        os.makedirs(self.output_dir, exist_ok=True)

    def save(self, frame, prefix):

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = f"{prefix}_{timestamp}.jpg"

        path = os.path.join(self.output_dir, filename)

        cv2.imwrite(path, frame)
        self.saved_paths.append(path)

        return path

    def cleanup_saved_files(self):

        for path in self.saved_paths:
            if os.path.exists(path):
                os.remove(path)

        self.saved_paths.clear()
