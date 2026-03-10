from ultralytics import YOLO


class WeaponDetector:

    def __init__(self, model_path=None, allowed_labels=None, model=None):

        self.model = model or YOLO(model_path)
        self.class_names = self.model.names
        self.allowed_labels = set(allowed_labels or [])

    def detect(self, frame):

        results = self.model(frame, verbose=False)

        detections = []

        for result in results:
            for box in result.boxes:

                class_id = int(box.cls[0])
                class_name = self.class_names.get(class_id, "unknown")

                if self.allowed_labels and class_name not in self.allowed_labels:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0]

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })

        return detections
