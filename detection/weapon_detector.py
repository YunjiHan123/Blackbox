from ultralytics import YOLO

from settings import DEVICE


class WeaponDetector:

    def __init__(self, model_path=None, allowed_labels=None, model=None):

        self.model = model or YOLO(model_path)
        self.class_names = self.model.names
        self.allowed_labels = set(allowed_labels or [])

    def detect(self, frame):

        results = self.model(frame, device=DEVICE, verbose=False)

        detections = []

        for result in results:
            for box in result.boxes:

                class_id = int(box.cls[0])
                class_name = self.class_names.get(class_id, "unknown")

                if class_name == "person":
                    continue

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

        return self._deduplicate_detections(detections)

    def _deduplicate_detections(self, detections, iou_threshold=0.6):
        if not detections:
            return []

        detections = sorted(
            detections,
            key=lambda det: det["confidence"],
            reverse=True,
        )

        kept = []

        for detection in detections:
            if all(
                self._iou(detection["bbox"], kept_detection["bbox"]) < iou_threshold
                for kept_detection in kept
            ):
                kept.append(detection)

        return kept

    def _iou(self, bbox_a, bbox_b):
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area
