from ultralytics import YOLO

from config import PERSON_CONFIDENCE, PERSON_IMAGE_SIZE, PERSON_MODEL_PATH


class PersonDetector:
    def __init__(
        self,
        model_path=PERSON_MODEL_PATH,
        confidence=PERSON_CONFIDENCE,
        image_size=PERSON_IMAGE_SIZE,
        model=None,
    ):
        self.model = model or YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, frame):
        results = self.model(
            frame,
            classes=[0],
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append(
                    {
                        "bbox": [x1, y1, x2, y2],
                        "class_name": "person",
                        "confidence": confidence,
                    }
                )

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
