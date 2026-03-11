from ultralytics import YOLO

from settings import DEVICE


class PoseDetector:

    def __init__(self, model_path=None, model=None):

        self.model = model or YOLO(model_path)

    def detect(self, frame):

        results = self.model(frame, device=DEVICE, verbose=False)

        poses = []

        for result in results:
            boxes = result.boxes
            keypoints = result.keypoints

            if boxes is None or keypoints is None:
                continue

            xy = keypoints.xy.cpu().numpy()
            conf = keypoints.conf.cpu().numpy()

            for index, box in enumerate(boxes):

                x1, y1, x2, y2 = box.xyxy[0]
                points = []

                for point, point_confidence in zip(xy[index], conf[index]):
                    points.append([
                        float(point[0]),
                        float(point[1]),
                        float(point_confidence)
                    ])

                poses.append({
                    "class_name": "person",
                    "confidence": float(box.conf[0]),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "keypoints": points
                })

        return poses
