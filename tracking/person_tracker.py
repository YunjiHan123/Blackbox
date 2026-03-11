import time

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class PersonTracker:

    def __init__(
        self,
        max_age=180,
        n_init=2,
        max_cosine_distance=0.3,
        reid_match_threshold=0.72,
        reid_ttl_seconds=120,
    ):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
        )
        self.reid_match_threshold = reid_match_threshold
        self.reid_ttl_seconds = reid_ttl_seconds
        self.next_person_id = 1
        self.track_to_person = {}
        self.person_profiles = {}
        self.orb = cv2.ORB_create(nfeatures=128)

    def update(self, detections, frame):
        ds_detections = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            ds_detections.append(([x1, y1, w, h], conf, "person"))

        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        now = time.time()
        self._expire_old_profiles(now)

        persons = []
        active_person_ids = set()

        for track in tracks:
            if not track.is_confirmed():
                continue

            if getattr(track, "time_since_update", 0) > 0:
                continue

            track_id = track.track_id
            bbox = [int(v) for v in track.to_ltrb()]
            appearance = self._extract_appearance(frame, bbox)
            person_id = self._resolve_person_id(track_id, appearance, bbox, now, active_person_ids)
            active_person_ids.add(person_id)

            profile = self.person_profiles[person_id]
            profile["last_seen"] = now
            profile["bbox"] = bbox
            profile["active"] = True
            if appearance["hist"] is not None:
                profile["hist"] = self._blend_hist(profile.get("hist"), appearance["hist"])
            if appearance["orb"] is not None:
                profile["orb"] = appearance["orb"]

            persons.append(
                {
                    "id": person_id,
                    "track_id": track_id,
                    "bbox": bbox,
                }
            )

        persons = self._deduplicate_persons(persons)
        self._mark_inactive_tracks({person["id"] for person in persons}, now)

        return persons

    def _resolve_person_id(self, track_id, appearance, bbox, now, active_person_ids):
        existing_person_id = self.track_to_person.get(track_id)
        if existing_person_id is not None:
            return existing_person_id

        matched_person_id = self._match_existing_person(appearance, bbox, now, active_person_ids)
        if matched_person_id is None:
            matched_person_id = self.next_person_id
            self.next_person_id += 1
            self.person_profiles[matched_person_id] = {
                "hist": appearance["hist"],
                "orb": appearance["orb"],
                "bbox": bbox,
                "last_seen": now,
                "active": True,
            }

        self.track_to_person[track_id] = matched_person_id
        return matched_person_id

    def _match_existing_person(self, appearance, bbox, now, active_person_ids):
        hist = appearance["hist"]
        orb = appearance["orb"]
        if hist is None and orb is None:
            return None

        best_person_id = None
        best_score = -1.0

        for person_id, profile in self.person_profiles.items():
            if person_id in active_person_ids:
                continue

            if profile.get("active", False):
                continue

            if now - profile["last_seen"] > self.reid_ttl_seconds:
                continue

            score = self._compute_match_score(appearance, bbox, profile)
            if score > best_score:
                best_score = score
                best_person_id = person_id

        if best_score >= self.reid_match_threshold:
            return best_person_id

        return None

    def _compute_match_score(self, appearance, bbox, profile):
        scores = []
        weights = []

        hist = appearance["hist"]
        profile_hist = profile.get("hist")
        if hist is not None and profile_hist is not None:
            hist_score = float(np.dot(hist, profile_hist))
            scores.append(hist_score)
            weights.append(0.45)

        orb = appearance["orb"]
        profile_orb = profile.get("orb")
        if orb is not None and profile_orb is not None:
            orb_score = self._orb_similarity(orb, profile_orb)
            scores.append(orb_score)
            weights.append(0.35)

        size_score = self._size_similarity(bbox, profile.get("bbox"))
        scores.append(size_score)
        weights.append(0.20)

        if not scores:
            return -1.0

        weighted_sum = sum(score * weight for score, weight in zip(scores, weights))
        total_weight = sum(weights)
        return weighted_sum / total_weight

    def _mark_inactive_tracks(self, active_person_ids, now):
        active_ids = set(active_person_ids)

        for person_id, profile in self.person_profiles.items():
            if person_id not in active_ids:
                profile["active"] = False

        for track_id, person_id in list(self.track_to_person.items()):
            profile = self.person_profiles.get(person_id)
            if profile is None:
                del self.track_to_person[track_id]
                continue

            if not profile["active"] and now - profile["last_seen"] > self.reid_ttl_seconds:
                del self.track_to_person[track_id]

    def _expire_old_profiles(self, now):
        expired_person_ids = [
            person_id
            for person_id, profile in self.person_profiles.items()
            if not profile.get("active", False)
            and now - profile["last_seen"] > self.reid_ttl_seconds
        ]

        for person_id in expired_person_ids:
            del self.person_profiles[person_id]

        for track_id, person_id in list(self.track_to_person.items()):
            if person_id not in self.person_profiles:
                del self.track_to_person[track_id]

    def _extract_appearance(self, frame, bbox):
        crop = self._crop_bbox(frame, bbox)
        if crop is None:
            return {"hist": None, "orb": None}

        return {
            "hist": self._extract_hist(crop),
            "orb": self._extract_orb(crop),
        }

    def _crop_bbox(self, frame, bbox):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return crop

    def _extract_hist(self, crop):
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)

        norm = np.linalg.norm(hist)
        if norm == 0:
            return None

        return hist / norm

    def _extract_orb(self, crop):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (96, 192))
        _, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) == 0:
            return None
        return descriptors

    def _orb_similarity(self, descriptors_a, descriptors_b):
        if descriptors_a is None or descriptors_b is None:
            return 0.0

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptors_a, descriptors_b)
        if not matches:
            return 0.0

        good_matches = [match for match in matches if match.distance < 48]
        max_count = max(len(descriptors_a), len(descriptors_b))
        if max_count == 0:
            return 0.0

        return min(1.0, len(good_matches) / max_count)

    def _size_similarity(self, bbox_a, bbox_b):
        if bbox_b is None:
            return 0.0

        width_a = max(1, bbox_a[2] - bbox_a[0])
        height_a = max(1, bbox_a[3] - bbox_a[1])
        width_b = max(1, bbox_b[2] - bbox_b[0])
        height_b = max(1, bbox_b[3] - bbox_b[1])

        width_ratio = min(width_a, width_b) / max(width_a, width_b)
        height_ratio = min(height_a, height_b) / max(height_a, height_b)
        return (width_ratio + height_ratio) / 2.0

    def _blend_hist(self, previous_hist, current_hist, alpha=0.30):
        if previous_hist is None:
            return current_hist

        blended = (1.0 - alpha) * previous_hist + alpha * current_hist
        norm = np.linalg.norm(blended)
        if norm == 0:
            return current_hist
        return blended / norm

    def _deduplicate_persons(self, persons, iou_threshold=0.65):
        if not persons:
            return []

        kept = []

        for person in persons:
            duplicate = False

            for kept_person in kept:
                if self._iou(person["bbox"], kept_person["bbox"]) >= iou_threshold:
                    duplicate = True
                    break

                if person["id"] == kept_person["id"]:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(person)

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
