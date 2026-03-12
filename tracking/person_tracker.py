import os
import time
from collections import deque

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class OnnxReIDEmbedder:

    def __init__(self, model_path, input_size=(128, 256)):
        self.model_path = model_path
        self.input_size = input_size
        self.net = None
        self.output_dim = None

        if not model_path or not os.path.exists(model_path):
            return

        try:
            self.net = cv2.dnn.readNetFromONNX(model_path)
            self.output_dim = self._infer_output_dim()
        except cv2.error:
            self.net = None
            self.output_dim = None

    @property
    def enabled(self):
        return self.net is not None

    def extract(self, crop):
        if not self.enabled or crop is None or crop.size == 0:
            return None

        embedding = self._forward(crop)
        if embedding is None:
            return None

        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None
        return embedding / norm

    def zero_vector(self):
        if not self.enabled or not self.output_dim:
            return None
        return np.zeros(self.output_dim, dtype=np.float32)

    def _infer_output_dim(self):
        dummy_crop = np.zeros((self.input_size[1], self.input_size[0], 3), dtype=np.uint8)
        embedding = self._forward(dummy_crop)
        if embedding is None:
            return None
        return int(embedding.size)

    def _forward(self, crop):
        try:
            width, height = self.input_size
            resized = cv2.resize(crop, (width, height))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            blob = cv2.dnn.blobFromImage(
                rgb,
                scalefactor=1.0 / 255.0,
                size=(width, height),
                mean=(0.485, 0.456, 0.406),
                swapRB=False,
                crop=False,
            )
            blob[:, 0, :, :] /= 0.229
            blob[:, 1, :, :] /= 0.224
            blob[:, 2, :, :] /= 0.225

            self.net.setInput(blob)
            return self.net.forward().flatten().astype(np.float32)
        except cv2.error:
            return None


class PersonTracker:

    def __init__(
        self,
        max_age=180,
        n_init=5,
        max_cosine_distance=0.3,
        reid_match_threshold=0.78,
        reid_ttl_seconds=45,
        embedding_weight=0.55,
        hist_weight=0.20,
        orb_weight=0.15,
        size_weight=0.10,
        hist_blend_alpha=0.20,
        orb_distance_threshold=42,
        profile_history=6,
        reid_model_path="models/person_reid.onnx",
        reid_input_size=(128, 256),
        min_embedding_similarity=0.82,
        min_size_similarity=0.55,
    ):
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            embedder=None,
        )
        self.reid_match_threshold = reid_match_threshold
        self.reid_ttl_seconds = reid_ttl_seconds
        self.embedding_weight = embedding_weight
        self.hist_weight = hist_weight
        self.orb_weight = orb_weight
        self.size_weight = size_weight
        self.hist_blend_alpha = hist_blend_alpha
        self.orb_distance_threshold = orb_distance_threshold
        self.profile_history = max(1, profile_history)
        self.min_embedding_similarity = min_embedding_similarity
        self.min_size_similarity = min_size_similarity
        self.next_person_id = 1
        self.track_to_person = {}
        self.person_profiles = {}
        self.orb = cv2.ORB_create(nfeatures=128)
        self.reid_embedder = OnnxReIDEmbedder(reid_model_path, input_size=reid_input_size)

    def update(self, detections, frame):
        ds_detections = []
        embeds = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            ds_detections.append(([x1, y1, w, h], conf, "person"))
            embeds.append(self._extract_detection_embedding(frame, det["bbox"]))

        tracks = self.tracker.update_tracks(ds_detections, embeds=embeds)

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
            if appearance["embedding"] is not None:
                profile["embeddings"].append(appearance["embedding"])
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
                "embeddings": deque(
                    [appearance["embedding"]] if appearance["embedding"] is not None else [],
                    maxlen=self.profile_history,
                ),
                "hist": appearance["hist"],
                "orb": appearance["orb"],
                "bbox": bbox,
                "last_seen": now,
                "active": True,
            }

        self.track_to_person[track_id] = matched_person_id
        return matched_person_id

    def _match_existing_person(self, appearance, bbox, now, active_person_ids):
        if (
            appearance["embedding"] is None
            and appearance["hist"] is None
            and appearance["orb"] is None
        ):
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

        size_score = self._size_similarity(bbox, profile.get("bbox"))
        if size_score < self.min_size_similarity:
            return -1.0

        embedding_score = self._embedding_similarity(
            appearance["embedding"],
            profile.get("embeddings"),
        )
        if (
            appearance["embedding"] is not None
            and profile.get("embeddings")
            and (embedding_score is None or embedding_score < self.min_embedding_similarity)
        ):
            return -1.0

        if embedding_score is not None:
            scores.append(embedding_score)
            weights.append(self.embedding_weight)

        hist = appearance["hist"]
        profile_hist = profile.get("hist")
        if hist is not None and profile_hist is not None:
            hist_score = float(np.dot(hist, profile_hist))
            scores.append(hist_score)
            weights.append(self.hist_weight)

        orb = appearance["orb"]
        profile_orb = profile.get("orb")
        if orb is not None and profile_orb is not None:
            orb_score = self._orb_similarity(orb, profile_orb)
            scores.append(orb_score)
            weights.append(self.orb_weight)

        scores.append(size_score)
        weights.append(self.size_weight)

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
            return {"embedding": None, "hist": None, "orb": None}

        return {
            "embedding": self._extract_reid_embedding(crop),
            "hist": self._extract_hist(crop),
            "orb": self._extract_orb(crop),
        }

    def _extract_detection_embedding(self, frame, bbox):
        crop = self._crop_bbox(frame, bbox)
        if self.reid_embedder.enabled:
            if crop is None:
                zero_embedding = self.reid_embedder.zero_vector()
                if zero_embedding is not None:
                    return zero_embedding
            else:
                reid_embedding = self._extract_reid_embedding(crop)
                if reid_embedding is not None:
                    return reid_embedding

                zero_embedding = self.reid_embedder.zero_vector()
                if zero_embedding is not None:
                    return zero_embedding

        if crop is None:
            return np.zeros(24 * 24, dtype=np.float32)

        hist = self._extract_hist(crop)
        if hist is None:
            return np.zeros(24 * 24, dtype=np.float32)

        return hist

    def _extract_reid_embedding(self, crop):
        return self.reid_embedder.extract(crop)

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

        good_matches = [
            match for match in matches if match.distance < self.orb_distance_threshold
        ]
        max_count = max(len(descriptors_a), len(descriptors_b))
        if max_count == 0:
            return 0.0

        return min(1.0, len(good_matches) / max_count)

    def _embedding_similarity(self, embedding, profile_embeddings):
        if embedding is None or not profile_embeddings:
            return None

        best_score = None
        for candidate in profile_embeddings:
            if candidate is None:
                continue
            score = float(np.dot(embedding, candidate))
            if best_score is None or score > best_score:
                best_score = score

        if best_score is None:
            return None

        return max(0.0, min(1.0, (best_score + 1.0) / 2.0))

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

    def _blend_hist(self, previous_hist, current_hist):
        if previous_hist is None:
            return current_hist

        alpha = self.hist_blend_alpha
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
