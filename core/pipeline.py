from ultralytics import YOLO

from settings import (
    CAMERA_BLOCK_CAPTURE_INTERVAL_SECONDS,
    CAMERA_BLOCK_CHANGE_RATIO_THRESHOLD,
    CAMERA_BLOCK_MIN_BRIGHTNESS,
    CAMERA_BLOCK_MIN_PIXEL_STD,
    DOOR_LOCK_COOLDOWN_SECONDS,
    DOOR_LOCK_CONTACT_ROI_HEIGHT_RATIO,
    DOOR_LOCK_CONTACT_ROI_WIDTH_RATIO,
    DOOR_LOCK_DECAY_FRAMES,
    DOOR_LOCK_HAND_EXTENSION_RATIO,
    DOOR_LOCK_MIN_PERSON_HEIGHT_RATIO,
    DOOR_LOCK_MIN_SHOULDER_WIDTH_RATIO,
    DOOR_LOCK_MOVEMENT_DELTA_THRESHOLD,
    DOOR_LOCK_MOVEMENT_HISTORY_SIZE,
    DOOR_LOCK_MIN_WRIST_DISTANCE_PIXELS,
    DOOR_LOCK_MISS_GRACE_SECONDS,
    DOOR_LOCK_REQUIRED_SECONDS,
    DOOR_LOCK_ROI_HEIGHT_RATIO,
    DOOR_LOCK_ROI_WIDTH_RATIO,
    DOOR_LOCK_ROI_X_RATIO,
    DOOR_LOCK_ROI_Y_RATIO,
    DOOR_LOCK_WRIST_DISTANCE_SHOULDER_RATIO,
    DOOR_LOCK_WRIST_CONFIDENCE_THRESHOLD,
    LOITERING_THRESHOLD_SECONDS,
    PERSON_NEAR_AREA_THRESHOLD,
    PERSON_NEAR_COOLDOWN_SECONDS,
    PERSON_NEAR_MIN_CONFIDENCE,
    PERSON_NEAR_REQUIRED_TIME_SECONDS,
    PERSON_MODEL_PATH,
    POSE_MODEL_PATH,
    REID_EMBEDDING_WEIGHT,
    REID_HIST_BLEND_ALPHA,
    REID_HIST_WEIGHT,
    REID_INPUT_HEIGHT,
    REID_INPUT_WIDTH,
    REID_MATCH_THRESHOLD,
    REID_MIN_EMBEDDING_SIMILARITY,
    REID_MIN_SIZE_SIMILARITY,
    REID_MODEL_PATH,
    REID_ORB_DISTANCE_THRESHOLD,
    REID_ORB_WEIGHT,
    REID_PROFILE_HISTORY,
    REID_SIZE_WEIGHT,
    REID_TTL_SECONDS,
    TRACKER_MAX_AGE,
    TRACKER_MAX_COSINE_DISTANCE,
    TRACKER_N_INIT,
    WEAPON_LABELS,
    WEAPON_MODEL_PATH,
)
from behaviors.loitering_behavior import LoiteringDetector
from detection.camera_block_detector import CameraBlockDetector
from detection.door_lock_detector import DoorLockDetector
from detection.person_detector import PersonDetector
from detection.person_proximity_detector import PersonProximityDetector
from detection.pose_detector import PoseDetector
from detection.weapon_detector import WeaponDetector
from tracking.person_tracker import PersonTracker
from utils.detection_gate import DetectionGate


def create_pipeline():

    # Use separate model instances to avoid shared overrides (e.g. classes filter).
    weapon_model = YOLO(WEAPON_MODEL_PATH)
    person_model = YOLO(PERSON_MODEL_PATH)

    return {
        "weapon_detector": WeaponDetector(
            allowed_labels=WEAPON_LABELS,
            model=weapon_model,
        ),
        "person_proximity_detector": PersonProximityDetector(
            model=person_model,
            area_threshold=PERSON_NEAR_AREA_THRESHOLD,
            required_time_seconds=PERSON_NEAR_REQUIRED_TIME_SECONDS,
            min_confidence=PERSON_NEAR_MIN_CONFIDENCE,
            cooldown_seconds=PERSON_NEAR_COOLDOWN_SECONDS,
        ),
        "person_detector": PersonDetector(model=person_model),
        "pose_detector": PoseDetector(POSE_MODEL_PATH),
        "camera_block_detector": CameraBlockDetector(
            change_ratio_threshold=CAMERA_BLOCK_CHANGE_RATIO_THRESHOLD,
            min_brightness=CAMERA_BLOCK_MIN_BRIGHTNESS,
            min_pixel_std=CAMERA_BLOCK_MIN_PIXEL_STD,
            capture_interval_seconds=CAMERA_BLOCK_CAPTURE_INTERVAL_SECONDS,
        ),
        "door_lock_detector": DoorLockDetector(
            roi_x_ratio=DOOR_LOCK_ROI_X_RATIO,
            roi_y_ratio=DOOR_LOCK_ROI_Y_RATIO,
            roi_width_ratio=DOOR_LOCK_ROI_WIDTH_RATIO,
            roi_height_ratio=DOOR_LOCK_ROI_HEIGHT_RATIO,
            contact_roi_width_ratio=DOOR_LOCK_CONTACT_ROI_WIDTH_RATIO,
            contact_roi_height_ratio=DOOR_LOCK_CONTACT_ROI_HEIGHT_RATIO,
            wrist_confidence_threshold=DOOR_LOCK_WRIST_CONFIDENCE_THRESHOLD,
            required_seconds=DOOR_LOCK_REQUIRED_SECONDS,
            miss_grace_seconds=DOOR_LOCK_MISS_GRACE_SECONDS,
            movement_history_size=DOOR_LOCK_MOVEMENT_HISTORY_SIZE,
            decay_frames=DOOR_LOCK_DECAY_FRAMES,
            movement_delta_threshold=DOOR_LOCK_MOVEMENT_DELTA_THRESHOLD,
            min_person_height_ratio=DOOR_LOCK_MIN_PERSON_HEIGHT_RATIO,
            min_shoulder_width_ratio=DOOR_LOCK_MIN_SHOULDER_WIDTH_RATIO,
            hand_extension_ratio=DOOR_LOCK_HAND_EXTENSION_RATIO,
            wrist_distance_shoulder_ratio=DOOR_LOCK_WRIST_DISTANCE_SHOULDER_RATIO,
            min_wrist_distance_pixels=DOOR_LOCK_MIN_WRIST_DISTANCE_PIXELS,
            cooldown_seconds=DOOR_LOCK_COOLDOWN_SECONDS,
        ),
        "person_tracker": PersonTracker(
            max_age=TRACKER_MAX_AGE,
            n_init=TRACKER_N_INIT,
            max_cosine_distance=TRACKER_MAX_COSINE_DISTANCE,
            reid_match_threshold=REID_MATCH_THRESHOLD,
            reid_ttl_seconds=REID_TTL_SECONDS,
            embedding_weight=REID_EMBEDDING_WEIGHT,
            hist_weight=REID_HIST_WEIGHT,
            orb_weight=REID_ORB_WEIGHT,
            size_weight=REID_SIZE_WEIGHT,
            hist_blend_alpha=REID_HIST_BLEND_ALPHA,
            orb_distance_threshold=REID_ORB_DISTANCE_THRESHOLD,
            profile_history=REID_PROFILE_HISTORY,
            reid_model_path=REID_MODEL_PATH,
            reid_input_size=(REID_INPUT_WIDTH, REID_INPUT_HEIGHT),
            min_embedding_similarity=REID_MIN_EMBEDDING_SIMILARITY,
            min_size_similarity=REID_MIN_SIZE_SIMILARITY,
        ),
        "loitering_detector": LoiteringDetector(threshold=LOITERING_THRESHOLD_SECONDS),
        "weapon_gate": DetectionGate(
            min_confidence=0.45,
            window_size=5,
            required_hits=3,
            cooldown_frames=60,
        ),
    }


def analyze_frame(pipeline, frame, timestamp=None):

    weapon_detections = pipeline["weapon_detector"].detect(frame)
    person_detections = pipeline["person_detector"].detect(frame)
    persons = pipeline["person_tracker"].update(person_detections, frame)
    poses = pipeline["pose_detector"].detect(frame)
    block_event = pipeline["camera_block_detector"].analyze(frame, timestamp=timestamp)
    person_near_event = pipeline["person_proximity_detector"].analyze(
        frame,
        timestamp=timestamp,
    )
    door_lock_event = pipeline["door_lock_detector"].analyze(
        frame,
        poses,
        timestamp=timestamp,
    )

    return {
        "weapon_detections": weapon_detections,
        "persons": persons,
        "poses": poses,
        "block_event": block_event,
        "person_near_event": person_near_event,
        "door_lock_event": door_lock_event,
    }
