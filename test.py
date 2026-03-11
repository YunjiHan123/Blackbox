import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from config import (
    CAMERA_BLOCK_CAPTURE_INTERVAL_SECONDS,
    CAMERA_BLOCK_CHANGE_RATIO_THRESHOLD,
    CAMERA_BLOCK_MIN_BRIGHTNESS,
    CAMERA_BLOCK_MIN_PIXEL_STD,
    DOOR_LOCK_COOLDOWN_SECONDS,
    DOOR_LOCK_DECAY_FRAMES,
    DOOR_LOCK_MOVEMENT_DELTA_THRESHOLD,
    DOOR_LOCK_MOVEMENT_HISTORY_SIZE,
    DOOR_LOCK_REQUIRED_SECONDS,
    DOOR_LOCK_ROI_HEIGHT_RATIO,
    DOOR_LOCK_ROI_WIDTH_RATIO,
    DOOR_LOCK_ROI_X_RATIO,
    DOOR_LOCK_ROI_Y_RATIO,
    DOOR_LOCK_WRIST_CONFIDENCE_THRESHOLD,
    PERSON_NEAR_AREA_THRESHOLD,
    PERSON_NEAR_COOLDOWN_SECONDS,
    PERSON_NEAR_MIN_CONFIDENCE,
    PERSON_NEAR_REQUIRED_TIME_SECONDS,
    POSE_COLOR,
    POSE_CONNECTIONS,
    POSE_LINE_COLOR,
    POSE_MODEL_PATH,
    POSE_POINT_COLOR,
    WEAPON_COLOR,
    WEAPON_LABELS,
    WEAPON_MODEL_PATH,
)
from events.event_types import (
    EVENT_CAMERA_BLOCK,
    EVENT_DOOR_LOCK_MANIPULATION,
    EVENT_FACE_NEAR,
    EVENT_WEAPON,
)
from detection.camera_block_detector import CameraBlockDetector
from detection.door_lock_detector import DoorLockDetector
from detection.person_proximity_detector import PersonProximityDetector
from detection.pose_detector import PoseDetector
from detection.weapon_detector import WeaponDetector
from utils.detection_gate import DetectionGate
from utils.visualization import draw_detection, draw_pose, draw_roi


ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT / "videos"
LABELS_PATH = ROOT / "labels.json"
WINDOW_NAME = "Evaluation"
RESULT_HOLD_MS = 800
SKIP_KEY = ord("s")

IMPLEMENTED_WARNINGS = {
    EVENT_WEAPON,
    EVENT_CAMERA_BLOCK,
    EVENT_FACE_NEAR,
    EVENT_DOOR_LOCK_MANIPULATION,
}


class EvaluationRuntime:

    def __init__(self):

        object_model = YOLO(WEAPON_MODEL_PATH)
        pose_model = YOLO(POSE_MODEL_PATH)
        self.object_model = object_model
        self.pose_model = pose_model

    def create_pipeline(self):

        return {
            "weapon_detector": WeaponDetector(
                allowed_labels=WEAPON_LABELS,
                model=self.object_model,
            ),
            "pose_detector": PoseDetector(model=self.pose_model),
            "camera_block_detector": CameraBlockDetector(
                change_ratio_threshold=CAMERA_BLOCK_CHANGE_RATIO_THRESHOLD,
                min_brightness=CAMERA_BLOCK_MIN_BRIGHTNESS,
                min_pixel_std=CAMERA_BLOCK_MIN_PIXEL_STD,
                capture_interval_seconds=CAMERA_BLOCK_CAPTURE_INTERVAL_SECONDS,
            ),
            "person_proximity_detector": PersonProximityDetector(
                model=self.object_model,
                area_threshold=PERSON_NEAR_AREA_THRESHOLD,
                required_time_seconds=PERSON_NEAR_REQUIRED_TIME_SECONDS,
                min_confidence=PERSON_NEAR_MIN_CONFIDENCE,
                cooldown_seconds=PERSON_NEAR_COOLDOWN_SECONDS,
            ),
            "door_lock_detector": DoorLockDetector(
                roi_x_ratio=DOOR_LOCK_ROI_X_RATIO,
                roi_y_ratio=DOOR_LOCK_ROI_Y_RATIO,
                roi_width_ratio=DOOR_LOCK_ROI_WIDTH_RATIO,
                roi_height_ratio=DOOR_LOCK_ROI_HEIGHT_RATIO,
                wrist_confidence_threshold=DOOR_LOCK_WRIST_CONFIDENCE_THRESHOLD,
                required_seconds=DOOR_LOCK_REQUIRED_SECONDS,
                movement_history_size=DOOR_LOCK_MOVEMENT_HISTORY_SIZE,
                decay_frames=DOOR_LOCK_DECAY_FRAMES,
                movement_delta_threshold=DOOR_LOCK_MOVEMENT_DELTA_THRESHOLD,
                cooldown_seconds=DOOR_LOCK_COOLDOWN_SECONDS,
            ),
            "weapon_gate": DetectionGate(
                min_confidence=0.45,
                window_size=5,
                required_hits=3,
                cooldown_frames=60,
            ),
        }


def load_dataset(labels_path=LABELS_PATH):

    with open(labels_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return payload["videos"]


def _frame_timestamp_seconds(capture, frame_index, fps):

    timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
    if timestamp_ms and timestamp_ms > 0:
        return timestamp_ms / 1000.0

    if fps and fps > 0:
        return frame_index / fps

    return float(frame_index)


def _draw_overlay(
    frame,
    video_name,
    expected_warnings,
    detected_warnings,
    status,
    frame_index,
    progress_text,
):

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    lines = [
        f"video: {video_name}",
        f"expected: {', '.join(expected_warnings)}",
        f"detected: {', '.join(detected_warnings) if detected_warnings else '-'}",
        f"status: {status} | frame: {frame_index} | S: skip | ESC: exit",
    ]

    color = (255, 255, 255)
    if status == "PASS":
        color = (0, 255, 0)
    elif status == "FAIL":
        color = (0, 0, 255)

    y = 28
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7 if index == 0 else 0.6,
            color if index == 3 else (255, 255, 255),
            2,
        )
        y += 25

    progress_box_width = 320
    progress_box_height = 36
    progress_x1 = max(0, frame.shape[1] - progress_box_width - 20)
    progress_y1 = 15
    progress_overlay = frame.copy()
    cv2.rectangle(
        progress_overlay,
        (progress_x1, progress_y1),
        (progress_x1 + progress_box_width, progress_y1 + progress_box_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(progress_overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(
        frame,
        progress_text,
        (progress_x1 + 10, progress_y1 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


def _show_result_frame(frame, video_name, expected_warnings, detected_warnings, status, progress_text):

    result_frame = frame.copy()
    _draw_overlay(
        result_frame,
        video_name=video_name,
        expected_warnings=expected_warnings,
        detected_warnings=detected_warnings,
        status=status,
        frame_index=-1,
        progress_text=progress_text,
    )
    cv2.imshow(WINDOW_NAME, result_frame)
    key = cv2.waitKey(RESULT_HOLD_MS) & 0xFF
    return key not in (27, SKIP_KEY), key


def model(video_path, runtime, expected_warnings, video_name, progress_text):

    pipeline = runtime.create_pipeline()
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_index = 0
    detected_warnings = []
    last_frame = None
    aborted = False
    skipped = False

    try:
        while True:
            ret, frame = capture.read()
            if not ret or frame is None:
                break

            frame_index += 1
            timestamp = _frame_timestamp_seconds(capture, frame_index, fps)
            last_frame = frame.copy()

            weapon_detections = pipeline["weapon_detector"].detect(frame)
            poses = pipeline["pose_detector"].detect(frame)
            block_event = pipeline["camera_block_detector"].analyze(frame, timestamp=timestamp)
            person_event = pipeline["person_proximity_detector"].analyze(frame, timestamp=timestamp)
            door_lock_event = pipeline["door_lock_detector"].analyze(
                frame,
                poses,
                timestamp=timestamp,
            )

            if pipeline["weapon_gate"].update(weapon_detections):
                if EVENT_WEAPON not in detected_warnings:
                    detected_warnings.append(EVENT_WEAPON)

            if block_event is not None and block_event["should_capture"]:
                if EVENT_CAMERA_BLOCK not in detected_warnings:
                    detected_warnings.append(EVENT_CAMERA_BLOCK)

            if person_event is not None and person_event["triggered"] and person_event["should_capture"]:
                if EVENT_FACE_NEAR not in detected_warnings:
                    detected_warnings.append(EVENT_FACE_NEAR)

            if door_lock_event["triggered"] and door_lock_event["should_capture"]:
                if door_lock_event["event_name"] not in detected_warnings:
                    detected_warnings.append(door_lock_event["event_name"])

            display_frame = frame.copy()

            for pose in poses:
                draw_pose(
                    display_frame,
                    pose,
                    box_color=POSE_COLOR,
                    line_color=POSE_LINE_COLOR,
                    point_color=POSE_POINT_COLOR,
                    connections=POSE_CONNECTIONS,
                )

            for detection in weapon_detections:
                draw_detection(display_frame, detection, WEAPON_COLOR)

            if person_event is not None:
                draw_detection(display_frame, person_event, (0, 0, 255))

            draw_roi(
                display_frame,
                door_lock_event["roi"],
                "Door Lock",
                door_lock_event["color"],
                thickness=3 if door_lock_event["target_wrist"] is not None else 2,
            )

            if door_lock_event["target_wrist"] is not None:
                cv2.circle(
                    display_frame,
                    door_lock_event["target_wrist"],
                    6,
                    door_lock_event["color"],
                    -1,
                )

            status = "RUNNING"
            if any(warning in expected_warnings for warning in detected_warnings):
                status = "PASS"
            elif detected_warnings:
                status = "DETECTED"

            _draw_overlay(
                display_frame,
                video_name=video_name,
                expected_warnings=expected_warnings,
                detected_warnings=detected_warnings,
                status=status,
                frame_index=frame_index,
                progress_text=progress_text,
            )
            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                aborted = True
                break
            if key == SKIP_KEY:
                skipped = True
                break

            if any(warning in expected_warnings for warning in detected_warnings):
                break
    finally:
        capture.release()

    return {
        "detected": detected_warnings,
        "last_frame": last_frame,
        "aborted": aborted,
        "skipped": skipped,
    }


def evaluate_case(video_name, expected_warnings, runtime):

    detected = model(VIDEOS_DIR / video_name, runtime)
    passed = detected in expected_warnings
    status = "PASS" if passed else "FAIL"

    return {
        "video": video_name,
        "status": status,
        "expected": expected_warnings,
        "unsupported": [],
        "detected": detected,
        "missing": [] if passed else expected_warnings,
        "unexpected": [] if detected is None or passed else [detected],
    }


def test(model_fn, dataset, runtime):

    results = []

    total_cases = len(dataset)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    pass_count = 0
    fail_count = 0
    skip_count = 0

    for index, item in enumerate(dataset, start=1):
        print(f"Processing [{index}/{total_cases}] {item['file']} ...")
        progress_text = (
            f"progress {index}/{total_cases} | "
            f"pass {pass_count} | fail {fail_count} | skip {skip_count}"
        )
        model_result = model_fn(
            VIDEOS_DIR / item["file"],
            runtime,
            expected_warnings=item["expected_warnings"],
            video_name=item["file"],
            progress_text=progress_text,
        )
        detected = model_result["detected"]
        passed = any(warning in item["expected_warnings"] for warning in detected)
        skipped = model_result["skipped"]
        result = {
            "video": item["file"],
            "status": "SKIP" if skipped else ("PASS" if passed else "FAIL"),
            "expected": item["expected_warnings"],
            "unsupported": [],
            "detected": detected,
            "missing": [] if passed or skipped else item["expected_warnings"],
            "unexpected": [] if passed or skipped else [
                warning for warning in detected if warning not in item["expected_warnings"]
            ],
        }
        print(
            f"  -> {result['status']} "
            f"(detected: {', '.join(detected) if detected else '-'})"
        )

        if result["status"] == "PASS":
            pass_count += 1
        elif result["status"] == "FAIL":
            fail_count += 1
        elif result["status"] == "SKIP":
            skip_count += 1

        result_progress_text = (
            f"progress {index}/{total_cases} | "
            f"pass {pass_count} | fail {fail_count} | skip {skip_count}"
        )

        if model_result["last_frame"] is not None:
            should_continue, hold_key = _show_result_frame(
                model_result["last_frame"],
                video_name=item["file"],
                expected_warnings=item["expected_warnings"],
                detected_warnings=detected,
                status=result["status"],
                progress_text=result_progress_text,
            )
            if not should_continue:
                results.append(result)
                if hold_key == 27:
                    break

        if model_result["aborted"]:
            results.append(result)
            break

        results.append(result)

    scored_results = [result for result in results if result["status"] != "SKIP"]
    passed = sum(1 for result in scored_results if result["status"] == "PASS")

    cv2.destroyWindow(WINDOW_NAME)

    return {
        "score": passed,
        "total": len(scored_results),
        "skipped": len(results) - len(scored_results),
        "results": results,
    }


def main():

    runtime = EvaluationRuntime()
    dataset = load_dataset()
    summary = test(model, dataset, runtime)

    for result in summary["results"]:
        print(f"[{result['status']}] {result['video']}")
        print(f"  expected: {', '.join(result['expected'])}")

        if result["unsupported"]:
            print(f"  unsupported: {', '.join(result['unsupported'])}")
            continue

        print(f"  detected: {', '.join(result['detected']) if result['detected'] else '-'}")
        print(f"  missing: {', '.join(result['missing']) if result['missing'] else '-'}")
        print(f"  unexpected: {', '.join(result['unexpected']) if result['unexpected'] else '-'}")

    print()
    print(f"Score: {summary['score']} / {summary['total']}")
    print(f"Skipped: {summary['skipped']}")


if __name__ == "__main__":
    main()
