import cv2
from ultralytics import YOLO

from config import (
    ALERT_BANNER_COLOR,
    ALERT_HOLD_SECONDS,
    ALERT_TEXT_COLOR,
    ALERT_WINDOW_NAME,
    ALERT_WINDOW_GAP,
    CAMERA_INDEX,
    CAMERA_BLOCK_CAPTURE_INTERVAL_SECONDS,
    CAMERA_BLOCK_CHANGE_RATIO_THRESHOLD,
    CAMERA_BLOCK_MIN_BRIGHTNESS,
    CAMERA_BLOCK_MIN_PIXEL_STD,
    CAMERA_BUFFER_SIZE,
    DELETE_OUTPUTS_ON_EXIT,
    DOOR_LOCK_COOLDOWN_SECONDS,
    DOOR_LOCK_DECAY_FRAMES,
    DOOR_LOCK_FPS_HINT,
    DOOR_LOCK_MOVEMENT_DELTA_THRESHOLD,
    DOOR_LOCK_MOVEMENT_HISTORY_SIZE,
    DOOR_LOCK_REQUIRED_FRAMES,
    DOOR_LOCK_ROI_HEIGHT_RATIO,
    DOOR_LOCK_ROI_WIDTH_RATIO,
    DOOR_LOCK_ROI_X_RATIO,
    DOOR_LOCK_ROI_Y_RATIO,
    DOOR_LOCK_WRIST_CONFIDENCE_THRESHOLD,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAIN_WINDOW_HEIGHT,
    MAIN_WINDOW_WIDTH,
    MAIN_WINDOW_X,
    MAIN_WINDOW_Y,
    PERSON_NEAR_AREA_THRESHOLD,
    PERSON_NEAR_COOLDOWN_SECONDS,
    PERSON_NEAR_MIN_CONFIDENCE,
    PERSON_NEAR_REQUIRED_TIME_SECONDS,
    POSE_MODEL_PATH,
    POSE_COLOR,
    POSE_CONNECTIONS,
    POSE_LINE_COLOR,
    POSE_POINT_COLOR,
    PORTRAIT_ROTATION,
    STATUS_COLOR,
    WEAPON_MODEL_PATH,
    WEAPON_COLOR,
    WEAPON_LABELS,
    WINDOW_NAME,
)
from detection.camera_block_detector import CameraBlockDetector
from detection.door_lock_detector import DoorLockDetector
from detection.person_proximity_detector import PersonProximityDetector
from detection.pose_detector import PoseDetector
from detection.weapon_detector import WeaponDetector
from utils.alert_state import AlertState
from utils.camera import CameraStream, resize_to_window
from utils.detection_gate import DetectionGate
from utils.frame_saver import FrameSaver
from utils.logger import Logger
from utils.visualization import (
    draw_detection,
    draw_pose,
    draw_roi,
    draw_status,
)


def ensure_alert_window():

    cv2.namedWindow(ALERT_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ALERT_WINDOW_NAME, MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT)
    cv2.moveWindow(
        ALERT_WINDOW_NAME,
        MAIN_WINDOW_X + MAIN_WINDOW_WIDTH + ALERT_WINDOW_GAP,
        MAIN_WINDOW_Y,
    )


def destroy_window_if_exists(window_name):

    try:
        cv2.getWindowImageRect(window_name)
    except cv2.error:
        return

    cv2.destroyWindow(window_name)


def main():

    object_model = YOLO(WEAPON_MODEL_PATH)
    weapon_detector = WeaponDetector(
        allowed_labels=WEAPON_LABELS,
        model=object_model,
    )
    person_proximity_detector = PersonProximityDetector(
        model=object_model,
        area_threshold=PERSON_NEAR_AREA_THRESHOLD,
        required_time_seconds=PERSON_NEAR_REQUIRED_TIME_SECONDS,
        min_confidence=PERSON_NEAR_MIN_CONFIDENCE,
        cooldown_seconds=PERSON_NEAR_COOLDOWN_SECONDS,
    )
    pose_detector = PoseDetector(POSE_MODEL_PATH)
    camera_block_detector = CameraBlockDetector(
        change_ratio_threshold=CAMERA_BLOCK_CHANGE_RATIO_THRESHOLD,
        min_brightness=CAMERA_BLOCK_MIN_BRIGHTNESS,
        min_pixel_std=CAMERA_BLOCK_MIN_PIXEL_STD,
        capture_interval_seconds=CAMERA_BLOCK_CAPTURE_INTERVAL_SECONDS,
    )
    door_lock_detector = DoorLockDetector(
        roi_x_ratio=DOOR_LOCK_ROI_X_RATIO,
        roi_y_ratio=DOOR_LOCK_ROI_Y_RATIO,
        roi_width_ratio=DOOR_LOCK_ROI_WIDTH_RATIO,
        roi_height_ratio=DOOR_LOCK_ROI_HEIGHT_RATIO,
        wrist_confidence_threshold=DOOR_LOCK_WRIST_CONFIDENCE_THRESHOLD,
        required_frames=DOOR_LOCK_REQUIRED_FRAMES,
        movement_history_size=DOOR_LOCK_MOVEMENT_HISTORY_SIZE,
        decay_frames=DOOR_LOCK_DECAY_FRAMES,
        movement_delta_threshold=DOOR_LOCK_MOVEMENT_DELTA_THRESHOLD,
        cooldown_seconds=DOOR_LOCK_COOLDOWN_SECONDS,
        fps_hint=DOOR_LOCK_FPS_HINT,
    )
    logger = Logger()
    saver = FrameSaver()
    alert_state = AlertState(ALERT_HOLD_SECONDS, ALERT_BANNER_COLOR, ALERT_TEXT_COLOR)
    gate = DetectionGate(
        min_confidence=0.45,
        window_size=5,
        required_hits=3,
        cooldown_frames=60,
    )
    camera = CameraStream(
        camera_index=CAMERA_INDEX,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        rotation=PORTRAIT_ROTATION,
        buffer_size=CAMERA_BUFFER_SIZE,
    )

    if camera.open() is None:
        print(f"Failed to open camera. index={CAMERA_INDEX}")
        return

    print(
        f"Camera connected: index={CAMERA_INDEX}, "
        f"resolution={FRAME_WIDTH}x{FRAME_HEIGHT}"
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT)
    cv2.moveWindow(WINDOW_NAME, MAIN_WINDOW_X, MAIN_WINDOW_Y)
    while True:

        ret, frame = camera.read()

        if not ret or frame is None:
            print("Failed to read frame from camera.")
            break

        weapon_detections = weapon_detector.detect(frame)
        poses = pose_detector.detect(frame)
        block_event = camera_block_detector.analyze(frame)
        person_near_event = person_proximity_detector.analyze(frame)
        door_lock_event = door_lock_detector.analyze(frame, poses)

        for pose in poses:
            draw_pose(
                frame,
                pose,
                box_color=POSE_COLOR,
                line_color=POSE_LINE_COLOR,
                point_color=POSE_POINT_COLOR,
                connections=POSE_CONNECTIONS,
            )

        for detection in weapon_detections:
            draw_detection(frame, detection, WEAPON_COLOR)

        draw_roi(
            frame,
            door_lock_event["roi"],
            "Door Lock",
            door_lock_event["color"],
            thickness=3 if door_lock_event["target_wrist"] is not None else 2,
        )

        if door_lock_event["target_wrist"] is not None:
            cv2.circle(
                frame,
                door_lock_event["target_wrist"],
                6,
                door_lock_event["color"],
                -1,
            )

            if not door_lock_event["triggered"]:
                cv2.putText(
                    frame,
                    f"{door_lock_event['countdown_seconds']:.1f}s",
                    (door_lock_event["roi"][0], max(door_lock_event["roi"][1] - 40, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    door_lock_event["color"],
                    2,
                )
            elif door_lock_event["should_capture"]:
                logger.log(
                    f"{door_lock_event['event_name']} "
                    f"(dx={door_lock_event['movement_dx']}, dy={door_lock_event['movement_dy']}, "
                    f"frames={door_lock_event['frames_in_zone']})"
                )
                saver.save(frame, door_lock_event["event_name"])
                alert_state.activate(
                    frame,
                    "ALERT  DOOR LOCK TAMPERING",
                    door_lock_event["subtitle"],
                )
                ensure_alert_window()

        if person_near_event is not None:
            draw_detection(
                frame,
                person_near_event,
                ALERT_BANNER_COLOR,
            )

            if person_near_event["triggered"]:
                if person_near_event["should_capture"]:
                    logger.log(
                        "person near camera detected "
                        f"(area_ratio={person_near_event['area_ratio']:.2f}, "
                        f"confidence={person_near_event['confidence']:.2f}, "
                        f"elapsed={person_near_event['elapsed']:.1f}s)"
                    )
                    saver.save(frame, "person_near_camera")
                    alert_state.activate(
                        frame,
                        "ALERT  PERSON NEAR CAMERA",
                        "A person stayed too close to the camera",
                    )
                    ensure_alert_window()
            else:
                cv2.putText(
                    frame,
                    f"{person_near_event['remaining']:.1f}s",
                    (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    3,
                )

        if block_event is not None:
            if block_event["should_capture"]:
                logger.log(
                    "camera blocking detected "
                    f"(brightness={block_event['metrics']['brightness']:.1f}, "
                    f"pixel_std={block_event['metrics']['pixel_std']:.1f}, "
                    f"change_ratio={block_event['metrics']['change_ratio']:.2f})"
                )
                saver.save(frame, "camera_blocking")
                alert_state.activate(
                    frame,
                    "ALERT  CAMERA BLOCKING",
                    "Camera blocking attempt detected",
                )
                ensure_alert_window()

        for detection in gate.update(weapon_detections):
            label = detection["class_name"]
            confidence = detection["confidence"]
            logger.log(f"{label} detected (confidence={confidence:.2f})")
            saver.save(frame, label)
            alert_state.activate(
                frame,
                "ALERT  WEAPON DETECTED",
                f"{label.upper()} detected ({confidence:.2f})",
            )
            ensure_alert_window()

        draw_status(
            frame,
            f"camera index: {CAMERA_INDEX} | landscape mode | esc to exit",
            STATUS_COLOR,
        )

        cv2.imshow(WINDOW_NAME, resize_to_window(frame, WINDOW_NAME))

        if alert_state.is_active():
            cv2.imshow(
                ALERT_WINDOW_NAME,
                resize_to_window(alert_state.alert_frame, ALERT_WINDOW_NAME),
            )
        else:
            alert_state.clear_if_expired()
            if alert_state.alert_frame is None:
                destroy_window_if_exists(ALERT_WINDOW_NAME)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    destroy_window_if_exists(ALERT_WINDOW_NAME)
    cv2.destroyAllWindows()

    if DELETE_OUTPUTS_ON_EXIT:
        saver.cleanup_saved_files()


if __name__ == "__main__":
    main()
