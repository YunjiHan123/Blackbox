import cv2
import time

from config import (
    ALERT_BANNER_COLOR,
    ALERT_HOLD_SECONDS,
    ALERT_TEXT_COLOR,
    ALERT_WINDOW_NAME,
    ALERT_WINDOW_GAP,
    CAMERA_INDEX,
    DELETE_OUTPUTS_ON_EXIT,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAIN_WINDOW_HEIGHT,
    MAIN_WINDOW_WIDTH,
    MAIN_WINDOW_X,
    MAIN_WINDOW_Y,
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
from detection.pose_detector import PoseDetector
from detection.weapon_detector import WeaponDetector
from utils.camera import CameraStream, resize_to_window
from utils.detection_gate import DetectionGate
from utils.frame_saver import FrameSaver
from utils.logger import Logger
from utils.visualization import (
    draw_alert_banner,
    draw_detection,
    draw_pose,
    draw_status,
)


def main():

    weapon_detector = WeaponDetector(WEAPON_MODEL_PATH, allowed_labels=WEAPON_LABELS)
    pose_detector = PoseDetector(POSE_MODEL_PATH)
    logger = Logger()
    saver = FrameSaver()
    gate = DetectionGate(
        min_confidence=0.45,
        window_size=10,
        required_hits=6,
        cooldown_frames=60,
    )
    camera = CameraStream(
        camera_index=CAMERA_INDEX,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        rotation=PORTRAIT_ROTATION,
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
    alert_until = 0.0
    alert_frame = None

    while True:

        ret, frame = camera.read()

        if not ret or frame is None:
            print("Failed to read frame from camera.")
            break

        weapon_detections = weapon_detector.detect(frame)
        poses = pose_detector.detect(frame)

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

        for detection in gate.update(weapon_detections):
            label = detection["class_name"]
            confidence = detection["confidence"]
            logger.log(f"{label} detected (confidence={confidence:.2f})")
            saver.save(frame, label)
            alert_until = time.time() + ALERT_HOLD_SECONDS
            alert_frame = frame.copy()
            draw_alert_banner(
                alert_frame,
                "ALERT  WEAPON DETECTED",
                f"{label.upper()} detected ({confidence:.2f})",
                ALERT_BANNER_COLOR,
                ALERT_TEXT_COLOR,
            )
            cv2.namedWindow(ALERT_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(ALERT_WINDOW_NAME, MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT)
            cv2.moveWindow(
                ALERT_WINDOW_NAME,
                MAIN_WINDOW_X + MAIN_WINDOW_WIDTH + ALERT_WINDOW_GAP,
                MAIN_WINDOW_Y,
            )

        draw_status(
            frame,
            f"camera index: {CAMERA_INDEX} | portrait mode | esc to exit",
            STATUS_COLOR,
        )

        cv2.imshow(WINDOW_NAME, resize_to_window(frame, WINDOW_NAME))

        if alert_frame is not None and time.time() < alert_until:
            cv2.imshow(ALERT_WINDOW_NAME, resize_to_window(alert_frame, ALERT_WINDOW_NAME))
        elif alert_frame is not None and time.time() >= alert_until:
            cv2.destroyWindow(ALERT_WINDOW_NAME)
            alert_frame = None

        if cv2.waitKey(1) & 0xFF == 27:
            break

    camera.release()
    if alert_frame is not None:
        cv2.destroyWindow(ALERT_WINDOW_NAME)
    cv2.destroyAllWindows()

    if DELETE_OUTPUTS_ON_EXIT:
        saver.cleanup_saved_files()


if __name__ == "__main__":
    main()
