import cv2

from settings import (
    ALERT_BANNER_COLOR,
    CAMERA_INDEX,
    PERSON_TRACK_COLOR,
    POSE_COLOR,
    POSE_CONNECTIONS,
    POSE_LINE_COLOR,
    POSE_POINT_COLOR,
    STATUS_COLOR,
    WEAPON_COLOR,
)
from utils.visualization import (
    draw_detection,
    draw_pose,
    draw_roi,
    draw_status,
    draw_tracked_person,
)


def render_frame(frame, frame_result, pipeline, show_status=True):

    weapon_detections = frame_result["weapon_detections"]
    persons = frame_result["persons"]
    poses = frame_result["poses"]
    person_near_event = frame_result["person_near_event"]
    door_lock_event = frame_result["door_lock_event"]

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

    for person in persons:
        person_id = person["id"]
        loitering_seconds = pipeline["loitering_detector"].get_duration(person_id)
        draw_tracked_person(
            frame,
            person,
            PERSON_TRACK_COLOR,
            loitering_seconds=loitering_seconds,
        )

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

    if person_near_event is not None:
        draw_detection(frame, person_near_event, ALERT_BANNER_COLOR)

        if not person_near_event["triggered"]:
            cv2.putText(
                frame,
                f"{person_near_event['remaining']:.1f}s",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                3,
            )

    if show_status:
        draw_status(
            frame,
            f"camera index: {CAMERA_INDEX} | landscape mode | esc to exit",
            STATUS_COLOR,
        )
