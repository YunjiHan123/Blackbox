import cv2

from settings import (
    ALERT_BANNER_COLOR,
    CAMERA_INDEX,
    CAMERA_MIRROR,
    PERSON_TRACK_COLOR,
    POSE_COLOR,
    POSE_CONNECTIONS,
    POSE_LINE_COLOR,
    POSE_POINT_COLOR,
    STATUS_COLOR,
    WEAPON_COLOR,
)
from utils.visualization import (
    draw_alert_flash,
    draw_alert_footer,
    draw_alert_header,
    draw_detection,
    draw_hud_event_log,
    draw_hud_status_panel,
    draw_pose,
    draw_roi,
    draw_tracked_person,
)


def _render_pose_overlays(frame, poses):
    for pose in poses:
        draw_pose(
            frame,
            pose,
            box_color=POSE_COLOR,
            line_color=POSE_LINE_COLOR,
            point_color=POSE_POINT_COLOR,
            connections=POSE_CONNECTIONS,
            draw_box=False,
        )


def _render_weapon_overlays(frame, weapon_detections):
    for detection in weapon_detections:
        draw_detection(frame, detection, WEAPON_COLOR, target_style=True)


def _render_tracked_persons(frame, persons, pipeline):
    for person in persons:
        person_id = person["id"]
        loitering_seconds = pipeline["loitering_detector"].get_duration(person_id)
        draw_tracked_person(
            frame,
            person,
            PERSON_TRACK_COLOR,
            loitering_seconds=loitering_seconds,
            target_style=True,
        )


def _render_door_lock_overlay(frame, door_lock_event):
    draw_roi(
        frame,
        door_lock_event["roi"],
        "Door Lock",
        door_lock_event["color"],
        thickness=3 if door_lock_event["target_wrist"] is not None else 2,
        extend_to_bottom=True,
    )
    draw_roi(
        frame,
        door_lock_event["contact_band"],
        "Lock Contact",
        door_lock_event["color"],
        thickness=2,
    )

    if door_lock_event["target_wrist"] is None:
        return

    cv2.circle(
        frame,
        door_lock_event["target_wrist"],
        6,
        door_lock_event["color"],
        -1,
    )

    if door_lock_event["triggered"]:
        return

    cv2.putText(
        frame,
        f"{door_lock_event['countdown_seconds']:.1f}s",
        (door_lock_event["roi"][0], max(door_lock_event["roi"][1] - 40, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        door_lock_event["color"],
        2,
    )


def _render_person_near_overlay(frame, person_near_event):
    if person_near_event is None:
        return

    draw_detection(frame, person_near_event, ALERT_BANNER_COLOR, target_style=True)
    if person_near_event["triggered"]:
        return

    cv2.putText(
        frame,
        f"{person_near_event['remaining']:.1f}s",
        (40, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        3,
    )


def _render_block_overlay(frame, block_event):
    if block_event is None:
        return

    metrics = block_event["metrics"]
    cv2.putText(
        frame,
        (
            "BLOCK RISK "
            f"BRI {metrics['brightness']:.0f}  "
            f"STD {metrics['pixel_std']:.0f}  "
            f"DELTA {metrics['change_ratio']:.2f}"
        ),
        (24, 160),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        ALERT_BANNER_COLOR,
        1,
        cv2.LINE_AA,
    )


def _render_status_overlays(frame, persons, weapon_detections, logger):
    draw_hud_status_panel(
        frame,
        [
            "Mode   LIVE SURVEILLANCE",
            f"Camera CAM-{CAMERA_INDEX:02d}   Mirror {'ON' if CAMERA_MIRROR else 'OFF'}",
            f"Targets {len(persons):02d}   Weapons {len(weapon_detections):02d}",
            "Exit   ESC",
        ],
        STATUS_COLOR,
        (230, 242, 255),
    )
    entries = logger.get_recent() if logger is not None else []
    draw_hud_event_log(frame, entries, STATUS_COLOR, (220, 235, 255))


def _render_active_alert(frame, alert_state):
    if alert_state is None or not alert_state.is_active():
        return

    draw_alert_flash(frame, ALERT_BANNER_COLOR, alert_state.get_flash_strength())
    draw_alert_header(
        frame,
        alert_state.title,
        alert_state.subtitle,
        ALERT_BANNER_COLOR,
        (255, 255, 255),
    )


def render_frame(frame, frame_result, pipeline, logger=None, alert_state=None, show_status=True):

    weapon_detections = frame_result["weapon_detections"]
    persons = frame_result["persons"]
    poses = frame_result["poses"]
    person_near_event = frame_result["person_near_event"]
    door_lock_event = frame_result["door_lock_event"]
    block_event = frame_result["block_event"]

    _render_pose_overlays(frame, poses)
    _render_weapon_overlays(frame, weapon_detections)
    _render_tracked_persons(frame, persons, pipeline)
    _render_door_lock_overlay(frame, door_lock_event)
    _render_person_near_overlay(frame, person_near_event)
    _render_block_overlay(frame, block_event)

    if show_status:
        _render_status_overlays(frame, persons, weapon_detections, logger)

    _render_active_alert(frame, alert_state)


def render_alert_frame(alert_state):

    if alert_state.alert_frame is None:
        return None

    alert_frame = alert_state.alert_frame.copy()
    strength = max(0.75, alert_state.get_flash_strength())
    draw_alert_flash(alert_frame, ALERT_BANNER_COLOR, strength)
    draw_alert_header(
        alert_frame,
        alert_state.title or "ALERT  THREAT DETECTED",
        alert_state.subtitle or "Immediate attention required",
        ALERT_BANNER_COLOR,
        (255, 255, 255),
    )
    draw_alert_footer(
        alert_frame,
        "WARNING  SECURITY EVENT ACTIVE",
        ALERT_BANNER_COLOR,
        (255, 255, 255),
    )
    return alert_frame
