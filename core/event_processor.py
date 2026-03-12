from core.event_handler import handle_event
from core.event_types import (
    EVENT_CAMERA_BLOCK,
    EVENT_DOOR_LOCK_MANIPULATION,
    EVENT_FACE_NEAR,
    EVENT_LABELS,
    EVENT_LOITERING,
    EVENT_WEAPON,
)


def process_frame_events(
    frame,
    frame_result,
    pipeline,
    logger,
    saver,
    alert_state,
    ensure_alert_window,
):

    _process_loitering_events(
        frame,
        frame_result["persons"],
        pipeline,
        logger,
        saver,
        alert_state,
        ensure_alert_window,
    )
    _process_door_lock_event(
        frame,
        frame_result["door_lock_event"],
        logger,
        saver,
        alert_state,
        ensure_alert_window,
    )
    _process_face_near_event(
        frame,
        frame_result["person_near_event"],
        logger,
        saver,
        alert_state,
        ensure_alert_window,
    )
    _process_camera_block_event(
        frame,
        frame_result["block_event"],
        logger,
        saver,
        alert_state,
        ensure_alert_window,
    )
    _process_weapon_events(
        frame,
        frame_result["weapon_alert_detections"],
        logger,
        saver,
        alert_state,
        ensure_alert_window,
    )


def _process_loitering_events(
    frame,
    persons,
    pipeline,
    logger,
    saver,
    alert_state,
    ensure_alert_window,
):

    for person in persons:
        person_id = person["id"]

        if pipeline["loitering_detector"].update(person_id):
            handle_event(
                frame,
                EVENT_LOITERING,
                logger,
                saver,
                alert_state,
                ensure_alert_window,
                subtitle=f"{EVENT_LABELS[EVENT_LOITERING]} (ID={person_id})",
                log_message=f"{EVENT_LOITERING} detected (id={person_id})",
            )


def _process_door_lock_event(
    frame,
    door_lock_event,
    logger,
    saver,
    alert_state,
    ensure_alert_window,
):

    if not door_lock_event["triggered"] or not door_lock_event["should_capture"]:
        return

    handle_event(
        frame,
        EVENT_DOOR_LOCK_MANIPULATION,
        logger,
        saver,
        alert_state,
        ensure_alert_window,
        subtitle=door_lock_event["subtitle"],
        log_message=(
            f"{door_lock_event['event_name']} "
            f"(dx={door_lock_event['movement_dx']}, dy={door_lock_event['movement_dy']}, "
            f"frames={door_lock_event['frames_in_zone']})"
        ),
        save_prefix=door_lock_event["event_name"],
    )


def _process_face_near_event(
    frame,
    person_near_event,
    logger,
    saver,
    alert_state,
    ensure_alert_window,
):

    if person_near_event is None:
        return

    if not person_near_event["triggered"] or not person_near_event["should_capture"]:
        return

    handle_event(
        frame,
        EVENT_FACE_NEAR,
        logger,
        saver,
        alert_state,
        ensure_alert_window,
        log_message=(
            f"{EVENT_FACE_NEAR} detected "
            f"(area_ratio={person_near_event['area_ratio']:.2f}, "
            f"confidence={person_near_event['confidence']:.2f}, "
            f"elapsed={person_near_event['elapsed']:.1f}s)"
        ),
    )


def _process_camera_block_event(
    frame,
    block_event,
    logger,
    saver,
    alert_state,
    ensure_alert_window,
):

    if block_event is None or not block_event["should_capture"]:
        return

    handle_event(
        frame,
        EVENT_CAMERA_BLOCK,
        logger,
        saver,
        alert_state,
        ensure_alert_window,
        log_message=(
            f"{EVENT_CAMERA_BLOCK} detected "
            f"(brightness={block_event['metrics']['brightness']:.1f}, "
            f"pixel_std={block_event['metrics']['pixel_std']:.1f}, "
            f"change_ratio={block_event['metrics']['change_ratio']:.2f})"
        ),
    )


def _process_weapon_events(
    frame,
    weapon_alert_detections,
    logger,
    saver,
    alert_state,
    ensure_alert_window,
):

    for detection in weapon_alert_detections:
        label = detection["class_name"]
        confidence = detection["confidence"]
        handle_event(
            frame,
            EVENT_WEAPON,
            logger,
            saver,
            alert_state,
            ensure_alert_window,
            subtitle=f"{EVENT_LABELS[EVENT_WEAPON]}: {label.upper()} ({confidence:.2f})",
            log_message=(
                f"{EVENT_WEAPON} detected "
                f"(label={label}, confidence={confidence:.2f})"
            ),
        )
