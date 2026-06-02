from core.event_types import (
    EVENT_CAMERA_BLOCK,
    EVENT_DOOR_LOCK_MANIPULATION,
    EVENT_FACE_NEAR,
    EVENT_LOITERING,
    EVENT_WEAPON,
)
from core.renderer import render_frame


def collect_detected_warnings(frame_result, pipeline, detected_warnings, timestamp=None):

    for person in frame_result["persons"]:
        person_id = person["id"]
        if timestamp is None:
            triggered = pipeline["loitering_detector"].update(person_id)
        else:
            triggered = pipeline["loitering_detector"].update_with_timestamp(person_id, timestamp)
        if triggered:
            _append_if_detected(detected_warnings, EVENT_LOITERING, True)

    _append_if_detected(
        detected_warnings,
        EVENT_WEAPON,
        bool(frame_result["weapon_alert_detections"]),
    )
    _append_if_detected(
        detected_warnings,
        EVENT_CAMERA_BLOCK,
        frame_result["block_event"] is not None and frame_result["block_event"]["should_capture"],
    )
    _append_if_detected(
        detected_warnings,
        EVENT_FACE_NEAR,
        frame_result["person_near_event"] is not None
        and frame_result["person_near_event"]["triggered"]
        and frame_result["person_near_event"]["should_capture"],
    )
    _append_if_detected(
        detected_warnings,
        EVENT_DOOR_LOCK_MANIPULATION,
        frame_result["door_lock_event"]["triggered"]
        and frame_result["door_lock_event"]["should_capture"],
    )


def render_evaluation_frame(frame, frame_result, pipeline):

    render_frame(frame, frame_result, pipeline, show_status=False)


def _append_if_detected(detected_warnings, event_type, detected):

    if detected and event_type not in detected_warnings:
        detected_warnings.append(event_type)
