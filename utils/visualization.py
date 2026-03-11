import cv2


def draw_detection(frame, detection, color):

    x1, y1, x2, y2 = detection["bbox"]
    label = detection["class_name"]
    confidence = detection["confidence"]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        f"{label} {confidence:.2f}",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def draw_pose(frame, pose, box_color, line_color, point_color, connections):

    draw_detection(frame, pose, box_color)

    for start_idx, end_idx in connections:
        x1, y1, confidence1 = pose["keypoints"][start_idx]
        x2, y2, confidence2 = pose["keypoints"][end_idx]

        if confidence1 < 0.4 or confidence2 < 0.4:
            continue

        cv2.line(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            line_color,
            2,
        )

    for x, y, confidence in pose["keypoints"]:
        if confidence < 0.4:
            continue
        cv2.circle(frame, (int(x), int(y)), 3, point_color, -1)


def draw_status(frame, message, color):

    cv2.putText(
        frame,
        message,
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def draw_roi(frame, bbox, label, color, thickness=2):

    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_alert_banner(frame, title, subtitle, banner_color, text_color):

    height, width = frame.shape[:2]
    banner_height = min(140, max(100, height // 6))

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, banner_height), banner_color, -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    cv2.putText(
        frame,
        title,
        (20, 45),
        cv2.FONT_HERSHEY_DUPLEX,
        1.0,
        text_color,
        2,
    )
    cv2.putText(
        frame,
        subtitle,
        (20, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        text_color,
        2,
    )


def draw_tracked_person(frame, person, color, loitering_seconds=None):

    x1, y1, x2, y2 = person["bbox"]
    person_id = person["id"]

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        f"ID:{person_id}",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    if loitering_seconds is not None:
        cv2.putText(
            frame,
            f"{loitering_seconds:.1f}s",
            (x1, min(y2 + 25, frame.shape[0] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
