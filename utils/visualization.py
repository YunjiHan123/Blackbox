import cv2


def _blend_region(frame, bbox, color, alpha):

    x1, y1, x2, y2 = bbox
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1], x2)
    y2 = min(frame.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return

    overlay = frame[y1:y2, x1:x2].copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, frame[y1:y2, x1:x2], 1.0 - alpha, 0, frame[y1:y2, x1:x2])


def draw_panel(frame, bbox, color, alpha=0.18, border_color=None):

    _blend_region(frame, bbox, color, alpha)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        border_color or color,
        1,
        cv2.LINE_AA,
    )


def draw_target_box(frame, bbox, color, label=None, value=None, thickness=2):

    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    corner = max(12, min(width, height) // 5)

    segments = [
        ((x1, y1), (x1 + corner, y1)),
        ((x1, y1), (x1, y1 + corner)),
        ((x2, y1), (x2 - corner, y1)),
        ((x2, y1), (x2, y1 + corner)),
        ((x1, y2), (x1 + corner, y2)),
        ((x1, y2), (x1, y2 - corner)),
        ((x2, y2), (x2 - corner, y2)),
        ((x2, y2), (x2, y2 - corner)),
    ]

    for start, end in segments:
        cv2.line(frame, start, end, color, thickness, cv2.LINE_AA)

    _blend_region(frame, bbox, color, 0.05)

    if label:
        text = label if value is None else f"{label}  {value}"
        label_width = max(120, len(text) * 9)
        chip_x1 = max(8, x1 + 6)
        chip_y1 = max(8, y1 + 6)
        chip_x2 = min(frame.shape[1] - 8, chip_x1 + label_width)
        chip_y2 = min(frame.shape[0] - 8, chip_y1 + 22)
        chip_bbox = (
            chip_x1,
            chip_y1,
            chip_x2,
            chip_y2,
        )
        draw_panel(frame, chip_bbox, color, alpha=0.2, border_color=color)
        cv2.putText(
            frame,
            text,
            (chip_bbox[0] + 8, chip_bbox[1] + 15),
            cv2.FONT_HERSHEY_DUPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_detection(frame, detection, color, target_style=False):

    x1, y1, x2, y2 = detection["bbox"]
    label = detection["class_name"]
    confidence = detection["confidence"]

    if target_style:
        draw_target_box(
            frame,
            (x1, y1, x2, y2),
            color,
            label=label.upper(),
            value=f"{confidence:.2f}",
        )
        return

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


def draw_pose(frame, pose, box_color, line_color, point_color, connections, draw_box=True):

    if draw_box:
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


def draw_roi(frame, bbox, label, color, thickness=2, extend_to_bottom=False):

    x1, y1, x2, y2 = bbox
    if extend_to_bottom:
        y2 = frame.shape[0] - 1

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


def draw_tracked_person(frame, person, color, loitering_seconds=None, target_style=False):

    x1, y1, x2, y2 = person["bbox"]
    person_id = person["id"]

    if target_style:
        value = f"T+{loitering_seconds:.1f}s" if loitering_seconds is not None else None
        draw_target_box(
            frame,
            (x1, y1, x2, y2),
            color,
            label=f"TRACK {person_id}",
            value=value,
        )
        return

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


def draw_hud_status_panel(frame, lines, accent_color, text_color):

    panel_bbox = (20, 20, min(frame.shape[1] - 20, 360), 146)
    draw_panel(frame, panel_bbox, (18, 30, 44), alpha=0.5, border_color=accent_color)
    cv2.putText(
        frame,
        "SYSTEM STATUS",
        (panel_bbox[0] + 14, panel_bbox[1] + 24),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        accent_color,
        1,
        cv2.LINE_AA,
    )

    for index, line in enumerate(lines):
        y = panel_bbox[1] + 50 + index * 24
        cv2.putText(
            frame,
            line,
            (panel_bbox[0] + 16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            1,
            cv2.LINE_AA,
        )


def draw_hud_event_log(frame, entries, accent_color, text_color):

    panel_height = 160
    y1 = max(20, frame.shape[0] - panel_height - 20)
    panel_bbox = (20, y1, min(frame.shape[1] - 20, 520), frame.shape[0] - 20)
    draw_panel(frame, panel_bbox, (12, 20, 33), alpha=0.58, border_color=accent_color)
    cv2.putText(
        frame,
        "EVENT LOG",
        (panel_bbox[0] + 14, panel_bbox[1] + 24),
        cv2.FONT_HERSHEY_DUPLEX,
        0.55,
        accent_color,
        1,
        cv2.LINE_AA,
    )

    if not entries:
        cv2.putText(
            frame,
            "No threats detected in this session.",
            (panel_bbox[0] + 16, panel_bbox[1] + 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            text_color,
            1,
            cv2.LINE_AA,
        )
        return

    for index, entry in enumerate(entries[:5]):
        y = panel_bbox[1] + 52 + index * 22
        cv2.putText(
            frame,
            f"[{entry['time']}] {entry['message']}",
            (panel_bbox[0] + 16, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            1,
            cv2.LINE_AA,
        )


def draw_alert_flash(frame, color, strength):

    if strength <= 0:
        return

    overlay = frame.copy()
    alpha = min(0.22, 0.08 + (0.14 * strength))
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

    border_thickness = max(4, int(8 * strength))
    cv2.rectangle(
        frame,
        (8, 8),
        (frame.shape[1] - 8, frame.shape[0] - 8),
        color,
        border_thickness,
        cv2.LINE_AA,
    )


def draw_alert_header(frame, title, subtitle, color, text_color):

    if not title:
        return

    width = min(frame.shape[1] - 40, 620)
    panel_bbox = (20, 20, 20 + width, 104)
    draw_panel(frame, panel_bbox, (40, 16, 18), alpha=0.62, border_color=color)
    cv2.putText(
        frame,
        title,
        (panel_bbox[0] + 16, panel_bbox[1] + 34),
        cv2.FONT_HERSHEY_DUPLEX,
        0.85,
        text_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        subtitle,
        (panel_bbox[0] + 16, panel_bbox[1] + 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        text_color,
        1,
        cv2.LINE_AA,
    )


def draw_alert_footer(frame, message, color, text_color):

    height = frame.shape[0]
    panel_bbox = (20, max(20, height - 86), min(frame.shape[1] - 20, 620), height - 20)
    draw_panel(frame, panel_bbox, (40, 12, 12), alpha=0.68, border_color=color)
    cv2.putText(
        frame,
        message,
        (panel_bbox[0] + 16, panel_bbox[1] + 42),
        cv2.FONT_HERSHEY_DUPLEX,
        0.72,
        text_color,
        2,
        cv2.LINE_AA,
    )
