import cv2
from ultralytics import YOLO

from config import (
    CAMERA_BUFFER_SIZE,
    CAMERA_INDEX,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAIN_WINDOW_HEIGHT,
    MAIN_WINDOW_WIDTH,
    MAIN_WINDOW_X,
    MAIN_WINDOW_Y,
    POSE_MODEL_PATH,
    PORTRAIT_ROTATION,
    WEAPON_MODEL_PATH,
    WINDOW_NAME,
)
from utils.camera import resize_to_window


mouse_state = {
    "window_x": None,
    "window_y": None,
    "frame_x": None,
    "frame_y": None,
}
display_state = {
    "width": None,
    "height": None,
}


def on_mouse(event, x, y, flags, param):

    if event != cv2.EVENT_MOUSEMOVE:
        return

    frame = param["frame"]
    displayed_width = param["display_width"]
    displayed_height = param["display_height"]

    mouse_state["window_x"] = x
    mouse_state["window_y"] = y

    if frame is None or displayed_width is None or displayed_height is None:
        mouse_state["frame_x"] = None
        mouse_state["frame_y"] = None
        return

    frame_height, frame_width = frame.shape[:2]

    if displayed_width <= 0 or displayed_height <= 0:
        mouse_state["frame_x"] = None
        mouse_state["frame_y"] = None
        return

    if x >= displayed_width or y >= displayed_height:
        mouse_state["frame_x"] = None
        mouse_state["frame_y"] = None
        return

    scale_x = frame_width / float(displayed_width)
    scale_y = frame_height / float(displayed_height)

    mouse_state["frame_x"] = int(x * scale_x)
    mouse_state["frame_y"] = int(y * scale_y)


def draw_debug_panel(frame, source_label, actual_capture_size):

    lines = [
        f"source: {source_label}",
        f"requested capture: {FRAME_WIDTH}x{FRAME_HEIGHT}",
        f"actual capture: {actual_capture_size[0]}x{actual_capture_size[1]}",
        f"rotated frame: {frame.shape[1]}x{frame.shape[0]}",
        f"window target: {MAIN_WINDOW_WIDTH}x{MAIN_WINDOW_HEIGHT} at ({MAIN_WINDOW_X}, {MAIN_WINDOW_Y})",
    ]

    try:
        _, _, window_width, window_height = cv2.getWindowImageRect(WINDOW_NAME)
        lines.append(f"current window: {window_width}x{window_height}")
    except cv2.error:
        lines.append("current window: unavailable")

    if mouse_state["frame_x"] is not None and mouse_state["frame_y"] is not None:
        lines.append(
            "mouse window/frame: "
            f"({mouse_state['window_x']}, {mouse_state['window_y']})"
            f" -> ({mouse_state['frame_x']}, {mouse_state['frame_y']})"
        )
    else:
        lines.append("mouse window/frame: move cursor over image")

    y = 30
    for line in lines:
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        y += 28


def main():

    source = CAMERA_INDEX
    source_label = str(source)

    pose_model = YOLO(POSE_MODEL_PATH)
    my_model = YOLO(WEAPON_MODEL_PATH)

    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)

    if not cap.isOpened():
        print(f"카메라 연결 실패: source={source_label}")
        return

    actual_capture_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_capture_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"camera source={source_label}")
    print(f"requested capture={FRAME_WIDTH}x{FRAME_HEIGHT}")
    print(f"actual capture={actual_capture_width}x{actual_capture_height}")
    print(f"rotation={PORTRAIT_ROTATION}")
    print(
        f"window target={MAIN_WINDOW_WIDTH}x{MAIN_WINDOW_HEIGHT} "
        f"at ({MAIN_WINDOW_X}, {MAIN_WINDOW_Y})"
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT)
    cv2.moveWindow(WINDOW_NAME, MAIN_WINDOW_X, MAIN_WINDOW_Y)

    callback_state = {
        "frame": None,
        "display_width": None,
        "display_height": None,
    }
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, callback_state)

    while True:

        ret, frame = cap.read()

        if not ret or frame is None:
            print("카메라 프레임 읽기 실패")
            break

        if PORTRAIT_ROTATION is not None:
            frame = cv2.rotate(frame, PORTRAIT_ROTATION)

        pose_results = pose_model(frame, verbose=False)
        my_results = my_model(frame, verbose=False)

        annotated_frame = pose_results[0].plot()
        annotated_frame = my_results[0].plot(img=annotated_frame)

        draw_debug_panel(
            annotated_frame,
            source_label=source_label,
            actual_capture_size=(actual_capture_width, actual_capture_height),
        )

        display_frame = resize_to_window(annotated_frame, WINDOW_NAME)
        display_state["width"] = display_frame.shape[1]
        display_state["height"] = display_frame.shape[0]
        callback_state["frame"] = annotated_frame
        callback_state["display_width"] = display_state["width"]
        callback_state["display_height"] = display_state["height"]

        cv2.imshow(WINDOW_NAME, display_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
