import cv2

from core.alert_state import AlertState
from core.event_processor import process_frame_events
from core.frame_saver import FrameSaver
from core.logger import Logger
from core.pipeline import analyze_frame
from core.renderer import render_frame
from settings import (
    ALERT_BANNER_COLOR,
    ALERT_HOLD_SECONDS,
    ALERT_TEXT_COLOR,
    ALERT_WINDOW_GAP,
    ALERT_WINDOW_NAME,
    CAMERA_BUFFER_SIZE,
    CAMERA_INDEX,
    DELETE_OUTPUTS_ON_EXIT,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    MAIN_WINDOW_HEIGHT,
    MAIN_WINDOW_WIDTH,
    MAIN_WINDOW_X,
    MAIN_WINDOW_Y,
    PORTRAIT_ROTATION,
    WINDOW_NAME,
)
from utils.camera import CameraStream, resize_to_window


def run_camera_runtime(pipeline):

    logger = Logger()
    saver = FrameSaver()
    alert_state = AlertState(ALERT_HOLD_SECONDS, ALERT_BANNER_COLOR, ALERT_TEXT_COLOR)
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

    _setup_main_window()

    try:
        while True:
            ret, frame = camera.read()

            if not ret or frame is None:
                print("Failed to read frame from camera.")
                break

            frame_result = analyze_frame(pipeline, frame)
            render_frame(frame, frame_result, pipeline)
            process_frame_events(
                frame,
                frame_result,
                pipeline,
                logger,
                saver,
                alert_state,
                ensure_alert_window,
            )
            _show_windows(frame, alert_state)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        camera.release()
        destroy_window_if_exists(ALERT_WINDOW_NAME)
        cv2.destroyAllWindows()

        if DELETE_OUTPUTS_ON_EXIT:
            saver.cleanup_saved_files()


def _setup_main_window():

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, MAIN_WINDOW_WIDTH, MAIN_WINDOW_HEIGHT)
    cv2.moveWindow(WINDOW_NAME, MAIN_WINDOW_X, MAIN_WINDOW_Y)


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


def _show_windows(frame, alert_state):

    cv2.imshow(WINDOW_NAME, resize_to_window(frame, WINDOW_NAME))

    if alert_state.is_active():
        cv2.imshow(
            ALERT_WINDOW_NAME,
            resize_to_window(alert_state.alert_frame, ALERT_WINDOW_NAME),
        )
        return

    alert_state.clear_if_expired()
    if alert_state.alert_frame is None:
        destroy_window_if_exists(ALERT_WINDOW_NAME)
