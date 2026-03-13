import json
from pathlib import Path
import threading
import argparse

import cv2

from core.evaluation import collect_detected_warnings, render_evaluation_frame
from core.pipeline import analyze_frame, create_pipeline
from core.event_types import (
    EVENT_CAMERA_BLOCK,
    EVENT_DOOR_LOCK_MANIPULATION,
    EVENT_FACE_NEAR,
    EVENT_LOITERING,
    EVENT_WEAPON,
)


ROOT = Path(__file__).resolve().parent
VIDEOS_DIR = ROOT / "test_data" / "videos"
LABELS_PATH = ROOT / "test_data" / "labels.json"
WINDOW_NAME = "Evaluation"
RESULT_HOLD_MS = 800
SKIP_KEY = ord("s")
SUMMARY_PLOT_PATH = ROOT / "test_data" / "summary_metrics.png"
ASYNC_PLAYBACK = True
EVAL_PERSON_NEAR_HEAD_AREA_THRESHOLD = 0.18

IMPLEMENTED_WARNINGS = {
    EVENT_WEAPON,
    EVENT_CAMERA_BLOCK,
    EVENT_FACE_NEAR,
    EVENT_DOOR_LOCK_MANIPULATION,
    EVENT_LOITERING,
}


class EvaluationRuntime:

    def __init__(self):
        pass

    def create_pipeline(self):
        pipeline = create_pipeline()
        # Make face-near overlays more likely in evaluation videos.
        pipeline["person_proximity_detector"].head_area_threshold = (
            EVAL_PERSON_NEAR_HEAD_AREA_THRESHOLD
        )
        return pipeline


def load_dataset(labels_path=LABELS_PATH):

    with open(labels_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return payload["videos"]


def _frame_timestamp_seconds(capture, frame_index, fps):

    timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
    if timestamp_ms and timestamp_ms > 0:
        return timestamp_ms / 1000.0

    if fps and fps > 0:
        return frame_index / fps

    return float(frame_index)


def _update_max_loitering_seconds(pipeline, persons, current_max):

    if not persons:
        return current_max

    for person in persons:
        duration = pipeline["loitering_detector"].get_duration(person["id"])
        if duration > current_max:
            current_max = duration

    return current_max


def _draw_overlay(
    frame,
    video_name,
    expected_warnings,
    detected_warnings,
    status,
    frame_index,
    progress_text,
):

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    lines = [
        f"video: {video_name}",
        f"expected: {', '.join(expected_warnings)}",
        f"detected: {', '.join(detected_warnings) if detected_warnings else '-'}",
        f"status: {status} | frame: {frame_index} | S: skip | ESC: exit",
    ]

    color = (255, 255, 255)
    if status == "PASS":
        color = (0, 255, 0)
    elif status == "FAIL":
        color = (0, 0, 255)

    y = 28
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7 if index == 0 else 0.6,
            color if index == 3 else (255, 255, 255),
            2,
        )
        y += 25

    progress_box_width = 320
    progress_box_height = 36
    progress_x1 = max(0, frame.shape[1] - progress_box_width - 20)
    progress_y1 = 15
    progress_overlay = frame.copy()
    cv2.rectangle(
        progress_overlay,
        (progress_x1, progress_y1),
        (progress_x1 + progress_box_width, progress_y1 + progress_box_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(progress_overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(
        frame,
        progress_text,
        (progress_x1 + 10, progress_y1 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )


def _show_result_frame(frame, video_name, expected_warnings, detected_warnings, status, progress_text):

    result_frame = frame.copy()
    _draw_overlay(
        result_frame,
        video_name=video_name,
        expected_warnings=expected_warnings,
        detected_warnings=detected_warnings,
        status=status,
        frame_index=-1,
        progress_text=progress_text,
    )
    cv2.imshow(WINDOW_NAME, result_frame)
    key = cv2.waitKey(RESULT_HOLD_MS) & 0xFF
    return key not in (27, SKIP_KEY), key


def model(video_path, runtime, expected_warnings, video_name, progress_text, display=True):

    if display and ASYNC_PLAYBACK:
        return _model_async_playback(
            video_path,
            runtime,
            expected_warnings,
            video_name,
            progress_text,
        )

    return _model_sync(
        video_path,
        runtime,
        expected_warnings,
        video_name,
        progress_text,
        display=display,
    )


def _model_sync(video_path, runtime, expected_warnings, video_name, progress_text, display=True):

    pipeline = runtime.create_pipeline()
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_delay_ms = int(1000 / fps) if fps and fps > 0 else 1
    frame_index = 0
    detected_warnings = []
    last_frame = None
    max_loitering_seconds = 0.0
    aborted = False
    skipped = False

    try:
        while True:
            ret, frame = capture.read()
            if not ret or frame is None:
                break

            frame_index += 1
            timestamp = _frame_timestamp_seconds(capture, frame_index, fps)
            last_frame = frame.copy()

            frame_result = analyze_frame(pipeline, frame, timestamp=timestamp)
            collect_detected_warnings(
                frame_result,
                pipeline,
                detected_warnings,
                timestamp=timestamp,
            )
            max_loitering_seconds = _update_max_loitering_seconds(
                pipeline,
                frame_result["persons"],
                max_loitering_seconds,
            )

            if display:
                display_frame = frame.copy()
                render_evaluation_frame(display_frame, frame_result, pipeline)

                status = "RUNNING"
                if any(warning in expected_warnings for warning in detected_warnings):
                    status = "HIT"
                elif detected_warnings:
                    status = "DETECTED"

                _draw_overlay(
                    display_frame,
                    video_name=video_name,
                    expected_warnings=expected_warnings,
                    detected_warnings=detected_warnings,
                    status=status,
                    frame_index=frame_index,
                    progress_text=progress_text,
                )
                cv2.imshow(WINDOW_NAME, display_frame)
                key = cv2.waitKey(frame_delay_ms) & 0xFF
                if key == 27:
                    aborted = True
                    break
                if key == SKIP_KEY:
                    skipped = True
                    break

    finally:
        capture.release()

    return {
        "detected": detected_warnings,
        "last_frame": last_frame,
        "aborted": aborted,
        "skipped": skipped,
        "loitering_max_seconds": max_loitering_seconds,
    }


def _model_async_playback(video_path, runtime, expected_warnings, video_name, progress_text):

    pipeline = runtime.create_pipeline()
    display_capture = cv2.VideoCapture(str(video_path))
    analysis_capture = cv2.VideoCapture(str(video_path))

    if not display_capture.isOpened() or not analysis_capture.isOpened():
        display_capture.release()
        analysis_capture.release()
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = display_capture.get(cv2.CAP_PROP_FPS)
    frame_delay_ms = int(1000 / fps) if fps and fps > 0 else 1

    detected_warnings = []
    last_frame = None
    max_loitering_seconds = 0.0
    aborted = False
    skipped = False

    lock = threading.Lock()
    stop_event = threading.Event()

    latest_rendered_frame = None

    def analysis_loop():
        nonlocal last_frame, max_loitering_seconds, latest_rendered_frame
        frame_index = 0
        analysis_fps = analysis_capture.get(cv2.CAP_PROP_FPS)

        try:
            while not stop_event.is_set():
                ret, frame = analysis_capture.read()
                if not ret or frame is None:
                    break

                frame_index += 1
                timestamp = _frame_timestamp_seconds(analysis_capture, frame_index, analysis_fps)

                frame_result = analyze_frame(pipeline, frame, timestamp=timestamp)
                with lock:
                    collect_detected_warnings(
                        frame_result,
                        pipeline,
                        detected_warnings,
                        timestamp=timestamp,
                    )
                    max_loitering_seconds = _update_max_loitering_seconds(
                        pipeline,
                        frame_result["persons"],
                        max_loitering_seconds,
                    )
                    rendered = frame.copy()
                    render_evaluation_frame(rendered, frame_result, pipeline)
                    latest_rendered_frame = rendered
                    last_frame = frame.copy()
        finally:
            analysis_capture.release()

    analysis_thread = threading.Thread(target=analysis_loop, daemon=True)
    analysis_thread.start()

    try:
        while True:
            ret, frame = display_capture.read()
            if not ret or frame is None:
                break

            with lock:
                detected_snapshot = list(detected_warnings)
                display_base = latest_rendered_frame.copy() if latest_rendered_frame is not None else frame.copy()

            status = "RUNNING"
            if any(warning in expected_warnings for warning in detected_snapshot):
                status = "HIT"
            elif detected_snapshot:
                status = "DETECTED"

            display_frame = display_base
            _draw_overlay(
                display_frame,
                video_name=video_name,
                expected_warnings=expected_warnings,
                detected_warnings=detected_snapshot,
                status=status,
                frame_index=-1,
                progress_text=progress_text,
            )
            cv2.imshow(WINDOW_NAME, display_frame)
            key = cv2.waitKey(frame_delay_ms) & 0xFF
            if key == 27:
                aborted = True
                stop_event.set()
                break
            if key == SKIP_KEY:
                skipped = True
                stop_event.set()
                break
    finally:
        display_capture.release()

    # If display ended but analysis is still running, keep the window responsive.
    while analysis_thread.is_alive() and not (aborted or skipped):
        with lock:
            detected_snapshot = list(detected_warnings)
            display_base = latest_rendered_frame.copy() if latest_rendered_frame is not None else None

        if display_base is None:
            display_base = last_frame.copy() if last_frame is not None else None

        if display_base is not None:
            display_frame = display_base
            _draw_overlay(
                display_frame,
                video_name=video_name,
                expected_warnings=expected_warnings,
                detected_warnings=detected_snapshot,
                status="ANALYZING",
                frame_index=-1,
                progress_text=progress_text,
            )
            cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            aborted = True
            stop_event.set()
            break
        if key == SKIP_KEY:
            skipped = True
            stop_event.set()
            break

    analysis_thread.join()

    with lock:
        detected_snapshot = list(detected_warnings)
        final_last_frame = last_frame
        final_max_loitering_seconds = max_loitering_seconds

    return {
        "detected": detected_snapshot,
        "last_frame": final_last_frame,
        "aborted": aborted,
        "skipped": skipped,
        "loitering_max_seconds": final_max_loitering_seconds,
    }


def evaluate_case(video_name, expected_warnings, runtime):

    detected = model(VIDEOS_DIR / video_name, runtime)
    passed = detected in expected_warnings
    status = "PASS" if passed else "FAIL"

    return {
        "video": video_name,
        "status": status,
        "expected": expected_warnings,
        "unsupported": [],
        "detected": detected,
        "missing": [] if passed else expected_warnings,
        "unexpected": [] if detected is None or passed else [detected],
    }


def test(model_fn, dataset, runtime):
    return test_with_options(model_fn, dataset, runtime, display=True)


def test_with_options(model_fn, dataset, runtime, display=True):

    results = []

    total_cases = len(dataset)
    if display:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    pass_count = 0
    fail_count = 0
    skip_count = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    for index, item in enumerate(dataset, start=1):
        print(f"Processing [{index}/{total_cases}] {item['file']} ...")
        progress_text = (
            f"progress {index}/{total_cases} | "
            f"pass {pass_count} | fail {fail_count} | skip {skip_count}"
        )
        model_result = model_fn(
            VIDEOS_DIR / item["file"],
            runtime,
            expected_warnings=item["expected_warnings"],
            video_name=item["file"],
            progress_text=progress_text,
            display=display,
        )
        detected = model_result["detected"]
        loitering_max_seconds = model_result["loitering_max_seconds"]
        expected_all = item["expected_warnings"]
        supported_expected = [warning for warning in expected_all if warning in IMPLEMENTED_WARNINGS]
        unsupported_expected = [warning for warning in expected_all if warning not in IMPLEMENTED_WARNINGS]
        expected_set = set(supported_expected)
        detected_set = set(detected)
        tp = len(expected_set & detected_set)
        fp = len(detected_set - expected_set)
        fn = len(expected_set - detected_set)
        tn = len(IMPLEMENTED_WARNINGS - expected_set - detected_set)
        passed = (fn == 0 and fp == 0)
        skipped = model_result["skipped"]
        result = {
            "video": item["file"],
            "status": "SKIP" if skipped else ("PASS" if passed else "FAIL"),
            "expected": expected_all,
            "unsupported": unsupported_expected,
            "detected": detected,
            "missing": [] if passed or skipped else [warning for warning in supported_expected if warning not in detected_set],
            "unexpected": [] if passed or skipped else [warning for warning in detected if warning not in expected_set],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "loitering_max_seconds": loitering_max_seconds,
        }
        print(
            f"  -> {result['status']} "
            f"(detected: {', '.join(detected) if detected else '-'} | "
            f"tp {tp} fp {fp} fn {fn} tn {tn})"
        )
        if not display:
            print(f"  expected: {', '.join(result['expected'])}")
            if result["unsupported"]:
                print(f"  unsupported: {', '.join(result['unsupported'])}")
            print(f"  detected: {', '.join(result['detected']) if result['detected'] else '-'}")
            print(f"  missing: {', '.join(result['missing']) if result['missing'] else '-'}")
            print(f"  unexpected: {', '.join(result['unexpected']) if result['unexpected'] else '-'}")
            print(f"  loitering_max_seconds: {result['loitering_max_seconds']:.1f}s")
            if result["status"] != "SKIP":
                print(
                    f"  counts: tp {result['tp']} fp {result['fp']} fn {result['fn']} tn {result['tn']}"
                )

        if result["status"] == "PASS":
            pass_count += 1
        elif result["status"] == "FAIL":
            fail_count += 1
        elif result["status"] == "SKIP":
            skip_count += 1
        if result["status"] != "SKIP":
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

        result_progress_text = (
            f"progress {index}/{total_cases} | "
            f"pass {pass_count} | fail {fail_count} | skip {skip_count}"
        )

        if display and model_result["last_frame"] is not None:
            should_continue, hold_key = _show_result_frame(
                model_result["last_frame"],
                video_name=item["file"],
                expected_warnings=item["expected_warnings"],
                detected_warnings=detected,
                status=result["status"],
                progress_text=result_progress_text,
            )
            if not should_continue:
                results.append(result)
                if hold_key == 27:
                    break

        if model_result["aborted"]:
            results.append(result)
            break

        results.append(result)

    scored_results = [result for result in results if result["status"] != "SKIP"]
    passed = sum(1 for result in scored_results if result["status"] == "PASS")

    if display:
        cv2.destroyWindow(WINDOW_NAME)

    return {
        "score": passed,
        "total": len(scored_results),
        "skipped": len(results) - len(scored_results),
        "results": results,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
    }


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run analysis without video display (faster).",
    )
    args = parser.parse_args()

    runtime = EvaluationRuntime()
    dataset = load_dataset()
    summary = test_with_options(model, dataset, runtime, display=not args.headless)

    for result in summary["results"]:
        print(f"[{result['status']}] {result['video']}")
        print(f"  expected: {', '.join(result['expected'])}")

        if result["unsupported"]:
            print(f"  unsupported: {', '.join(result['unsupported'])}")
            continue

        print(f"  detected: {', '.join(result['detected']) if result['detected'] else '-'}")
        print(f"  missing: {', '.join(result['missing']) if result['missing'] else '-'}")
        print(f"  unexpected: {', '.join(result['unexpected']) if result['unexpected'] else '-'}")
        print(f"  loitering_max_seconds: {result['loitering_max_seconds']:.1f}s")
        if result["status"] != "SKIP":
            print(
                f"  counts: tp {result['tp']} fp {result['fp']} fn {result['fn']} tn {result['tn']}"
            )

    print()
    print(f"Score: {summary['score']} / {summary['total']}")
    print(f"Skipped: {summary['skipped']}")
    print(f"Totals: tp {summary['tp']} fp {summary['fp']} fn {summary['fn']} tn {summary['tn']}")

    _save_summary_plot(summary, SUMMARY_PLOT_PATH)
    print(f"Summary plot saved: {SUMMARY_PLOT_PATH}")


def _save_summary_plot(summary, output_path):

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping summary plot.")
        return

    labels = ["TP", "FP", "FN", "TN"]
    values = [summary["tp"], summary["fp"], summary["fn"], summary["tn"]]
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#1f77b4"]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=colors)
    plt.title("Evaluation Summary (All Videos)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    main()
