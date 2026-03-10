import cv2


CAMERA_CANDIDATES = [0, 1, 2, 3, 4, 5]


def open_camera(index):

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        return None

    for _ in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            return cap

    cap.release()
    return None


def main():

    current = 0
    cap = None

    while current < len(CAMERA_CANDIDATES):
        camera_index = CAMERA_CANDIDATES[current]

        if cap is None:
            cap = open_camera(camera_index)

        if cap is None:
            print(f"index={camera_index} failed")
            current += 1
            continue

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"index={camera_index} failed to read frame")
                break

            cv2.putText(
                frame,
                f"camera index: {camera_index}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                "n: next camera | s: select | esc: exit",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("Camera Probe", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                cap.release()
                cv2.destroyAllWindows()
                return

            if key == ord("s"):
                print(f"Selected camera index={camera_index}")
                cap.release()
                cv2.destroyAllWindows()
                return

            if key == ord("n"):
                break

        cap.release()
        cap = None
        current += 1

    cv2.destroyAllWindows()
    print("No available camera was found.")


if __name__ == "__main__":
    main()
