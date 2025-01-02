import cv2
import numpy as np

SOURCE = np.array([[976, 123], [1150, 147], [800, 630], [0, 520]])


def show_mouse_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"X: {x}, Y: {y}")


if __name__ == "__main__":
    video_path = r"resources\highway.mp4"

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture the first frame of the video.")
        exit()

    # Draw the polygon on the frame
    cv2.polylines(frame, [SOURCE], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.namedWindow("First Frame")
    param = {"frame": frame}
    cv2.setMouseCallback("First Frame", show_mouse_coordinates, param)

    while True:
        cv2.imshow("First Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
