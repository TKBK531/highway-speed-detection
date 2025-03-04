import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque
import os
from datetime import datetime

SOURCE = np.array([[976, 123], [1150, 147], [800, 630], [0, 520]])

LENGTH_OF_ONE_LINE = 3
LENGTH_BETWEEN_LINES = 9
WIDTH_OF_ROAD = 6

target_length = LENGTH_OF_ONE_LINE * 6 + LENGTH_BETWEEN_LINES * 6
target_width = WIDTH_OF_ROAD

TARGET = np.array(
    [
        [1, 1],
        [target_width, 1],
        [target_width, target_length],
        [1, target_length],
    ]
)


class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.m = cv2.getPerspectiveTransform(
            source.astype(np.float32), target.astype(np.float32)
        )

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return (
            transformed_points.reshape(-1, 2)
            if transformed_points is not None
            else np.array([])
        )


def get_middle_line(source: np.ndarray, transformer: ViewTransformer) -> np.ndarray:
    transformed = transformer.transform_points(source)
    top_mid = (transformed[0] + transformed[1]) / 2
    bottom_mid = (transformed[2] + transformed[3]) / 2
    return np.array([top_mid, bottom_mid])


def main():
    video_path = r"resources\highway.mp4"

    # Generate a unique identifier using the current date and time
    unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directories
    output_dir = os.path.join("outputs", unique_id)
    log_dir = os.path.join(output_dir, "log")
    video_dir = os.path.join(output_dir, "video")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    output_file = os.path.join(log_dir, "vehicle_speeds.txt")
    output_video = os.path.join(video_dir, "annotated_video.mp4")

    video_info = sv.VideoInfo.from_video_path(video_path)
    # model = YOLO("yolov8x.pt")
    model = YOLO("yolov8n.pt")
    # model = YOLO("yolov7n.pt")
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    thickness = 1

    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

    frame_generator = sv.get_video_frames_generator(video_path)
    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    speeds = defaultdict(list)
    recorded_ids = set()
    exit_times = {}

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_video, fourcc, video_info.fps, video_info.resolution_wh
    )

    with open(output_file, "w") as f:
        f.write("Vehicle Speed Records\n")
        f.write("=====================\n\n")

        for frame_number, frame in enumerate(frame_generator):
            result = model.predict(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[polygon_zone.trigger(detections)]
            detections = byte_track.update_with_detections(detections=detections)

            points = view_transformer.transform_points(
                detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            ).astype(int)

            labels = []
            middle_line = get_middle_line(SOURCE, view_transformer)

            for tracker_id, (x, y), class_id in zip(
                detections.tracker_id, points, detections.class_id
            ):
                coordinates[tracker_id].append(y)

                side = "Left Lane" if x < middle_line[:, 0].mean() else "Right Lane"

                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id} ({side})")
                else:
                    distance = abs(
                        coordinates[tracker_id][-1] - coordinates[tracker_id][0]
                    )
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6
                    speeds[tracker_id].append(speed)
                    avg_speed = sum(speeds[tracker_id]) / len(speeds[tracker_id])
                    vehicle_type = model.names[class_id]

                    labels.append(f"#{tracker_id} {int(avg_speed)} km/h ({side})")

                    if tracker_id not in recorded_ids:
                        exit_time = frame_number / video_info.fps  # Time in seconds
                        exit_times[tracker_id] = exit_time
                        f.write(
                            f"Tracker ID: {tracker_id}\n"
                            f"Average Speed: {int(avg_speed)} km/h\n"
                            f"Vehicle Type: {vehicle_type}\n"
                            f"Lane: {side}\n"
                            f"Exit Time: {exit_time:.2f} seconds\n"
                            f"-----------------------------\n"
                        )
                        recorded_ids.add(tracker_id)

            annotated_frame = frame.copy()
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )

            out.write(annotated_frame)
            cv2.imshow("Annotated Frame", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
