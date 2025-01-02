import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque

SOURCE = np.array([[976, 123], [1150, 147], [800, 630], [0, 520]])

LENGTH_OF_ONE_LINE = 3
LENGTH_BETWEEN_LINES = 9
WIDTH_OF_ROAD = 5

target_length = LENGTH_OF_ONE_LINE * 5 + LENGTH_BETWEEN_LINES * 6
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


def main():
    video_path = r"resources\highway.mp4"
    output_file = "vehicle_speeds.txt"
    output_video = "annotated_video2.mp4"

    video_info = sv.VideoInfo.from_video_path(video_path)
    model = YOLO("yolov8x.pt")
    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh=video_info.resolution_wh
    )
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)
    box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness)

    frame_generator = sv.get_video_frames_generator(video_path)
    polygon_zone = sv.PolygonZone(polygon=SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
    speeds = defaultdict(list)
    recorded_ids = set()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        output_video, fourcc, video_info.fps, video_info.resolution_wh
    )

    with open(output_file, "w") as f:
        f.write("Vehicle Speed Records\n")
        f.write("=====================\n\n")

        for frame in frame_generator:
            result = model.predict(frame)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[polygon_zone.trigger(detections)]
            detections = byte_track.update_with_detections(detections=detections)

            points = view_transformer.transform_points(
                detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            ).astype(int)

            labels = []
            for tracker_id, [_, y], class_id in zip(
                detections.tracker_id, points, detections.class_id
            ):
                coordinates[tracker_id].append(y)
                if len(coordinates[tracker_id]) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    distance = abs(
                        coordinates[tracker_id][-1] - coordinates[tracker_id][0]
                    )
                    time = len(coordinates[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6
                    speeds[tracker_id].append(speed)
                    avg_speed = sum(speeds[tracker_id]) / len(speeds[tracker_id])
                    vehicle_type = model.names[class_id]
                    labels.append(f"#{tracker_id} {int(avg_speed)} km/h")
                    if tracker_id not in recorded_ids:
                        f.write(
                            f"Tracker ID: {tracker_id}\n"
                            f"Average Speed: {int(avg_speed)} km/h\n"
                            f"Vehicle Type: {vehicle_type}\n"
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
