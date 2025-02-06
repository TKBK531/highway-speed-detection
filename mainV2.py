import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque

SOURCE_RIGHT_LANE = np.array([[976, 123], [1060, 131], [368, 571], [0, 520]])
SOURCE_LEFT_LANE = np.array([[1060, 131], [1150, 147], [800, 630], [368, 571]])

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
    polygon_zone_left = sv.PolygonZone(polygon=SOURCE_LEFT_LANE)
    polygon_zone_right = sv.PolygonZone(polygon=SOURCE_RIGHT_LANE)
    view_transformer_left = ViewTransformer(source=SOURCE_LEFT_LANE, target=TARGET)
    view_transformer_right = ViewTransformer(source=SOURCE_RIGHT_LANE, target=TARGET)

    coordinates_left = defaultdict(lambda: deque(maxlen=video_info.fps))
    coordinates_right = defaultdict(lambda: deque(maxlen=video_info.fps))
    speeds_left = defaultdict(list)
    speeds_right = defaultdict(list)
    recorded_ids_left = set()
    recorded_ids_right = set()

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
            detections_left = detections[polygon_zone_left.trigger(detections)]
            detections_right = detections[polygon_zone_right.trigger(detections)]
            detections_left = byte_track.update_with_detections(
                detections=detections_left
            )
            detections_right = byte_track.update_with_detections(
                detections=detections_right
            )

            points_left = view_transformer_left.transform_points(
                detections_left.get_anchors_coordinates(
                    anchor=sv.Position.BOTTOM_CENTER
                )
            ).astype(int)
            points_right = view_transformer_right.transform_points(
                detections_right.get_anchors_coordinates(
                    anchor=sv.Position.BOTTOM_CENTER
                )
            ).astype(int)

            labels_left = []
            labels_right = []

            for tracker_id, [_, y], class_id in zip(
                detections_left.tracker_id, points_left, detections_left.class_id
            ):
                coordinates_left[tracker_id].append(y)
                if len(coordinates_left[tracker_id]) < video_info.fps / 2:
                    labels_left.append(f"#{tracker_id}")
                else:
                    distance = abs(
                        coordinates_left[tracker_id][-1]
                        - coordinates_left[tracker_id][0]
                    )
                    time = len(coordinates_left[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6
                    speeds_left[tracker_id].append(speed)
                    avg_speed = sum(speeds_left[tracker_id]) / len(
                        speeds_left[tracker_id]
                    )
                    vehicle_type = model.names[class_id]
                    labels_left.append(f"#{tracker_id} {int(avg_speed)} km/h")
                    if tracker_id not in recorded_ids_left:
                        f.write(
                            f"Left Lane - Tracker ID: {tracker_id}\n"
                            f"Average Speed: {int(avg_speed)} km/h\n"
                            f"Vehicle Type: {vehicle_type}\n"
                            f"-----------------------------\n"
                        )
                        recorded_ids_left.add(tracker_id)

            for tracker_id, [_, y], class_id in zip(
                detections_right.tracker_id, points_right, detections_right.class_id
            ):
                coordinates_right[tracker_id].append(y)
                if len(coordinates_right[tracker_id]) < video_info.fps / 2:
                    labels_right.append(f"#{tracker_id}")
                else:
                    distance = abs(
                        coordinates_right[tracker_id][-1]
                        - coordinates_right[tracker_id][0]
                    )
                    time = len(coordinates_right[tracker_id]) / video_info.fps
                    speed = distance / time * 3.6
                    speeds_right[tracker_id].append(speed)
                    avg_speed = sum(speeds_right[tracker_id]) / len(
                        speeds_right[tracker_id]
                    )
                    vehicle_type = model.names[class_id]
                    labels_right.append(f"#{tracker_id} {int(avg_speed)} km/h")
                    if tracker_id not in recorded_ids_right:
                        f.write(
                            f"Right Lane - Tracker ID: {tracker_id}\n"
                            f"Average Speed: {int(avg_speed)} km/h\n"
                            f"Vehicle Type: {vehicle_type}\n"
                            f"-----------------------------\n"
                        )
                        recorded_ids_right.add(tracker_id)

            annotated_frame = frame.copy()
            cv2.polylines(
                annotated_frame,
                [SOURCE_LEFT_LANE],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2,
            )
            cv2.polylines(
                annotated_frame,
                [SOURCE_RIGHT_LANE],
                isClosed=True,
                color=(255, 0, 0),
                thickness=2,
            )
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections_left
            )
            annotated_frame = box_annotator.annotate(
                scene=annotated_frame, detections=detections_right
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections_left, labels=labels_left
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections_right, labels=labels_right
            )

            out.write(annotated_frame)
            cv2.imshow("Annotated Frame", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
