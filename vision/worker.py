# vision/worker.py

from datetime import datetime, timezone
import time

import cv2
import requests
from ultralytics import YOLO, solutions

from vision.config import (
    MODEL_PATH,
    POSE_MODEL_PATH,
    VIDEO_SOURCE,
    API_EVENTS_URL,
    QUEUE_COUNT_URL,
    CAMERA_ID,
    CONFIDENCE_THRESHOLD,
    LINE_REGION,
    PICKUP_REGION,
    MIN_BOX_HEIGHT,
    MAX_BOX_HEIGHT,
    EVENT_COOLDOWN_SECONDS,
    PERSON_CLASS_ID,
    TRACKER,
    QUEUE_COUNT_POST_INTERVAL_SECONDS,
)

from vision.geometry import (
    point_in_polygon,
    box_center,
    box_height,
    box_in_valid_height_range,
    draw_polygon,
)


# COCO pose indexes
LEFT_WRIST = 9
RIGHT_WRIST = 10


class EventMemory:
    """
    Prevents the worker from sending the same line_enter or pickup event
    repeatedly for the same tracked person.
    """

    def __init__(self):
        self.last_line_enter = {}
        self.last_pickup = {}
        self.active_tracks = set()

    def can_emit(self, event_dict, track_id: int, now: float) -> bool:
        last_time = event_dict.get(track_id)

        if last_time is None:
            return True

        return now - last_time >= EVENT_COOLDOWN_SECONDS

    def mark_line_enter(self, track_id: int, now: float):
        self.last_line_enter[track_id] = now
        self.active_tracks.add(track_id)

    def mark_pickup(self, track_id: int, now: float):
        self.last_pickup[track_id] = now

        if track_id in self.active_tracks:
            self.active_tracks.remove(track_id)


def post_event(
    track_id: int,
    event_type: str,
    box_h: int | None = None,
    notes: str | None = None,
):
    """
    Sends a line_enter or pickup event to the backend.
    """

    payload = {
        "camera_id": str(CAMERA_ID),
        "track_id": int(track_id),
        "event_type": str(event_type),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "box_height": int(box_h) if box_h is not None else None,
        "notes": str(notes) if notes is not None else None,
    }

    try:
        response = requests.post(
            API_EVENTS_URL,
            json=payload,
            timeout=2,
        )
        response.raise_for_status()
        print(f"Sent event: {payload}")
    except requests.RequestException as exc:
        print(f"Failed to send event: {exc}")


def post_queue_count(queue_count: int):
    """
    Sends the current number of people in line to the backend.
    This number comes from QueueManager.
    """

    payload = {
        "camera_id": str(CAMERA_ID),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "queue_count": int(queue_count),
    }

    try:
        response = requests.post(
            QUEUE_COUNT_URL,
            json=payload,
            timeout=2,
        )
        response.raise_for_status()
        print(f"Sent queue count: {queue_count}")
    except requests.RequestException as exc:
        print(f"Failed to send queue count: {exc}")


def keypoint_visible(keypoints, index: int, min_conf: float = 0.3) -> bool:
    if keypoints is None:
        return False

    if index >= len(keypoints):
        return False

    x, y, conf = keypoints[index]
    return conf >= min_conf and x > 0 and y > 0


def get_keypoint_xy(keypoints, index: int):
    x, y, conf = keypoints[index]
    return int(x), int(y)


def wrist_inside_pickup_region(keypoints) -> bool:
    """
    Returns True if either wrist is inside the pickup region.
    """

    left_inside = False
    right_inside = False

    if keypoint_visible(keypoints, LEFT_WRIST):
        left_wrist = get_keypoint_xy(keypoints, LEFT_WRIST)
        left_inside = point_in_polygon(left_wrist, PICKUP_REGION)

    if keypoint_visible(keypoints, RIGHT_WRIST):
        right_wrist = get_keypoint_xy(keypoints, RIGHT_WRIST)
        right_inside = point_in_polygon(right_wrist, PICKUP_REGION)

    return left_inside or right_inside


def draw_point(frame, point, label, color):
    x, y = point

    cv2.circle(frame, (x, y), 6, color, -1)

    cv2.putText(
        frame,
        label,
        (x + 8, y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )


def main():
    # QueueManager counts people in line.
    queue_manager = solutions.QueueManager(
        model=MODEL_PATH,
        region=LINE_REGION,
        line_width=3,
        conf=CONFIDENCE_THRESHOLD,
        classes=[PERSON_CLASS_ID],
        show=False,
    )

    # Pose model is used for wrist-in-pickup-region logic.
    pose_model = YOLO(POSE_MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

    memory = EventMemory()
    last_queue_count_post = 0.0

    print("Starting vision worker with QueueManager.")
    print("Press q or Esc to quit.")

    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to read frame.")
            break

        # Run QueueManager on a copy so it can annotate/count the line region.
        queue_frame = frame.copy()
        queue_results = queue_manager(queue_frame)

        queue_count = queue_results.queue_count if hasattr(queue_results, "queue_count") else 0
        display_frame = frame.copy()

        now = time.time()

        if now - last_queue_count_post >= QUEUE_COUNT_POST_INTERVAL_SECONDS:
            post_queue_count(queue_count)
            last_queue_count_post = now

        # Run YOLO pose tracking separately for line_enter and pickup events.
        pose_results = pose_model.track(
            source=frame,
            persist=True,
            tracker=TRACKER,
            conf=CONFIDENCE_THRESHOLD,
            classes=[PERSON_CLASS_ID],
            verbose=False,
        )

        result = pose_results[0]

        # Draw regions manually so they are always visible.
        draw_polygon(display_frame, LINE_REGION, "Line Region", (255, 0, 0))
        draw_polygon(display_frame, PICKUP_REGION, "Pickup Region", (255, 0, 255))

        if result.boxes is not None and result.keypoints is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
            keypoints_all = result.keypoints.data.cpu().numpy()

            track_ids = result.boxes.id

            if track_ids is not None:
                for box, keypoints, raw_track_id in zip(
                    boxes_xyxy,
                    keypoints_all,
                    track_ids,
                ):
                    track_id = int(raw_track_id)

                    x1, y1, x2, y2 = map(int, box)
                    person_box = (x1, y1, x2, y2)

                    center = box_center(person_box)
                    h = box_height(person_box)

                    draw_point(display_frame, center, "center", (255, 255, 255))

                    in_line_region = point_in_polygon(center, LINE_REGION)

                    valid_pickup_distance = box_in_valid_height_range(
                        person_box,
                        MIN_BOX_HEIGHT,
                        MAX_BOX_HEIGHT,
                    )

                    wrist_in_pickup = wrist_inside_pickup_region(keypoints)

                    if keypoint_visible(keypoints, LEFT_WRIST):
                        left_wrist = get_keypoint_xy(keypoints, LEFT_WRIST)
                        draw_point(display_frame, left_wrist, "L wrist", (0, 255, 255))

                    if keypoint_visible(keypoints, RIGHT_WRIST):
                        right_wrist = get_keypoint_xy(keypoints, RIGHT_WRIST)
                        draw_point(display_frame, right_wrist, "R wrist", (0, 165, 255))

                    # Event 1: person enters line region.
                    if (
                        in_line_region
                        and track_id not in memory.active_tracks
                        and memory.can_emit(
                            memory.last_line_enter,
                            track_id,
                            now,
                        )
                    ):
                        post_event(
                            track_id=track_id,
                            event_type="line_enter",
                            box_h=h,
                            notes="Person center entered line region.",
                        )
                        memory.mark_line_enter(track_id, now)

                    # Event 2: same tracked person reaches pickup region.
                    if (
                        track_id in memory.active_tracks
                        and wrist_in_pickup
                        and valid_pickup_distance
                        and memory.can_emit(
                            memory.last_pickup,
                            track_id,
                            now,
                        )
                    ):
                        post_event(
                            track_id=track_id,
                            event_type="pickup",
                            box_h=h,
                            notes=(
                                "Wrist entered pickup region and person "
                                "was within valid box-height range."
                            ),
                        )
                        memory.mark_pickup(track_id, now)

                    # Draw debugging info from pose tracking.
                    is_active = track_id in memory.active_tracks
                    color = (0, 255, 0) if is_active else (0, 0, 255)

                    cv2.rectangle(
                        display_frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2,
                    )

                    label = (
                        f"ID {track_id} | h={h} | "
                        f"line={in_line_region} | "
                        f"pickup={wrist_in_pickup}"
                    )

                    cv2.putText(
                        display_frame,
                        label,
                        (x1, max(y1 - 10, 25)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

        cv2.imshow("Vision Worker Debug", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            print("Exiting vision worker.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()