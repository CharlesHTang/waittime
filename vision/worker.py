# vision/worker.py

from datetime import datetime, timezone, timedelta
import time
import json
import subprocess
import threading
import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
from ultralytics import YOLO, solutions

from vision.config import (
    MODEL_PATH,
    POSE_MODEL_PATH,
    VIDEO_SOURCE,
    CONFIDENCE_THRESHOLD,
    LINE_REGION,
    PICKUP_REGION,
    MIN_BOX_HEIGHT,
    MAX_BOX_HEIGHT,
    EVENT_COOLDOWN_SECONDS,
    PERSON_CLASS_ID,
    TRACKER,
    QUEUE_COUNT_POST_INTERVAL_SECONDS,
    OUTPUT_JSON_PATH,
    S3_JSON_URI,
    S3_UPLOAD_INTERVAL_SECONDS,
    STORE_ID,
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


RECENT_WAIT_WINDOW_MINUTES = 5
STALE_WAIT_CUTOFF_MINUTES = 20


class LocalWaitState:
    def __init__(self):
        self.active_line_entries = {}
        self.completed_waits = []
        self.latest_queue_count = 0

    def record_line_enter(self, track_id: int, timestamp: datetime):
        if track_id not in self.active_line_entries:
            self.active_line_entries[track_id] = timestamp

    def record_pickup(self, track_id: int, timestamp: datetime):
        if track_id in self.active_line_entries:
            start_time = self.active_line_entries.pop(track_id)
            wait_seconds = (timestamp - start_time).total_seconds()

            if wait_seconds >= 0:
                self.completed_waits.append((wait_seconds, timestamp))

    def update_queue_count(self, queue_count: int):
        self.latest_queue_count = int(queue_count)

    def get_estimated_wait_seconds(self):
        if not self.completed_waits:
            return None

        now = datetime.now(timezone.utc)

        recent_cutoff = now - timedelta(minutes=RECENT_WAIT_WINDOW_MINUTES)
        stale_cutoff = now - timedelta(minutes=STALE_WAIT_CUTOFF_MINUTES)

        recent_waits = [
            wait_seconds
            for wait_seconds, timestamp in self.completed_waits
            if timestamp >= recent_cutoff
        ]

        if recent_waits:
            return sum(recent_waits) / len(recent_waits)

        latest_wait_seconds, latest_timestamp = self.completed_waits[-1]

        if latest_timestamp >= stale_cutoff:
            return latest_wait_seconds

        return None

    def to_json_data(self, store_id: str):
        return {
            "store_id": store_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "active_people_in_line": self.latest_queue_count,
            "average_wait_seconds": self.get_estimated_wait_seconds(),
        }
    

def write_status_json(state: LocalWaitState):
    """
    Writes the current wait status to a local JSON file, which can then be uploaded to S3.
    """
    data = state.to_json_data(STORE_ID)

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Wrote {OUTPUT_JSON_PATH}: {data}")


def upload_status_json_to_s3():
    """
    Uploads the local JSON file to S3 using the AWS CLI.
    """
    command = [
        "aws",
        "s3",
        "cp",
        OUTPUT_JSON_PATH,
        S3_JSON_URI,
        "--content-type",
        "application/json",
        "--cache-control",
        "no-cache",
    ]

    try:
        subprocess.run(command, check=True)
        print(f"Uploaded {OUTPUT_JSON_PATH} to {S3_JSON_URI}")
    except FileNotFoundError:
        print("Failed to upload JSON: AWS CLI is not installed or not in PATH.")
    except subprocess.CalledProcessError as exc:
        print(f"Failed to upload JSON to S3: {exc}")


upload_in_progress = False
upload_lock = threading.Lock()


def upload_status_json_to_s3_async():
    global upload_in_progress

    with upload_lock:
        if upload_in_progress:
            return

        upload_in_progress = True

    def worker():
        global upload_in_progress

        try:
            upload_status_json_to_s3()
        finally:
            with upload_lock:
                upload_in_progress = False

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()


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


def get_queue_count(queue_manager, frame) -> int:
    """
    Runs the QueueManager on the given frame to get the live queue count.
    """
    queue_frame = frame.copy()
    queue_results = queue_manager(queue_frame)

    return queue_results.queue_count if hasattr(queue_results, "queue_count") else 0


def run_pose_tracking(pose_model, frame):
    """
    Run YOLO pose tracking for line_enter and pickup events.
    """
    pose_results = pose_model.track(
        source=frame,
        persist=True,
        tracker=TRACKER,
        conf=CONFIDENCE_THRESHOLD,
        classes=[PERSON_CLASS_ID],
        verbose=False,
    )

    if not pose_results:
        return None

    return pose_results[0]


def process_pose_results(result, display_frame, memory, local_state, now):
    # Draw regions manually so they are always visible.
    draw_polygon(display_frame, LINE_REGION, "Line Region", (255, 0, 0))
    draw_polygon(display_frame, PICKUP_REGION, "Pickup Region", (255, 0, 255))

    if result is None:
        return

    if result.boxes is None or result.keypoints is None:
        return

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    keypoints_all = result.keypoints.data.cpu().numpy()

    track_ids = result.boxes.id

    if track_ids is None:
        return

    for box, keypoints, raw_track_id in zip(boxes_xyxy, keypoints_all, track_ids):
        process_person_detection(box=box, keypoints=keypoints, raw_track_id=raw_track_id, display_frame=display_frame, memory=memory, local_state=local_state, now=now)


def process_person_detection(box, keypoints, raw_track_id, display_frame, memory, local_state, now):
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
        event_time = datetime.now(timezone.utc)
        local_state.record_line_enter(track_id, event_time)
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
        event_time = datetime.now(timezone.utc)
        local_state.record_pickup(track_id, event_time)
        memory.mark_pickup(track_id, now)

    draw_person_debug_box(
        display_frame,
        person_box,
        track_id,
        h,
        in_line_region,
        wrist_in_pickup,
        memory,
    )


def draw_person_debug_box(frame, box, track_id, height, in_line_region, wrist_in_pickup, memory: EventMemory):
    """
    Draws a bounding box around the detected person, along with debug info like track ID, box height, and whether they are in the line or pickup regions.
    """
    x1, y1, x2, y2 = box

    is_active = track_id in memory.active_tracks
    color = (0, 255, 0) if is_active else (0, 0, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = (
        f"ID {track_id} | h={height} | "
        f"line={in_line_region} | "
        f"pickup={wrist_in_pickup}"
    )

    cv2.putText(
        frame,
        label,
        (x1, max(y1 - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
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
        verbose=False,
    )

    # Pose model is used for wrist-in-pickup-region logic.
    pose_model = YOLO(POSE_MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

    # EventMemory prevents duplicate events for the same track_id within a short time window.
    memory = EventMemory()
    last_queue_count_post = 0.0

    # Initialize local state and event memory.
    local_state = LocalWaitState()
    last_s3_upload = 0.0

    print("Starting vision worker with QueueManager.")
    print("Press q or Esc to quit.")

    while True:
        success, frame = cap.read()

        if not success:
            print("Failed to read frame.")
            break


        display_frame = frame.copy()
        now = time.time()

        if now - last_queue_count_post >= QUEUE_COUNT_POST_INTERVAL_SECONDS:
            queue_count = get_queue_count(queue_manager, frame)
            local_state.update_queue_count(queue_count)
            last_queue_count_post = now

        result = run_pose_tracking(pose_model, frame)

        process_pose_results(result=result, display_frame=display_frame, memory=memory, local_state=local_state, now=now)

        # Periodically write local state to JSON and upload to S3.
        if now - last_s3_upload >= S3_UPLOAD_INTERVAL_SECONDS:
            write_status_json(local_state)
            upload_status_json_to_s3_async()
            last_s3_upload = now

        cv2.imshow("Vision Worker Debug", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            print("Exiting vision worker.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()