# vision/config.py

POSE_MODEL_PATH = "yolo26n-pose.pt"

# For webcam testing:
VIDEO_SOURCE = 0

# For IP camera:
VIDEO_SOURCE = "rtsp://admin:dtc2s2026@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0"

QUEUE_COUNT_POST_INTERVAL_SECONDS = 1.0

CONFIDENCE_THRESHOLD = 0.45

# Region where someone enters the line.
# Tune these points for your camera view.
LINE_REGION = [
    (50, 250),
    (450, 250),
    (450, 700),
    (50, 700),
]

# Region where wrist entering means pickup interaction.
PICKUP_REGION = [
    (700, 200),
    (1200, 200),
    (1200, 650),
    (700, 650),
]

# Bounding box height filter.
# Person must appear within this height range to count as pickup.
MIN_BOX_HEIGHT = 200
MAX_BOX_HEIGHT = 550

# Prevent repeated events every frame.
EVENT_COOLDOWN_SECONDS = 5

# COCO person class.
PERSON_CLASS_ID = 0

TRACKER = "bytetrack.yaml"

# Queue Management
MODEL_PATH = "yolo26n.pt"

# S3 setup
STORE_ID = "store1"

OUTPUT_JSON_PATH = f"{STORE_ID}data.json"

S3_JSON_URI = f"s3://nu-s26-dtc2-s33t4-website-data/{STORE_ID}data.json"

S3_UPLOAD_INTERVAL_SECONDS = 30.0