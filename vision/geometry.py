# vision/geometry.py

import cv2
import numpy as np


def point_in_polygon(point, polygon) -> bool:
    polygon_np = np.array(polygon, dtype=np.int32)
    return cv2.pointPolygonTest(polygon_np, point, False) >= 0


def box_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def box_height(box) -> int:
    x1, y1, x2, y2 = box
    return int(y2 - y1)


def box_in_valid_height_range(
    box,
    min_height: int,
    max_height: int,
) -> bool:
    height = box_height(box)
    return min_height <= height <= max_height


def draw_polygon(frame, polygon, label, color):
    points = np.array(polygon, dtype=np.int32)

    cv2.polylines(
        frame,
        [points],
        isClosed=True,
        color=color,
        thickness=3,
    )

    cv2.putText(
        frame,
        label,
        polygon[0],
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )