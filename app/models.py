# app/models.py

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel


class EventType(str, Enum):
    LINE_ENTER = "line_enter"
    PICKUP = "pickup"
    LINE_EXIT = "line_exit"


class VisionEvent(BaseModel):
    camera_id: str
    track_id: int
    event_type: EventType
    timestamp: datetime
    box_height: Optional[int] = None
    notes: Optional[str] = None


class QueueCountUpdate(BaseModel):
    camera_id: str
    timestamp: datetime
    queue_count: int


class StatusResponse(BaseModel):
    active_people_in_line: int
    completed_waits_count: int
    average_wait_seconds: Optional[float]