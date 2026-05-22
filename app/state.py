# app/state.py

from datetime import datetime
from typing import Dict, List, Optional

from app.models import EventType, QueueCountUpdate, VisionEvent


class WaitTimeState:
    """
    In-memory state for MVP.

    Queue count comes from QueueManager.
    Wait time comes from tracked line_enter -> pickup events.
    
    Replace with AWS later.
    """

    def __init__(self):
        self.active_line_entries: Dict[int, datetime] = {}
        self.completed_waits: List[float] = []
        self.events: List[VisionEvent] = []

        self.latest_queue_count: int = 0
        self.latest_queue_count_time: Optional[datetime] = None

    def add_event(self, event: VisionEvent) -> None:
        self.events.append(event)

        track_id = event.track_id

        if event.event_type == EventType.LINE_ENTER:
            if track_id not in self.active_line_entries:
                self.active_line_entries[track_id] = event.timestamp

        elif event.event_type in {EventType.PICKUP, EventType.LINE_EXIT}:
            if track_id in self.active_line_entries:
                start_time = self.active_line_entries.pop(track_id)
                wait_seconds = (event.timestamp - start_time).total_seconds()

                if wait_seconds >= 0:
                    self.completed_waits.append(wait_seconds)

    def update_queue_count(self, update: QueueCountUpdate) -> None:
        self.latest_queue_count = int(update.queue_count)
        self.latest_queue_count_time = update.timestamp

    def get_status(self):
        avg_wait: Optional[float] = None

        if self.completed_waits:
            avg_wait = sum(self.completed_waits) / len(self.completed_waits)

        return {
            "active_people_in_line": self.latest_queue_count,
            "completed_waits_count": len(self.completed_waits),
            "average_wait_seconds": avg_wait,
        }

    def get_recent_events(self, limit: int = 50):
        return self.events[-limit:]


wait_state = WaitTimeState()