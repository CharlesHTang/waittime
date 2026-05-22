# app/state.py

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from app.models import EventType, QueueCountUpdate, VisionEvent


RECENT_WAIT_WINDOW_MINUTES = 5


class WaitTimeState:
    """
    In-memory state for MVP.

    Queue count comes from QueueManager.
    Wait time comes from tracked line_enter -> pickup events.
    
    Replace with AWS later.
    """

    def __init__(self):
        self.active_line_entries: Dict[int, datetime] = {}
        self.completed_waits: List[Tuple[float, datetime]] = []
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
                    self.completed_waits.append((wait_seconds, event.timestamp))

    def update_queue_count(self, update: QueueCountUpdate) -> None:
        self.latest_queue_count = int(update.queue_count)
        self.latest_queue_count_time = update.timestamp

    def get_estimated_wait_seconds(self) -> Optional [float]:
        """
        Estimate wait time using completed waits from the last RECENT_WAIT_WINDOW_MINUTES minutes.
        If there are no recent completed waits, fall back to the latest
        completed wait.
        """

        if not self.completed_waits:
            return None
        
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=RECENT_WAIT_WINDOW_MINUTES)

        recent_waits = [wait_seconds for wait_seconds, timestamp in self.completed_waits if timestamp >= cutoff]

        if recent_waits:
            return sum(recent_waits) / len(recent_waits)
        
        # Fallback: latest completed wait.
        latest_wait_seconds, _ = self.completed_waits[-1]
        return latest_wait_seconds
    
    def get_status(self):
        estimated_wait = self.get_estimated_wait_seconds()

        return {
            "active_people_in_line": self.latest_queue_count,
            "completed_waits_count": len(self.completed_waits),
            "average_wait_seconds": estimated_wait,
        }

    def get_recent_events(self, limit: int = 50):
        return self.events[-limit:]


wait_state = WaitTimeState()