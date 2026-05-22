# app/main.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.models import VisionEvent, QueueCountUpdate, StatusResponse
from app.state import wait_state

app = FastAPI(title="Northwestern Dining Wait-Time API")

app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def homepage():
    return FileResponse("app/static/index.html")


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/events")
def receive_event(event: VisionEvent):
    """
    Called by the computer-vision worker whenever a tracked person
    enters the line or reaches pickup/exit.
    """
    wait_state.add_event(event)
    return {"status": "received", "event": event}


@app.post("/api/queue-count")
def receive_queue_count(update: QueueCountUpdate):
    """
    Called by the vision worker to update the live number of people in line.
    """
    wait_state.update_queue_count(update)
    return {"status": "received", "queue_count": update.queue_count}


@app.get("/api/status", response_model=StatusResponse)
def get_status():
    """
    Called by the website to display current line count and wait time.
    """
    return wait_state.get_status()


@app.get("/api/events")
def get_recent_events(limit: int = 50):
    """
    Useful for debugging.
    """
    return wait_state.get_recent_events(limit=limit)