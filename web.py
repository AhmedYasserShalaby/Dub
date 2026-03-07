import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse


app = FastAPI(title="Dub Downloader")

# Ensure static directory exists
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Store the global active process
active_process = None
PROJECT_ROOT = Path(__file__).parent


class DubRequest(BaseModel):
    url: str
    browser: str = "chrome"
    extractor_mode: str = "hybrid"
    platform: str = "auto"


def latest_run_state_path() -> Path | None:
    candidates = sorted(
        PROJECT_ROOT.glob(".course_work/*/run_state.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def latest_manifest_path() -> Path | None:
    candidates = sorted(
        PROJECT_ROOT.glob(".course_work/*/course_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def estimate_progress_percent(statuses: list[str]) -> int:
    weights = {
        "queued_download": 0.05,
        "downloading": 0.3,
        "queued_dub": 0.55,
        "dubbing": 0.8,
        "dubbed": 1.0,
        "skipped_existing_dub": 1.0,
        "skipped_filtered": 1.0,
        "skipped_by_user": 1.0,
        "failed_download": 0.3,
        "failed_dub": 0.8,
    }
    if not statuses:
        return 0
    score = sum(weights.get(status, 0.0) for status in statuses)
    return max(0, min(100, round((score / len(statuses)) * 100)))


def read_progress_payload() -> dict:
    state_path = latest_run_state_path()
    manifest_path = latest_manifest_path()
    process_running = bool(active_process and active_process.poll() is None)

    if not state_path or not state_path.exists():
        return {
            "has_run": False,
            "process_running": process_running,
            "progress_percent": 0,
            "summary": {},
            "videos": [],
        }

    state = json.loads(state_path.read_text(encoding="utf-8"))
    manifest = {}
    if manifest_path and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    videos_map = state.get("videos", {})
    videos = []
    for key, value in sorted(videos_map.items()):
        videos.append(
            {
                "key": key,
                "lesson_title": value.get("lesson_title", key),
                "status": value.get("status", "unknown"),
                "error": value.get("error", ""),
                "final_video": value.get("final_video", ""),
            }
        )

    statuses = [video["status"] for video in videos]
    summary = {
        "total": len(videos),
        "queued_download": sum(status == "queued_download" for status in statuses),
        "downloading": sum(status == "downloading" for status in statuses),
        "queued_dub": sum(status == "queued_dub" for status in statuses),
        "dubbing": sum(status == "dubbing" for status in statuses),
        "done": sum(status in {"dubbed", "skipped_existing_dub"} for status in statuses),
        "failed": sum(status in {"failed_download", "failed_dub"} for status in statuses),
    }
    return {
        "has_run": True,
        "process_running": process_running,
        "course_name": state.get("course_name") or manifest.get("course_name", ""),
        "course_slug": state.get("course_slug") or manifest.get("course_slug", ""),
        "course_url": state.get("course_url") or manifest.get("course_url", ""),
        "state_path": str(state_path),
        "progress_percent": estimate_progress_percent(statuses),
        "summary": summary,
        "videos": videos,
    }


@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("Static frontend not found. Create static/index.html")
    with open(index_file, "r") as f:
        return f.read()


@app.get("/api/progress")
async def get_progress():
    return read_progress_payload()


@app.post("/api/dub")
async def start_dubbing(req: DubRequest):
    global active_process
    
    if active_process and active_process.poll() is None:
        return {"status": "error", "message": "A dubbing process is already running"}

    # Construct the command to run course_pipeline.py
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "course_pipeline.py"),
        "--course-url", req.url,
        "--browser", getattr(req, "browser", "chrome"),
        "--extractor-mode", getattr(req, "extractor_mode", "hybrid"),
        "--platform", getattr(req, "platform", "auto"),
        "--save-extraction-debug",
    ]

    try:
        # Start the subprocess globally
        active_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1, # Line buffered
            cwd=str(Path(__file__).parent)
        )
        return {"status": "started", "message": "Dubbing process launched"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/stream")
async def stream_logs(request: Request):
    """Streams terminal output of the active running process using Server-Sent Events"""
    global active_process

    async def event_generator() -> AsyncGenerator[dict, None]:
        if not active_process or active_process.poll() is not None:
            yield {
                "event": "message",
                "data": "No active dubbing process running."
            }
            return

        yield {
            "event": "message", 
            "data": "Connected to dubbing process output stream..."
        }

        while True:
            # If client disconnected
            if await request.is_disconnected():
                break

            # Check if process is done
            if active_process.poll() is not None:
                # Read any remaining output
                for line in active_process.stdout:
                    yield {"event": "message", "data": line.strip()}
                
                exit_code = active_process.returncode
                yield {
                    "event": "message", 
                    "data": f"Process exited with code {exit_code}."
                }
                break
                
            # Read line asynchronously using asyncio loop to avoid blocking server
            loop = asyncio.get_event_loop()
            line = await loop.run_in_executor(None, active_process.stdout.readline)
            
            if line:
                yield {"event": "message", "data": line.strip()}
            else:
                await asyncio.sleep(0.1)

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn
    # Make sure to run inside .venv
    uvicorn.run("web:app", host="0.0.0.0", port=8000, reload=True)
