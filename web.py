import asyncio
import json
import shlex
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

EventSourceResponse: Any

try:
    from sse_starlette.sse import EventSourceResponse as ImportedEventSourceResponse
except ModuleNotFoundError:  # pragma: no cover - test/runtime fallback
    from fastapi.responses import StreamingResponse

    class FallbackEventSourceResponse(StreamingResponse):
        def __init__(self, content, **kwargs):
            super().__init__(content, media_type="text/event-stream", **kwargs)

    EventSourceResponse = FallbackEventSourceResponse
else:
    EventSourceResponse = ImportedEventSourceResponse


app = FastAPI(title="Dub Downloader")

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

PROJECT_ROOT = Path(__file__).parent
ORCHESTRATOR_WEB_DIR = PROJECT_ROOT / ".orchestrator" / "web"
ORCHESTRATOR_WEB_DIR.mkdir(parents=True, exist_ok=True)
SESSION_ROOT = ORCHESTRATOR_WEB_DIR / "sessions"
SESSION_ROOT.mkdir(parents=True, exist_ok=True)


class DubRequest(BaseModel):
    url: str
    browser: str = "chrome"
    extractor_mode: str = "hybrid"
    platform: str = "auto"
    auto_repair: bool = True


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.fromtimestamp(0, tz=UTC)


def model_dump(model: BaseModel) -> dict[str, Any]:
    dump = getattr(model, "model_dump", None)
    if dump is not None:
        return dump()
    return model.dict()


def read_json_file(path: Path | None, default: Any = None) -> Any:
    if not path or not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def session_dir_for(session_id: str) -> Path:
    return SESSION_ROOT / session_id


def session_record_path(session_id: str) -> Path:
    return session_dir_for(session_id) / "session.json"


def session_log_path(session_id: str) -> Path:
    return session_dir_for(session_id) / "events.log"


def discover_run_root(session_id: str) -> Path | None:
    candidates = sorted(
        PROJECT_ROOT.glob(f".course_work/*/runs/{session_id}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


@dataclass
class SessionArtifacts:
    session_id: str
    course_slug: str = ""
    run_root: Path | None = None
    manifest_path: Path | None = None
    state_path: Path | None = None
    report_path: Path | None = None
    lesson_reports: list[Path] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "course_slug": self.course_slug,
            "run_root": str(self.run_root) if self.run_root else "",
            "manifest_path": str(self.manifest_path) if self.manifest_path else "",
            "state_path": str(self.state_path) if self.state_path else "",
            "report_path": str(self.report_path) if self.report_path else "",
            "lesson_reports": [str(path) for path in self.lesson_reports],
        }


def discover_session_artifacts(session_id: str) -> SessionArtifacts:
    run_root = discover_run_root(session_id)
    if not run_root:
        return SessionArtifacts(session_id=session_id)

    course_slug = run_root.parent.parent.name
    manifest_path = run_root / "course_manifest.json"
    state_path = run_root / "run_state.json"
    report_path = run_root / "run_report.json"
    lesson_reports = sorted(run_root.glob("lessons/**/lesson_report.json"))
    return SessionArtifacts(
        session_id=session_id,
        course_slug=course_slug,
        run_root=run_root,
        manifest_path=manifest_path if manifest_path.exists() else None,
        state_path=state_path if state_path.exists() else None,
        report_path=report_path if report_path.exists() else None,
        lesson_reports=lesson_reports,
    )


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


def normalize_summary(summary: dict[str, Any] | None, statuses: list[str]) -> dict[str, Any]:
    summary = summary or {}
    if summary:
        return summary
    return {
        "total": len(statuses),
        "queued_download": sum(status == "queued_download" for status in statuses),
        "downloading": sum(status == "downloading" for status in statuses),
        "queued_dub": sum(status == "queued_dub" for status in statuses),
        "dubbing": sum(status == "dubbing" for status in statuses),
        "done": sum(status in {"dubbed", "skipped_existing_dub"} for status in statuses),
        "failed": sum(status in {"failed_download", "failed_dub"} for status in statuses),
    }


@dataclass
class WebSession:
    session_id: str
    request: DubRequest
    pipeline_cmd: list[str]
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    phase: str = "starting"
    status: str = "running"
    history: list[str] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)
    finished: bool = False
    session_dir: Path = field(default_factory=Path)
    record_path: Path = field(default_factory=Path)
    log_path: Path = field(default_factory=Path)
    context_path: Path | None = None
    pipeline_process: subprocess.Popen[str] | None = None
    repair_process: subprocess.Popen[str] | None = None
    pipeline_exit_code: int | None = None
    repair_exit_code: int | None = None
    repair_run_id: str | None = None
    repair_state_path: Path | None = None
    pipeline_state_path: Path | None = None
    pipeline_manifest_path: Path | None = None
    pipeline_report_path: Path | None = None
    run_artifacts: SessionArtifacts | None = None

    @classmethod
    def from_record(cls, payload: dict[str, Any]) -> "WebSession":
        request = DubRequest(**payload.get("request", {}))
        session = cls(
            session_id=str(payload["session_id"]),
            request=request,
            pipeline_cmd=list(payload.get("pipeline_cmd", [])),
            created_at=str(payload.get("created_at") or utc_now()),
            updated_at=str(payload.get("updated_at") or utc_now()),
            phase=str(payload.get("phase") or "starting"),
            status=str(payload.get("status") or "running"),
            finished=bool(payload.get("finished", False)),
            session_dir=Path(payload.get("session_dir") or session_dir_for(str(payload["session_id"]))),
            record_path=Path(payload.get("record_path") or session_record_path(str(payload["session_id"]))),
            log_path=Path(payload.get("log_path") or session_log_path(str(payload["session_id"]))),
            repair_run_id=payload.get("repair_run_id"),
            repair_state_path=Path(payload["repair_state_path"]) if payload.get("repair_state_path") else None,
            pipeline_state_path=Path(payload["pipeline_state_path"]) if payload.get("pipeline_state_path") else None,
            pipeline_manifest_path=Path(payload["pipeline_manifest_path"]) if payload.get("pipeline_manifest_path") else None,
            pipeline_report_path=Path(payload["pipeline_report_path"]) if payload.get("pipeline_report_path") else None,
        )
        history = payload.get("history")
        if isinstance(history, list):
            session.history = [str(item) for item in history]
        artifacts = payload.get("run_artifacts") or {}
        if artifacts:
            session.run_artifacts = SessionArtifacts(
                session_id=session.session_id,
                course_slug=str(artifacts.get("course_slug") or ""),
                run_root=Path(artifacts["run_root"]) if artifacts.get("run_root") else None,
                manifest_path=Path(artifacts["manifest_path"]) if artifacts.get("manifest_path") else None,
                state_path=Path(artifacts["state_path"]) if artifacts.get("state_path") else None,
                report_path=Path(artifacts["report_path"]) if artifacts.get("report_path") else None,
                lesson_reports=[Path(path) for path in artifacts.get("lesson_reports", [])],
            )
        return session

    def _base_record(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "request": model_dump(self.request),
            "pipeline_cmd": list(self.pipeline_cmd),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "phase": self.phase,
            "status": self.status,
            "finished": self.finished,
            "session_dir": str(self.session_dir),
            "record_path": str(self.record_path),
            "log_path": str(self.log_path),
            "pipeline_exit_code": self.pipeline_exit_code,
            "repair_exit_code": self.repair_exit_code,
            "repair_run_id": self.repair_run_id,
            "repair_state_path": str(self.repair_state_path) if self.repair_state_path else "",
            "pipeline_state_path": str(self.pipeline_state_path) if self.pipeline_state_path else "",
            "pipeline_manifest_path": str(self.pipeline_manifest_path) if self.pipeline_manifest_path else "",
            "pipeline_report_path": str(self.pipeline_report_path) if self.pipeline_report_path else "",
            "run_artifacts": self.run_artifacts.to_dict() if self.run_artifacts else {},
        }

    def persist(self) -> None:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.updated_at = utc_now()
        write_json_file(self.record_path, self._base_record())

    def append(self, message: str) -> None:
        line = message.rstrip("\n")
        with self.lock:
            self.history.append(line)
            self.updated_at = utc_now()
            self.session_dir.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            self.persist()

    def snapshot(self, start_index: int) -> tuple[list[str], bool]:
        with self.lock:
            return self.history[start_index:].copy(), self.finished

    def mark_finished(self, status: str) -> None:
        with self.lock:
            self.status = status
            self.finished = True
            self.updated_at = utc_now()
            self.persist()

    def attach_artifacts(self, artifacts: SessionArtifacts) -> None:
        with self.lock:
            self.run_artifacts = artifacts
            if artifacts.run_root:
                self.pipeline_state_path = artifacts.state_path
                self.pipeline_manifest_path = artifacts.manifest_path
                self.pipeline_report_path = artifacts.report_path
            self.updated_at = utc_now()
            self.persist()

    def session_urls(self) -> dict[str, str]:
        return {
            "detail_url": f"/api/sessions/{self.session_id}",
            "progress_url": f"/api/sessions/{self.session_id}/progress",
            "stream_url": f"/api/sessions/{self.session_id}/stream",
        }

    def to_record(self) -> dict[str, Any]:
        record = self._base_record()
        record.update(self.session_urls())
        record["run_id"] = self.session_id
        return record

    def to_summary(self) -> dict[str, Any]:
        record = self.to_record()
        record["history_tail"] = self.history[-10:]
        return record

    def to_detail(self) -> dict[str, Any]:
        artifacts = self.run_artifacts or discover_session_artifacts(self.session_id)
        detail = self.to_summary()
        detail["artifacts"] = artifacts.to_dict()
        return detail


class SessionRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, WebSession] = {}

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()

    def register(self, session: WebSession) -> WebSession:
        with self._lock:
            self._sessions[session.session_id] = session
        session.persist()
        return session

    def get(self, session_id: str) -> WebSession | None:
        with self._lock:
            return self._sessions.get(session_id)

    def _sorted_sessions(self, sessions: list[WebSession]) -> list[WebSession]:
        return sorted(
            sessions,
            key=lambda item: (
                parse_utc_timestamp(item.updated_at),
                parse_utc_timestamp(item.created_at),
                item.session_id,
            ),
            reverse=True,
        )

    def load_from_disk(self) -> None:
        if not SESSION_ROOT.exists():
            return
        for record_path in sorted(SESSION_ROOT.glob("*/session.json"), key=lambda path: path.stat().st_mtime):
            payload = read_json_file(record_path, {})
            if not payload:
                continue
            session_id = str(payload.get("session_id") or record_path.parent.name)
            updated_at = str(payload.get("updated_at") or "")
            with self._lock:
                existing = self._sessions.get(session_id)
                if existing and existing.updated_at >= updated_at:
                    continue
            session = WebSession.from_record(payload)
            with self._lock:
                existing = self._sessions.get(session_id)
                if existing and existing.updated_at >= session.updated_at:
                    continue
                self._sessions[session_id] = session

    def get_or_load(self, session_id: str) -> WebSession | None:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is not None:
            return session
        record_path = session_record_path(session_id)
        payload = read_json_file(record_path, {})
        if not payload:
            return None
        session = WebSession.from_record(payload)
        with self._lock:
            existing = self._sessions.get(session_id)
            if existing is None or existing.updated_at <= session.updated_at:
                self._sessions[session_id] = session
                return session
            return existing

    def list(self) -> list[WebSession]:
        self.load_from_disk()
        with self._lock:
            sessions = list(self._sessions.values())
        return self._sorted_sessions(sessions)

    def latest(self) -> WebSession | None:
        sessions = self.list()
        return sessions[0] if sessions else None


session_registry = SessionRegistry()


def get_session_or_404(session_id: str) -> WebSession:
    session = session_registry.get_or_load(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return session


def latest_session_or_empty() -> WebSession | None:
    return session_registry.latest()


def read_orchestrator_payload(session: WebSession | None) -> dict[str, Any]:
    if not session or not session.request.auto_repair:
        return {}

    state_path = session.repair_state_path
    if not state_path or not state_path.exists():
        return {
            "enabled": True,
            "active": session.phase == "repairing" and not session.finished,
            "status": "waiting" if session.phase != "repairing" else "running",
            "job_id": session.repair_run_id or "",
            "iteration": 0,
            "max_iterations": 0,
            "last_decision": "",
            "last_reason": "",
        }

    state = read_json_file(state_path, {})
    latest = state.get("iterations", [])[-1] if state.get("iterations") else {}
    return {
        "enabled": True,
        "active": session.phase == "repairing" and not session.finished,
        "status": state.get("status", "unknown"),
        "job_id": state.get("job_id", ""),
        "iteration": state.get("iteration", 0),
        "max_iterations": state.get("max_iterations", 0),
        "last_decision": latest.get("decision", ""),
        "last_reason": latest.get("decision_reason", ""),
    }


def read_session_progress_payload(session_id: str) -> dict[str, Any]:
    session = session_registry.get_or_load(session_id)
    artifacts = discover_session_artifacts(session_id)
    if session and artifacts.run_root:
        session.attach_artifacts(artifacts)

    state = read_json_file(artifacts.state_path, {}) if artifacts.state_path else {}
    report = read_json_file(artifacts.report_path, {}) if artifacts.report_path else {}
    manifest = read_json_file(artifacts.manifest_path, {}) if artifacts.manifest_path else {}
    videos_map = state.get("videos") or report.get("videos") or {}
    videos: list[dict[str, Any]] = []
    if isinstance(videos_map, dict):
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
    elif isinstance(videos_map, list):
        for value in videos_map:
            videos.append(
                {
                    "key": value.get("key", value.get("lesson_title", "")),
                    "lesson_title": value.get("lesson_title", ""),
                    "status": value.get("status", "unknown"),
                    "error": value.get("error", ""),
                    "final_video": value.get("final_video", ""),
                }
            )

    statuses = [video["status"] for video in videos]
    summary = normalize_summary(report.get("summary"), statuses)
    payload: dict[str, Any] = {
        "session_id": session_id,
        "run_id": session_id,
        "has_run": bool(artifacts.run_root or state or report),
        "process_running": bool(session and not session.finished),
        "session_status": session.status if session else report.get("status", "unknown"),
        "course_name": state.get("course_name") or manifest.get("course_name", ""),
        "course_slug": state.get("course_slug") or manifest.get("course_slug") or artifacts.course_slug,
        "course_url": state.get("course_url") or manifest.get("course_url", session.request.url if session else ""),
        "state_path": str(artifacts.state_path) if artifacts.state_path else "",
        "report_path": str(artifacts.report_path) if artifacts.report_path else "",
        "manifest_path": str(artifacts.manifest_path) if artifacts.manifest_path else "",
        "progress_percent": estimate_progress_percent(statuses),
        "summary": summary,
        "videos": videos,
        "qa": report.get("qa", {}),
        "timings": report.get("timings", {}),
        "final_video": report.get("final_video") or state.get("final_video", ""),
        "repair": read_orchestrator_payload(session),
    }
    if session:
        payload["session"] = session.to_summary()
    return payload


def read_compat_progress_payload() -> dict[str, Any]:
    latest = latest_session_or_empty()
    if not latest:
        return {
            "has_run": False,
            "process_running": False,
            "progress_percent": 0,
            "summary": {},
            "videos": [],
            "repair": {},
        }
    return read_session_progress_payload(latest.session_id)


def build_course_pipeline_command(req: DubRequest, run_id: str) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / "course_pipeline.py"),
        "--course-url",
        req.url,
        "--run-id",
        run_id,
        "--browser",
        req.browser,
        "--extractor-mode",
        req.extractor_mode,
        "--platform",
        req.platform,
        "--save-extraction-debug",
    ]


def build_repair_context(session: WebSession) -> Path:
    artifacts = session.run_artifacts or discover_session_artifacts(session.session_id)
    session.attach_artifacts(artifacts)

    progress_payload = read_session_progress_payload(session.session_id)
    failed_videos = [
        video for video in progress_payload.get("videos", []) if video["status"] in {"failed_download", "failed_dub"}
    ]
    failed_lines = []
    for video in failed_videos[:10]:
        failed_lines.append(
            f"- {video['lesson_title']} [{video['status']}]: {video['error'] or 'no error text captured'}"
        )
    history_tail = "\n".join(session.history[-80:])
    state_path = artifacts.state_path
    manifest_path = artifacts.manifest_path
    report_path = artifacts.report_path
    state_excerpt = state_path.read_text(encoding="utf-8")[:4000] if state_path and state_path.exists() else "missing"
    manifest_excerpt = (
        manifest_path.read_text(encoding="utf-8")[:4000] if manifest_path and manifest_path.exists() else "missing"
    )
    report_excerpt = report_path.read_text(encoding="utf-8")[:4000] if report_path and report_path.exists() else "missing"
    replay_command = (
        f"{shlex.quote(sys.executable)} course_pipeline.py "
        f"--course-url {shlex.quote(session.request.url)} "
        f"--run-id {shlex.quote(session.session_id)} "
        f"--browser {shlex.quote(session.request.browser)} "
        f"--extractor-mode {shlex.quote(session.request.extractor_mode)} "
        f"--platform {shlex.quote(session.request.platform)} "
        "--max-videos 1 --download-workers 1 --dub-workers 1 --save-extraction-debug"
    )
    context = f"""
    Website-triggered course dubbing run failed.

    Request:
    - course_url: {session.request.url}
    - browser: {session.request.browser}
    - extractor_mode: {session.request.extractor_mode}
    - platform: {session.request.platform}

    Course pipeline command:
    - {' '.join(session.pipeline_cmd)}

    Current run state path:
    - {state_path or 'missing'}

    Current manifest path:
    - {manifest_path or 'missing'}

    Replay command:
    - {replay_command}

    Failed videos:
    {chr(10).join(failed_lines) or '- none captured'}

    Run state excerpt:
    {state_excerpt}

    Manifest excerpt:
    {manifest_excerpt}

    Report excerpt:
    {report_excerpt}

    Recent website log tail:
    {history_tail or 'no logs captured'}

    Repair objective:
    - Fix the repository so the website flow can successfully process this course URL.
    - Preserve existing successful behavior for supported platforms.
    - Prefer durable fixes in extraction, retries, output routing, and failure handling over hacks.
    """
    context_path = ORCHESTRATOR_WEB_DIR / f"{session.session_id}-repair-context.md"
    context_path.write_text(context.strip() + "\n", encoding="utf-8")
    session.context_path = context_path
    return context_path


def start_repair_process(session: WebSession) -> subprocess.Popen[str]:
    context_path = build_repair_context(session)
    py = shlex.quote(sys.executable)
    quoted_url = shlex.quote(session.request.url)
    quoted_browser = shlex.quote(session.request.browser)
    quoted_mode = shlex.quote(session.request.extractor_mode)
    quoted_platform = shlex.quote(session.request.platform)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "orchestrate.py"),
        "run",
        "--goal",
        f"Make the local website workflow successfully dub the submitted course URL {session.request.url!r} and harden the failure path that just broke.",
        "--context-file",
        str(context_path),
        "--verify",
        f"{py} -m unittest tests.test_orchestrate",
        "--verify",
        f"{py} course_pipeline.py --help",
        "--verify",
        (
            f"{py} course_pipeline.py --course-url {quoted_url} "
            f"--browser {quoted_browser} "
            f"--extractor-mode {quoted_mode} "
            f"--platform {quoted_platform} "
            "--max-videos 1 --download-workers 1 --dub-workers 1"
        ),
        "--max-iterations",
        "4",
    ]
    session.append("[auto_repair] Launching orchestrator repair loop...")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_ROOT),
    )
    session.repair_process = process
    session.phase = "repairing"
    return process


def drain_process_output(session: WebSession, process: subprocess.Popen[str], prefix: str) -> int:
    assert process.stdout is not None
    for line in process.stdout:
        rendered = f"{prefix}{line.rstrip()}"
        session.append(rendered)
        if prefix == "[repair] " and rendered.startswith("[repair] run_id:"):
            session.repair_run_id = rendered.split(":", 1)[1].strip()
            session.repair_state_path = PROJECT_ROOT / ".orchestrator" / "runs" / session.repair_run_id / "state.json"
    process.wait()
    session.append(f"{prefix}Process exited with code {process.returncode}.")
    return process.returncode


def refresh_repair_state_path(session: WebSession) -> None:
    state_path = session.repair_state_path
    if not state_path or not state_path.exists():
        return
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    session.repair_run_id = state.get("job_id")


def session_worker(session: WebSession) -> None:
    try:
        session.append("[pipeline] Starting course pipeline...")
        pipeline = subprocess.Popen(
            session.pipeline_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PROJECT_ROOT),
        )
        session.pipeline_process = pipeline
        pipeline_exit = drain_process_output(session, pipeline, "[pipeline] ")
        session.pipeline_exit_code = pipeline_exit
        session.attach_artifacts(discover_session_artifacts(session.session_id))

        progress_payload = read_session_progress_payload(session.session_id)
        has_failed_videos = any(
            video["status"] in {"failed_download", "failed_dub"} for video in progress_payload.get("videos", [])
        )
        if pipeline_exit == 0 and not has_failed_videos:
            session.phase = "done"
            session.mark_finished("success")
            return

        if not session.request.auto_repair:
            session.phase = "failed"
            session.mark_finished("error")
            return

        repair = start_repair_process(session)
        repair_exit = drain_process_output(session, repair, "[repair] ")
        session.repair_exit_code = repair_exit
        refresh_repair_state_path(session)
        if repair_exit == 0:
            session.phase = "done"
            session.mark_finished("success")
        else:
            session.phase = "failed"
            session.mark_finished("error")
    except Exception as exc:  # pragma: no cover - runtime guard
        session.append(f"[system] Unexpected web session error: {exc}")
        session.phase = "failed"
        session.mark_finished("error")


@app.get("/", response_class=HTMLResponse)
async def get_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        return HTMLResponse("Static frontend not found. Create static/index.html")
    return index_file.read_text(encoding="utf-8")


@app.get("/api/progress")
async def get_progress(response: Response):
    response.headers["Deprecation"] = "true"
    response.headers["X-Session-Scope"] = "latest-session-shim"
    return read_compat_progress_payload()


@app.post("/api/dub")
async def start_dubbing(req: DubRequest):
    session_id = uuid.uuid4().hex[:10]
    session_dir = session_dir_for(session_id)
    log_path = session_log_path(session_id)
    record_path = session_record_path(session_id)
    pipeline_cmd = build_course_pipeline_command(req, session_id)
    session = WebSession(
        session_id=session_id,
        request=req,
        pipeline_cmd=pipeline_cmd,
        session_dir=session_dir,
        record_path=record_path,
        log_path=log_path,
    )
    session_registry.register(session)
    thread = threading.Thread(target=session_worker, args=(session,), daemon=True)
    thread.start()
    message = "Dubbing process launched"
    if req.auto_repair:
        message += " with automatic repair enabled"
    response = {
        "status": "started",
        "message": message,
        "session_id": session_id,
    }
    response.update(session.session_urls())
    return response


@app.get("/api/sessions")
async def list_sessions():
    return [session.to_summary() for session in session_registry.list()]


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    session = get_session_or_404(session_id)
    return session.to_detail()


@app.get("/api/sessions/{session_id}/progress")
async def get_session_progress(session_id: str):
    get_session_or_404(session_id)
    return read_session_progress_payload(session_id)


def build_stream_generator(request: Request, session_id: str) -> AsyncGenerator[dict, None]:
    async def event_generator() -> AsyncGenerator[dict, None]:
        session = session_registry.get_or_load(session_id)
        if session is None:
            yield {"event": "message", "data": f"Session not found: {session_id}"}
            return

        cursor = 0
        log_path = session.log_path
        while True:
            if await request.is_disconnected():
                break

            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8").splitlines()
                new_lines = lines[cursor:]
                for line in new_lines:
                    cursor += 1
                    yield {"event": "message", "data": line}

            session = session_registry.get_or_load(session_id) or session
            if session.finished:
                return

            await asyncio.sleep(0.25)

    return event_generator()


@app.get("/api/sessions/{session_id}/stream")
async def stream_session_logs(session_id: str, request: Request):
    get_session_or_404(session_id)
    return EventSourceResponse(build_stream_generator(request, session_id))


@app.get("/api/stream")
async def stream_logs(request: Request):
    latest = latest_session_or_empty()
    if latest is None:
        async def empty_generator() -> AsyncGenerator[dict, None]:
            yield {"event": "message", "data": "No active dubbing process running."}
        return EventSourceResponse(empty_generator(), headers={"Deprecation": "true", "X-Session-Scope": "latest-session-shim"})
    return EventSourceResponse(build_stream_generator(request, latest.session_id), headers={"Deprecation": "true", "X-Session-Scope": "latest-session-shim"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web:app", host="0.0.0.0", port=8000, reload=True)
