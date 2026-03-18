import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from web import (
    DubRequest,
    WebSession,
    build_course_pipeline_command,
    get_session_or_404,
    list_sessions,
    latest_session_or_empty,
    read_compat_progress_payload,
    read_session_progress_payload,
    session_registry,
)


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp(prefix="web-state-test-"))
        self.project_root = self.temp_dir
        self.orchestrator_dir = self.temp_dir / ".orchestrator" / "web"
        self.session_root = self.orchestrator_dir / "sessions"
        self.session_root.mkdir(parents=True, exist_ok=True)
        session_registry.clear()

        self.project_patch = patch("web.PROJECT_ROOT", self.project_root)
        self.orchestrator_patch = patch("web.ORCHESTRATOR_WEB_DIR", self.orchestrator_dir)
        self.session_root_patch = patch("web.SESSION_ROOT", self.session_root)
        self.project_patch.start()
        self.orchestrator_patch.start()
        self.session_root_patch.start()

    def tearDown(self):
        self.project_patch.stop()
        self.orchestrator_patch.stop()
        self.session_root_patch.stop()
        session_registry.clear()

    def test_build_course_pipeline_command_includes_request_fields(self):
        req = DubRequest(
            url="https://example.com/course",
            browser="chrome",
            extractor_mode="hybrid",
            platform="auto",
            auto_repair=True,
        )

        cmd = build_course_pipeline_command(req, "session-123")

        self.assertIn("--course-url", cmd)
        self.assertIn("https://example.com/course", cmd)
        self.assertIn("--run-id", cmd)
        self.assertIn("session-123", cmd)
        self.assertIn("--browser", cmd)
        self.assertIn("chrome", cmd)

    def test_session_registry_and_compat_progress_use_session_ids(self):
        session_one = self._make_session("session-001", "https://example.com/a")
        session_two = self._make_session("session-002", "https://example.com/b")
        session_registry.register(session_one)
        session_registry.register(session_two)

        self.assertIs(session_registry.get("session-001"), session_one)
        self.assertIs(session_registry.get("session-002"), session_two)
        self.assertEqual(latest_session_or_empty().session_id, "session-002")

        run_root = self.project_root / ".course_work" / "course-b" / "runs" / "session-002"
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "course_manifest.json").write_text(
            json.dumps({
                "course_name": "Course B",
                "course_slug": "course-b",
                "course_url": "https://example.com/b",
            }),
            encoding="utf-8",
        )
        (run_root / "run_state.json").write_text(
            json.dumps({
                "course_name": "Course B",
                "course_slug": "course-b",
                "course_url": "https://example.com/b",
                "videos": {
                    "001": {
                        "lesson_title": "Intro",
                        "status": "dubbed",
                        "final_video": "/tmp/final.mp4",
                    }
                },
            }),
            encoding="utf-8",
        )
        (run_root / "run_report.json").write_text(
            json.dumps({
                "summary": {"total": 1, "done": 1, "failed": 0},
                "final_video": "/tmp/final.mp4",
                "qa": {"summary": {"glossary_misses": 0}},
                "timings": {"pipeline_seconds": 1.5},
            }),
            encoding="utf-8",
        )

        detail = get_session_or_404("session-002")
        self.assertEqual(detail.session_id, "session-002")

        payload = read_session_progress_payload("session-002")
        self.assertEqual(payload["session_id"], "session-002")
        self.assertEqual(payload["course_slug"], "course-b")
        self.assertEqual(payload["summary"]["total"], 1)
        self.assertEqual(payload["videos"][0]["lesson_title"], "Intro")
        self.assertEqual(payload["report_path"], str(run_root / "run_report.json"))

        compat_payload = read_compat_progress_payload()
        self.assertEqual(compat_payload["session_id"], "session-002")

        sessions = asyncio.run(list_sessions())
        self.assertEqual([item["session_id"] for item in sessions], ["session-002", "session-001"])

    def test_web_session_history_snapshot(self):
        session = WebSession(
            session_id="abc123",
            request=DubRequest(url="https://example.com/course"),
            pipeline_cmd=["python", "course_pipeline.py"],
            session_dir=self.session_root / "abc123",
            record_path=self.session_root / "abc123" / "session.json",
            log_path=self.session_root / "abc123" / "events.log",
        )
        session.append("one")
        session.append("two")

        lines, finished = session.snapshot(1)

        self.assertEqual(lines, ["two"])
        self.assertFalse(finished)

    def _make_session(self, session_id: str, url: str) -> WebSession:
        session_dir = self.session_root / session_id
        return WebSession(
            session_id=session_id,
            request=DubRequest(url=url),
            pipeline_cmd=["python", "course_pipeline.py", "--run-id", session_id],
            session_dir=session_dir,
            record_path=session_dir / "session.json",
            log_path=session_dir / "events.log",
        )


if __name__ == "__main__":
    unittest.main()
