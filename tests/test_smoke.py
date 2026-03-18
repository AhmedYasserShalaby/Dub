import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import course_pipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_VIDEO = REPO_ROOT / "tests" / "fixtures" / "tiny_fixture.mp4"
FIXTURE_AUDIO = REPO_ROOT / "tests" / "fixtures" / "tiny_fixture.wav"


class SmokeTests(unittest.TestCase):
    def test_dub_cli_smoke_and_cached_rerun(self):
        temp_root = Path(tempfile.mkdtemp(prefix="dub-smoke-"))
        project_dir = temp_root / ".course_work" / "smoke-course" / "runs" / "smoke-run" / "lessons" / "01_smoke-lesson" / "video_01"
        shared_root = temp_root / ".course_work" / "smoke-course" / "shared"
        final_video = temp_root / "courses" / "smoke-course" / "01-smoke-lesson.arabic-dub.mp4"
        env = os.environ.copy()
        env["DUB_SMOKE_MODE"] = "1"
        cmd = [
            sys.executable,
            str(REPO_ROOT / "dub.py"),
            "--input",
            str(FIXTURE_VIDEO),
            "--audio-input",
            str(FIXTURE_AUDIO),
            "--project-dir",
            str(project_dir),
            "--shared-root",
            str(shared_root),
            "--final-video-path",
            str(final_video),
            "--course-slug",
            "smoke-course",
            "--run-id",
            "smoke-run",
            "--video-key",
            "smoke-video-01",
            "--lesson-position",
            "1",
            "--lesson-title",
            "Smoke Lesson",
            "--translation-provider",
            "google",
            "--voice",
            "ar-EG-ShakirNeural",
        ]

        first = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
        self.assertEqual(first.returncode, 0, first.stderr or first.stdout)
        self.assertTrue(final_video.exists())
        lesson_report_path = project_dir / "lesson_report.json"
        self.assertTrue(lesson_report_path.exists())
        lesson_report = json.loads(lesson_report_path.read_text(encoding="utf-8"))
        self.assertEqual(lesson_report["status"], "success")
        self.assertTrue((project_dir / "outputs" / "quality_diagnostics.json").exists())

        second = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True, check=False)
        self.assertEqual(second.returncode, 0, second.stderr or second.stdout)
        rerun_report = json.loads(lesson_report_path.read_text(encoding="utf-8"))
        self.assertEqual(rerun_report["status"], "skipped_cached")

    def test_course_pipeline_smoke_writes_run_report(self):
        temp_root = Path(tempfile.mkdtemp(prefix="course-smoke-"))
        manifest = {
            "platform": "generic",
            "course_url": "https://example.com/course",
            "course_name": "Smoke Course",
            "course_slug": "smoke-course",
            "student_course_id": "generic_01",
            "videos": [
                {
                    "lesson_id": "lesson-1",
                    "lesson_position": 1,
                    "lesson_title": "Smoke Lesson",
                    "lesson_url": "https://example.com/course/lesson-1",
                    "video_position": 1,
                    "player_url": "https://example.com/player/lesson-1",
                    "stream_url": "https://example.com/video.mp4",
                    "stream_title": "Smoke Lesson",
                    "referer": "https://example.com/course/lesson-1",
                    "download_url": "https://example.com/video.mp4",
                    "source_kind": "stream",
                    "extractor_strategy": "smoke",
                    "subtitle_url": "",
                    "subtitle_language": "",
                }
            ],
        }

        env = os.environ.copy()
        env["DUB_SMOKE_MODE"] = "1"

        def fake_prepare(video, work_root, args):
            return FIXTURE_VIDEO

        argv = [
            "course_pipeline.py",
            "--course-url",
            "https://example.com/course",
            "--project-dir",
            str(temp_root),
            "--browser",
            "chrome",
            "--disable-auto-select-voice",
            "--translation-provider",
            "google",
            "--download-workers",
            "1",
            "--dub-workers",
            "1",
            "--max-videos",
            "1",
            "--run-id",
            "smoke-run",
        ]

        with patch.dict(os.environ, env, clear=False), patch.object(sys, "argv", argv), patch(
            "course_pipeline.detect_platform",
            return_value="generic",
        ), patch(
            "course_pipeline.extract_course_manifest",
            return_value=manifest,
        ), patch(
            "course_pipeline.prepare_source_video",
            side_effect=fake_prepare,
        ), patch(
            "course_pipeline.resolve_course_voice",
            return_value="ar-EG-ShakirNeural",
        ):
            course_pipeline.main()

        run_root = temp_root / ".course_work" / "smoke-course" / "runs" / "smoke-run"
        state_path = run_root / "run_state.json"
        report_path = run_root / "run_report.json"
        final_video = temp_root / "courses" / "smoke-course" / "01-smoke-lesson.arabic-dub.mp4"
        self.assertTrue(state_path.exists())
        self.assertTrue(report_path.exists())
        self.assertTrue(final_video.exists())
        report = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(report["report_type"], "run")
        self.assertEqual(report["summary"]["total"], 1)
        self.assertEqual(report["summary"]["dubbed"], 1)
        self.assertEqual(len(report["lesson_reports"]), 1)


if __name__ == "__main__":
    unittest.main()
