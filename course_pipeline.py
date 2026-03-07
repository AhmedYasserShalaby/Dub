#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import browser_cookie3
import requests

from dub import (
    DEFAULT_BG_GAIN_DB,
    DEFAULT_DUB_GAIN,
    DEFAULT_MAX_DUB_SPEED,
    DEFAULT_MODEL,
    DEFAULT_NARRATION_RATE,
    DEFAULT_TTS_CONCURRENCY,
    DEFAULT_TIMED_RATE,
    DEFAULT_VOICE,
    PROJECT_ROOT,
    load_dotenv,
)


from universal_extractor import detect_platform, extract_course_manifest

DOTENV_PATH = PROJECT_ROOT / ".env"
COURSE_TTS_DEFAULT_STRATEGY = "timed"
ITEMS_FIELDS = """{
        id,
        parentItemId,
        title,
        type,
        position,
        availability,
        theoryVisible,
        testsVisible,
        tasksVisible,
        progress{
            id,
            beginDate,
            completeDate,
            precompleteDate,
            status,
            theoryStatus,
            testStatus,
            taskStatus,
            numberUncheckedTasks,
            numberCorrectTasks,
            numberIncorrectTasks
        }
    }"""
COURSE_FIELDS = """{
            id,
            beginDate,
            endDate,
            state,
            flowBeginDate,
            accessBeginDate,
            accessEndDate,
            isSubscription,
            course{
                id,
                name,
                dontAskFeedback
            },
            currentCourseItem{
                id
            },
            feedback,
            progress,
            student{
                id
            }
        }"""
PLAYER_URL_RE = re.compile(r"https://v\.xlsite\.ai/v/[A-Za-z0-9_-]+(?:\?[^\s\"'>]+)?")
STREAM_URL_RE = re.compile(r"url:\s*['\"]([^'\"]+master\.m3u8[^'\"]*)['\"]")
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)
LEARN_ID_RE = re.compile(r"/learn/([A-Za-z0-9_-]+)")
REQUIRED_DUB_SPEED_RE = re.compile(
    r"Dub would require ([0-9]+(?:\.[0-9]+)?)x speed, above the allowed max of ([0-9]+(?:\.[0-9]+)?)x",
    re.IGNORECASE,
)


load_dotenv(DOTENV_PATH)


@dataclass
class CourseVideo:
    lesson_id: str
    lesson_position: int
    lesson_title: str
    lesson_url: str
    video_position: int
    player_url: str
    stream_url: str
    stream_title: str
    referer: str = ""
    download_url: str = ""
    source_kind: str = "stream"
    extractor_strategy: str = ""
    subtitle_url: str = ""
    subtitle_language: str = ""

    @property
    def key(self) -> str:
        return f"{self.lesson_position:03d}-{self.lesson_id}-video-{self.video_position:02d}"

    @property
    def resolved_download_url(self) -> str:
        return self.download_url or self.stream_url

    @property
    def resolved_referer(self) -> str:
        return self.referer or self.player_url or self.lesson_url or "https://v.xlsite.ai/"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def run_capture_result(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def run_capture(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"Command failed: {' '.join(cmd)}")
    return result.stdout.strip()


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "item"


def parse_filter_list(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def matches_filter(video: CourseVideo, tokens: list[str]) -> bool:
    if not tokens:
        return False
    title = video.lesson_title.lower()
    slug = slugify(video.lesson_title)
    return any(token in title or token in slug for token in tokens)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def find_media_files(directory: Path) -> list[Path]:
    candidates = [
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".mov", ".webm"}
    ]
    return sorted(candidates)


def load_browser_cookies(browser: str, domain: str):
    loaders = {
        "chrome": browser_cookie3.chrome,
        "chromium": browser_cookie3.chromium,
        "brave": browser_cookie3.brave,
        "edge": browser_cookie3.edge,
        "firefox": browser_cookie3.firefox,
        "opera": browser_cookie3.opera,
        "safari": browser_cookie3.safari,
    }
    try:
        loader = loaders[browser.lower()]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported browser for cookies: {browser}") from exc
    return loader(domain_name=domain)


class FPPlatformAdapter:
    def __init__(self, course_url: str, browser: str):
        parsed = urlparse(course_url)
        self.course_url = course_url
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"
        self.browser = browser
        self.session = requests.Session()
        self.session.cookies = load_browser_cookies(browser, parsed.netloc)
        self.access_token = self._extract_access_token(parsed.netloc)
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
                "Referer": self.base_url + "/",
                "User-Agent": "Mozilla/5.0",
            }
        )

    def _extract_access_token(self, domain: str) -> str:
        for cookie in self.session.cookies:
            if cookie.name == "student_access_token" and domain in cookie.domain:
                return cookie.value
        raise RuntimeError(
            f"Could not find student_access_token cookie for {domain}. Make sure you are logged in."
        )

    def _api_get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.session.get(urljoin(self.base_url, path), params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if not payload.get("success"):
            raise RuntimeError(f"API error for {path}: {payload.get('errors')}")
        return payload["body"]

    def _seed_course_item_id(self) -> str:
        match = LEARN_ID_RE.search(urlparse(self.course_url).path)
        if match:
            return match.group(1)

        response = self.session.get(self.course_url, timeout=30)
        response.raise_for_status()
        match = LEARN_ID_RE.search(response.text)
        if match:
            return match.group(1)
        raise RuntimeError(
            "Could not determine a lesson id from the provided course URL. Provide a course URL that exposes /learn/<id>."
        )

    def fetch_student_course(self) -> dict[str, Any]:
        seed_item_id = self._seed_course_item_id()
        return self._api_get(f"/api/v1/student-course/{seed_item_id}", {"fields": COURSE_FIELDS})

    def fetch_course_items(self, student_course_id: str) -> list[dict[str, Any]]:
        candidates = [
            {"studentCourseId": student_course_id, "fields": ITEMS_FIELDS, "take": 1000},
            {"contactCourseId": student_course_id, "fields": ITEMS_FIELDS, "take": 1000},
            {"studentCourseId": student_course_id, "fields": ITEMS_FIELDS},
        ]
        last_error: Exception | None = None
        for params in candidates:
            try:
                body = self._api_get("/api/v1/student-course-item", params)
                items = body.get("items")
                if isinstance(items, list):
                    return items
            except Exception as exc:  # pragma: no cover - fallback handling
                last_error = exc
        raise RuntimeError("Could not fetch course items") from last_error

    def fetch_course_item_detail(self, student_course_id: str, course_item_id: str) -> dict[str, Any]:
        return self._api_get(f"/api/v1/student-course-item/{student_course_id}/{course_item_id}")

    def resolve_stream(self, player_url: str) -> tuple[str, str]:
        response = self.session.get(player_url, timeout=30)
        response.raise_for_status()
        html = response.text
        match = STREAM_URL_RE.search(html)
        if not match:
            raise RuntimeError(f"Could not resolve stream URL from player page: {player_url}")
        title_match = TITLE_RE.search(html)
        title = title_match.group(1).strip() if title_match else "video"
        return match.group(1), title

    def discover(self) -> dict[str, Any]:
        course_body = self.fetch_student_course()
        student_course = course_body["studentCourse"]
        student_course_id = student_course["id"]
        course_name = student_course["course"]["name"]
        items = sorted(
            self.fetch_course_items(student_course_id),
            key=lambda item: item.get("position") or 0,
        )

        videos: list[CourseVideo] = []
        for item in items:
            lesson_id = item["id"]
            detail = self.fetch_course_item_detail(student_course_id, lesson_id)
            detail_strings = json.dumps(detail, ensure_ascii=False)
            player_urls = list(dict.fromkeys(PLAYER_URL_RE.findall(detail_strings)))
            if not player_urls:
                continue
            lesson_title = detail.get("title") or item.get("title") or lesson_id
            lesson_url = urljoin(self.base_url, f"/learn/{lesson_id}/theory")
            for video_index, player_url in enumerate(player_urls, start=1):
                stream_url, stream_title = self.resolve_stream(player_url)
                videos.append(
                    CourseVideo(
                        lesson_id=lesson_id,
                        lesson_position=int(item.get("position") or 0),
                        lesson_title=lesson_title,
                        lesson_url=lesson_url,
                        video_position=video_index,
                        player_url=player_url,
                        stream_url=stream_url,
                        stream_title=stream_title,
                        referer=lesson_url,
                        download_url=stream_url,
                        source_kind="stream",
                        extractor_strategy="fp-platform-api",
                    )
                )

        course_slug = slugify(course_name)
        return {
            "platform": "fp-platform",
            "course_url": self.course_url,
            "course_name": course_name,
            "course_slug": course_slug,
            "student_course_id": student_course_id,
            "videos": [asdict(video) for video in videos],
        }


def course_paths(root: Path, course_slug: str) -> dict[str, Path]:
    base = root / "courses" / course_slug
    paths = {
        "base": base,
        "work": ensure_dir(root / ".course_work" / course_slug),
    }
    ensure_dir(base)
    return paths


def with_course_suffix(course_slug: str, suffix: str) -> str:
    clean_suffix = suffix.strip().strip("-_")
    if not clean_suffix:
        return course_slug
    return f"{course_slug}-{clean_suffix}"


def build_lesson_video_counts(videos: list[CourseVideo]) -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for video in videos:
        key = (video.lesson_id, slugify(video.lesson_title))
        counts[key] = counts.get(key, 0) + 1
    return counts


def lesson_video_dir(lessons_dir: Path, video: CourseVideo) -> Path:
    lesson_dir = ensure_dir(
        lessons_dir / f"{video.lesson_position:03d}_{slugify(video.lesson_title)}"
    )
    return ensure_dir(lesson_dir / f"video_{video.video_position:02d}")


def build_format_selector(max_video_height: int) -> str:
    if max_video_height > 0:
        return (
            f"bestvideo[height<={max_video_height}]+bestaudio/"
            f"best[height<={max_video_height}]"
        )
    return "bestvideo+bestaudio/best"


def build_download_attempts(args: argparse.Namespace) -> list[dict[str, int]]:
    attempts: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    configs = [
        (int(args.max_video_height), int(args.yt_concurrent_fragments)),
        (480 if int(args.max_video_height or 0) != 480 else int(args.max_video_height), 4),
        (360, 2),
    ]
    for max_height, fragments in configs:
        key = (max_height, fragments)
        if key in seen:
            continue
        seen.add(key)
        attempts.append(
            {
                "max_video_height": max_height,
                "concurrent_fragments": max(1, fragments),
            }
        )
    return attempts


def download_video(
    video: CourseVideo,
    browser: str,
    video_dir: Path,
    concurrent_fragments: int,
    max_video_height: int,
) -> Path:
    source_dir = ensure_dir(video_dir / "source")
    output_template = str(
        source_dir
        / f"{video.lesson_position:03d}-{slugify(video.lesson_title)}-v{video.video_position:02d}.%(ext)s"
    )
    cmd = [
        "yt-dlp",
        "--cookies-from-browser",
        browser,
        "--referer",
        video.resolved_referer,
        "--concurrent-fragments",
        str(concurrent_fragments),
        "--merge-output-format",
        "mp4",
        "-f",
        build_format_selector(max_video_height),
        "--write-info-json",
        "-o",
        output_template,
        video.resolved_download_url,
    ]
    try:
        run(cmd)
    except Exception as exc:
        media_files = find_media_files(source_dir)
        if media_files:
            return media_files[-1]
        raise RuntimeError(f"{exc}") from exc
    media_files = find_media_files(source_dir)
    if not media_files:
        raise RuntimeError(f"No media file downloaded for {video.resolved_download_url}")
    return media_files[-1]


def build_video_state(video: CourseVideo) -> dict[str, Any]:
    return {
        "key": video.key,
        "lesson_title": video.lesson_title,
        "lesson_url": video.lesson_url,
        "player_url": video.player_url,
        "stream_url": video.stream_url,
        "download_url": video.resolved_download_url,
        "source_kind": video.source_kind,
        "extractor_strategy": video.extractor_strategy,
        "subtitle_url": video.subtitle_url,
        "subtitle_language": video.subtitle_language,
    }


def final_video_path(
    course_output_dir: Path,
    video: CourseVideo,
    lesson_video_counts: dict[tuple[str, str], int],
) -> Path:
    lesson_slug = slugify(video.lesson_title)
    lesson_key = (video.lesson_id, lesson_slug)
    if lesson_video_counts.get(lesson_key, 1) > 1:
        return course_output_dir / f"{lesson_slug}.v{video.video_position:02d}.arabic-dub.mp4"
    return course_output_dir / f"{lesson_slug}.arabic-dub.mp4"


def migrate_legacy_multi_video_outputs(
    course_output_dir: Path,
    work_root: Path,
    lesson_video_counts: dict[tuple[str, str], int],
) -> None:
    legacy_dir = ensure_dir(work_root / "legacy_migrated" / "final_outputs")
    for (lesson_id, lesson_slug), count in lesson_video_counts.items():
        if count <= 1:
            continue
        legacy_output = course_output_dir / f"{lesson_slug}.arabic-dub.mp4"
        if not legacy_output.exists():
            continue
        target = legacy_dir / legacy_output.name
        if target.exists():
            target.unlink()
        shutil.move(str(legacy_output), str(target))
        print(f"[migrated_legacy_output] {lesson_id} {legacy_output.name}")


def build_failure_state(video: CourseVideo, status: str, error: Exception | str) -> dict[str, Any]:
    state = build_video_state(video)
    state["status"] = status
    state["error"] = str(error)
    return state


def persist_video_state(
    state: dict[str, Any],
    state_path: Path,
    result: dict[str, Any],
    *,
    emit: bool = True,
) -> None:
    state["videos"][result["key"]] = result
    write_json(state_path, state)
    if emit:
        print(f"[{result['status']}] {result['key']} {result['lesson_title']}")


def next_download_candidate(
    pending_downloads: list[CourseVideo],
    active_downloads: dict[Future[Path], CourseVideo],
    frontier_lessons: int,
) -> CourseVideo | None:
    if not pending_downloads:
        return None

    frontier_positions: list[int] = []
    unfinished = sorted(
        list(active_downloads.values()) + pending_downloads,
        key=lambda video: (video.lesson_position, video.video_position, video.key),
    )
    for video in unfinished:
        if video.lesson_position in frontier_positions:
            continue
        frontier_positions.append(video.lesson_position)
        if len(frontier_positions) >= max(1, frontier_lessons):
            break

    allowed_positions = set(frontier_positions)
    for index, video in enumerate(pending_downloads):
        if video.lesson_position in allowed_positions:
            return pending_downloads.pop(index)
    return pending_downloads.pop(0)


def prepare_source_video(
    video: CourseVideo,
    work_root: Path,
    args: argparse.Namespace,
) -> Path:
    video_dir = lesson_video_dir(work_root, video)
    inputs_dir = ensure_dir(video_dir / "inputs")
    source_dir = ensure_dir(video_dir / "source")

    if args.force_redownload and inputs_dir.exists():
        for item in inputs_dir.iterdir():
            if item.is_file():
                item.unlink()
        for item in source_dir.iterdir():
            if item.is_file():
                item.unlink()

    source_files = find_media_files(inputs_dir)
    if source_files and not args.force_redownload:
        return source_files[-1]

    downloaded_source_files = find_media_files(source_dir)
    if downloaded_source_files and not args.force_redownload:
        source_video = downloaded_source_files[-1]
        target = inputs_dir / source_video.name
        if source_video != target:
            shutil.copy2(source_video, target)
            source_video = target
        return source_video

    attempt_configs = build_download_attempts(args)
    max_attempts = max(args.download_retries, len(attempt_configs))
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        config = attempt_configs[min(attempt - 1, len(attempt_configs) - 1)]
        try:
            if attempt > 1:
                print(
                    f"[download_retry] {video.lesson_title} "
                    f"height={config['max_video_height']} "
                    f"fragments={config['concurrent_fragments']}"
                )
            source_video = download_video(
                video,
                args.browser,
                video_dir,
                config["concurrent_fragments"],
                config["max_video_height"],
            )
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                raise
            print(
                f"Retrying download for {video.lesson_title} "
                f"({attempt}/{max_attempts - 1} retries used)..."
            )
            time.sleep(args.retry_wait_seconds)

    if last_error is not None:
        raise last_error

    if source_video.parent != inputs_dir:
        target = inputs_dir / source_video.name
        shutil.copy2(source_video, target)
        source_video = target
    return source_video


def convert_webvtt_to_srt(vtt_text: str) -> str:
    lines = [line.rstrip("\n") for line in vtt_text.replace("\r\n", "\n").split("\n")]
    cues: list[tuple[str, list[str]]] = []
    current_time: str | None = None
    current_text: list[str] = []

    for raw_line in lines:
        line = raw_line.strip("\ufeff")
        if not line or line == "WEBVTT" or line.startswith(("NOTE", "STYLE", "REGION")):
            if current_time and current_text:
                cues.append((current_time, current_text))
                current_time = None
                current_text = []
            continue
        if "-->" in line:
            if current_time and current_text:
                cues.append((current_time, current_text))
                current_text = []
            current_time = line.replace(".", ",")
            continue
        if current_time:
            current_text.append(line)

    if current_time and current_text:
        cues.append((current_time, current_text))

    output_lines: list[str] = []
    for index, (timing, text_lines) in enumerate(cues, start=1):
        output_lines.append(str(index))
        output_lines.append(timing)
        output_lines.extend(text_lines)
        output_lines.append("")
    return "\n".join(output_lines).strip() + "\n"


def ensure_english_subtitles(video: CourseVideo, video_dir: Path) -> None:
    if not video.subtitle_url:
        return
    outputs_dir = ensure_dir(video_dir / "outputs")
    english_srt = outputs_dir / "english.srt"
    if english_srt.exists():
        return
    response = requests.get(video.subtitle_url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    english_srt.write_text(convert_webvtt_to_srt(response.text), encoding="utf-8")


def build_dub_attempts(args: argparse.Namespace) -> list[dict[str, float]]:
    attempts: list[dict[str, float]] = [
        {
            "video_slowdown": float(args.video_slowdown),
            "timed_collision_max_speed": float(args.timed_collision_max_speed),
        }
    ]
    max_auto_slowdown = max(float(args.video_slowdown), 2.0)
    slowdown = float(args.video_slowdown)
    while slowdown + 0.099 < max_auto_slowdown:
        slowdown = round(slowdown + 0.10, 2)
        attempts.append(
            {
                "video_slowdown": slowdown,
                "timed_collision_max_speed": float(args.timed_collision_max_speed),
            }
        )
    return attempts


def dub_video(
    video: CourseVideo,
    source_video: Path,
    course_output_dir: Path,
    work_root: Path,
    args: argparse.Namespace,
    lesson_video_counts: dict[tuple[str, str], int],
) -> dict[str, Any]:
    video_dir = lesson_video_dir(work_root, video)
    outputs_dir = ensure_dir(video_dir / "outputs")
    state = build_video_state(video)
    final_video = final_video_path(course_output_dir, video, lesson_video_counts)

    ensure_english_subtitles(video, video_dir)

    if final_video.exists() and not args.force_redub:
        state["status"] = "skipped_existing_dub"
        state["final_video"] = str(final_video)
        return state

    attempt_configs = build_dub_attempts(args)
    max_attempts = max(args.dub_retries, len(attempt_configs))
    last_error: Exception | None = None
    current_collision_speed = float(args.timed_collision_max_speed)
    for attempt in range(1, max_attempts + 1):
        config = attempt_configs[min(attempt - 1, len(attempt_configs) - 1)].copy()
        config["timed_collision_max_speed"] = current_collision_speed
        dub_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "dub.py"),
            "--input",
            str(source_video),
            "--project-dir",
            str(video_dir),
            "--voice",
            args.voice,
            "--model",
            args.model,
            "--deepl-api-key",
            args.deepl_api_key,
            "--tts-strategy",
            args.tts_strategy,
            "--narration-rate",
            args.narration_rate,
            "--timed-rate",
            args.timed_rate,
            "--video-slowdown",
            str(config["video_slowdown"]),
            "--max-dub-speed",
            str(args.max_dub_speed),
            "--dub-gain",
            str(args.dub_gain),
            "--bg-gain-db",
            str(args.bg_gain_db),
            "--tts-concurrency",
            str(args.tts_concurrency),
            "--chunk-max-chars",
            str(args.chunk_max_chars),
            "--timed-max-late-drift-ms",
            str(args.timed_max_late_drift_ms),
            "--timed-collision-max-speed",
            str(config["timed_collision_max_speed"]),
        ]
        if args.use_reviewed_arabic:
            dub_cmd.append("--use-reviewed-arabic")
        if args.deepl_api_url:
            dub_cmd.extend(["--deepl-api-url", args.deepl_api_url])
        try:
            if attempt > 1:
                print(
                    f"[dub_retry] {video.lesson_title} "
                    f"slowdown={config['video_slowdown']:.2f} "
                    f"collision_max={config['timed_collision_max_speed']:.2f}"
                )
            result = run_capture_result(dub_cmd, cwd=PROJECT_ROOT)
            if result.stdout:
                print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
            if result.returncode != 0:
                error_output = "\n".join(
                    part for part in [result.stderr.strip(), result.stdout.strip()] if part
                ).strip()
                raise RuntimeError(error_output or f"Command failed: {' '.join(dub_cmd)}")
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            error_text = str(exc)
            speed_match = REQUIRED_DUB_SPEED_RE.search(error_text)
            if speed_match:
                required_speed = float(speed_match.group(1))
                if required_speed <= args.max_dub_speed:
                    current_collision_speed = max(
                        current_collision_speed,
                        min(args.max_dub_speed, round(required_speed + 0.05, 2)),
                    )
            if attempt == max_attempts:
                raise
            print(
                f"Retrying dub for {video.lesson_title} "
                f"({attempt}/{max_attempts - 1} retries used)..."
            )
            time.sleep(args.retry_wait_seconds)

    if last_error is not None:
        raise last_error

    produced = outputs_dir / f"{source_video.stem}.arabic-dub.mp4"
    if produced.exists():
        shutil.copy2(produced, final_video)
    state["status"] = "dubbed"
    state["source_video"] = str(source_video)
    state["final_video"] = str(final_video if final_video.exists() else produced)

    if args.clean_work and state["status"] == "dubbed":
        shutil.rmtree(video_dir, ignore_errors=True)
    return state


def process_video(
    video: CourseVideo,
    course_output_dir: Path,
    work_root: Path,
    args: argparse.Namespace,
    lesson_video_counts: dict[tuple[str, str], int],
) -> dict[str, Any]:
    source_video = prepare_source_video(video, work_root, args)
    return dub_video(video, source_video, course_output_dir, work_root, args, lesson_video_counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and Arabic-dub an entire course")
    parser.add_argument("--course-url", required=True, help="Course URL on the supported platform")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=PROJECT_ROOT,
        help="Root Dub workspace directory",
    )
    parser.add_argument(
        "--browser",
        default=os.environ.get("DEFAULT_BROWSER", "chrome"),
        help="Browser to reuse for authentication cookies",
    )
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Arabic neural voice")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="MLX Whisper model")
    parser.add_argument(
        "--deepl-api-key",
        default=os.environ.get("DEEPL_API_KEY"),
        help="DeepL API key",
    )
    parser.add_argument(
        "--deepl-api-url",
        default=os.environ.get("DEEPL_API_URL"),
        help="DeepL API URL (auto-detected from API key when omitted)",
    )
    parser.add_argument(
        "--tts-strategy",
        choices=("timed",),
        default=COURSE_TTS_DEFAULT_STRATEGY,
        help="Timed only (non-sped final output)",
    )
    parser.add_argument(
        "--narration-rate",
        default=DEFAULT_NARRATION_RATE,
        help="Base narration rate for chunked TTS",
    )
    parser.add_argument(
        "--timed-rate",
        default=DEFAULT_TIMED_RATE,
        help="Fixed speaking rate for timed TTS, e.g. +0%% (no speed-up)",
    )
    parser.add_argument(
        "--video-slowdown",
        type=float,
        default=1.0,
        help="Stretch the final video timeline by this factor to give the dub more breathing room",
    )
    parser.add_argument(
        "--max-dub-speed",
        type=float,
        default=DEFAULT_MAX_DUB_SPEED,
        help="Maximum allowed dub speedup factor before the lesson fails",
    )
    parser.add_argument(
        "--dub-gain",
        type=float,
        default=DEFAULT_DUB_GAIN,
        help="Arabic dub gain multiplier during final mix",
    )
    parser.add_argument(
        "--bg-gain-db",
        type=int,
        default=DEFAULT_BG_GAIN_DB,
        help="Original background audio gain in dB during final mix",
    )
    parser.add_argument(
        "--tts-concurrency",
        type=int,
        default=DEFAULT_TTS_CONCURRENCY,
        help="Concurrent chunk synthesis requests",
    )
    parser.add_argument(
        "--chunk-max-chars",
        type=int,
        default=2600,
        help="Maximum characters per chunked TTS request",
    )
    parser.add_argument(
        "--timed-max-late-drift-ms",
        type=int,
        default=2200,
        help="Maximum local timing drift allowed before timed segments are forced tighter",
    )
    parser.add_argument(
        "--timed-collision-max-speed",
        type=float,
        default=1.38,
        help="Maximum selective speed-up allowed for collision-only timed fitting",
    )
    parser.add_argument(
        "--use-reviewed-arabic",
        action="store_true",
        help="Use outputs/arabic_reviewed.srt when present for each lesson video",
    )
    parser.add_argument(
        "--course-slug-suffix",
        default="",
        help="Optional suffix for the course output/work folder, e.g. natural-sync",
    )
    parser.add_argument(
        "--yt-concurrent-fragments",
        type=int,
        default=12,
        help="yt-dlp concurrent fragment count",
    )
    parser.add_argument(
        "--max-video-height",
        type=int,
        default=0,
        help="Cap future downloads to this video height (0 keeps the best available)",
    )
    parser.add_argument(
        "--only-lessons",
        default="",
        help="Comma-separated substrings to include (matches lesson title or slug)",
    )
    parser.add_argument(
        "--skip-lessons",
        default="",
        help="Comma-separated substrings to skip (matches lesson title or slug)",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Process only the first N discovered videos (0 means all)",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        default=6,
        help="Number of lesson downloads to run in parallel",
    )
    parser.add_argument(
        "--dub-workers",
        type=int,
        default=4,
        help="Number of lesson dub jobs to run in parallel",
    )
    parser.add_argument(
        "--download-frontier-lessons",
        type=int,
        default=2,
        help="Only prioritize downloads from the first N unfinished lesson positions",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Re-download source videos even if they already exist",
    )
    parser.add_argument(
        "--force-redub",
        action="store_true",
        help="Re-run dubbing even if a dubbed file already exists",
    )
    parser.add_argument(
        "--clean-work",
        action="store_true",
        help="Delete per-video working folders after successful dub (default: on)",
    )
    parser.add_argument(
        "--dub-retries",
        type=int,
        default=3,
        help="Number of dub attempts per lesson before marking it failed",
    )
    parser.add_argument(
        "--download-retries",
        type=int,
        default=3,
        help="Number of download attempts per lesson before marking it failed",
    )
    parser.add_argument(
        "--retry-wait-seconds",
        type=int,
        default=5,
        help="Seconds to wait between retry attempts",
    )
    parser.add_argument(
        "--keep-work",
        action="store_false",
        dest="clean_work",
        help="Keep per-video working folders under .course_work",
    )
    parser.add_argument(
        "--extractor-mode",
        choices=("hybrid", "auto", "interactive"),
        default="hybrid",
        help="Non-FP extraction mode: automatic first, interactive only, or hybrid fallback",
    )
    parser.add_argument(
        "--platform",
        choices=("auto", "fp-platform", "getcourse", "generic"),
        default="auto",
        help="Force a platform adapter or auto-detect from the URL",
    )
    parser.add_argument(
        "--extract-timeout-seconds",
        type=int,
        default=45,
        help="Timeout for direct extraction/network resolution attempts",
    )
    parser.add_argument(
        "--manual-play-timeout-seconds",
        type=int,
        default=180,
        help="How long the interactive browser fallback waits for media activity",
    )
    parser.add_argument(
        "--save-extraction-debug",
        action="store_true",
        help="Always keep extraction diagnostics under .course_work/<course-slug>/extraction",
    )
    parser.set_defaults(clean_work=True)
    args = parser.parse_args()
    args.only_lessons = parse_filter_list(args.only_lessons)
    args.skip_lessons = parse_filter_list(args.skip_lessons)

    if not args.deepl_api_key:
        raise SystemExit("Missing DeepL API key. Set DEEPL_API_KEY in .env or pass --deepl-api-key.")

    detected_platform = detect_platform(args.course_url) if args.platform == "auto" else args.platform

    # Determine extraction strategy based on URL domain
    if detected_platform == "fp-platform":
        adapter = FPPlatformAdapter(args.course_url, args.browser)
        manifest = adapter.discover()
    else:
        preview_slug = slugify(urlparse(args.course_url).netloc or "course")
        preview_work_dir = args.project_dir.resolve() / ".course_work" / preview_slug
        debug_dir = preview_work_dir / "extraction"
        print(
            f"Non FP-Platform URL detected ({detected_platform}). "
            f"Starting {args.extractor_mode} universal extraction for {args.course_url}"
        )
        manifest = extract_course_manifest(
            args.course_url,
            browser=args.browser,
            mode=args.extractor_mode,
            platform=detected_platform,
            extract_timeout_seconds=args.extract_timeout_seconds,
            manual_play_timeout_seconds=args.manual_play_timeout_seconds,
            debug_dir=debug_dir if args.save_extraction_debug or detected_platform != "fp-platform" else None,
        )

    effective_course_slug = with_course_suffix(manifest["course_slug"], args.course_slug_suffix)
    paths = course_paths(args.project_dir.resolve(), effective_course_slug)
    if detected_platform != "fp-platform" and debug_dir.exists() and paths["work"] != preview_work_dir:
        ensure_dir(paths["work"])
        target_debug_dir = paths["work"] / "extraction"
        if target_debug_dir.exists():
            shutil.rmtree(target_debug_dir, ignore_errors=True)
        shutil.move(str(debug_dir), str(target_debug_dir))
        if preview_work_dir.exists() and not any(preview_work_dir.iterdir()):
            preview_work_dir.rmdir()
        print(f"[extractor] saved diagnostics to {target_debug_dir}")
    manifest_path = paths["work"] / "course_manifest.json"
    state_path = paths["work"] / "run_state.json"
    manifest["course_slug"] = effective_course_slug
    write_json(manifest_path, manifest)

    state = read_json(state_path, {"videos": {}})
    videos = [CourseVideo(**video_data) for video_data in manifest["videos"]]
    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    lesson_video_counts = build_lesson_video_counts(videos)
    state["course_name"] = manifest["course_name"]
    state["course_slug"] = manifest["course_slug"]
    state["course_url"] = manifest["course_url"]
    state["selected_video_count"] = len(videos)
    migrate_legacy_multi_video_outputs(paths["base"], paths["work"], lesson_video_counts)
    pending_downloads: list[CourseVideo] = []
    for video in videos:
        if args.only_lessons and not matches_filter(video, args.only_lessons):
            result = {
                "key": video.key,
                "lesson_title": video.lesson_title,
                "status": "skipped_filtered",
                "reason": "only_lessons",
            }
            persist_video_state(state, state_path, result)
            continue
        if args.skip_lessons and matches_filter(video, args.skip_lessons):
            result = {
                "key": video.key,
                "lesson_title": video.lesson_title,
                "status": "skipped_by_user",
                "reason": "skip_lessons",
            }
            persist_video_state(state, state_path, result)
            continue
        final_video = final_video_path(paths["base"], video, lesson_video_counts)
        if final_video.exists() and not args.force_redub:
            result = build_video_state(video)
            result["status"] = "skipped_existing_dub"
            result["final_video"] = str(final_video)
            persist_video_state(state, state_path, result)
            continue
        result = build_video_state(video)
        result["status"] = "queued_download"
        persist_video_state(state, state_path, result, emit=False)
        pending_downloads.append(video)

    pending_dubs: deque[tuple[CourseVideo, Path]] = deque()
    active_downloads: dict[Future[Path], CourseVideo] = {}
    active_dubs: dict[Future[dict[str, Any]], CourseVideo] = {}

    with (
        ThreadPoolExecutor(max_workers=max(1, args.download_workers)) as download_executor,
        ThreadPoolExecutor(max_workers=max(1, args.dub_workers)) as dub_executor,
    ):
        while pending_downloads or pending_dubs or active_downloads or active_dubs:
            while pending_downloads and len(active_downloads) < max(1, args.download_workers):
                video = next_download_candidate(
                    pending_downloads,
                    active_downloads,
                    args.download_frontier_lessons,
                )
                if video is None:
                    break
                result = build_video_state(video)
                result["status"] = "downloading"
                persist_video_state(state, state_path, result)
                future = download_executor.submit(prepare_source_video, video, paths["work"], args)
                active_downloads[future] = video

            while pending_dubs and len(active_dubs) < max(1, args.dub_workers):
                video, source_video = pending_dubs.popleft()
                result = build_video_state(video)
                result["status"] = "dubbing"
                result["source_video"] = str(source_video)
                persist_video_state(state, state_path, result)
                future = dub_executor.submit(
                    dub_video,
                    video,
                    source_video,
                    paths["base"],
                    paths["work"],
                    args,
                    lesson_video_counts,
                )
                active_dubs[future] = video

            active_futures = list(active_downloads.keys()) + list(active_dubs.keys())
            if not active_futures:
                break

            done_futures, _ = wait(active_futures, return_when=FIRST_COMPLETED)
            for future in done_futures:
                if future in active_downloads:
                    video = active_downloads.pop(future)
                    try:
                        source_video = future.result()
                    except Exception as exc:
                        result = build_failure_state(video, "failed_download", exc)
                    else:
                        result = build_video_state(video)
                        result["status"] = "queued_dub"
                        result["source_video"] = str(source_video)
                        pending_dubs.append((video, source_video))
                    persist_video_state(state, state_path, result)
                    continue

                video = active_dubs.pop(future)
                try:
                    result = future.result()
                except Exception as exc:  # pragma: no cover - executor guard
                    result = build_failure_state(video, "failed_dub", exc)
                persist_video_state(state, state_path, result)

    print(f"Manifest: {manifest_path}")
    print(f"State: {state_path}")


if __name__ == "__main__":
    main()
