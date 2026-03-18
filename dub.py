#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import edge_tts
import pysrt
import requests
from deep_translator import GoogleTranslator
from pipeline_support import (
    CacheStore,
    DENSE_REWRITE_VERSION,
    SEGMENTATION_CLEANUP_VERSION,
    SPOKEN_REWRITE_VERSION,
    STAGE_MIX,
    STAGE_SYNTHESIS,
    STAGE_TRANSLATION,
    STAGE_TRANSCRIPTION,
    hash_file,
    read_json,
    stable_hash_json,
    stable_hash_text,
    stages_requiring_refresh,
    utc_timestamp,
    write_json,
)

try:
    from mlx_whisper import transcribe as mlx_transcribe
except ImportError:  # pragma: no cover - handled at runtime
    mlx_transcribe = None


DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_VOICE = os.environ.get("DEFAULT_ARABIC_VOICE", "ar-EG-ShakirNeural")
DEFAULT_BG_GAIN_DB = -30
DEFAULT_DUB_GAIN = 4.2
DEFAULT_TTS_STRATEGY = "timed"
DEFAULT_NARRATION_RATE = "-6%"
DEFAULT_TTS_CONCURRENCY = 4
DEFAULT_CHUNK_MAX_CHARS = 2600
DEFAULT_TIMED_RATE = "+0%"
DEFAULT_TRANSLATION_PROVIDER = "auto"
DEFAULT_QUALITY_PROFILE = "natural"
DEFAULT_VIDEO_SLOWDOWN = 1.10
DEFAULT_TIMED_MAX_LATE_DRIFT_MS = 1700
DEFAULT_TIMED_COLLISION_MAX_SPEED = 1.22
DEFAULT_SPOKEN_STYLE = "egyptian"
TIMED_MIX_BATCH_SIZE = 48
TIMED_IDEAL_GAP_MS = 70
TIMED_MIN_GAP_MS = 20
TIMED_FIT_TAIL_PADDING_MS = 10
TIMED_FIT_FADE_OUT_MS = 70
TIMED_MIN_FIT_MS = 120
TIMED_FIT_TRIGGER_MS = 220
TIMED_MAX_LATE_DRIFT_MS = DEFAULT_TIMED_MAX_LATE_DRIFT_MS
DEFAULT_MAX_DUB_SPEED = 1.5
SAMPLE_RATE = 24000
DEEPL_DEFAULT_URL = "https://api-free.deepl.com/v2/translate"
DEEPL_PRO_URL = "https://api.deepl.com/v2/translate"
DEEPL_CHUNK_SIZE = 20
DEFAULT_VOICE_CANDIDATES = [
    "ar-EG-ShakirNeural",
    "ar-EG-SalmaNeural",
    "ar-SA-HamedNeural",
    "ar-SA-ZariyahNeural",
]
DEFAULT_VOICE_SELECTION_TEXT = (
    "اهلا بيكم. النهارده هنشرح الفكرة بشكل بسيط وواضح خطوة بخطوة عشان التطبيق يبقى سهل."
)
ENGLISH_TOKEN_RE = re.compile(r"[A-Za-z]{3,}")
PROJECT_ROOT = Path(__file__).resolve().parent
DOTENV_PATH = PROJECT_ROOT / ".env"
SPOKEN_ARABIC_PHRASE_REPLACEMENTS = (
    ("في هذا الدرس", "في الدرس ده"),
    ("في هذه الحالة", "في الحالة دي"),
    ("على سبيل المثال", "مثلا"),
    ("بذلك", "بكده"),
    ("بعد ذلك", "بعد كده"),
    ("لذلك", "علشان كده"),
    ("في النهاية", "في الاخر"),
    ("من أجل", "عشان"),
    ("أيضا", "كمان"),
    ("أيضًا", "كمان"),
    ("الآن", "دلوقتي"),
    ("يمكنك", "تقدر"),
    ("يمكننا", "نقدر"),
    ("يجب", "لازم"),
    ("الذي", "اللي"),
    ("التي", "اللي"),
    ("الذين", "اللي"),
    ("هذا", "ده"),
    ("هذه", "دي"),
    ("ذلك", "ده"),
)
SPOKEN_ARABIC_REGEX_REPLACEMENTS = (
    (re.compile(r"\bسوف نقوم\b"), "هنعمل"),
    (re.compile(r"\bسنقوم\b"), "هنعمل"),
    (re.compile(r"\bسوف نبدأ\b"), "هنبدا"),
    (re.compile(r"\bلنبدأ\b"), "يلا نبدا"),
    (re.compile(r"\bسوف يكون\b"), "هيكون"),
    (re.compile(r"\bسوف يتم\b"), "هيتم"),
)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value and key not in os.environ:
                os.environ[key] = value


load_dotenv(DOTENV_PATH)


@dataclass
class Segment:
    index: int
    start_ms: int
    end_ms: int
    text: str

    @property
    def duration_ms(self) -> int:
        return max(1, self.end_ms - self.start_ms)


@dataclass
class QualityProfile:
    density_score: float
    density_label: str
    video_slowdown: float
    timed_max_late_drift_ms: int
    timed_collision_max_speed: float


@dataclass
class VoiceSelectionResult:
    selected_voice: str
    sample_text: str
    sample_dir: str
    ranking: list[dict[str, float | str]]


@dataclass
class WorkspaceLayout:
    project_dir: Path
    final_video: Path


@dataclass
class TranslationRunStats:
    provider: str
    cache_hits: int = 0
    cache_misses: int = 0
    glossary_hits: int = 0
    glossary_misses: int = 0

    @property
    def total(self) -> int:
        return self.cache_hits + self.cache_misses

    @property
    def hit_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return round(self.cache_hits / self.total, 3)


def run(cmd: list[str], cwd: Path | None = None) -> None:
    result = subprocess.run(cmd, cwd=cwd, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


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


def ensure_tool(name: str) -> None:
    if shutil.which(name):
        return
    raise RuntimeError(f"Missing required tool: {name}")


def ms_from_seconds(value: float) -> int:
    return int(round(value * 1000))


def scale_segments_timing(segments: list[Segment], factor: float) -> list[Segment]:
    if math.isclose(factor, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        return segments
    scaled: list[Segment] = []
    for segment in segments:
        scaled.append(
            Segment(
                index=segment.index,
                start_ms=ms_from_seconds(segment.start_ms / 1000 * factor),
                end_ms=ms_from_seconds(segment.end_ms / 1000 * factor),
                text=segment.text,
            )
        )
    return scaled


def to_srt_time(ms: int) -> pysrt.SubRipTime:
    return pysrt.SubRipTime(milliseconds=ms)


def load_glossary(path: Path) -> dict[str, str]:
    glossary: dict[str, str] = {}
    if not path.exists():
        return glossary
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            src = (row.get("source") or "").strip()
            dst = (row.get("target") or "").strip()
            if src and dst:
                glossary[src] = dst
    return glossary


def apply_glossary(text: str, glossary: dict[str, str]) -> str:
    output = text
    for source, target in glossary.items():
        output = output.replace(source, target)
    return output


def estimate_segment_density(segments: list[Segment]) -> float:
    if not segments:
        return 0.0
    weighted_scores: list[float] = []
    weights: list[float] = []
    for segment in segments:
        text = " ".join(segment.text.split())
        if not text:
            continue
        duration_seconds = max(segment.duration_ms / 1000, 0.15)
        chars_per_second = len(text) / duration_seconds
        words_per_second = len(text.split()) / duration_seconds
        score = chars_per_second + (words_per_second * 2.5)
        weight = max(1.0, duration_seconds)
        weighted_scores.append(score * weight)
        weights.append(weight)
    if not weights:
        return 0.0
    return sum(weighted_scores) / sum(weights)


def quality_profile_for_segments(
    segments: list[Segment],
    requested_video_slowdown: float,
    requested_timed_max_late_drift_ms: int,
    requested_timed_collision_max_speed: float,
    quality_profile: str,
) -> QualityProfile:
    density_score = estimate_segment_density(segments)
    density_label = "light"
    recommended_slowdown = requested_video_slowdown
    recommended_drift = requested_timed_max_late_drift_ms
    recommended_collision = requested_timed_collision_max_speed

    if quality_profile == "natural":
        if density_score >= 19.0:
            density_label = "extreme"
            recommended_slowdown = max(requested_video_slowdown, 1.18)
            recommended_drift = max(requested_timed_max_late_drift_ms, 1900)
            recommended_collision = max(requested_timed_collision_max_speed, 1.26)
        elif density_score >= 15.5:
            density_label = "dense"
            recommended_slowdown = max(requested_video_slowdown, 1.15)
            recommended_drift = max(requested_timed_max_late_drift_ms, 1800)
            recommended_collision = max(requested_timed_collision_max_speed, 1.24)
        elif density_score >= 12.0:
            density_label = "medium"
            recommended_slowdown = max(requested_video_slowdown, 1.12)
            recommended_drift = max(requested_timed_max_late_drift_ms, 1800)
            recommended_collision = max(requested_timed_collision_max_speed, 1.24)

    return QualityProfile(
        density_score=round(density_score, 2),
        density_label=density_label,
        video_slowdown=recommended_slowdown,
        timed_max_late_drift_ms=recommended_drift,
        timed_collision_max_speed=recommended_collision,
    )


def rewrite_spoken_arabic_text(text: str, style: str = DEFAULT_SPOKEN_STYLE) -> str:
    output = " ".join(text.split())
    if not output or style != "egyptian":
        return output
    for source, target in SPOKEN_ARABIC_PHRASE_REPLACEMENTS:
        output = output.replace(source, target)
    for pattern, replacement in SPOKEN_ARABIC_REGEX_REPLACEMENTS:
        output = pattern.sub(replacement, output)
    output = re.sub(r"\s+([،,.!?])", r"\1", output)
    return output.strip()


def rewrite_spoken_arabic_segments(
    segments: list[Segment],
    style: str = DEFAULT_SPOKEN_STYLE,
) -> list[Segment]:
    rewritten: list[Segment] = []
    for segment in segments:
        rewritten.append(
            Segment(
                index=segment.index,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                text=rewrite_spoken_arabic_text(segment.text, style),
            )
        )
    return rewritten


def choose_translation_provider(
    requested_provider: str,
    deepl_api_key: str | None,
) -> str:
    if requested_provider == "google":
        return "google"
    if requested_provider == "deepl":
        if not deepl_api_key:
            raise RuntimeError(
                "Translation provider set to DeepL, but no DeepL API key was provided."
            )
        return "deepl"
    if deepl_api_key:
        return "deepl"
    return "google"


def extract_audio(video_path: Path, wav_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]
    try:
        run(cmd)
    except Exception as exc:
        error_text = str(exc).lower()
        if "does not contain any stream" not in error_text and "stream map" not in error_text:
            raise
        duration_seconds = max(0.1, probe_duration_ms(video_path) / 1000)
        print("Source video has no audio track. Creating silent background audio...")
        run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"anullsrc=r={SAMPLE_RATE}:cl=mono",
                "-t",
                f"{duration_seconds:.3f}",
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                str(wav_path),
            ]
        )


def find_sidecar_audio(video_path: Path) -> Path | None:
    parent = video_path.parent
    stem = video_path.stem.split(".f")[0]
    candidates = sorted(parent.glob(f"{stem}.faudio*"))
    return candidates[0] if candidates else None


def transcribe_audio(audio_path: Path, model: str, language: str) -> list[Segment]:
    if smoke_mode_enabled():
        return smoke_mode_english_segments()
    if mlx_transcribe is None:
        raise RuntimeError(
            "mlx-whisper is not installed. Run setup first: ./setup.sh"
        )

    result = mlx_transcribe(
        str(audio_path),
        path_or_hf_repo=model,
        language=language,
    )

    segments: list[Segment] = []
    for index, item in enumerate(result["segments"], start=1):
        text = " ".join(str(item["text"]).split())
        if not text:
            continue
        segments.append(
            Segment(
                index=index,
                start_ms=ms_from_seconds(float(item["start"])),
                end_ms=ms_from_seconds(float(item["end"])),
                text=text,
            )
        )
    return segments


def write_srt(path: Path, segments: list[Segment]) -> None:
    subs = pysrt.SubRipFile()
    for segment in segments:
        subs.append(
            pysrt.SubRipItem(
                index=segment.index,
                start=to_srt_time(segment.start_ms),
                end=to_srt_time(segment.end_ms),
                text=segment.text,
            )
        )
    subs.save(str(path), encoding="utf-8")


def read_srt(path: Path) -> list[Segment]:
    subs = pysrt.open(str(path), encoding="utf-8")
    return [
        Segment(
            index=item.index,
            start_ms=item.start.ordinal,
            end_ms=item.end.ordinal,
            text=item.text.strip(),
        )
        for item in subs
        if item.text.strip()
    ]


def write_segments_json(path: Path, segments: list[Segment]) -> None:
    path.write_text(
        json.dumps([asdict(segment) for segment in segments], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_segments_json(path: Path) -> list[Segment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        Segment(
            index=int(item["index"]),
            start_ms=int(item["start_ms"]),
            end_ms=int(item["end_ms"]),
            text=str(item["text"]).strip(),
        )
        for item in payload
        if str(item.get("text", "")).strip()
    ]


def segments_signature(segments: list[Segment]) -> str:
    payload = [
        {
            "index": segment.index,
            "start_ms": segment.start_ms,
            "end_ms": segment.end_ms,
            "text": segment.text,
        }
        for segment in segments
    ]
    return stable_hash_json(payload)


def infer_shared_root(project_dir: Path) -> Path:
    parts = list(project_dir.parts)
    if ".course_work" in parts:
        course_index = parts.index(".course_work") + 1
        if course_index < len(parts):
            course_root = Path(*parts[: course_index + 1])
            return course_root / "shared"
    return project_dir / "shared"


def build_video_key(lesson_position: int, lesson_title: str, video_position: int) -> str:
    return f"{lesson_position:03d}-{slugify(lesson_title)}-video-{video_position:02d}"


def build_processing_manifest(
    *,
    course_slug: str,
    video_key: str,
    source_hash: str,
    audio_hash: str,
    transcription_config: dict[str, Any],
    translation_config: dict[str, Any],
    synthesis_config: dict[str, Any],
    mix_config: dict[str, Any],
) -> dict[str, Any]:
    transcription_hash = stable_hash_json(
        {
            "source_hash": source_hash,
            "audio_hash": audio_hash,
            "config": transcription_config,
        }
    )
    translation_hash = stable_hash_json(
        {
            "upstream": transcription_hash,
            "config": translation_config,
        }
    )
    synthesis_hash = stable_hash_json(
        {
            "upstream": translation_hash,
            "config": synthesis_config,
        }
    )
    mix_hash = stable_hash_json(
        {
            "upstream": synthesis_hash,
            "config": mix_config,
        }
    )
    return {
        "course_slug": course_slug,
        "video_key": video_key,
        "source_hash": source_hash,
        "audio_hash": audio_hash,
        "stages": {
            STAGE_TRANSCRIPTION: transcription_hash,
            STAGE_TRANSLATION: translation_hash,
            STAGE_SYNTHESIS: synthesis_hash,
            STAGE_MIX: mix_hash,
        },
        "configs": {
            STAGE_TRANSCRIPTION: transcription_config,
            STAGE_TRANSLATION: translation_config,
            STAGE_SYNTHESIS: synthesis_config,
            STAGE_MIX: mix_config,
        },
    }


def subtitle_cleanup_reason(text: str) -> str:
    if len(text) <= 1:
        return "micro_fragment"
    return "cleanup"


def split_text_at_pause(text: str, parts: int = 2) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned or parts <= 1:
        return [cleaned]
    for separator in (". ", "! ", "? ", "; ", ", ", " - ", " — "):
        if separator.strip() and separator in cleaned:
            chunks = [chunk.strip(" ,.-") for chunk in cleaned.split(separator) if chunk.strip(" ,.-")]
            if len(chunks) >= parts:
                return chunks[:parts]
    words = cleaned.split()
    if len(words) < parts * 2:
        return [cleaned]
    step = max(1, len(words) // parts)
    split_chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + step)
        if len(split_chunks) == parts - 1:
            end = len(words)
        split_chunks.append(" ".join(words[start:end]).strip())
        start = end
    return [chunk for chunk in split_chunks if chunk]


def cleanup_english_segments(segments: list[Segment]) -> tuple[list[Segment], list[dict[str, Any]]]:
    cleaned: list[Segment] = []
    changes: list[dict[str, Any]] = []
    pending_text = ""
    pending_start_ms: int | None = None

    def flush_pending(end_ms: int) -> None:
        nonlocal pending_text, pending_start_ms
        text = " ".join(pending_text.split())
        if not text or pending_start_ms is None:
            pending_text = ""
            pending_start_ms = None
            return
        cleaned.append(
            Segment(
                index=len(cleaned) + 1,
                start_ms=pending_start_ms,
                end_ms=max(pending_start_ms + 1, end_ms),
                text=text,
            )
        )
        pending_text = ""
        pending_start_ms = None

    for segment in segments:
        text = " ".join(segment.text.split())
        if not text:
            continue
        current_start = segment.start_ms if pending_start_ms is None else pending_start_ms
        current_text = (pending_text + " " + text).strip() if pending_text else text
        short_fragment = len(text) <= 2 or len(text.split()) <= 1
        if short_fragment:
            changes.append(
                {
                    "kind": "merge_micro_fragment",
                    "index": segment.index,
                    "original_text": text,
                    "reason": subtitle_cleanup_reason(text),
                }
            )
            pending_text = current_text
            pending_start_ms = current_start
            continue
        if pending_text:
            text = current_text
        split_needed = len(text) > 84 or len(text.split()) > 16
        if split_needed:
            parts = split_text_at_pause(text)
            if len(parts) > 1:
                total_duration = max(1, segment.end_ms - current_start)
                chunk_duration = max(1, total_duration // len(parts))
                for index, chunk in enumerate(parts):
                    part_start = current_start + (index * chunk_duration)
                    part_end = segment.end_ms if index == len(parts) - 1 else part_start + chunk_duration
                    cleaned.append(
                        Segment(
                            index=len(cleaned) + 1,
                            start_ms=part_start,
                            end_ms=max(part_start + 1, part_end),
                            text=chunk,
                        )
                    )
                changes.append(
                    {
                        "kind": "split_long_segment",
                        "index": segment.index,
                        "original_text": text,
                        "new_parts": parts,
                        "reason": "pause_split",
                    }
                )
                pending_text = ""
                pending_start_ms = None
                continue
        cleaned.append(
            Segment(
                index=len(cleaned) + 1,
                start_ms=current_start,
                end_ms=max(current_start + 1, segment.end_ms),
                text=text,
            )
        )
        pending_text = ""
        pending_start_ms = None

    if pending_text and pending_start_ms is not None:
        flush_pending(max(pending_start_ms + 1, segments[-1].end_ms if segments else pending_start_ms + 1))

    for index, segment in enumerate(cleaned, start=1):
        segment.index = index
        if index > 1:
            segment.start_ms = max(segment.start_ms, cleaned[index - 2].end_ms)
        if segment.end_ms <= segment.start_ms:
            segment.end_ms = segment.start_ms + 1
    return cleaned, changes


def shorten_dense_arabic_text(text: str) -> tuple[str, str | None]:
    shortened = " ".join(text.split())
    replacements = (
        ("من فضلك", ""),
        ("الرجاء", ""),
        ("بشكل", ""),
        ("للغاية", "جدا"),
        ("في الدرس ده", "الدرس ده"),
        ("بعد كده", "وبعدها"),
    )
    for source, target in replacements:
        candidate = shortened.replace(source, target).strip()
        candidate = " ".join(candidate.split())
        if candidate and len(candidate) < len(shortened):
            return candidate, f"replace:{source}"
    return shortened, None


def auto_rewrite_dense_arabic_segments(segments: list[Segment]) -> tuple[list[Segment], list[dict[str, Any]]]:
    rewritten: list[Segment] = []
    changes: list[dict[str, Any]] = []
    for segment in segments:
        text = " ".join(segment.text.split())
        duration_seconds = max(segment.duration_ms / 1000, 0.15)
        chars_per_second = len(text) / duration_seconds
        new_text = text
        reason: str | None = None
        if chars_per_second > 12.8 or len(text) > max(46, int(duration_seconds * 12.0)):
            new_text, reason = shorten_dense_arabic_text(text)
        rewritten.append(
            Segment(
                index=segment.index,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                text=new_text,
            )
        )
        if reason and new_text != text:
            changes.append(
                {
                    "index": segment.index,
                    "original_text": text,
                    "rewritten_text": new_text,
                    "reason": reason,
                    "cps_before": round(chars_per_second, 2),
                }
            )
    return rewritten, changes


def remove_stage_outputs(
    outputs_dir: Path,
    work_dir: Path,
    stages_to_refresh: set[str],
    *,
    strict_translation_qa: bool = False,
) -> None:
    if STAGE_TRANSCRIPTION in stages_to_refresh:
        for path in (
            outputs_dir / "english.srt",
            outputs_dir / "segments.json",
            work_dir / "source_audio.wav",
        ):
            path.unlink(missing_ok=True)
    if STAGE_TRANSLATION in stages_to_refresh:
        for path in (
            outputs_dir / "arabic_auto.srt",
            outputs_dir / "quality_diagnostics.json",
        ):
            path.unlink(missing_ok=True)
    if STAGE_SYNTHESIS in stages_to_refresh:
        for path in (
            outputs_dir / "arabic_dub.wav",
            outputs_dir / "arabic_full_narration.mp3",
            outputs_dir / "arabic_full_narration.wav",
            outputs_dir / "arabic_full_script.txt",
        ):
            path.unlink(missing_ok=True)
        shutil.rmtree(work_dir / "timed_fitted_cache", ignore_errors=True)
        shutil.rmtree(work_dir / "timed_mix_stems", ignore_errors=True)
        shutil.rmtree(work_dir / "full_tts_chunks", ignore_errors=True)
    if STAGE_MIX in stages_to_refresh:
        for path in outputs_dir.glob("*.arabic-dub.mp4"):
            path.unlink(missing_ok=True)
    if strict_translation_qa and STAGE_TRANSLATION in stages_to_refresh:
        (outputs_dir / "qa_blocked.flag").unlink(missing_ok=True)


def translate_segments(
    segments: list[Segment],
    glossary: dict[str, str],
    deepl_api_key: str | None,
    deepl_api_url: str,
    translation_provider: str,
    *,
    cache_store: CacheStore | None = None,
    source_language: str = "en",
    target_language: str = "ar",
    glossary_hash: str = "",
    rewrite_version: str = SPOKEN_REWRITE_VERSION,
    spoken_style: str = DEFAULT_SPOKEN_STYLE,
) -> tuple[list[Segment], TranslationRunStats]:
    def google_translate_chunk(chunk: list[Segment]) -> list[Segment]:
        translator = GoogleTranslator(source="en", target="ar")
        translated_chunk: list[Segment] = []
        for segment in chunk:
            try:
                arabic = translator.translate(segment.text)
            except Exception as exc:  # pragma: no cover - network/provider failure
                raise RuntimeError(f"Google Translate fallback failed: {exc}") from exc
            translated_chunk.append(
                Segment(
                    index=segment.index,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    text=apply_glossary((arabic or "").strip(), glossary),
                )
            )
        return translated_chunk

    resolved_provider = choose_translation_provider(translation_provider, deepl_api_key)
    stats = TranslationRunStats(provider=resolved_provider)
    if smoke_mode_enabled():
        translated = [
            Segment(
                index=segment.index,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                text=apply_glossary(f"ترجمة تجريبية {segment.index}", glossary).strip(),
            )
            for segment in segments
        ]
        stats.cache_misses = len(translated)
        return translated, stats

    def translation_cache_key(segment_text: str, provider_name: str) -> str:
        return stable_hash_json(
            {
                "text": segment_text,
                "source_lang": source_language,
                "target_lang": target_language,
                "provider": provider_name,
                "glossary_hash": glossary_hash,
                "rewrite_version": rewrite_version,
                "style": spoken_style,
            }
        )

    def resolve_chunk_with_cache(chunk: list[Segment], provider_name: str) -> list[Segment]:
        cached: list[Segment] = []
        misses: list[Segment] = []
        miss_keys: list[str] = []
        for segment in chunk:
            cache_key = translation_cache_key(segment.text, provider_name)
            cached_payload = cache_store.get_translation(cache_key) if cache_store else None
            if cached_payload:
                stats.cache_hits += 1
                cached.append(
                    Segment(
                        index=segment.index,
                        start_ms=segment.start_ms,
                        end_ms=segment.end_ms,
                        text=str(cached_payload["translated_text"]).strip(),
                    )
                )
                continue
            stats.cache_misses += 1
            misses.append(segment)
            miss_keys.append(cache_key)

        translated_misses: list[Segment] = []
        if misses:
            if provider_name == "google":
                translated_misses = google_translate_chunk(misses)
            else:
                translated_misses = resolve_deepl_chunk(misses)
            if cache_store:
                for segment, translated_segment, cache_key in zip(misses, translated_misses, miss_keys):
                    cache_store.put_translation(
                        cache_key,
                        source_text=segment.text,
                        translated_text=translated_segment.text,
                        provider=provider_name,
                        source_lang=source_language,
                        target_lang=target_language,
                        glossary_hash=glossary_hash,
                        rewrite_version=rewrite_version,
                        style=spoken_style,
                    )

        translated_map = {segment.index: segment for segment in cached + translated_misses}
        return [translated_map[segment.index] for segment in chunk if segment.index in translated_map]

    def resolve_deepl_chunk(chunk: list[Segment]) -> list[Segment]:
        translated: list[Segment] = []
        data = [("text", segment.text) for segment in chunk]
        try:
            response = requests.post(
                deepl_api_url,
                headers=headers,
                params=params,
                data=data,
                timeout=30,
            )
        except requests.RequestException:
            print("DeepL request failed; using Google Translate fallback.")
            return google_translate_chunk(chunk)
        if response.status_code != 200:
            if response.status_code in {429, 456}:
                print(f"DeepL rate/quota limit hit ({response.status_code}); using Google Translate fallback.")
                return google_translate_chunk(chunk)
            raise RuntimeError(
                f"DeepL API responded with {response.status_code}: {response.text}"
            )

        payload = response.json()
        translations = payload.get("translations", [])
        if len(translations) != len(chunk):
            raise RuntimeError(
                "DeepL returned an unexpected number of translations."
            )

        for segment, translation in zip(chunk, translations):
            arabic = apply_glossary(translation.get("text", ""), glossary)
            translated.append(
                Segment(
                    index=segment.index,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                    text=arabic.strip(),
                )
            )
        return translated

    if resolved_provider == "google":
        print("Translating subtitles to Arabic with Google Translate...")
        translated_segments: list[Segment] = []
        for start in range(0, len(segments), DEEPL_CHUNK_SIZE):
            translated_segments.extend(
                resolve_chunk_with_cache(segments[start : start + DEEPL_CHUNK_SIZE], "google")
            )
        return translated_segments, stats

    deepl_translated_segments: list[Segment] = []
    print("Translating subtitles to Arabic with DeepL...")
    headers = {"Authorization": f"DeepL-Auth-Key {deepl_api_key}"}
    params = {
        "target_lang": "AR",
        "source_lang": "EN",
        "split_sentences": "1",
        "preserve_formatting": "0",
    }

    for start in range(0, len(segments), DEEPL_CHUNK_SIZE):
        chunk = segments[start : start + DEEPL_CHUNK_SIZE]
        deepl_translated_segments.extend(resolve_chunk_with_cache(chunk, "deepl"))
    return deepl_translated_segments, stats


def resolve_deepl_url(deepl_api_key: str, deepl_api_url: str | None) -> str:
    if deepl_api_url:
        return deepl_api_url
    # DeepL Free keys usually end with :fx; Pro keys use api.deepl.com.
    if deepl_api_key.endswith(":fx"):
        return DEEPL_DEFAULT_URL
    return DEEPL_PRO_URL


def validate_deepl_credentials(deepl_api_key: str, deepl_api_url: str) -> None:
    usage_url = deepl_api_url.replace("/v2/translate", "/v2/usage")
    try:
        response = requests.get(
            usage_url,
            headers={"Authorization": f"DeepL-Auth-Key {deepl_api_key}"},
            timeout=20,
        )
    except requests.RequestException as exc:
        raise RuntimeError("DeepL usage check failed. Check network/connectivity.") from exc
    if response.status_code == 200:
        return
    if response.status_code in {429, 456}:
        print(
            f"DeepL usage check returned {response.status_code}; continuing with Google Translate fallback when needed."
        )
        return
    hint = ""
    if response.status_code in {401, 403}:
        hint = " (possible key/endpoint mismatch; set DEEPL_API_URL if needed)"
    raise RuntimeError(
        f"DeepL usage check failed with {response.status_code}: {response.text}{hint}"
    )


def build_quality_diagnostics(
    english_segments: list[Segment],
    arabic_segments: list[Segment],
    glossary: dict[str, str],
    compression_events: list[dict[str, int | float | str]],
    quality_profile: QualityProfile,
    *,
    translation_stats: TranslationRunStats | None = None,
    strict_translation_qa: bool = False,
    cleanup_changes: list[dict[str, Any]] | None = None,
    density_rewrites: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "density": asdict(quality_profile),
        "segments_checked": len(arabic_segments),
        "long_segments": [],
        "untranslated_tokens": [],
        "glossary_misses": [],
        "heavy_compression_segments": compression_events,
        "subtitle_cleanup_changes": cleanup_changes or [],
        "density_rewrites": density_rewrites or [],
        "needs_attention": False,
        "blocking_issues": [],
    }

    for english, arabic in zip(english_segments, arabic_segments):
        text = " ".join(arabic.text.split())
        if not text:
            continue
        duration_seconds = max(arabic.duration_ms / 1000, 0.15)
        chars_per_second = len(text) / duration_seconds
        if chars_per_second > 13.5 or len(text) > max(42, int(duration_seconds * 12.5)):
            diagnostics["long_segments"].append(
                {
                    "index": arabic.index,
                    "slot_ms": arabic.duration_ms,
                    "chars": len(text),
                    "chars_per_second": round(chars_per_second, 2),
                    "text": text,
                }
            )
        english_tokens = {
            token.lower()
            for token in ENGLISH_TOKEN_RE.findall(arabic.text)
            if len(token) >= 4
        }
        if english_tokens:
            diagnostics["untranslated_tokens"].append(
                {
                    "index": arabic.index,
                    "tokens": sorted(english_tokens),
                    "text": text,
                }
            )

        for source, target in glossary.items():
            if source.lower() in english.text.lower() and target not in arabic.text:
                diagnostics["glossary_misses"].append(
                    {
                        "index": arabic.index,
                        "source": source,
                        "target": target,
                        "english": english.text,
                        "arabic": arabic.text,
                    }
                )

    diagnostics["summary"] = {
        "long_segments": len(diagnostics["long_segments"]),
        "untranslated_segments": len(diagnostics["untranslated_tokens"]),
        "glossary_misses": len(diagnostics["glossary_misses"]),
        "heavy_compression_segments": len(compression_events),
        "subtitle_cleanup_changes": len(diagnostics["subtitle_cleanup_changes"]),
        "density_rewrites": len(diagnostics["density_rewrites"]),
    }
    glossary_hits = 0
    for english, arabic in zip(english_segments, arabic_segments):
        if any(source.lower() in english.text.lower() and target in arabic.text for source, target in glossary.items()):
            glossary_hits += 1
    diagnostics["qa"] = {
        "strict_translation_qa": strict_translation_qa,
        "glossary_hits": translation_stats.glossary_hits if translation_stats else glossary_hits,
        "glossary_misses": len(diagnostics["glossary_misses"]),
        "english_token_leaks": len(diagnostics["untranslated_tokens"]),
        "translation_cache_hits": translation_stats.cache_hits if translation_stats else 0,
        "translation_cache_misses": translation_stats.cache_misses if translation_stats else 0,
        "translation_cache_hit_rate": translation_stats.hit_rate if translation_stats else 0.0,
    }
    if diagnostics["untranslated_tokens"]:
        diagnostics["blocking_issues"].append("english_token_leak")
    if diagnostics["glossary_misses"]:
        diagnostics["blocking_issues"].append("glossary_miss")
    diagnostics["needs_attention"] = any(diagnostics["summary"].values())
    return diagnostics


def smoke_mode_enabled() -> bool:
    return os.environ.get("DUB_SMOKE_MODE", "").strip() == "1"


def smoke_mode_english_segments() -> list[Segment]:
    payload = os.environ.get("DUB_SMOKE_SEGMENTS_JSON", "").strip()
    if payload:
        try:
            raw_segments = json.loads(payload)
            return [
                Segment(
                    index=index + 1,
                    start_ms=int(item.get("start_ms", index * 1000)),
                    end_ms=int(item.get("end_ms", (index + 1) * 1000)),
                    text=str(item.get("text", "")).strip(),
                )
                for index, item in enumerate(raw_segments)
                if str(item.get("text", "")).strip()
            ]
        except json.JSONDecodeError:
            pass
    return [
        Segment(index=1, start_ms=0, end_ms=900, text="Open the lesson slowly."),
        Segment(index=2, start_ms=1000, end_ms=1800, text="Keep your shoulders relaxed."),
    ]


def write_smoke_audio(out_path: Path, duration_ms: int = 180) -> None:
    duration_seconds = max(0.12, duration_ms / 1000)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=660:duration={duration_seconds:.3f}",
            "-ar",
            str(SAMPLE_RATE),
            str(out_path),
        ]
    )


def lesson_report_path(project_dir: Path) -> Path:
    return project_dir / "lesson_report.json"


def maybe_remove_empty(path: Path) -> None:
    if path.exists() and not any(path.iterdir()):
        path.rmdir()


def shared_stage_manifest_identity(
    args: argparse.Namespace,
    source_video: Path,
    source_audio_input: Path | None,
    glossary: dict[str, str],
) -> tuple[Path, CacheStore, str, str, str, dict[str, Any]]:
    shared_root = args.shared_root.resolve() if args.shared_root else infer_shared_root(args.project_dir.resolve())
    cache_store = CacheStore(shared_root / "cache.db")
    course_slug = slugify(args.course_slug or source_video.stem)
    video_key = args.video_key or build_video_key(args.lesson_position, args.lesson_title or source_video.stem, args.video_position)
    source_hash = stable_hash_text("missing")
    audio_hash = ""
    if source_video.exists():
        source_hash = hash_file(source_video)
    if source_audio_input and source_audio_input.exists():
        audio_hash = hash_file(source_audio_input)
    glossary_hash = stable_hash_json(glossary)
    stage_manifest = build_processing_manifest(
        course_slug=course_slug,
        video_key=video_key,
        source_hash=source_hash,
        audio_hash=audio_hash,
        transcription_config={
            "model": args.model,
            "source_language": args.source_language,
            "segmentation_cleanup_version": SEGMENTATION_CLEANUP_VERSION,
            "smoke_mode": smoke_mode_enabled(),
        },
        translation_config={
            "provider": args.translation_provider,
            "glossary_hash": glossary_hash,
            "spoken_style": args.spoken_style,
            "spoken_rewrite_enabled": not args.disable_spoken_rewrite,
            "spoken_rewrite_version": SPOKEN_REWRITE_VERSION,
            "dense_rewrite_version": DENSE_REWRITE_VERSION,
            "strict_translation_qa": args.strict_translation_qa,
            "use_reviewed_arabic": args.use_reviewed_arabic,
        },
        synthesis_config={
            "tts_strategy": args.tts_strategy,
            "voice": args.voice,
            "timed_rate": args.timed_rate,
            "narration_rate": args.narration_rate,
            "tts_concurrency": args.tts_concurrency,
            "timed_max_late_drift_ms": args.timed_max_late_drift_ms,
            "timed_collision_max_speed": args.timed_collision_max_speed,
            "quality_profile": args.quality_profile,
            "disable_timed_compression": args.disable_timed_compression,
        },
        mix_config={
            "dub_gain": args.dub_gain,
            "bg_gain_db": args.bg_gain_db,
            "video_slowdown": args.video_slowdown,
            "max_dub_speed": args.max_dub_speed,
        },
    )
    return shared_root, cache_store, course_slug, video_key, glossary_hash, stage_manifest


def write_lesson_report(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


async def synthesize_tts(
    text: str,
    voice: str,
    rate: str,
    out_path: Path,
    pitch: str = "+0Hz",
) -> None:
    if smoke_mode_enabled():
        write_smoke_audio(out_path)
        return
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch,
    )
    await communicate.save(str(out_path))


async def render_voice_candidate_samples(
    text: str,
    voices: list[str],
    out_dir: Path,
) -> list[tuple[str, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[tuple[str, Path]] = []
    for voice in voices:
        out_path = out_dir / f"{voice}.mp3"
        await synthesize_tts(text, voice, "-2%", out_path)
        rendered.append((voice, out_path))
    return rendered


def select_best_voice(
    sample_text: str,
    sample_dir: Path,
    candidates: list[str] | None = None,
) -> VoiceSelectionResult:
    candidates = candidates or list(DEFAULT_VOICE_CANDIDATES)
    rendered = asyncio.run(render_voice_candidate_samples(sample_text, candidates, sample_dir))
    durations = {voice: probe_duration_ms(path) for voice, path in rendered}
    cps_values = [
        len(sample_text) / max(duration_ms / 1000, 0.25)
        for duration_ms in durations.values()
    ]
    target_cps = statistics.median(cps_values) if cps_values else 9.5
    ranking: list[dict[str, float | str]] = []
    for voice, path in rendered:
        duration_ms = durations[voice]
        cps = len(sample_text) / max(duration_ms / 1000, 0.25)
        locale_penalty = 0 if voice.startswith("ar-EG-") else 20
        score = abs(cps - target_cps) + locale_penalty
        ranking.append(
            {
                "voice": voice,
                "duration_ms": duration_ms,
                "chars_per_second": round(cps, 2),
                "score": round(score, 3),
                "sample_path": str(path),
            }
        )
    ranking.sort(key=lambda item: float(item["score"]))
    return VoiceSelectionResult(
        selected_voice=str(ranking[0]["voice"]),
        sample_text=sample_text,
        sample_dir=str(sample_dir),
        ranking=ranking,
    )


def probe_duration_ms(path: Path) -> int:
    output = run_capture(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    ).strip()
    if output and output != "N/A":
        return ms_from_seconds(float(output))

    stream_output = run_capture(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    ).strip()
    for line in stream_output.splitlines():
        line = line.strip()
        if line and line != "N/A":
            return ms_from_seconds(float(line))
    raise RuntimeError(f"Could not determine audio duration for {path}")


def timed_tts_cache_path(
    cache_dir: Path,
    text: str,
    voice: str,
    rate: str,
    pitch: str = "+0Hz",
) -> Path:
    payload = json.dumps(
        {
            "text": text,
            "voice": voice,
            "rate": rate,
            "pitch": pitch,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.mp3"


async def render_timed_segments(
    segments: list[Segment],
    voice: str,
    rate: str,
    cache_dir: Path,
    concurrency: int,
    pitch: str = "+0Hz",
) -> list[tuple[Segment, Path]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    ordered_segments: list[tuple[Segment, Path]] = []
    pending_renders: dict[Path, str] = {}

    for segment in segments:
        cache_path = timed_tts_cache_path(cache_dir, segment.text, voice, rate, pitch)
        ordered_segments.append((segment, cache_path))
        if cache_path.exists() and cache_path.stat().st_size > 0:
            continue
        pending_renders.setdefault(cache_path, segment.text)

    async def render(cache_path: Path, text: str) -> None:
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return
        temp_path = cache_path.with_suffix(".tmp.mp3")
        temp_path.unlink(missing_ok=True)
        try:
            async with semaphore:
                await synthesize_tts(text, voice, rate, temp_path, pitch)
            temp_path.replace(cache_path)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    await asyncio.gather(*(render(path, text) for path, text in pending_renders.items()))
    return ordered_segments


def build_timed_dub_audio(
    segments: list[Segment],
    voice: str,
    total_ms: int,
    out_path: Path,
    timed_rate: str,
    cache_dir: Path,
    workspace_dir: Path,
    tts_concurrency: int,
    allow_timed_compression: bool,
    timed_max_late_drift_ms: int,
    timed_collision_max_speed: float,
) -> list[dict[str, int | float | str]]:
    compression_events: list[dict[str, int | float | str]] = []

    def mix_audio_layers(
        layers: list[tuple[int, Path, int, bool]],
        output_path: Path,
        volume_scale: float,
    ) -> None:
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r={SAMPLE_RATE}:cl=mono",
            "-t",
            f"{total_ms / 1000:.3f}",
        ]
        for _, file_path, _, _ in layers:
            cmd.extend(["-i", str(file_path)])

        filter_parts = [f"[0:a]atrim=0:{total_ms / 1000:.3f}[base]"]
        mix_labels = ["[base]"]
        for index, (delay_ms, _, clip_duration_ms, fitted) in enumerate(layers, start=1):
            label = f"s{index}"
            layer_filter = ""
            if fitted:
                fade_duration_sec = TIMED_FIT_FADE_OUT_MS / 1000
                fade_start_sec = max(0.0, clip_duration_ms / 1000 - fade_duration_sec)
                layer_filter = f"afade=t=out:st={fade_start_sec:.3f}:d={fade_duration_sec:.3f},"
            if delay_ms > 0:
                filter_parts.append(
                    f"[{index}:a]{layer_filter}adelay={delay_ms}|{delay_ms},apad[{label}]"
                )
            else:
                filter_parts.append(f"[{index}:a]{layer_filter}apad[{label}]")
            mix_labels.append(f"[{label}]")
        filter_parts.append(
            f"{''.join(mix_labels)}amix=inputs={len(mix_labels)}:duration=first:"
            "dropout_transition=0:normalize=0[mix]"
        )
        output_label = "[mix]"
        if not math.isclose(volume_scale, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            filter_parts.append(f"[mix]volume={volume_scale:.12f}[out]")
            output_label = "[out]"

        cmd.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                output_label,
                "-ar",
                str(SAMPLE_RATE),
                str(output_path),
            ]
        )
        run(cmd)

    active_segments = [
        segment
        for segment in segments
        if segment.text and segment.start_ms < total_ms
    ]
    segment_files = asyncio.run(
        render_timed_segments(
            active_segments,
            voice,
            timed_rate,
            cache_dir,
            tts_concurrency,
        )
    )
    if not segment_files:
        mix_audio_layers([], out_path, 1.0)
        return []

    fitted_cache_dir = workspace_dir / "timed_fitted_cache"
    fitted_cache_dir.mkdir(parents=True, exist_ok=True)
    layers: list[tuple[int, Path, int, bool]] = []
    previous_end_ms = 0
    for index, (segment, file_path) in enumerate(segment_files):
        source_duration_ms = probe_duration_ms(file_path)
        next_start_ms = total_ms
        if index + 1 < len(segment_files):
            next_start_ms = segment_files[index + 1][0].start_ms
        raw_slot_ms = max(1, next_start_ms - segment.start_ms)
        dynamic_gap_ms = min(
            TIMED_IDEAL_GAP_MS,
            max(TIMED_MIN_GAP_MS, int(raw_slot_ms * 0.06)),
        )
        scheduled_start_ms = max(segment.start_ms, previous_end_ms + dynamic_gap_ms)
        late_drift_ms = max(0, scheduled_start_ms - segment.start_ms)
        if late_drift_ms > timed_max_late_drift_ms:
            scheduled_start_ms = segment.start_ms + timed_max_late_drift_ms
        available_ms = max(1, next_start_ms - scheduled_start_ms - dynamic_gap_ms)
        overlap_ms = max(0, source_duration_ms - available_ms)
        resolved_path = file_path
        fitted = False
        if allow_timed_compression and overlap_ms > TIMED_FIT_TRIGGER_MS:
            target_ms = max(TIMED_MIN_FIT_MS, available_ms - TIMED_FIT_TAIL_PADDING_MS)
            compression_ratio = source_duration_ms / max(target_ms, 1)
            fitted_key = hashlib.sha1(
                (
                    f"{file_path}:{target_ms}:"
                    f"{timed_collision_max_speed:.3f}:{int(allow_timed_compression)}"
                ).encode("utf-8")
            ).hexdigest()
            resolved_path = fitted_cache_dir / f"{fitted_key}.wav"
            needs_refit = not resolved_path.exists() or resolved_path.stat().st_size == 0
            if not needs_refit:
                try:
                    probe_duration_ms(resolved_path)
                except RuntimeError:
                    needs_refit = True
            if needs_refit:
                resolved_path.unlink(missing_ok=True)
                fit_audio_to_duration(
                    file_path,
                    target_ms,
                    resolved_path,
                    max_speed=timed_collision_max_speed,
                )
            fitted = True
            if compression_ratio >= 1.18:
                compression_events.append(
                    {
                        "index": segment.index,
                        "text": segment.text,
                        "source_duration_ms": source_duration_ms,
                        "target_duration_ms": target_ms,
                        "speed_ratio": round(compression_ratio, 3),
                    }
                )
        resolved_duration_ms = source_duration_ms if resolved_path == file_path else probe_duration_ms(resolved_path)
        layers.append(
            (
                max(0, min(scheduled_start_ms, max(0, total_ms - 1))),
                resolved_path,
                resolved_duration_ms,
                fitted,
            )
        )
        previous_end_ms = min(total_ms, scheduled_start_ms + resolved_duration_ms)
    stem_dir = workspace_dir / "timed_mix_stems"
    shutil.rmtree(stem_dir, ignore_errors=True)
    stem_dir.mkdir(parents=True, exist_ok=True)

    level = 0
    while len(layers) > TIMED_MIX_BATCH_SIZE:
        next_layers: list[tuple[int, Path, int, bool]] = []
        for batch_index, start in enumerate(range(0, len(layers), TIMED_MIX_BATCH_SIZE), start=1):
            batch_layers = layers[start : start + TIMED_MIX_BATCH_SIZE]
            stem_path = stem_dir / f"level_{level:02d}_batch_{batch_index:03d}.wav"
            mix_audio_layers(batch_layers, stem_path, 1.0)
            next_layers.append((0, stem_path, probe_duration_ms(stem_path), False))
        layers = next_layers
        level += 1

    mix_audio_layers(layers, out_path, 1.0)
    return compression_events


def build_full_script(segments: list[Segment]) -> str:
    script_lines: list[str] = []
    previous = ""
    for segment in segments:
        text = " ".join(segment.text.split())
        if not text or text == previous:
            continue
        script_lines.append(text)
        previous = text
    return "\n\n".join(script_lines).strip()


def split_script_blocks(script: str, max_chars: int) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in script.split("\n\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        addition = len(paragraph) + (2 if current else 0)
        if current and current_len + addition > max_chars:
            blocks.append("\n\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len += addition
    if current:
        blocks.append("\n\n".join(current))
    return blocks


async def render_script_chunks(
    blocks: list[str],
    voice: str,
    rate: str,
    out_dir: Path,
    concurrency: int,
) -> list[Path]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def render(index: int, block: str) -> Path:
        out_path = out_dir / f"chunk_{index:03d}.mp3"
        if out_path.exists():
            return out_path
        async with semaphore:
            await synthesize_tts(block, voice, rate, out_path)
        return out_path

    tasks = [render(index, block) for index, block in enumerate(blocks, start=1)]
    return list(await asyncio.gather(*tasks))


def concat_audio_chunks(chunk_paths: list[Path], output_mp3: Path) -> None:
    concat_file = output_mp3.with_suffix(".concat.txt")
    concat_file.write_text(
        "\n".join(f"file '{path}'" for path in chunk_paths),
        encoding="utf-8",
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            str(output_mp3),
        ]
    )


def convert_audio(src: Path, dst: Path) -> None:
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ar",
            "48000",
            "-ac",
            "2",
            str(dst),
        ]
    )


def build_atempo_filter(speed_ratio: float) -> str:
    if speed_ratio <= 0:
        raise ValueError("speed_ratio must be positive")
    remaining = speed_ratio
    factors: list[float] = []
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    if not math.isclose(remaining, 1.0, rel_tol=1e-5, abs_tol=1e-5):
        factors.append(remaining)
    if not factors:
        return "anull"
    return ",".join(f"atempo={factor:.5f}" for factor in factors)


def fit_audio_to_duration(src: Path, target_ms: int, dst: Path, max_speed: float | None) -> None:
    current_ms = probe_duration_ms(src)
    if current_ms <= 0:
        raise RuntimeError("Source audio duration is invalid.")
    speed_ratio = current_ms / target_ms
    if math.isclose(speed_ratio, 1.0, rel_tol=0.01, abs_tol=0.01):
        shutil.copy2(src, dst)
        return
    if max_speed is not None and speed_ratio > max_speed:
        raise RuntimeError(
            f"Dub would require {speed_ratio:.2f}x speed, above the allowed max of {max_speed:.2f}x. "
            "Increase --video-slowdown or --max-dub-speed."
        )
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-filter:a",
            build_atempo_filter(speed_ratio),
            str(dst),
        ]
    )


def build_chunked_dub_audio(
    segments: list[Segment],
    voice: str,
    total_ms: int,
    out_path: Path,
    work_dir: Path,
    narration_script_path: Path,
    raw_narration_mp3_path: Path,
    raw_narration_wav_path: Path,
    narration_rate: str,
    chunk_max_chars: int,
    tts_concurrency: int,
    max_dub_speed: float,
) -> None:
    script = build_full_script(segments)
    narration_script_path.write_text(script, encoding="utf-8")
    blocks = split_script_blocks(script, chunk_max_chars)
    if not blocks:
        raise RuntimeError("Arabic narration script is empty.")

    chunk_dir = work_dir / "full_tts_chunks"
    chunk_paths = asyncio.run(
        render_script_chunks(
            blocks,
            voice,
            narration_rate,
            chunk_dir,
            tts_concurrency,
        )
    )
    concat_audio_chunks(chunk_paths, raw_narration_mp3_path)
    convert_audio(raw_narration_mp3_path, raw_narration_wav_path)
    fit_audio_to_duration(raw_narration_wav_path, total_ms, out_path, max_dub_speed)


def mux_final_video(
    video_path: Path,
    original_audio_path: Path | None,
    dubbed_audio_path: Path,
    out_path: Path,
    bg_gain_db: int,
    dub_gain: float,
    video_slowdown: float,
) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(video_path)]
    if original_audio_path is not None:
        cmd.extend(["-i", str(original_audio_path)])
    cmd.extend(["-i", str(dubbed_audio_path)])

    def audio_slowdown_filter(factor: float) -> str:
        return build_atempo_filter(1.0 / factor)

    if original_audio_path is not None:
        if not math.isclose(video_slowdown, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            filter_complex = (
                f"[0:v]setpts={video_slowdown:.6f}*PTS[vout];"
                f"[1:a]{audio_slowdown_filter(video_slowdown)},volume={bg_gain_db}dB[bgbase];"
                f"[2:a]volume={dub_gain},loudnorm=I=-18:LRA=7:TP=-1.5,acompressor=threshold=-19dB:ratio=2.1:attack=8:release=85:makeup=2,alimiter=limit=0.96[dub];"
                "[bgbase][dub]sidechaincompress=threshold=0.02:ratio=7:attack=25:release=260:makeup=1[bg];"
                "[bg][dub]amix=inputs=2:duration=first:dropout_transition=2,"
                "loudnorm=I=-16:LRA=11:TP=-1.5[aout]"
            )
            video_map = "[vout]"
            video_codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
        else:
            filter_complex = (
                f"[1:a]volume={bg_gain_db}dB[bgbase];"
                f"[2:a]volume={dub_gain},loudnorm=I=-18:LRA=7:TP=-1.5,acompressor=threshold=-19dB:ratio=2.1:attack=8:release=85:makeup=2,alimiter=limit=0.96[dub];"
                "[bgbase][dub]sidechaincompress=threshold=0.02:ratio=7:attack=25:release=260:makeup=1[bg];"
                "[bg][dub]amix=inputs=2:duration=first:dropout_transition=2,"
                "loudnorm=I=-16:LRA=11:TP=-1.5[aout]"
            )
            video_map = "0:v"
            video_codec_args = ["-c:v", "copy"]
    else:
        if not math.isclose(video_slowdown, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            filter_complex = (
                f"[0:v]setpts={video_slowdown:.6f}*PTS[vout];"
                f"[1:a]volume={dub_gain},loudnorm=I=-18:LRA=7:TP=-1.5,acompressor=threshold=-20dB:ratio=2.5:attack=5:release=50:makeup=4,alimiter=limit=0.97,"
                "loudnorm=I=-16:LRA=11:TP=-1.5[aout]"
            )
            video_map = "[vout]"
            video_codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
        else:
            filter_complex = (
                f"[1:a]volume={dub_gain},loudnorm=I=-18:LRA=7:TP=-1.5,"
                "acompressor=threshold=-20dB:ratio=2.5:attack=5:release=50:makeup=4,alimiter=limit=0.97,"
                "loudnorm=I=-16:LRA=11:TP=-1.5[aout]"
            )
            video_map = "0:v"
            video_codec_args = ["-c:v", "copy"]

    cmd.extend(
        [
            "-filter_complex",
            filter_complex,
            "-map",
            video_map,
            "-map",
            "[aout]",
        ]
    )
    cmd.extend(video_codec_args)
    cmd.extend(
        [
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-ac",
            "2",
            str(out_path),
        ]
    )
    run(cmd)


def ensure_workspace(project_dir: Path) -> tuple[Path, Path, Path]:
    inputs_dir = project_dir / "inputs"
    outputs_dir = project_dir / "outputs"
    work_dir = project_dir / "work"
    for path in (inputs_dir, outputs_dir, work_dir):
        path.mkdir(parents=True, exist_ok=True)
    return inputs_dir, outputs_dir, work_dir


def maybe_copy_input(source: Path, destination_dir: Path) -> Path:
    copied = destination_dir / source.name
    if copied != source and not copied.exists():
        shutil.copy2(source, copied)
    return copied if copied.exists() else source


def build_lesson_output_stem(lesson_position: int, lesson_title: str) -> str:
    if lesson_position < 1:
        raise RuntimeError("--lesson-position must be at least 1.")
    return f"{lesson_position:02d}-{slugify(lesson_title)}"


def build_final_video_name(
    lesson_position: int,
    lesson_title: str,
    video_position: int = 1,
    lesson_video_count: int = 1,
) -> str:
    if video_position < 1:
        raise RuntimeError("--video-position must be at least 1.")
    if lesson_video_count < 1:
        raise RuntimeError("--lesson-video-count must be at least 1.")
    stem = build_lesson_output_stem(lesson_position, lesson_title)
    if lesson_video_count > 1:
        return f"{stem}.v{video_position:02d}.arabic-dub.mp4"
    return f"{stem}.arabic-dub.mp4"


def build_default_workspace_layout(
    project_root: Path,
    input_video: Path,
    course_slug: str = "",
    lesson_position: int = 1,
    lesson_title: str = "",
    lesson_video_count: int = 1,
    video_position: int = 1,
) -> WorkspaceLayout:
    resolved_lesson_title = lesson_title.strip() or input_video.stem or "lesson"
    resolved_course_slug = slugify(course_slug or resolved_lesson_title)
    lesson_stem = build_lesson_output_stem(lesson_position, resolved_lesson_title)
    project_dir = (
        project_root
        / ".course_work"
        / resolved_course_slug
        / lesson_stem
        / f"video_{video_position:02d}"
    )
    final_video = (
        project_root
        / "courses"
        / resolved_course_slug
        / build_final_video_name(
            lesson_position,
            resolved_lesson_title,
            video_position=video_position,
            lesson_video_count=lesson_video_count,
        )
    )
    return WorkspaceLayout(project_dir=project_dir, final_video=final_video)


def resolve_workspace_layout(args: argparse.Namespace, input_video: Path) -> tuple[Path, Path]:
    default_layout = build_default_workspace_layout(
        PROJECT_ROOT,
        input_video,
        course_slug=args.course_slug,
        lesson_position=args.lesson_position,
        lesson_title=args.lesson_title,
        lesson_video_count=args.lesson_video_count,
        video_position=args.video_position,
    )
    project_dir = args.project_dir.resolve() if args.project_dir else default_layout.project_dir
    final_video = args.final_video_path.resolve() if args.final_video_path else default_layout.final_video
    final_video.parent.mkdir(parents=True, exist_ok=True)
    return project_dir, final_video


def load_project_glossary(project_dir: Path) -> dict[str, str]:
    for candidate in (project_dir / "glossary.csv", PROJECT_ROOT / "glossary.csv"):
        if candidate.exists():
            return load_glossary(candidate)
    return {}


def cleanup_local_final_outputs(outputs_dir: Path, keep_path: Path | None = None) -> None:
    keep_resolved = keep_path.resolve() if keep_path else None
    for candidate in outputs_dir.glob("*.arabic-dub.mp4"):
        if keep_resolved is not None and candidate.resolve() == keep_resolved:
            continue
        candidate.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="High-quality local Arabic dubbing pipeline")
    parser.add_argument("--input", required=True, type=Path, help="Path to source video")
    parser.add_argument("--audio-input", type=Path, help="Optional separate source audio file")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        help="Working directory for inputs/outputs/work (defaults to .course_work/<course-slug>/...)",
    )
    parser.add_argument("--final-video-path", type=Path, help="Explicit destination for the final dubbed MP4")
    parser.add_argument("--course-slug", default="", help="Visible course folder slug for ad hoc single-video runs")
    parser.add_argument("--lesson-position", type=int, default=1, help="Lesson position used in the final filename")
    parser.add_argument("--lesson-title", default="", help="Lesson title used in the final filename")
    parser.add_argument(
        "--lesson-video-count",
        type=int,
        default=1,
        help="Number of videos in the lesson; values above 1 enable .vNN final naming",
    )
    parser.add_argument(
        "--video-position",
        type=int,
        default=1,
        help="Video position within a multi-video lesson",
    )
    parser.add_argument("--video-key", default="", help="Stable lesson/video key for cache manifests")
    parser.add_argument("--run-id", default="", help="Optional run identifier for lesson reporting")
    parser.add_argument("--shared-root", type=Path, help="Shared course cache root (.course_work/<course-slug>/shared)")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Arabic neural voice")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="MLX Whisper model")
    parser.add_argument("--source-language", default="en", help="Source language")
    parser.add_argument(
        "--deepl-api-key",
        default=os.environ.get("DEEPL_API_KEY"),
        help="DeepL API key (or set DEEPL_API_KEY)",
    )
    parser.add_argument(
        "--deepl-api-url",
        default=os.environ.get("DEEPL_API_URL"),
        help="DeepL endpoint (auto-detected from API key when omitted)",
    )
    parser.add_argument(
        "--translation-provider",
        choices=("auto", "deepl", "google"),
        default=DEFAULT_TRANSLATION_PROVIDER,
        help="Translation provider selection: prefer DeepL when available, or use Google Translate automatically",
    )
    parser.add_argument(
        "--review-first",
        action="store_true",
        help="Stop after generating Arabic subtitles so you can review them",
    )
    parser.add_argument(
        "--use-reviewed-arabic",
        action="store_true",
        help="Use outputs/arabic_reviewed.srt if present instead of machine translation",
    )
    parser.add_argument(
        "--tts-strategy",
        choices=("chunked", "timed"),
        default=DEFAULT_TTS_STRATEGY,
        help="Use chunked narration for faster batch dubbing or timed segments for tighter sync",
    )
    parser.add_argument(
        "--narration-rate",
        default=DEFAULT_NARRATION_RATE,
        help="Base narration speaking rate for chunked TTS, e.g. -6%%",
    )
    parser.add_argument(
        "--timed-rate",
        default=DEFAULT_TIMED_RATE,
        help="Fixed speaking rate for timed TTS, e.g. +0%% (no speed-up)",
    )
    parser.add_argument(
        "--video-slowdown",
        type=float,
        default=DEFAULT_VIDEO_SLOWDOWN,
        help="Stretch the final video timeline by this factor to create more room for speech (1.0 keeps original speed)",
    )
    parser.add_argument(
        "--max-dub-speed",
        type=float,
        default=DEFAULT_MAX_DUB_SPEED,
        help="Maximum allowed dub speedup factor before the run fails and asks for more video room",
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
        help="Number of concurrent TTS requests",
    )
    parser.add_argument(
        "--chunk-max-chars",
        type=int,
        default=DEFAULT_CHUNK_MAX_CHARS,
        help="Maximum characters per chunked TTS request",
    )
    parser.add_argument(
        "--timed-max-late-drift-ms",
        type=int,
        default=DEFAULT_TIMED_MAX_LATE_DRIFT_MS,
        help="Maximum local timing drift allowed in timed mode before forcing a segment earlier",
    )
    parser.add_argument(
        "--disable-timed-compression",
        action="store_true",
        help="Never speed-compress timed segments; allow local drift instead",
    )
    parser.add_argument(
        "--timed-collision-max-speed",
        type=float,
        default=DEFAULT_TIMED_COLLISION_MAX_SPEED,
        help="Maximum speed-up allowed for selective timed collision fitting",
    )
    parser.add_argument(
        "--quality-profile",
        choices=("natural", "manual"),
        default=DEFAULT_QUALITY_PROFILE,
        help="Natural auto-fit profile or manual-only timing values",
    )
    parser.add_argument(
        "--spoken-style",
        choices=("egyptian",),
        default=DEFAULT_SPOKEN_STYLE,
        help="Spoken Arabic rewrite style applied to machine-translated subtitles",
    )
    parser.add_argument(
        "--disable-spoken-rewrite",
        action="store_true",
        help="Skip the spoken Arabic rewrite pass and keep raw machine-translated subtitles",
    )
    parser.add_argument(
        "--strict-translation-qa",
        action="store_true",
        help="Fail the lesson when glossary misses or leaked English tokens are detected",
    )
    args = parser.parse_args()
    if args.video_slowdown <= 0:
        raise RuntimeError("--video-slowdown must be greater than 0.")
    if args.max_dub_speed < 1.0:
        raise RuntimeError("--max-dub-speed must be at least 1.0.")
    if args.lesson_video_count < 1:
        raise RuntimeError("--lesson-video-count must be at least 1.")
    if args.video_position < 1:
        raise RuntimeError("--video-position must be at least 1.")
    if args.video_position > args.lesson_video_count:
        raise RuntimeError("--video-position cannot be greater than --lesson-video-count.")

    ensure_tool("ffmpeg")
    resolved_translation_provider = choose_translation_provider(
        args.translation_provider,
        args.deepl_api_key,
    )
    resolved_deepl_url = ""
    if resolved_translation_provider == "deepl":
        resolved_deepl_url = resolve_deepl_url(args.deepl_api_key or "", args.deepl_api_url)
        validate_deepl_credentials(args.deepl_api_key or "", resolved_deepl_url)

    input_video = args.input.resolve()
    project_dir, final_video = resolve_workspace_layout(args, input_video)
    args.project_dir = project_dir
    inputs_dir, outputs_dir, work_dir = ensure_workspace(project_dir)
    source_video = maybe_copy_input(input_video, inputs_dir)

    source_audio_input: Path | None = None
    if args.audio_input:
        source_audio_input = maybe_copy_input(args.audio_input.resolve(), inputs_dir)
    else:
        source_audio_input = find_sidecar_audio(source_video)

    extracted_audio = work_dir / "source_audio.wav"
    english_srt = outputs_dir / "english.srt"
    arabic_srt = outputs_dir / "arabic_auto.srt"
    reviewed_arabic_srt = outputs_dir / "arabic_reviewed.srt"
    dubbed_audio = outputs_dir / "arabic_dub.wav"
    transcript_json = outputs_dir / "segments.json"
    raw_narration_mp3 = outputs_dir / "arabic_full_narration.mp3"
    raw_narration_wav = outputs_dir / "arabic_full_narration.wav"
    narration_script_path = outputs_dir / "arabic_full_script.txt"
    diagnostics_json = outputs_dir / "quality_diagnostics.json"

    glossary = load_project_glossary(project_dir)
    shared_root, cache_store, course_slug, video_key, glossary_hash, current_manifest = shared_stage_manifest_identity(
        args,
        source_video,
        source_audio_input,
        glossary,
    )
    shared_tts_cache_dir = shared_root / "timed_tts_cache"
    previous_manifest = cache_store.get_stage_manifest(course_slug, video_key)
    previous_report = read_json(lesson_report_path(project_dir), {})
    refresh_stages = stages_requiring_refresh(previous_manifest, current_manifest)
    preexisting_english_seed = read_srt(english_srt) if english_srt.exists() and not transcript_json.exists() else []
    remove_stage_outputs(
        outputs_dir,
        work_dir,
        refresh_stages,
        strict_translation_qa=args.strict_translation_qa,
    )

    stage_cache = {
        stage_name: stage_name not in refresh_stages
        for stage_name in (STAGE_TRANSCRIPTION, STAGE_TRANSLATION, STAGE_SYNTHESIS, STAGE_MIX)
    }
    timings = {stage_name: 0.0 for stage_name in ("transcription", "translation", "synthesis", "mix", "total")}
    report_path = lesson_report_path(project_dir)
    lesson_started_at = utc_timestamp()
    lesson_started_perf = time.perf_counter()
    translation_stats = TranslationRunStats(provider=resolved_translation_provider)
    cleanup_changes: list[dict[str, Any]] = list(previous_report.get("cleanup_changes", []))
    density_rewrites: list[dict[str, Any]] = list(previous_report.get("density_rewrites", []))
    compression_events: list[dict[str, Any]] = list(previous_report.get("compression_events", []))
    report_payload: dict[str, Any] = {
        "report_type": "lesson",
        "course_slug": course_slug,
        "run_id": args.run_id or "",
        "video_key": video_key,
        "lesson_title": args.lesson_title or source_video.stem,
        "lesson_position": args.lesson_position,
        "video_position": args.video_position,
        "paths": {
            "project_dir": str(project_dir),
            "shared_root": str(shared_root),
            "source_video": str(source_video),
            "source_audio": str(source_audio_input) if source_audio_input else "",
            "final_video": str(final_video),
            "lesson_report": str(report_path),
        },
        "cache_summary": {
            "stage_hits": [stage_name for stage_name, hit in stage_cache.items() if hit],
            "stages_refreshed": sorted(refresh_stages),
        },
        "started_at": lesson_started_at,
        "status": "running",
    }

    if not refresh_stages and final_video.exists():
        report_payload["status"] = "skipped_cached"
        report_payload["completed_at"] = utc_timestamp()
        report_payload["timings"] = timings
        write_lesson_report(report_path, report_payload)
        print(f"Done. Final video: {final_video}")
        return

    english_segments: list[Segment] = []
    arabic_segments: list[Segment] = []
    diagnostics: dict[str, Any] = {}

    try:
        use_reviewed_arabic = args.use_reviewed_arabic and reviewed_arabic_srt.exists()
        transcription_started = time.perf_counter()
        if english_srt.exists() and transcript_json.exists() and STAGE_TRANSCRIPTION not in refresh_stages:
            print("Reusing existing English subtitles...")
            english_segments = read_srt(english_srt)
        elif transcript_json.exists() and STAGE_TRANSCRIPTION not in refresh_stages:
            print("Reusing existing transcript segments...")
            english_segments = read_segments_json(transcript_json)
            write_srt(english_srt, english_segments)
        else:
            if preexisting_english_seed:
                print("Cleaning existing English subtitles...")
                english_segments = preexisting_english_seed
            else:
                if extracted_audio.exists():
                    print("Reusing extracted source audio...")
                else:
                    print("Extracting source audio...")
                    extract_audio(source_audio_input or source_video, extracted_audio)

                print("Transcribing with MLX Whisper...")
                english_segments = transcribe_audio(extracted_audio, args.model, args.source_language)
            english_segments, cleanup_changes = cleanup_english_segments(english_segments)
            write_srt(english_srt, english_segments)
            write_segments_json(transcript_json, english_segments)
        timings["transcription"] = round(time.perf_counter() - transcription_started, 3)

        translation_started = time.perf_counter()
        if use_reviewed_arabic:
            print("Using reviewed Arabic subtitles...")
            arabic_segments = read_srt(reviewed_arabic_srt)
            translation_stats = TranslationRunStats(provider="reviewed")
        elif arabic_srt.exists() and STAGE_TRANSLATION not in refresh_stages:
            print("Reusing existing Arabic subtitles...")
            arabic_segments = read_srt(arabic_srt)
            translation_stats = TranslationRunStats(
                provider=str(previous_report.get("translation", {}).get("provider", resolved_translation_provider)),
                cache_hits=int(previous_report.get("translation", {}).get("cache_hits", 0)),
                cache_misses=int(previous_report.get("translation", {}).get("cache_misses", 0)),
                glossary_hits=int(previous_report.get("translation", {}).get("glossary_hits", 0)),
                glossary_misses=int(previous_report.get("translation", {}).get("glossary_misses", 0)),
            )
        else:
            arabic_segments, translation_stats = translate_segments(
                english_segments,
                glossary,
                args.deepl_api_key,
                resolved_deepl_url,
                args.translation_provider,
                cache_store=cache_store,
                source_language=args.source_language,
                target_language="ar",
                glossary_hash=glossary_hash,
                rewrite_version=SPOKEN_REWRITE_VERSION,
                spoken_style=args.spoken_style,
            )
            if not args.disable_spoken_rewrite:
                arabic_segments = rewrite_spoken_arabic_segments(arabic_segments, args.spoken_style)
            arabic_segments, density_rewrites = auto_rewrite_dense_arabic_segments(arabic_segments)
            write_srt(arabic_srt, arabic_segments)
        timings["translation"] = round(time.perf_counter() - translation_started, 3)

        if args.review_first and not use_reviewed_arabic:
            report_payload["status"] = "review_required"
            report_payload["completed_at"] = utc_timestamp()
            report_payload["translation"] = {
                "provider": translation_stats.provider,
                "cache_hits": translation_stats.cache_hits,
                "cache_misses": translation_stats.cache_misses,
                "glossary_hits": translation_stats.glossary_hits,
                "glossary_misses": translation_stats.glossary_misses,
                "hit_rate": translation_stats.hit_rate,
            }
            report_payload["cleanup_changes"] = cleanup_changes
            report_payload["density_rewrites"] = density_rewrites
            report_payload["timings"] = timings
            write_lesson_report(report_path, report_payload)
            print(f"Arabic draft ready for review: {arabic_srt}")
            print(f"Edit and save reviewed subtitles here: {reviewed_arabic_srt}")
            return

        effective_profile = quality_profile_for_segments(
            arabic_segments,
            args.video_slowdown,
            args.timed_max_late_drift_ms,
            args.timed_collision_max_speed,
            args.quality_profile,
        )
        total_duration_ms = probe_duration_ms(source_video)
        arabic_segments_for_mix = arabic_segments
        effective_video_slowdown = effective_profile.video_slowdown
        if not math.isclose(effective_video_slowdown, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            total_duration_ms = ms_from_seconds((total_duration_ms / 1000) * effective_video_slowdown)
            arabic_segments_for_mix = scale_segments_timing(arabic_segments_for_mix, effective_video_slowdown)
        print(
            f"Quality profile: {effective_profile.density_label} density "
            f"(score={effective_profile.density_score}), "
            f"slowdown={effective_profile.video_slowdown:.2f}, "
            f"late_drift={effective_profile.timed_max_late_drift_ms}, "
            f"collision_max={effective_profile.timed_collision_max_speed:.2f}"
        )

        synthesis_started = time.perf_counter()
        if STAGE_SYNTHESIS in refresh_stages or not dubbed_audio.exists():
            print(f"Synthesizing Arabic dub track with {args.tts_strategy} TTS...")
            if args.tts_strategy == "timed":
                compression_events = build_timed_dub_audio(
                    arabic_segments_for_mix,
                    args.voice,
                    total_duration_ms,
                    dubbed_audio,
                    args.timed_rate,
                    shared_tts_cache_dir,
                    work_dir,
                    args.tts_concurrency,
                    not args.disable_timed_compression,
                    effective_profile.timed_max_late_drift_ms,
                    effective_profile.timed_collision_max_speed,
                )
            else:
                build_chunked_dub_audio(
                    arabic_segments_for_mix,
                    args.voice,
                    total_duration_ms,
                    dubbed_audio,
                    work_dir,
                    narration_script_path,
                    raw_narration_mp3,
                    raw_narration_wav,
                    args.narration_rate,
                    args.chunk_max_chars,
                    args.tts_concurrency,
                    args.max_dub_speed,
                )
                compression_events = []
        timings["synthesis"] = round(time.perf_counter() - synthesis_started, 3)

        diagnostics = build_quality_diagnostics(
            english_segments,
            arabic_segments,
            glossary,
            compression_events,
            effective_profile,
            translation_stats=translation_stats,
            strict_translation_qa=args.strict_translation_qa,
            cleanup_changes=cleanup_changes,
            density_rewrites=density_rewrites,
        )
        diagnostics["qa"]["blocking_enabled"] = args.strict_translation_qa
        translation_stats.glossary_hits = int(diagnostics["qa"]["glossary_hits"])
        translation_stats.glossary_misses = int(diagnostics["qa"]["glossary_misses"])
        write_json(diagnostics_json, diagnostics)
        if diagnostics["needs_attention"]:
            print(f"Quality diagnostics flagged issues: {diagnostics_json}")
        if args.strict_translation_qa and diagnostics["blocking_issues"]:
            raise RuntimeError(
                "Strict translation QA failed: " + ", ".join(diagnostics["blocking_issues"])
            )

        mix_started = time.perf_counter()
        if STAGE_MIX in refresh_stages or not final_video.exists():
            print("Muxing final dubbed video...")
            mux_final_video(
                source_video,
                source_audio_input,
                dubbed_audio,
                final_video,
                args.bg_gain_db,
                args.dub_gain,
                effective_video_slowdown,
            )
        timings["mix"] = round(time.perf_counter() - mix_started, 3)
        cleanup_local_final_outputs(outputs_dir, keep_path=final_video)

        current_manifest["updated_at"] = utc_timestamp()
        current_manifest["paths"] = {
            "project_dir": str(project_dir),
            "lesson_report": str(report_path),
            "diagnostics": str(diagnostics_json),
            "final_video": str(final_video),
        }
        cache_store.put_stage_manifest(course_slug, video_key, current_manifest)

        report_payload["status"] = "success"
        report_payload["completed_at"] = utc_timestamp()
        report_payload["timings"] = timings
        report_payload["effective_profile"] = asdict(effective_profile)
        report_payload["compression_events"] = compression_events
        report_payload["cleanup_changes"] = cleanup_changes
        report_payload["density_rewrites"] = density_rewrites
        report_payload["translation"] = {
            "provider": translation_stats.provider,
            "cache_hits": translation_stats.cache_hits,
            "cache_misses": translation_stats.cache_misses,
            "glossary_hits": translation_stats.glossary_hits,
            "glossary_misses": translation_stats.glossary_misses,
            "hit_rate": translation_stats.hit_rate,
        }
        report_payload["qa"] = diagnostics.get("qa", {})
        report_payload["diagnostics_path"] = str(diagnostics_json)
        report_payload["final_video"] = str(final_video)
    except Exception as exc:
        report_payload["status"] = "failed"
        report_payload["completed_at"] = utc_timestamp()
        report_payload["timings"] = timings
        report_payload["error"] = str(exc)
        if diagnostics:
            report_payload["qa"] = diagnostics.get("qa", {})
        write_lesson_report(report_path, report_payload)
        raise

    timings["total"] = round(time.perf_counter() - lesson_started_perf, 3)
    report_payload["timings"] = timings
    write_lesson_report(report_path, report_payload)
    print(f"Done. Final video: {final_video}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI safeguard
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
