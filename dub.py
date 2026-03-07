#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import edge_tts
import pysrt
import requests
from deep_translator import GoogleTranslator

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
TIMED_MIX_BATCH_SIZE = 48
TIMED_IDEAL_GAP_MS = 70
TIMED_MIN_GAP_MS = 20
TIMED_FIT_TAIL_PADDING_MS = 10
TIMED_FIT_FADE_OUT_MS = 70
TIMED_MIN_FIT_MS = 120
TIMED_FIT_TRIGGER_MS = 220
TIMED_MAX_LATE_DRIFT_MS = 700
DEFAULT_VIDEO_SLOWDOWN = 1.0
DEFAULT_MAX_DUB_SPEED = 1.5
DEFAULT_TIMED_COLLISION_MAX_SPEED = 1.12
SAMPLE_RATE = 24000
DEEPL_DEFAULT_URL = "https://api-free.deepl.com/v2/translate"
DEEPL_PRO_URL = "https://api.deepl.com/v2/translate"
DEEPL_CHUNK_SIZE = 20
PROJECT_ROOT = Path(__file__).resolve().parent
DOTENV_PATH = PROJECT_ROOT / ".env"


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


def translate_segments(
    segments: list[Segment],
    glossary: dict[str, str],
    deepl_api_key: str | None,
    deepl_api_url: str,
) -> list[Segment]:
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

    if not deepl_api_key:
        raise RuntimeError(
            "Missing DeepL API key. Provide --deepl-api-key or set DEEPL_API_KEY."
        )

    translated: list[Segment] = []
    headers = {"Authorization": f"DeepL-Auth-Key {deepl_api_key}"}
    params = {
        "target_lang": "AR",
        "source_lang": "EN",
        "split_sentences": "1",
        "preserve_formatting": "0",
    }

    for start in range(0, len(segments), DEEPL_CHUNK_SIZE):
        chunk = segments[start : start + DEEPL_CHUNK_SIZE]
        data = [("text", segment.text) for segment in chunk]
        try:
            response = requests.post(
                deepl_api_url,
                headers=headers,
                params=params,
                data=data,
                timeout=30,
            )
        except requests.RequestException as exc:
            print(f"DeepL request failed for chunk {start // DEEPL_CHUNK_SIZE + 1}; using Google Translate fallback.")
            translated.extend(google_translate_chunk(chunk))
            continue
        if response.status_code != 200:
            if response.status_code in {429, 456}:
                print(
                    f"DeepL rate/quota limit hit ({response.status_code}) for chunk "
                    f"{start // DEEPL_CHUNK_SIZE + 1}; using Google Translate fallback."
                )
                translated.extend(google_translate_chunk(chunk))
                continue
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


async def synthesize_tts(
    text: str,
    voice: str,
    rate: str,
    out_path: Path,
    pitch: str = "+0Hz",
) -> None:
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch,
    )
    await communicate.save(str(out_path))


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
    tts_concurrency: int,
    allow_timed_compression: bool,
    timed_max_late_drift_ms: int,
    timed_collision_max_speed: float,
) -> None:
    duration_cache: dict[Path, int] = {}

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
        return

    fitted_cache_dir = cache_dir.parent / "timed_fitted_cache"
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
    stem_dir = cache_dir.parent / "timed_mix_stems"
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
        bg_input_label = "1:a"
        dub_input_label = "2:a"
        if not math.isclose(video_slowdown, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            filter_complex = (
                f"[0:v]setpts={video_slowdown:.6f}*PTS[vout];"
                f"[1:a]{audio_slowdown_filter(video_slowdown)},volume={bg_gain_db}dB[bg];"
                f"[2:a]volume={dub_gain},acompressor=threshold=-20dB:ratio=2.5:attack=5:release=50:makeup=4,alimiter=limit=0.97[dub];"
                "[bg][dub]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )
            video_map = "[vout]"
            video_codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
        else:
            filter_complex = (
                f"[1:a]volume={bg_gain_db}dB[bg];"
                f"[2:a]volume={dub_gain},acompressor=threshold=-20dB:ratio=2.5:attack=5:release=50:makeup=4,alimiter=limit=0.97[dub];"
                "[bg][dub]amix=inputs=2:duration=first:dropout_transition=2[aout]"
            )
            video_map = "0:v"
            video_codec_args = ["-c:v", "copy"]
    else:
        if not math.isclose(video_slowdown, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            filter_complex = (
                f"[0:v]setpts={video_slowdown:.6f}*PTS[vout];"
                f"[1:a]volume={dub_gain},acompressor=threshold=-20dB:ratio=2.5:attack=5:release=50:makeup=4,alimiter=limit=0.97[aout]"
            )
            video_map = "[vout]"
            video_codec_args = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18"]
        else:
            filter_complex = f"[1:a]volume={dub_gain},acompressor=threshold=-20dB:ratio=2.5:attack=5:release=50:makeup=4,alimiter=limit=0.97[aout]"
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


def main() -> None:
    parser = argparse.ArgumentParser(description="High-quality local Arabic dubbing pipeline")
    parser.add_argument("--input", required=True, type=Path, help="Path to source video")
    parser.add_argument("--audio-input", type=Path, help="Optional separate source audio file")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Dub project directory",
    )
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
        default=1.0,
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
        default=TIMED_MAX_LATE_DRIFT_MS,
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
    args = parser.parse_args()
    if args.video_slowdown <= 0:
        raise RuntimeError("--video-slowdown must be greater than 0.")
    if args.max_dub_speed < 1.0:
        raise RuntimeError("--max-dub-speed must be at least 1.0.")

    ensure_tool("ffmpeg")
    if not args.deepl_api_key:
        raise RuntimeError("Missing DeepL API key. Set DEEPL_API_KEY or pass --deepl-api-key.")
    resolved_deepl_url = resolve_deepl_url(args.deepl_api_key, args.deepl_api_url)
    validate_deepl_credentials(args.deepl_api_key, resolved_deepl_url)

    project_dir = args.project_dir.resolve()
    input_video = args.input.resolve()
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
    final_video = outputs_dir / f"{source_video.stem}.arabic-dub.mp4"
    transcript_json = outputs_dir / "segments.json"
    raw_narration_mp3 = outputs_dir / "arabic_full_narration.mp3"
    raw_narration_wav = outputs_dir / "arabic_full_narration.wav"
    narration_script_path = outputs_dir / "arabic_full_script.txt"

    glossary = load_glossary(project_dir / "glossary.csv")

    if args.use_reviewed_arabic and reviewed_arabic_srt.exists():
        print("Using reviewed Arabic subtitles...")
        arabic_segments = read_srt(reviewed_arabic_srt)
    else:
        if english_srt.exists():
            print("Reusing existing English subtitles...")
            english_segments = read_srt(english_srt)
            if not transcript_json.exists():
                write_segments_json(transcript_json, english_segments)
        elif transcript_json.exists():
            print("Reusing existing transcript segments...")
            english_segments = read_segments_json(transcript_json)
            write_srt(english_srt, english_segments)
        else:
            if extracted_audio.exists():
                print("Reusing extracted source audio...")
            else:
                print("Extracting source audio...")
                extract_audio(source_audio_input or source_video, extracted_audio)

            print("Transcribing with MLX Whisper...")
            english_segments = transcribe_audio(extracted_audio, args.model, args.source_language)
            write_srt(english_srt, english_segments)
            write_segments_json(transcript_json, english_segments)

        if arabic_srt.exists():
            print("Reusing existing Arabic subtitles...")
            arabic_segments = read_srt(arabic_srt)
        else:
            print("Translating subtitles to Arabic with DeepL...")
            arabic_segments = translate_segments(
                english_segments,
                glossary,
                args.deepl_api_key,
                resolved_deepl_url,
            )
            write_srt(arabic_srt, arabic_segments)
        if args.review_first:
            print(f"Arabic draft ready for review: {arabic_srt}")
            print(f"Edit and save reviewed subtitles here: {reviewed_arabic_srt}")
            return

    total_duration_ms = probe_duration_ms(source_video)
    arabic_segments_for_mix = arabic_segments
    if not math.isclose(args.video_slowdown, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        total_duration_ms = ms_from_seconds((total_duration_ms / 1000) * args.video_slowdown)
        arabic_segments_for_mix = scale_segments_timing(arabic_segments_for_mix, args.video_slowdown)
    print(f"Synthesizing Arabic dub track with {args.tts_strategy} TTS...")
    if args.tts_strategy == "timed":
        build_timed_dub_audio(
            arabic_segments_for_mix,
            args.voice,
            total_duration_ms,
            dubbed_audio,
            args.timed_rate,
            work_dir / "timed_tts_cache",
            args.tts_concurrency,
            not args.disable_timed_compression,
            args.timed_max_late_drift_ms,
            args.timed_collision_max_speed,
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

    print("Muxing final dubbed video...")
    mux_final_video(
        source_video,
        source_audio_input,
        dubbed_audio,
        final_video,
        args.bg_gain_db,
        args.dub_gain,
        args.video_slowdown,
    )
    print(f"Done. Final video: {final_video}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI safeguard
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
