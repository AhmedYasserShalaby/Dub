#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import subprocess
from pathlib import Path

import edge_tts


DEFAULT_VOICE = "ar-EG-ShakirNeural"
DEFAULT_RATE = "-6%"
DEFAULT_PITCH = "+0Hz"


async def save_tts(text: str, voice: str, rate: str, pitch: str, out_path: Path) -> None:
    communicate = edge_tts.Communicate(
        text=text,
        voice=voice,
        rate=rate,
        pitch=pitch,
    )
    await communicate.save(str(out_path))


def convert_to_wav(src: Path, dst: Path) -> None:
    subprocess.run(
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
        ],
        check=True,
        text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate high-quality Arabic TTS narration")
    parser.add_argument("--input", required=True, type=Path, help="Arabic text file")
    parser.add_argument("--output", required=True, type=Path, help="Output audio path (.mp3 or .wav)")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="Arabic neural voice")
    parser.add_argument("--rate", default=DEFAULT_RATE, help="Speech rate, e.g. -8%% or +5%%")
    parser.add_argument("--pitch", default=DEFAULT_PITCH, help="Pitch, e.g. +0Hz")
    args = parser.parse_args()

    text = args.input.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit("Input text file is empty")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() == ".wav":
        temp_mp3 = output.with_suffix(".tmp.mp3")
        asyncio.run(save_tts(text, args.voice, args.rate, args.pitch, temp_mp3))
        convert_to_wav(temp_mp3, output)
        temp_mp3.unlink(missing_ok=True)
    else:
        asyncio.run(save_tts(text, args.voice, args.rate, args.pitch, output))

    print(f"Created: {output}")


if __name__ == "__main__":
    main()
