#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import edge_tts


DEFAULT_VOICES = [
    "ar-EG-ShakirNeural",
    "ar-EG-SalmaNeural",
    "ar-SA-HamedNeural",
    "ar-SA-ZariyahNeural",
]


async def render_sample(text: str, voice: str, out_path: Path) -> None:
    communicate = edge_tts.Communicate(text=text, voice=voice, rate="-4%")
    await communicate.save(str(out_path))


async def main_async(text_path: Path, out_dir: Path) -> None:
    text = text_path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit("Sample text file is empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    for voice in DEFAULT_VOICES:
        await render_sample(text, voice, out_dir / f"{voice}.mp3")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Arabic voice comparison samples")
    parser.add_argument("--input", required=True, type=Path, help="Arabic sample text file")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory for sample files")
    args = parser.parse_args()
    asyncio.run(main_async(args.input.resolve(), args.out_dir.resolve()))
    print(f"Created voice samples in: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
