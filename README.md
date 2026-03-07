# Dub

End-to-end Arabic dubbing workspace for course videos.

## What it does

- Downloads a full course from the supported platform with `yt-dlp`
- Reuses your logged-in browser cookies for access
- Transcribes English audio with `mlx-whisper`
- Translates to Arabic with DeepL
- Generates Arabic dub audio with `ar-EG-ShakirNeural`
- Muxes the Arabic dub back into each lesson video

## Current default quality/speed strategy

- Translation: DeepL only
- Voice: `ar-EG-ShakirNeural`
- Dubbing mode: `timed`
- Timed speaking rate: `+0%` (no speed-up)
- Mix defaults: dub gain `2.6`, background `-24dB`

`timed` is now the default for better sync and more natural pacing, with no forced speed-up.

## Setup

```bash
cd /Users/ahmedyasser/Documents/Dub
chmod +x setup.sh dub.py tts_only.py voice_samples.py course_pipeline.py
./setup.sh
source .venv/bin/activate
cp .env.example .env
```

Then edit `.env` and fill in:

- `DEEPL_API_KEY`
- optionally `DEEPL_API_URL` (usually not needed; endpoint is auto-detected from your key)
- optionally `DEFAULT_BROWSER`
- optionally `DEFAULT_ARABIC_VOICE`

## Full-course pipeline

Run the full course workflow with:

```bash
cd /Users/ahmedyasser/Documents/Dub
source .venv/bin/activate
python course_pipeline.py \
  --course-url "PASTE_COURSE_URL_HERE" \
  --browser chrome
```

If you want to enforce exact natural timed speed explicitly:

```bash
python course_pipeline.py \
  --course-url "PASTE_COURSE_URL_HERE" \
  --browser chrome \
  --tts-strategy timed \
  --timed-rate "+0%"
```

This will create a per-course folder under:

```text
courses/<course-slug>/
```

Inside it, you will get final videos directly (flat output):

- `<lesson-title>.arabic-dub.mp4`
- `<another-lesson>.arabic-dub.mp4`

Working folders are written under `.course_work/<course-slug>/` and are deleted after each successful video by default.
Use `--keep-work` if you want to keep those intermediate files for debugging.

## Single-video pipeline

Run one video through the dubber with:

```bash
cd /Users/ahmedyasser/Documents/Dub
source .venv/bin/activate
python dub.py \
  --input "/path/to/video.mp4" \
  --voice ar-EG-ShakirNeural \
  --tts-strategy chunked
```

If the source has separate audio:

```bash
python dub.py \
  --input "/path/to/video.mp4" \
  --audio-input "/path/to/audio.mp4" \
  --voice ar-EG-ShakirNeural \
  --tts-strategy chunked
```

## Review-first workflow

If you want to stop after Arabic subtitle generation:

```bash
python dub.py \
  --input "/path/to/video.mp4" \
  --review-first
```

That creates:

- `outputs/english.srt`
- `outputs/arabic_auto.srt`

Then edit and save:

- `outputs/arabic_reviewed.srt`

And rerun:

```bash
python dub.py \
  --input "/path/to/video.mp4" \
  --use-reviewed-arabic
```

## Voice samples

Generate comparison clips with:

```bash
python voice_samples.py \
  --input "/Users/ahmedyasser/Documents/Dub/arabic_script.txt" \
  --out-dir "/Users/ahmedyasser/Documents/Dub/outputs/voice_samples"
```

## Fastest useful batch settings

```bash
python course_pipeline.py \
  --course-url "PASTE_COURSE_URL_HERE" \
  --browser chrome \
  --tts-strategy chunked \
  --tts-concurrency 4 \
  --yt-concurrent-fragments 8
```

## Files

- `course_pipeline.py`: course scraper/downloader/dubber entrypoint
- `dub.py`: reusable per-video dubbing pipeline
- `tts_only.py`: direct Arabic narration generation from script text
- `voice_samples.py`: quick voice comparison generator
- `glossary.csv`: post-translation terminology replacements
