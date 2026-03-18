# Dub

End-to-end Arabic dubbing workspace for course videos.

## What it does

- Downloads a full course from the supported platform with `yt-dlp`
- Reuses your logged-in browser cookies for access
- Transcribes English audio with `mlx-whisper`
- Translates to Arabic with DeepL when available, or Google Translate automatically
- Auto-selects a course voice from Arabic neural candidates and persists it under `.course_work/`
- Muxes the Arabic dub back into each lesson video

## Current default quality/speed strategy

- Translation: `auto` (`DeepL -> Google fallback`, or Google when no DeepL key is set)
- Voice: automatic course-level selection, starting from Egyptian voices
- Dubbing mode: `timed`
- Timed speaking rate: `+0%` (no speed-up)
- Quality profile: `natural`
- Polished pacing defaults: `--video-slowdown 1.10`, `--timed-max-late-drift-ms 1700`, `--timed-collision-max-speed 1.22`
- Mix defaults: speech-priority ducking with dubbed voice sidechained over the source audio

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

- optionally `DEEPL_API_KEY`
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

If you want to make the natural profile explicit:

```bash
python course_pipeline.py \
  --course-url "PASTE_COURSE_URL_HERE" \
  --browser chrome \
  --quality-profile natural \
  --timed-rate "+0%" \
  --video-slowdown 1.10 \
  --timed-max-late-drift-ms 1700 \
  --timed-collision-max-speed 1.22
```

This will create a per-course folder under:

```text
courses/<course-slug>/
```

Inside it, you will get final videos directly (flat output):

- `<lesson-position>-<lesson-title>.arabic-dub.mp4`
- `<lesson-position>-<lesson-title>.v01.arabic-dub.mp4` for multi-video lessons

Working folders are written under `.course_work/<course-slug>/` and are deleted after each successful video by default.
Use `--keep-work` if you want to keep those intermediate files for debugging; final MP4s still stay in `courses/<course-slug>/`.
Each video work folder now also includes `quality_diagnostics.json`, and the selected course voice is saved in `.course_work/<course-slug>/voice_selection.json`.

## Single-video pipeline

Run one video through the dubber with:

```bash
cd /Users/ahmedyasser/Documents/Dub
source .venv/bin/activate
python dub.py \
  --input "/path/to/video.mp4" \
  --course-slug "my-ad-hoc-run" \
  --lesson-position 1 \
  --lesson-title "My Lesson" \
  --voice ar-EG-ShakirNeural \
  --tts-strategy timed \
  --translation-provider auto \
  --quality-profile natural
```

That writes the final MP4 to `courses/my-ad-hoc-run/01-my-lesson.arabic-dub.mp4` and keeps subtitles, WAVs, caches, and review files under `.course_work/my-ad-hoc-run/`.

If the source has separate audio:

```bash
python dub.py \
  --input "/path/to/video.mp4" \
  --audio-input "/path/to/audio.mp4" \
  --voice ar-EG-ShakirNeural \
  --tts-strategy timed \
  --translation-provider auto \
  --quality-profile natural
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
  --disable-auto-select-voice \
  --quality-profile manual \
  --video-slowdown 1.0 \
  --tts-concurrency 4 \
  --yt-concurrent-fragments 8
```

## Files

- `course_pipeline.py`: course scraper/downloader/dubber entrypoint
- `dub.py`: reusable per-video dubbing pipeline
- `tts_only.py`: direct Arabic narration generation from script text
- `voice_samples.py`: quick voice comparison generator
- `glossary.csv`: post-translation terminology replacements
- `orchestrate.py`: terminal supervisor that drives a local Codex worker loop

## Terminal Orchestrator

Use the local Codex CLI as a worker agent and let the supervisor verify and retry automatically:

```bash
cd /Users/ahmedyasser/Documents/Dub
source .venv/bin/activate
python orchestrate.py run \
  --goal "Make course dubbing work for any supported input course URL" \
  --verify "python -m unittest tests.test_universal_extractor tests.test_orchestrate" \
  --verify "python course_pipeline.py --help"
```

Useful follow-up commands:

```bash
python orchestrate.py status
python orchestrate.py logs --last 1
python orchestrate.py resume --force
```

The supervisor stores its own artifacts in `.orchestrator/runs/` and keeps course output folders unchanged.
