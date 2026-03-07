# AGENTS.md

## Goal
Keep this workspace predictable for course-level Arabic dubbing runs:
- Course folders should stay clean and contain final MP4 outputs only.
- Intermediate artifacts should live outside course folders.
- Final quality should default to subtitle-timed dubbing, not the fast chunked mode.
- Final delivery should prioritize natural spoken Arabic over raw throughput.

## Project defaults
- Course pipeline entrypoint: `course_pipeline.py`
- Per-video pipeline entrypoint: `dub.py`
- Default course TTS strategy: `timed`
- Timed speaking rate default: `+0%` (no speed-up)
- Translation provider: DeepL via `DEEPL_API_KEY`
- Translation fallback: Google Translate when DeepL is rate-limited or quota-limited
- Spoken rewrite default: phrase-first spoken Arabic rewrite
- Spoken style default: Egyptian spoken Arabic
- Preferred polished pacing recipe:
  - normal timed speech by default
  - shorten only dense Arabic lines when needed
  - allow mild selective collision speed-up, not global speed-up
  - allow some local drift before fitting
  - if a lesson is still too dense, use a small global video slowdown
- Current polished anti-swelling recipe:
  - `--timed-rate +0%`
  - `--video-slowdown 1.40`
  - `--timed-max-late-drift-ms 2200`
  - `--timed-collision-max-speed 1.38`
  - `--use-reviewed-arabic`

## Output structure (required)
- Single-video lesson finals: `courses/<course-slug>/<lesson-position>-<lesson-title>.arabic-dub.mp4`
- Multi-video lesson finals: `courses/<course-slug>/<lesson-position>-<lesson-title>.v01.arabic-dub.mp4`
- Working files and run metadata: `.course_work/<course-slug>/`

Do not write `manifest/` or `lessons/` inside `courses/<course-slug>/`.

## Runtime rules
- Load config from `.env` in repo root.
- Keep `.env` out of source control.
- If work folders are needed for debugging, use `--keep-work`; otherwise keep cleanup enabled.
- For final delivery, prefer non-sped timed output and avoid chunked mode unless explicitly requested.
- Do not collapse multi-video lessons into a single final filename.
- If old ambiguous single-file outputs exist for multi-video lessons, move them into `.course_work/<course-slug>/legacy_migrated/final_outputs/` before writing new finals.
- For dense lessons, prefer rewriting dense Arabic lines before increasing speed caps.
- Prefer selective timed fitting over global dub speed-up.
- If a user wants a separate tuning pass, use `--course-slug-suffix` so outputs go into a separate course folder instead of overwriting the base run.
- Keep course folders clean: final MP4s only. Put manifests, review files, caches, and debug artifacts under `.course_work/<course-slug>/`.

## Common commands
```bash
source .venv/bin/activate

# Full course run (default: timed, clean output folders)
python course_pipeline.py --course-url "<COURSE_URL>" --browser chrome

# Keep intermediate working files for debugging
python course_pipeline.py --course-url "<COURSE_URL>" --browser chrome --keep-work

# Polished natural-sync style run in a separate variant folder
python course_pipeline.py \
  --course-url "<COURSE_URL>" \
  --browser chrome \
  --course-slug-suffix natural-sync \
  --timed-rate "+0%" \
  --video-slowdown 1.40 \
  --timed-max-late-drift-ms 2200 \
  --timed-collision-max-speed 1.38 \
  --use-reviewed-arabic \
  --keep-work
```

## Migration guideline
If legacy course folders contain `lessons/` and `manifest/`:
1. Copy final desired `*.arabic-dub.mp4` files into `courses/<course-slug>/`.
2. Move old `lessons/` and `manifest/` trees into `.course_work/<course-slug>/legacy_migrated/`.
3. Leave only final MP4 files in `courses/<course-slug>/`.
