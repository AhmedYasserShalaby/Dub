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
- Adaptive pacing default: keep original speed unless auto slowdown is needed
- Max dub speed default: `1.5x`

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
- Treat `--video-slowdown` as an exception for problem lessons, not the default path.
- Prefer the original foot-healing-style timed dub flow unless a user explicitly asks for experimental timing changes.

## Common commands
```bash
source .venv/bin/activate

# Full course run (default: timed, clean output folders)
python course_pipeline.py --course-url "<COURSE_URL>" --browser chrome

# Keep intermediate working files for debugging
python course_pipeline.py --course-url "<COURSE_URL>" --browser chrome --keep-work

# Timed default with manual slowdown only if explicitly requested
python course_pipeline.py \
  --course-url "<COURSE_URL>" \
  --browser chrome \
  --timed-rate "+0%"
```

## Migration guideline
If legacy course folders contain `lessons/` and `manifest/`:
1. Copy final desired `*.arabic-dub.mp4` files into `courses/<course-slug>/`.
2. Move old `lessons/` and `manifest/` trees into `.course_work/<course-slug>/legacy_migrated/`.
3. Leave only final MP4 files in `courses/<course-slug>/`.
