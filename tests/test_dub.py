import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.modules.setdefault("edge_tts", SimpleNamespace(Communicate=object))
sys.modules.setdefault("pysrt", SimpleNamespace())
sys.modules.setdefault(
    "deep_translator",
    SimpleNamespace(GoogleTranslator=object),
)

from dub import (
    DEFAULT_SPOKEN_STYLE,
    CacheStore,
    QualityProfile,
    Segment,
    build_default_workspace_layout,
    build_final_video_name,
    build_processing_manifest,
    build_quality_diagnostics,
    choose_translation_provider,
    quality_profile_for_segments,
    translate_segments,
    rewrite_spoken_arabic_text,
    select_best_voice,
)
from pipeline_support import STAGE_MIX, STAGE_SYNTHESIS, STAGE_TRANSLATION, stages_requiring_refresh


class DubHelpersTests(unittest.TestCase):
    def test_build_final_video_name_uses_lesson_position_prefix(self):
        self.assertEqual(
            build_final_video_name(1, "Welcome Lesson"),
            "01-welcome-lesson.arabic-dub.mp4",
        )
        self.assertEqual(
            build_final_video_name(12, "Welcome Lesson", video_position=1, lesson_video_count=2),
            "12-welcome-lesson.v01.arabic-dub.mp4",
        )

    def test_build_default_workspace_layout_uses_courses_and_course_work(self):
        temp_root = Path(tempfile.mkdtemp(prefix="dub-layout-test-"))
        layout = build_default_workspace_layout(
            temp_root,
            Path("/tmp/My Lesson.mp4"),
            course_slug="demo-course",
            lesson_position=3,
            lesson_title="My Lesson",
        )
        self.assertEqual(
            layout.project_dir,
            temp_root / ".course_work" / "demo-course" / "03-my-lesson" / "video_01",
        )
        self.assertEqual(
            layout.final_video,
            temp_root / "courses" / "demo-course" / "03-my-lesson.arabic-dub.mp4",
        )

    def test_rewrite_spoken_arabic_text_prefers_egyptian_phrasing(self):
        rewritten = rewrite_spoken_arabic_text("في هذا الدرس سنقوم بذلك الآن أيضًا")
        self.assertEqual(rewritten, "في الدرس ده هنعمل بكده دلوقتي كمان")

    def test_choose_translation_provider_uses_google_without_key(self):
        self.assertEqual(choose_translation_provider("auto", None), "google")
        self.assertEqual(choose_translation_provider("google", "abc"), "google")

    def test_choose_translation_provider_requires_key_for_deepl(self):
        with self.assertRaises(RuntimeError):
            choose_translation_provider("deepl", None)

    def test_quality_profile_scales_dense_lessons(self):
        segments = [
            Segment(
                index=1,
                start_ms=0,
                end_ms=1200,
                text="هذا سطر عربي طويل جدا للاختبار ومعرفة كيف يتم حساب كثافة الكلام بشكل تلقائي",
            )
        ]
        profile = quality_profile_for_segments(segments, 1.0, 700, 1.12, "natural")
        self.assertIn(profile.density_label, {"dense", "extreme"})
        self.assertGreaterEqual(profile.video_slowdown, 1.15)
        self.assertGreaterEqual(profile.timed_max_late_drift_ms, 1800)
        self.assertGreaterEqual(profile.timed_collision_max_speed, 1.24)

    def test_build_quality_diagnostics_flags_attention_items(self):
        english = [Segment(index=1, start_ms=0, end_ms=1000, text="Open the Dashboard now")]
        arabic = [Segment(index=1, start_ms=0, end_ms=1000, text="افتح Dashboard الآن بسرعة جدا")]
        profile = QualityProfile(
            density_score=16.2,
            density_label="dense",
            video_slowdown=1.4,
            timed_max_late_drift_ms=2200,
            timed_collision_max_speed=1.38,
        )
        diagnostics = build_quality_diagnostics(
            english,
            arabic,
            {"Dashboard": "لوحة التحكم"},
            [{"index": 1, "speed_ratio": 1.25}],
            profile,
        )
        self.assertTrue(diagnostics["needs_attention"])
        self.assertEqual(diagnostics["summary"]["heavy_compression_segments"], 1)
        self.assertEqual(diagnostics["summary"]["untranslated_segments"], 1)
        self.assertEqual(diagnostics["summary"]["glossary_misses"], 1)

    def test_select_best_voice_prefers_egyptian_candidate(self):
        temp_dir = Path(tempfile.mkdtemp(prefix="voice-select-test-"))

        async def fake_render(text, voices, out_dir):
            rendered = []
            for voice in voices:
                path = out_dir / f"{voice}.mp3"
                path.write_bytes(b"fake")
                rendered.append((voice, path))
            return rendered

        duration_map = {
            "ar-EG-ShakirNeural.mp3": 5000,
            "ar-EG-SalmaNeural.mp3": 5200,
            "ar-SA-HamedNeural.mp3": 4900,
        }

        def fake_probe(path):
            return duration_map[path.name]

        with patch("dub.render_voice_candidate_samples", side_effect=fake_render), patch(
            "dub.probe_duration_ms",
            side_effect=fake_probe,
        ):
            result = select_best_voice(
                "اهلا بيكم في الدرس",
                temp_dir,
                ["ar-EG-ShakirNeural", "ar-EG-SalmaNeural", "ar-SA-HamedNeural"],
            )

        self.assertEqual(result.selected_voice, "ar-EG-ShakirNeural")
        self.assertEqual(result.ranking[0]["voice"], "ar-EG-ShakirNeural")

    def test_stages_requiring_refresh_marks_downstream_stages(self):
        previous = build_processing_manifest(
            course_slug="demo",
            video_key="video-1",
            source_hash="src",
            audio_hash="",
            transcription_config={"model": "a"},
            translation_config={"provider": "google"},
            synthesis_config={"voice": "x"},
            mix_config={"dub_gain": 1},
        )
        current = build_processing_manifest(
            course_slug="demo",
            video_key="video-1",
            source_hash="src",
            audio_hash="",
            transcription_config={"model": "a"},
            translation_config={"provider": "deepl"},
            synthesis_config={"voice": "x"},
            mix_config={"dub_gain": 1},
        )
        self.assertEqual(
            stages_requiring_refresh(previous, current),
            {STAGE_TRANSLATION, STAGE_SYNTHESIS, STAGE_MIX},
        )

    def test_translate_segments_uses_translation_memory_cache(self):
        temp_dir = Path(tempfile.mkdtemp(prefix="translation-cache-test-"))
        store = CacheStore(temp_dir / "cache.db")
        segments = [Segment(index=1, start_ms=0, end_ms=500, text="Hello world")]

        class FakeTranslator:
            calls = 0

            def __init__(self, source, target):
                self.source = source
                self.target = target

            def translate(self, text):
                FakeTranslator.calls += 1
                return "اهلا بالعالم"

        with patch("dub.GoogleTranslator", FakeTranslator):
            first, first_stats = translate_segments(
                segments,
                {},
                None,
                "",
                "google",
                cache_store=store,
                source_language="en",
                target_language="ar",
                glossary_hash="g1",
                rewrite_version="v1",
                spoken_style=DEFAULT_SPOKEN_STYLE,
            )
            second, second_stats = translate_segments(
                segments,
                {},
                None,
                "",
                "google",
                cache_store=store,
                source_language="en",
                target_language="ar",
                glossary_hash="g1",
                rewrite_version="v1",
                spoken_style=DEFAULT_SPOKEN_STYLE,
            )

        self.assertEqual(first[0].text, "اهلا بالعالم")
        self.assertEqual(second[0].text, "اهلا بالعالم")
        self.assertEqual(FakeTranslator.calls, 1)
        self.assertEqual(first_stats.cache_misses, 1)
        self.assertEqual(second_stats.cache_hits, 1)


if __name__ == "__main__":
    unittest.main()
