import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from universal_extractor import (
    ExtractionOptions,
    UniversalCourseExtractor,
    detect_platform,
    extract_js_object,
    is_getcourse_lesson_url,
)


class UniversalExtractorTests(unittest.TestCase):
    def make_extractor(self, url: str) -> UniversalCourseExtractor:
        temp_dir = Path(tempfile.mkdtemp(prefix="extractor-test-"))
        options = ExtractionOptions(browser="chrome", debug_dir=temp_dir)
        with patch.object(UniversalCourseExtractor, "_load_cookies", return_value=None):
            extractor = UniversalCourseExtractor(url, options)
        extractor.session = Mock()
        return extractor

    def test_detect_platform(self):
        self.assertEqual(detect_platform("https://course.fp-platform.online/learn/abc"), "fp-platform")
        self.assertEqual(
            detect_platform("https://example.com/pl/teach/control/lesson/view?id=123"),
            "getcourse",
        )
        self.assertEqual(detect_platform("https://example.com/course"), "generic")

    def test_is_getcourse_lesson_url(self):
        self.assertTrue(is_getcourse_lesson_url("https://example.com/pl/teach/control/lesson/view?id=123"))
        self.assertFalse(is_getcourse_lesson_url("https://example.com/pl/teach/control/stream/view?id=123"))

    def test_extract_getcourse_lesson_links(self):
        extractor = self.make_extractor("https://example.com/pl/teach/control/lesson/view?id=1")
        html = """
        <a href="/pl/teach/control/lesson/view?id=10">Lesson 1</a>
        <a href="https://example.com/pl/teach/control/lesson/view?id=11&editMode=0">Lesson 2</a>
        <a href="/other">Other</a>
        """
        links = extractor._extract_getcourse_lesson_links(html)
        self.assertEqual(
            links,
            [
                "https://example.com/pl/teach/control/lesson/view?id=10",
                "https://example.com/pl/teach/control/lesson/view?id=11&editMode=0",
            ],
        )

    def test_extract_js_object(self):
        html = '<script>var playerOptions = {"playlist":[{"sources":{"hls":{"src":"https://cdn.example.com/master.m3u8"}}}]};</script>'
        payload = extract_js_object(html, "playerOptions")
        self.assertEqual(
            payload["playlist"][0]["sources"]["hls"]["src"],
            "https://cdn.example.com/master.m3u8",
        )

    def test_extract_with_ytdlp_uses_playlist_entries(self):
        extractor = self.make_extractor("https://example.com/course")
        payload = {
            "_type": "playlist",
            "entries": [
                {"url": "https://cdn.example.com/one.m3u8", "title": "Part 1"},
                {"url": "https://cdn.example.com/two.mp4", "title": "Part 2"},
            ],
        }
        with patch("universal_extractor.shutil.which", return_value="/usr/local/bin/yt-dlp"), patch(
            "universal_extractor.subprocess.run",
            return_value=Mock(returncode=0, stdout=json.dumps(payload), stderr=""),
        ):
            videos = extractor._extract_with_ytdlp("https://example.com/lesson", "lesson-1", 1, "Lesson title")
        self.assertEqual(len(videos), 2)
        self.assertEqual(videos[0].stream_url, "https://cdn.example.com/one.m3u8")
        self.assertEqual(videos[1].video_position, 2)
        self.assertEqual(videos[0].extractor_strategy, "yt-dlp-playlist")


if __name__ == "__main__":
    unittest.main()
