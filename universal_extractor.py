from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from html import unescape
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import parse_qs, urljoin, urlparse

import browser_cookie3
import requests
from playwright.async_api import BrowserContext, Request, async_playwright


MEDIA_URL_RE = re.compile(r"https?://[^\s\"'<>]+(?:m3u8|mpd|mp4)(?:\?[^\s\"'<>]+)?", re.IGNORECASE)
LESSON_URL_RE = re.compile(
    r"https?://[^\s\"'<>]+/pl/teach/control/lesson/view\?id=\d+(?:[^\s\"'<>]*)?",
    re.IGNORECASE,
)
GETCOURSE_LESSON_PATH_RE = re.compile(
    r"/(?:pl/)?teach/control/lesson/view(?:\?id=\d+[^\s\"'<>]*|/id/\d+[^\s\"'<>]*)",
    re.IGNORECASE,
)
GETCOURSE_COURSE_PATH_RE = re.compile(
    r"/teach/control/stream/view/id/\d+[^\s\"'<>]*",
    re.IGNORECASE,
)
IFRAME_SRC_RE = re.compile(r"<iframe[^>]+src=[\"']([^\"']+)[\"']", re.IGNORECASE)
ANCHOR_HREF_RE = re.compile(r"<a[^>]+href=[\"']([^\"']+)[\"']", re.IGNORECASE)
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)
OG_TITLE_RE = re.compile(
    r"<meta[^>]+property=[\"']og:title[\"'][^>]+content=[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
GETCOURSE_HEADER_RE = re.compile(
    r"<div[^>]+class=[\"'][^\"']*header[^\"']*[\"'][^>]*>(.*?)</div>",
    re.IGNORECASE | re.DOTALL,
)
SCRIPT_JSON_RE = re.compile(
    r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
    re.IGNORECASE | re.DOTALL,
)


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "item"


def get_browser_cookie_loader(browser: str):
    loaders = {
        "chrome": browser_cookie3.chrome,
        "chromium": browser_cookie3.chromium,
        "brave": browser_cookie3.brave,
        "edge": browser_cookie3.edge,
        "firefox": browser_cookie3.firefox,
        "opera": browser_cookie3.opera,
        "safari": browser_cookie3.safari,
    }
    try:
        return loaders[browser.lower()]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported browser for cookies: {browser}") from exc


def load_browser_cookies(browser: str, domain: str):
    loader = get_browser_cookie_loader(browser)
    return loader(domain_name=domain)


def path_looks_like_media(url: str) -> bool:
    lower = url.lower()
    return any(ext in lower for ext in (".m3u8", ".mpd", ".mp4"))


def extract_title(html: str) -> str:
    for pattern in (OG_TITLE_RE, TITLE_RE):
        match = pattern.search(html)
        if match:
            return unescape(re.sub(r"\s+", " ", match.group(1))).strip()
    return "video"


def getcourse_lesson_id(url: str) -> str:
    parsed = urlparse(url)
    lesson_id = parse_qs(parsed.query).get("id", [""])[0]
    if not lesson_id:
        match = re.search(r"/id/(\d+)", parsed.path)
        if match:
            lesson_id = match.group(1)
    return lesson_id or slugify(url)


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def lesson_html_has_media_markers(html: str) -> bool:
    return bool(
        MEDIA_URL_RE.search(html)
        or IFRAME_SRC_RE.search(html)
        or re.search(r"<video\b|<audio\b", html, re.IGNORECASE)
    )


def strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", unescape(text))).strip()


def normalize_video_heading(text: str) -> str:
    match = re.search(r"(Video\s+\d+[^<]*)", text, re.IGNORECASE)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    return text


def extract_getcourse_embed_titles(html: str) -> dict[str, str]:
    headings: list[tuple[int, str]] = []
    for match in GETCOURSE_HEADER_RE.finditer(html):
        clean_title = normalize_video_heading(strip_html(match.group(1)))
        if clean_title and "video" in clean_title.lower():
            headings.append((match.start(), clean_title))

    titles: dict[str, str] = {}
    for iframe_match in IFRAME_SRC_RE.finditer(html):
        iframe_src = iframe_match.group(1)
        iframe_url = urljoin("https://placeholder.invalid", iframe_src)
        prior_headings = [title for position, title in headings if position < iframe_match.start()]
        if prior_headings:
            titles[iframe_url] = prior_headings[-1]
    return titles


def extract_js_object(source: str, variable_name: str) -> dict[str, Any] | None:
    marker = f"var {variable_name} ="
    start = source.find(marker)
    if start == -1:
        return None
    brace_start = source.find("{", start)
    if brace_start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    string_char = ""
    for index in range(brace_start, len(source)):
        char = source[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == string_char:
                in_string = False
            continue
        if char in ("'", '"'):
            in_string = True
            string_char = char
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(source[brace_start:index + 1])
    return None


@dataclass
class ExtractionOptions:
    browser: str
    mode: str = "hybrid"
    platform: str = "auto"
    extract_timeout_seconds: int = 45
    manual_play_timeout_seconds: int = 180
    debug_dir: Path | None = None


@dataclass
class DiscoveredVideo:
    lesson_id: str
    lesson_position: int
    lesson_title: str
    lesson_url: str
    video_position: int
    player_url: str
    stream_url: str
    stream_title: str
    referer: str = ""
    download_url: str = ""
    source_kind: str = "stream"
    extractor_strategy: str = ""
    subtitle_url: str = ""
    subtitle_language: str = ""


class DebugRecorder:
    def __init__(self, root: Path | None):
        self.root = root
        if self.root:
            self.root.mkdir(parents=True, exist_ok=True)

    def write_text(self, name: str, content: str) -> None:
        if not self.root:
            return
        (self.root / name).write_text(content, encoding="utf-8")

    def write_json(self, name: str, payload: Any) -> None:
        if not self.root:
            return
        (self.root / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class BrowserNetworkSniffer:
    def __init__(self, session: requests.Session, options: ExtractionOptions, debug: DebugRecorder):
        self.session = session
        self.options = options
        self.debug = debug
        self.captured_urls: list[str] = []
        self.captured_title = "video"

    def _note_candidate(self, url: str) -> None:
        if not url:
            return
        lower = url.lower()
        if any(token in lower for token in ("segment", "chunk", ".ts?", ".m4s", "frag")):
            return
        if path_looks_like_media(url) and url not in self.captured_urls:
            self.captured_urls.append(url)

    async def _attempt_autoplay(self, page) -> None:
        for frame in page.frames:
            try:
                await frame.evaluate(
                    """
                    () => {
                        const playSelectors = [
                            '[aria-label*="play" i]',
                            '[title*="play" i]',
                            '[data-testid*="play" i]',
                            '[class*="play" i]',
                            'button',
                            '.vjs-big-play-button'
                        ];
                        for (const selector of playSelectors) {
                            const node = document.querySelector(selector);
                            if (node instanceof HTMLElement) {
                                node.click();
                            }
                        }
                        for (const media of document.querySelectorAll('video, audio')) {
                            media.muted = true;
                            media.autoplay = true;
                            const playPromise = media.play?.();
                            if (playPromise && typeof playPromise.catch === 'function') {
                                playPromise.catch(() => {});
                            }
                        }
                    }
                    """
                )
            except Exception:
                continue

    def _intercept_request(self, request: Request) -> None:
        self._note_candidate(request.url)

    async def sniff(self, url: str) -> list[tuple[str, str]]:
        self.debug.write_text("interactive_target.txt", url)
        print("Interactive browser fallback started.")
        print("A browser window will open for manual recovery if automatic extraction cannot see the media.")
        print("If the lesson does not autoplay, click Play and keep the tab visible.")
        async with async_playwright() as p:
            browser_type, launch_kwargs = self._playwright_launcher(p)
            context = await browser_type.launch(**launch_kwargs)
            browser_context: BrowserContext | None = None
            try:
                browser_context = await context.new_context(ignore_https_errors=True)
                await self._inject_cookies(browser_context, url)
                page = await browser_context.new_page()
                page.on("request", self._intercept_request)
                page.on("response", lambda response: self._note_candidate(response.url))
                page.on("framenavigated", lambda frame: self._note_candidate(frame.url))

                await page.goto(url, wait_until="domcontentloaded", timeout=self.options.extract_timeout_seconds * 1000)

                try:
                    title = await page.title()
                    if title:
                        self.captured_title = title.strip()
                except Exception:
                    pass

                try:
                    await page.wait_for_load_state("networkidle", timeout=5000)
                except Exception:
                    pass

                wait_time = 0
                while not self.captured_urls and wait_time < self.options.manual_play_timeout_seconds:
                    await self._attempt_autoplay(page)
                    media_sources = await page.evaluate(
                        """
                        () => {
                            const found = [];
                            for (const media of document.querySelectorAll('video, audio')) {
                                if (media.currentSrc) found.push(media.currentSrc);
                                if (media.src) found.push(media.src);
                            }
                            for (const source of document.querySelectorAll('video source, audio source')) {
                                if (source.src) found.push(source.src);
                            }
                            return found;
                        }
                        """
                    )
                    for media_url in media_sources:
                        self._note_candidate(media_url)
                    if self.captured_urls:
                        break
                    await asyncio.sleep(1)
                    wait_time += 1

                if not self.captured_urls:
                    raise RuntimeError(
                        "Interactive browser fallback timed out without seeing a playable media URL."
                    )
                print(f"Interactive browser fallback captured {len(self.captured_urls)} media URL(s).")
                return [(captured_url, self.captured_title) for captured_url in self.captured_urls]
            finally:
                if browser_context:
                    await browser_context.close()
                await context.close()

    async def _inject_cookies(self, context: BrowserContext, url: str) -> None:
        parsed = urlparse(url)
        domain = parsed.netloc
        try:
            jar = load_browser_cookies(self.options.browser, domain)
        except Exception as exc:
            self.debug.write_text("cookie_load_error.txt", str(exc))
            return

        cookies = []
        for cookie in jar:
            if domain not in cookie.domain:
                continue
            cookie_payload: dict[str, Any] = {
                "name": cookie.name,
                "value": cookie.value,
                "domain": cookie.domain,
                "path": cookie.path or "/",
                "secure": bool(cookie.secure),
                "httpOnly": bool(cookie._rest.get("HttpOnly")),
            }
            if cookie.expires:
                cookie_payload["expires"] = float(cookie.expires)
            cookies.append(cookie_payload)
        if cookies:
            await context.add_cookies(cookies)

    def _playwright_launcher(self, playwright):
        browser_name = self.options.browser.lower()
        if browser_name == "firefox":
            return playwright.firefox, {"headless": False}
        launch_kwargs: dict[str, Any] = {"headless": False}
        if browser_name == "chrome":
            launch_kwargs["channel"] = "chrome"
        elif browser_name == "edge":
            launch_kwargs["channel"] = "msedge"
        return playwright.chromium, launch_kwargs


class UniversalCourseExtractor:
    def __init__(self, course_url: str, options: ExtractionOptions):
        self.course_url = course_url
        self.options = options
        self.parsed_url = urlparse(course_url)
        self.base_url = f"{self.parsed_url.scheme}://{self.parsed_url.netloc}"
        self.debug = DebugRecorder(options.debug_dir)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0",
                "Referer": course_url,
            }
        )
        self._load_cookies()

    def _load_cookies(self) -> None:
        diagnostics = {
            "browser": self.options.browser,
            "domain": self.parsed_url.netloc,
            "yt_dlp_found": shutil.which("yt-dlp") is not None,
            "ffmpeg_found": shutil.which("ffmpeg") is not None,
        }
        try:
            self.session.cookies = load_browser_cookies(self.options.browser, self.parsed_url.netloc)
            diagnostics["cookie_status"] = "loaded"
            diagnostics["cookie_count"] = len(list(self.session.cookies))
        except Exception as exc:
            diagnostics["cookie_status"] = "failed"
            diagnostics["cookie_error"] = str(exc)
        self.debug.write_json("preflight.json", diagnostics)

    def extract_manifest(self) -> dict[str, Any]:
        platform = detect_platform(self.course_url) if self.options.platform == "auto" else self.options.platform
        print(f"[extractor] platform={platform} mode={self.options.mode} browser={self.options.browser}")
        self.debug.write_json(
            "extractor_selection.json",
            {
                "url": self.course_url,
                "browser": self.options.browser,
                "mode": self.options.mode,
                "platform": platform,
            },
        )

        if platform == "getcourse":
            print("[extractor] using GetCourse adapter")
            manifest = self._extract_getcourse_manifest()
        else:
            print("[extractor] using generic adapter")
            manifest = self._extract_generic_manifest(platform)

        self.debug.write_json("manifest_preview.json", manifest)
        return manifest

    def _extract_getcourse_manifest(self) -> dict[str, Any]:
        response = self.session.get(self.course_url, timeout=self.options.extract_timeout_seconds)
        response.raise_for_status()
        html = response.text
        self.debug.write_text("root_page.html", html)
        root_title = extract_title(html)
        root_is_lesson = is_getcourse_lesson_url(self.course_url)
        lesson_urls = []
        if root_is_lesson:
            course_page_url = self._extract_getcourse_course_page_url(html)
            if course_page_url:
                course_response = self.session.get(
                    course_page_url,
                    timeout=self.options.extract_timeout_seconds,
                )
                course_response.raise_for_status()
                course_html = course_response.text
                self.debug.write_text("getcourse_course_page.html", course_html)
                lesson_urls = self._extract_getcourse_lesson_links(course_html)
                root_title = extract_title(course_html) or root_title
        else:
            lesson_urls = self._extract_getcourse_lesson_links(html)
        print(f"[extractor] discovered {len(lesson_urls)} lesson link(s) on the root page")

        videos: list[DiscoveredVideo] = []
        if lesson_urls:
            for lesson_position, lesson_url in enumerate(lesson_urls, start=1):
                videos.extend(self._extract_lesson_videos(lesson_url, lesson_position))
        else:
            videos.extend(self._extract_lesson_videos(self.course_url, 1))

        course_title = root_title
        if videos:
            course_title = root_title if len(lesson_urls) > 1 else videos[0].lesson_title
        return self._build_manifest("getcourse", course_title, videos)

    def _extract_generic_manifest(self, platform: str) -> dict[str, Any]:
        videos = self._extract_videos_for_url(
            self.course_url,
            lesson_id=slugify(self.course_url),
            lesson_position=1,
            lesson_title="video",
        )
        if not videos:
            raise RuntimeError(f"No playable media found for {self.course_url}")
        return self._build_manifest(platform, videos[0].lesson_title, videos)

    def _extract_getcourse_lesson_links(self, html: str) -> list[str]:
        candidates = [urljoin(self.base_url, href) for href in ANCHOR_HREF_RE.findall(html)]
        candidates.extend(LESSON_URL_RE.findall(html))
        candidates.extend(urljoin(self.base_url, path) for path in GETCOURSE_LESSON_PATH_RE.findall(html))
        lesson_urls = [url for url in candidates if is_getcourse_lesson_url(url)]
        return unique_preserve_order(lesson_urls)

    def _extract_getcourse_course_page_url(self, html: str) -> str | None:
        candidates = [urljoin(self.base_url, href) for href in ANCHOR_HREF_RE.findall(html)]
        candidates.extend(urljoin(self.base_url, path) for path in GETCOURSE_COURSE_PATH_RE.findall(html))
        for candidate in unique_preserve_order(candidates):
            if re.search(r"/teach/control/stream/view/id/\d+", urlparse(candidate).path):
                return candidate
        return None

    def _extract_lesson_videos(self, lesson_url: str, lesson_position: int) -> list[DiscoveredVideo]:
        response = self.session.get(lesson_url, timeout=self.options.extract_timeout_seconds)
        response.raise_for_status()
        html = response.text
        lesson_title = extract_title(html)
        lesson_id = getcourse_lesson_id(lesson_url)
        self.debug.write_text(f"lesson_{lesson_id}.html", html)
        if not lesson_html_has_media_markers(html):
            print(f"[extractor] skipping non-video lesson {lesson_url}")
            return []
        videos = self._extract_videos_for_url(
            lesson_url,
            lesson_id=lesson_id,
            lesson_position=lesson_position,
            lesson_title=lesson_title,
            html=html,
        )
        if videos:
            return videos
        raise RuntimeError(f"Could not find playable media for lesson {lesson_url}")

    def _extract_videos_for_url(
        self,
        url: str,
        lesson_id: str,
        lesson_position: int,
        lesson_title: str,
        html: str | None = None,
    ) -> list[DiscoveredVideo]:
        automatic_candidates = []

        if self.options.mode in {"hybrid", "auto"}:
            print(f"[extractor] trying yt-dlp direct extraction for {url}")
            automatic_candidates.extend(self._extract_with_ytdlp(url, lesson_id, lesson_position, lesson_title))
            if automatic_candidates:
                print(f"[extractor] yt-dlp resolved {len(automatic_candidates)} video candidate(s)")
                return automatic_candidates

        if html:
            media_urls = unique_preserve_order(MEDIA_URL_RE.findall(html))
            iframe_urls = unique_preserve_order(urljoin(url, src) for src in IFRAME_SRC_RE.findall(html))
            iframe_titles = extract_getcourse_embed_titles(html)
            if media_urls:
                print(f"[extractor] found {len(media_urls)} media URL(s) directly in HTML")
                return self._videos_from_urls(
                    media_urls,
                    lesson_id,
                    lesson_position,
                    lesson_title,
                    player_url=url,
                    strategy="html-media",
                )
            aggregated_iframe_results: list[DiscoveredVideo] = []
            for iframe_url in iframe_urls:
                embed_results = self._extract_known_embed(
                    iframe_url,
                    url,
                    lesson_id,
                    lesson_position,
                    iframe_titles.get(iframe_url, lesson_title),
                )
                if embed_results:
                    print(f"[extractor] resolved {len(embed_results)} media candidate(s) through known embed {iframe_url}")
                    aggregated_iframe_results.extend(embed_results)
                    continue
                iframe_results = self._extract_with_ytdlp(
                    iframe_url,
                    lesson_id,
                    lesson_position,
                    iframe_titles.get(iframe_url, lesson_title),
                    referer=url,
                )
                if iframe_results:
                    print(f"[extractor] resolved {len(iframe_results)} media candidate(s) through iframe {iframe_url}")
                    aggregated_iframe_results.extend(iframe_results)
            if aggregated_iframe_results:
                for index, video in enumerate(aggregated_iframe_results, start=1):
                    video.video_position = index
                return aggregated_iframe_results

        if self.options.mode in {"hybrid", "interactive"}:
            print(f"[extractor] falling back to interactive browser capture for {url}")
            return self._extract_with_interactive_browser(url, lesson_id, lesson_position, lesson_title)
        return []

    def _extract_known_embed(
        self,
        iframe_url: str,
        lesson_url: str,
        lesson_id: str,
        lesson_position: int,
        lesson_title: str,
    ) -> list[DiscoveredVideo]:
        parsed = urlparse(iframe_url)
        if "kinescope.io" in parsed.netloc and "/embed/" in parsed.path:
            return self._extract_kinescope_embed(
                iframe_url,
                lesson_url,
                lesson_id,
                lesson_position,
                lesson_title,
            )
        return []

    def _extract_kinescope_embed(
        self,
        iframe_url: str,
        lesson_url: str,
        lesson_id: str,
        lesson_position: int,
        lesson_title: str,
    ) -> list[DiscoveredVideo]:
        lesson_parsed = urlparse(lesson_url)
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": lesson_url,
            "Origin": f"{lesson_parsed.scheme}://{lesson_parsed.netloc}",
        }
        try:
            response = self.session.get(
                iframe_url,
                headers=headers,
                timeout=self.options.extract_timeout_seconds,
            )
            response.raise_for_status()
        except Exception as exc:
            self.debug.write_text("kinescope_error.txt", str(exc))
            return []

        self.debug.write_text("kinescope_embed.html", response.text)
        player_options = extract_js_object(response.text, "playerOptions")
        if not player_options:
            return []

        playlist = player_options.get("playlist") or []
        results: list[DiscoveredVideo] = []
        for index, item in enumerate(playlist, start=1):
            if not isinstance(item, dict):
                continue
            sources = item.get("sources") or {}
            hls = sources.get("hls") or {}
            shakahls = sources.get("shakahls") or {}
            src = hls.get("src") or shakahls.get("src")
            if not src:
                continue
            subtitle_url = ""
            subtitle_language = ""
            for subtitle in item.get("vtt") or []:
                if not isinstance(subtitle, dict):
                    continue
                candidate_url = subtitle.get("src") or ""
                candidate_lang = subtitle.get("srcLang") or ""
                if not candidate_url:
                    continue
                subtitle_url = candidate_url
                subtitle_language = candidate_lang
                if candidate_lang.lower() == "en":
                    break
            results.append(
                DiscoveredVideo(
                    lesson_id=lesson_id,
                    lesson_position=lesson_position,
                    lesson_title=lesson_title,
                    lesson_url=lesson_url,
                    video_position=index,
                    player_url=iframe_url,
                    stream_url=src,
                    stream_title=item.get("title") or lesson_title,
                    referer=lesson_url,
                    download_url=src,
                    source_kind="stream",
                    extractor_strategy="kinescope-embed",
                    subtitle_url=subtitle_url,
                    subtitle_language=subtitle_language,
                )
            )
        return results

    def _extract_with_ytdlp(
        self,
        url: str,
        lesson_id: str,
        lesson_position: int,
        lesson_title: str,
        referer: str | None = None,
    ) -> list[DiscoveredVideo]:
        if shutil.which("yt-dlp") is None:
            return []

        cmd = [
            "yt-dlp",
            "--dump-single-json",
            "--no-warnings",
            "--cookies-from-browser",
            self.options.browser,
        ]
        if referer:
            cmd.extend(["--referer", referer])
        cmd.append(url)

        try:
            result = subprocess.run(
                cmd,
                text=True,
                capture_output=True,
                check=False,
                timeout=self.options.extract_timeout_seconds,
            )
        except Exception as exc:
            self.debug.write_text("ytdlp_error.txt", str(exc))
            return []

        if result.returncode != 0 or not result.stdout.strip():
            self.debug.write_text("ytdlp_stderr.txt", result.stderr.strip())
            if result.stderr.strip():
                print(f"[extractor] yt-dlp failed for {url}: {result.stderr.strip()}")
            return []

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            self.debug.write_text("ytdlp_stdout.txt", result.stdout)
            return []

        self.debug.write_json("ytdlp_payload.json", payload)
        if payload.get("_type") == "playlist" and isinstance(payload.get("entries"), list):
            videos: list[DiscoveredVideo] = []
            for index, entry in enumerate(payload["entries"], start=1):
                if not isinstance(entry, dict):
                    continue
                entry_url = entry.get("url") or entry.get("webpage_url") or url
                entry_title = entry.get("title") or f"{lesson_title} {index}"
                videos.append(
                    DiscoveredVideo(
                        lesson_id=lesson_id,
                        lesson_position=lesson_position,
                        lesson_title=lesson_title,
                        lesson_url=url,
                        video_position=index,
                        player_url=entry.get("webpage_url") or url,
                        stream_url=entry_url,
                        stream_title=entry_title,
                        referer=referer or url,
                        download_url=entry_url,
                        source_kind="page" if not path_looks_like_media(entry_url) else "stream",
                        extractor_strategy="yt-dlp-playlist",
                    )
                )
            return videos

        resolved_url = payload.get("url") or payload.get("webpage_url") or url
        title = payload.get("title") or lesson_title
        return [
            DiscoveredVideo(
                lesson_id=lesson_id,
                lesson_position=lesson_position,
                lesson_title=title if lesson_title == "video" else lesson_title,
                lesson_url=url,
                video_position=1,
                player_url=payload.get("webpage_url") or url,
                stream_url=resolved_url,
                stream_title=title,
                referer=referer or url,
                download_url=resolved_url,
                source_kind="page" if not path_looks_like_media(resolved_url) else "stream",
                extractor_strategy="yt-dlp",
            )
        ]

    def _extract_with_interactive_browser(
        self,
        url: str,
        lesson_id: str,
        lesson_position: int,
        lesson_title: str,
    ) -> list[DiscoveredVideo]:
        sniffer = BrowserNetworkSniffer(self.session, self.options, self.debug)
        captured = asyncio.run(sniffer.sniff(url))
        return self._videos_from_urls(
            [item[0] for item in captured],
            lesson_id,
            lesson_position,
            lesson_title if lesson_title != "video" else (captured[0][1] if captured else "video"),
            player_url=url,
            strategy="interactive-browser",
            stream_titles=[item[1] for item in captured],
        )

    def _videos_from_urls(
        self,
        urls: list[str],
        lesson_id: str,
        lesson_position: int,
        lesson_title: str,
        *,
        player_url: str,
        strategy: str,
        stream_titles: list[str] | None = None,
    ) -> list[DiscoveredVideo]:
        deduped = unique_preserve_order(urls)
        videos: list[DiscoveredVideo] = []
        for index, resolved_url in enumerate(deduped, start=1):
            title = stream_titles[index - 1] if stream_titles and index - 1 < len(stream_titles) else lesson_title
            videos.append(
                DiscoveredVideo(
                    lesson_id=lesson_id,
                    lesson_position=lesson_position,
                    lesson_title=lesson_title,
                    lesson_url=player_url,
                    video_position=index,
                    player_url=player_url,
                    stream_url=resolved_url,
                    stream_title=title,
                    referer=player_url,
                    download_url=resolved_url,
                    source_kind="page" if not path_looks_like_media(resolved_url) else "stream",
                    extractor_strategy=strategy,
                )
            )
        return videos

    def _build_manifest(self, platform: str, course_name: str, videos: list[DiscoveredVideo]) -> dict[str, Any]:
        course_slug = slugify(course_name or self.parsed_url.netloc)
        return {
            "platform": platform,
            "course_url": self.course_url,
            "course_name": course_name or self.parsed_url.netloc,
            "course_slug": course_slug,
            "student_course_id": f"{platform}_01",
            "videos": [asdict(video) for video in videos],
        }


def is_getcourse_lesson_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.path in {"/pl/teach/control/lesson/view", "/teach/control/lesson/view"} and "id" in parse_qs(parsed.query):
        return True
    return re.search(r"/teach/control/lesson/view/id/\d+", parsed.path) is not None


def detect_platform(url: str) -> str:
    lowered = url.lower()
    if "fp-platform" in lowered or "v.xlsite.ai" in lowered:
        return "fp-platform"
    if "/pl/teach/control/" in lowered:
        return "getcourse"
    return "generic"


def extract_course_manifest(
    url: str,
    *,
    browser: str,
    mode: str = "hybrid",
    platform: str = "auto",
    extract_timeout_seconds: int = 45,
    manual_play_timeout_seconds: int = 180,
    debug_dir: Path | None = None,
) -> dict[str, Any]:
    extractor = UniversalCourseExtractor(
        url,
        ExtractionOptions(
            browser=browser,
            mode=mode,
            platform=platform,
            extract_timeout_seconds=extract_timeout_seconds,
            manual_play_timeout_seconds=manual_play_timeout_seconds,
            debug_dir=debug_dir,
        ),
    )
    return extractor.extract_manifest()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        manifest = extract_course_manifest(sys.argv[1], browser="chrome")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
