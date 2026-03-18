from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


STAGE_TRANSCRIPTION = "transcription"
STAGE_TRANSLATION = "translation"
STAGE_SYNTHESIS = "synthesis"
STAGE_MIX = "mix"
STAGE_ORDER = (
    STAGE_TRANSCRIPTION,
    STAGE_TRANSLATION,
    STAGE_SYNTHESIS,
    STAGE_MIX,
)
SEGMENTATION_CLEANUP_VERSION = "v1"
SPOKEN_REWRITE_VERSION = "v1"
DENSE_REWRITE_VERSION = "v1"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def now_ts() -> float:
    return time.time()


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_hash_json(payload: Any) -> str:
    return stable_hash_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stage_changed(previous: dict[str, Any] | None, current: dict[str, Any], stage_name: str) -> bool:
    if not previous:
        return True
    previous_stage = (previous.get("stages") or {}).get(stage_name)
    current_stage = (current.get("stages") or {}).get(stage_name)
    return previous_stage != current_stage


def stages_requiring_refresh(previous: dict[str, Any] | None, current: dict[str, Any]) -> set[str]:
    changed_from: str | None = None
    for stage_name in STAGE_ORDER:
        if stage_changed(previous, current, stage_name):
            changed_from = stage_name
            break
    if changed_from is None:
        return set()
    start_index = STAGE_ORDER.index(changed_from)
    return set(STAGE_ORDER[start_index:])


@contextmanager
def locked_path(lock_path: Path) -> Iterator[None]:
    ensure_dir(lock_path.parent)
    with lock_path.open("a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(str(os.getpid()))
            handle.flush()
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class CacheStore:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        ensure_dir(db_path.parent)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS stage_manifests (
                    course_slug TEXT NOT NULL,
                    video_key TEXT NOT NULL,
                    manifest_json TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (course_slug, video_key)
                );

                CREATE TABLE IF NOT EXISTS translation_memory (
                    cache_key TEXT PRIMARY KEY,
                    source_text TEXT NOT NULL,
                    translated_text TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    source_lang TEXT NOT NULL,
                    target_lang TEXT NOT NULL,
                    glossary_hash TEXT NOT NULL,
                    rewrite_version TEXT NOT NULL,
                    style TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                """
            )
            connection.commit()

    def get_stage_manifest(self, course_slug: str, video_key: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT manifest_json FROM stage_manifests WHERE course_slug = ? AND video_key = ?",
                (course_slug, video_key),
            ).fetchone()
        if not row:
            return None
        return json.loads(str(row["manifest_json"]))

    def put_stage_manifest(self, course_slug: str, video_key: str, manifest: dict[str, Any]) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO stage_manifests (course_slug, video_key, manifest_json, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(course_slug, video_key)
                DO UPDATE SET manifest_json = excluded.manifest_json, updated_at = excluded.updated_at
                """,
                (course_slug, video_key, json.dumps(manifest, ensure_ascii=False, sort_keys=True), now_ts()),
            )
            connection.commit()

    def get_translation(self, cache_key: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT source_text, translated_text, provider, source_lang, target_lang,
                       glossary_hash, rewrite_version, style, updated_at
                FROM translation_memory
                WHERE cache_key = ?
                """,
                (cache_key,),
            ).fetchone()
        if not row:
            return None
        return dict(row)

    def put_translation(
        self,
        cache_key: str,
        *,
        source_text: str,
        translated_text: str,
        provider: str,
        source_lang: str,
        target_lang: str,
        glossary_hash: str,
        rewrite_version: str,
        style: str,
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO translation_memory (
                    cache_key, source_text, translated_text, provider, source_lang, target_lang,
                    glossary_hash, rewrite_version, style, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key)
                DO UPDATE SET
                    translated_text = excluded.translated_text,
                    provider = excluded.provider,
                    source_lang = excluded.source_lang,
                    target_lang = excluded.target_lang,
                    glossary_hash = excluded.glossary_hash,
                    rewrite_version = excluded.rewrite_version,
                    style = excluded.style,
                    updated_at = excluded.updated_at
                """,
                (
                    cache_key,
                    source_text,
                    translated_text,
                    provider,
                    source_lang,
                    target_lang,
                    glossary_hash,
                    rewrite_version,
                    style,
                    now_ts(),
                ),
            )
            connection.commit()
