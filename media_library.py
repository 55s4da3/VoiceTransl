"""Read-only media discovery and the persistent VoiceTransl work catalogue.

Scanning never renames, moves, tags, or writes beside a user's media.  A scan
first produces an :class:`ImportPreview`; persisting it into the independent
SQLite catalogue is an explicit second operation.
"""

from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import shutil
import sqlite3
import threading
import time
import unicodedata
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SCHEMA_VERSION = 2
PRODUCT_ID_RE = re.compile(r"(?<![A-Z0-9])((?:RJ|VJ|BJ)0*\d{6,10})(?!\d)", re.IGNORECASE)
TRACK_PREFIX_RE = re.compile(r"^\s*(?:track\s*)?\d{1,3}(?:[_ .\-、]+|$)", re.IGNORECASE)
QUALITY_TOKEN_RE = re.compile(
    r"(?:^|[\[\]（）() _.-])(wav|flac|mp3|m4a|aac|opus|ogg|lossless|hi[- ]?res|\d{2,3}k)(?=$|[\[\]（）() _.-])",
    re.IGNORECASE,
)

AUDIO_EXTENSIONS = {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"}
VIDEO_EXTENSIONS = {".mkv", ".mp4", ".ts", ".webm"}
SUBTITLE_EXTENSIONS = {".ass", ".lrc", ".srt", ".ssa", ".vtt"}
IMAGE_EXTENSIONS = {".avif", ".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
TEXT_EXTENSIONS = {".csv", ".html", ".md", ".nfo", ".pdf", ".txt"}
ARCHIVE_EXTENSIONS = {".001", ".002", ".7z", ".rar", ".zip"}
INCOMPLETE_EXTENSIONS = {".aria2", ".crdownload", ".download", ".part"}
MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
DERIVED_DIRECTORY_NAMES = {"已翻译", "translated", "translations", "字幕", "subtitles"}
GENERIC_COLLECTION_NAMES = {
    "audio", "voice", "music", "youtube", "download", "downloads", "misc", "other",
    "新建文件夹", "新建文件夹 3", "未分类", "其他", "其它", "音声", "音乐", "插画", "イラスト",
    "fc", "fyz", "ts", "输出", "output",
}

BONUS_RE = re.compile(
    r"(?:特典|おまけ|オマケ|omake|bonus|extra|エクストラ|free[ _-]?talk|フリートーク|"
    r"キャストコメント|収録後|附赠|附錄|附录|同梱|ng(?:版|テイク)?|outtake)",
    re.IGNORECASE,
)
COVER_RE = re.compile(r"(?:cover|package|pake|jacket|ジャケット|封面|表紙|ロゴあり|logo)", re.IGNORECASE)
ILLUSTRATION_RE = re.compile(r"(?:illustration|イラスト|插画|插圖|差分|立ち絵|スチル|special)", re.IGNORECASE)
SCRIPT_RE = re.compile(r"(?:script|scenario|台本|剧本|腳本|スクリプト)", re.IGNORECASE)
README_RE = re.compile(r"(?:read[ _-]?me|りーどみー|説明|说明)", re.IGNORECASE)
PROMO_RE = re.compile(r"(?:招聘|交流群|投票群|参与成员|參與成員|补票|補票|目录\d*|目録)", re.IGNORECASE)
SUBTITLE_LANGUAGE_SUFFIX_RE = re.compile(
    r"(?:\.(?P<language>zh(?:-cn|-hans)?|ja|jp|en|ko|tg)|\.(?P<combined>combine|combined|bilingual))$",
    re.IGNORECASE,
)
GENERATED_SUFFIX_RE = re.compile(r"\.(?:resegmented|aligned|translated|tg)$", re.IGNORECASE)
VARIANT_RE = re.compile(
    r"(?:SE(?:無|な)し|環境音(?:無|な)し|効果音(?:無|な)し|no[ _-]?(?:se|bgm)|"
    r"バイノーラル|モノラル|左右|右耳|左耳)",
    re.IGNORECASE,
)


def normalize_product_id(value: str) -> str:
    match = PRODUCT_ID_RE.search(str(value or ""))
    return match.group(1).upper() if match else ""


def product_ids(value: str) -> list[str]:
    return list(dict.fromkeys(match.group(1).upper() for match in PRODUCT_ID_RE.finditer(value or "")))


def nearest_product_id(value: str) -> str:
    """Return the product id closest to the file (the last id in its path)."""
    matches = list(PRODUCT_ID_RE.finditer(str(value or "")))
    return matches[-1].group(1).upper() if matches else ""


def normalized_title(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = PRODUCT_ID_RE.sub(" ", text)
    text = QUALITY_TOKEN_RE.sub(" ", text)
    text = re.sub(r"[\[\]【】（）(){}<>「」『』]", " ", text)
    text = re.sub(r"[\s_.\-—–]+", " ", text).strip().casefold()
    return text


def stable_local_id(relative_name: str) -> str:
    normalized = normalized_title(relative_name) or unicodedata.normalize("NFKC", relative_name).casefold()
    return "local:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]


def asset_kind(path: Path) -> str:
    extension = path.suffix.casefold()
    if extension in AUDIO_EXTENSIONS:
        return "audio"
    if extension in VIDEO_EXTENSIONS:
        return "video"
    if extension in SUBTITLE_EXTENSIONS:
        return "subtitle"
    if extension in IMAGE_EXTENSIONS:
        return "image"
    if extension in TEXT_EXTENSIONS:
        return "document"
    if extension in ARCHIVE_EXTENSIONS:
        return "archive"
    return "attachment"


def chapter_title(path: Path) -> str:
    title = logical_stem(path)
    title = PRODUCT_ID_RE.sub("", title)
    title = TRACK_PREFIX_RE.sub("", title)
    title = re.sub(r"\s+", " ", title).strip(" _.-")
    return title or path.stem


def logical_stem(path: Path) -> str:
    title = unicodedata.normalize("NFKC", path.stem)
    while True:
        reduced = GENERATED_SUFFIX_RE.sub("", title)
        reduced = SUBTITLE_LANGUAGE_SUFFIX_RE.sub("", reduced)
        if reduced == title:
            break
        title = reduced
    return title.strip(" _.-") or path.stem


def subtitle_language(path: Path) -> str:
    if path.suffix.casefold() not in SUBTITLE_EXTENSIONS:
        return ""
    stem = unicodedata.normalize("NFKC", path.stem)
    match = SUBTITLE_LANGUAGE_SUFFIX_RE.search(stem)
    if not match:
        return "source"
    if match.group("combined"):
        return "bilingual"
    language = (match.group("language") or "").casefold()
    if language.startswith("zh") or language == "tg":
        return "zh"
    if language in {"ja", "jp"}:
        return "ja"
    return language


def resource_section(path: Path) -> str:
    text = unicodedata.normalize("NFKC", str(path))
    if BONUS_RE.search(text):
        return "bonus"
    if path.suffix.casefold() in IMAGE_EXTENSIONS:
        return "artwork"
    if path.suffix.casefold() in TEXT_EXTENSIONS:
        return "document"
    if path.suffix.casefold() in ARCHIVE_EXTENSIONS:
        return "archive"
    if path.suffix.casefold() in INCOMPLETE_EXTENSIONS:
        return "incomplete"
    return "main"


def resource_role(path: Path) -> str:
    text = unicodedata.normalize("NFKC", str(path))
    extension = path.suffix.casefold()
    if extension in INCOMPLETE_EXTENSIONS:
        return "incomplete"
    if extension in SUBTITLE_EXTENSIONS:
        return "subtitle"
    if extension in AUDIO_EXTENSIONS:
        return "bonus_audio" if BONUS_RE.search(text) else "chapter_audio"
    if extension in VIDEO_EXTENSIONS:
        return "bonus_video" if BONUS_RE.search(text) else "chapter_video"
    if extension in IMAGE_EXTENSIONS:
        if PROMO_RE.search(text):
            return "promo"
        if COVER_RE.search(text):
            return "cover"
        return "illustration" if ILLUSTRATION_RE.search(text) else "image"
    if extension in TEXT_EXTENSIONS:
        if SCRIPT_RE.search(text):
            return "script"
        if README_RE.search(text):
            return "readme"
        return "document"
    if extension in ARCHIVE_EXTENSIONS:
        return "archive"
    return "attachment"


def resource_group_key(path: Path) -> str:
    stem = PRODUCT_ID_RE.sub(" ", logical_stem(path))
    stem = QUALITY_TOKEN_RE.sub(" ", stem)
    stem = VARIANT_RE.sub(" ", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" _.-").casefold()
    return stem or normalized_title(path.stem) or path.stem.casefold()


def resource_variant(path: Path, nested: Sequence[str] = ()) -> str:
    values = [part for part in nested if QUALITY_TOKEN_RE.search(part) or VARIANT_RE.search(part)]
    match = VARIANT_RE.search(unicodedata.normalize("NFKC", path.stem))
    if match:
        values.append(match.group(0))
    return " / ".join(dict.fromkeys(values)) or path.suffix.casefold().lstrip(".")


def resource_sort_order(path: Path) -> int:
    stem = PRODUCT_ID_RE.sub("", unicodedata.normalize("NFKC", path.stem))
    match = re.search(r"(?:^|[ _.-])(?:track[ _.-]*)?(\d{1,3})(?=$|[ _.-])", stem, re.IGNORECASE)
    return int(match.group(1)) if match else 1_000_000


def _path_key(path: Path) -> str:
    return str(path.resolve()).casefold()


def _path_fingerprint(path: Path) -> str:
    stat = path.stat()
    value = f"{_path_key(path)}\0{max(0, int(stat.st_size))}\0{max(0, int(stat.st_mtime_ns))}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_component(value: str, fallback: str = "Untitled") -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text).strip(" .")
    return (text or fallback)[:120]


def _inside(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


@dataclass(slots=True)
class AssetPreview:
    path: str
    relative_path: str
    kind: str
    extension: str
    size: int
    modified_ns: int
    chapter: str
    version: str
    derived: bool = False
    source_work_id: str = ""
    section: str = "main"
    role: str = "attachment"
    group_key: str = ""
    language: str = ""
    variant: str = ""
    sort_order: int = 1_000_000
    content_fingerprint: str = ""
    file_identity: str = ""

    @property
    def fingerprint(self) -> str:
        value = f"{_path_key(Path(self.path))}\0{self.size}\0{self.modified_ns}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class WorkPreview:
    work_id: str
    product_id: str
    title: str
    normalized_title: str
    source_roots: list[str] = field(default_factory=list)
    assets: list[AssetPreview] = field(default_factory=list)
    confidence: float = 0.0
    classification: str = "unclassified"
    reasons: list[str] = field(default_factory=list)
    series_hint: str = ""

    def public(self, include_assets: bool = True) -> dict[str, Any]:
        value = {
            "id": self.work_id,
            "product_id": self.product_id,
            "title": self.title,
            "normalized_title": self.normalized_title,
            "source_roots": list(self.source_roots),
            "confidence": round(self.confidence, 3),
            "classification": self.classification,
            "reasons": list(self.reasons),
            "series_hint": self.series_hint,
            "asset_count": len(self.assets),
            "audio_count": sum(asset.kind in {"audio", "video"} for asset in self.assets),
            "subtitle_count": sum(asset.kind == "subtitle" for asset in self.assets),
        }
        if include_assets:
            value["assets"] = [asdict(asset) | {"fingerprint": asset.fingerprint} for asset in self.assets]
        return value


@dataclass(slots=True)
class ImportPreview:
    root: str
    created_at: float
    works: list[WorkPreview]
    ignored: list[dict[str, str]] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, Any]:
        assets = [asset for work in self.works for asset in work.assets]
        return {
            "root": self.root,
            "works": len(self.works),
            "assets": sum(len(work.assets) for work in self.works),
            "audio": sum(asset.kind in {"audio", "video"} for work in self.works for asset in work.assets),
            "subtitles": sum(asset.kind == "subtitle" for work in self.works for asset in work.assets),
            "images": sum(asset.kind == "image" for asset in assets),
            "documents": sum(asset.kind in {"document", "archive", "attachment"} for asset in assets),
            "bonus": sum(asset.section == "bonus" for asset in assets),
            "incomplete": sum(asset.section == "incomplete" for asset in assets),
            "official_ids": sum(bool(work.product_id) for work in self.works),
            "auto_grouped": sum(work.classification == "auto" for work in self.works),
            "inbox": sum(work.classification != "auto" for work in self.works),
            "ignored": len(self.ignored),
        }

    def public(self, include_assets: bool = True) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "created_at": self.created_at,
            "summary": self.summary,
            "works": [work.public(include_assets=include_assets) for work in self.works],
            "ignored": list(self.ignored),
        }


class MediaLibraryScanner:
    """Discover work candidates while treating the media root as read-only."""

    def __init__(
        self,
        derived_names: Iterable[str] = DERIVED_DIRECTORY_NAMES,
        known_titles: Mapping[str, str] | None = None,
    ):
        self.derived_names = {unicodedata.normalize("NFKC", name).casefold() for name in derived_names}
        self.known_titles = {
            str(work_id): str(title)
            for work_id, title in (known_titles or {}).items()
            if str(work_id).strip() and str(title).strip()
        }

    def scan(self, root: Path | str) -> ImportPreview:
        base = Path(root).expanduser().resolve(strict=True)
        if not base.is_dir():
            raise ValueError(f"media root is not a directory: {base}")

        candidates: dict[str, WorkPreview] = {}
        duplicate_sources: dict[str, set[str]] = {}
        ignored: list[dict[str, str]] = []
        discovered: list[tuple[Path, Path, bool]] = []
        for entry in sorted(base.iterdir(), key=lambda item: item.name.casefold()):
            if entry.is_symlink():
                ignored.append({"path": str(entry), "reason": "symlink"})
                continue
            if entry.is_dir():
                paths = list(self._files(entry))
                if not paths:
                    ignored.append({"path": str(entry), "reason": "empty_directory"})
                    continue
                derived = unicodedata.normalize("NFKC", entry.name).casefold() in self.derived_names
                discovered.extend((path, entry, derived) for path in paths)
            elif entry.is_file():
                discovered.append((entry, base, False))

        for path, source_root, derived in discovered:
            try:
                relative = path.relative_to(base)
            except ValueError:
                relative = Path(path.name)
            product_id = nearest_product_id(str(relative))
            reason = "official_product_id"
            if not product_id:
                product_id = self._match_known_title(path)
                reason = "unique_title_match" if product_id else ""
            if product_id:
                title = self._product_title(relative, product_id)
                candidate = self._candidate(candidates, product_id, product_id, title)
                if reason and reason not in candidate.reasons:
                    candidate.reasons.append(reason)
                source_work_id = product_id
                key = product_id
            else:
                title = self._local_title(relative)
                key = stable_local_id(title)
                candidate = self._candidate(candidates, key, "", title)
                source_work_id = ""
            source = str(source_root)
            if source not in candidate.source_roots:
                candidate.source_roots.append(source)
            duplicate_sources.setdefault(key, set()).add(source)
            candidate.assets.append(
                self._asset(
                    base,
                    path,
                    source_root,
                    derived=derived and asset_kind(path) == "subtitle",
                    source_work_id=source_work_id,
                )
            )

        works = list(candidates.values())
        self._classify(works, duplicate_sources)
        works.sort(key=lambda item: (not bool(item.product_id), item.product_id or item.normalized_title, item.title))
        return ImportPreview(str(base), time.time(), works, ignored)

    def _match_known_title(self, path: Path) -> str:
        phrase = logical_stem(path)
        phrase = COVER_RE.sub(" ", phrase)
        phrase = ILLUSTRATION_RE.sub(" ", phrase)
        phrase = re.sub(r"(?:ロゴ|logo)(?:あり|なし|有り|無し)?", " ", phrase, flags=re.IGNORECASE)
        phrase = normalized_title(phrase)
        if len(phrase) < 6:
            return ""
        matches = [
            work_id
            for work_id, title in self.known_titles.items()
            if phrase in normalized_title(title)
        ]
        return matches[0] if len(matches) == 1 and normalize_product_id(matches[0]) else ""

    @staticmethod
    def _product_title(relative: Path, product_id: str) -> str:
        matching_parts = [part for part in relative.parts[:-1] if product_id in part.upper()]
        if matching_parts:
            return matching_parts[-1]
        stem = logical_stem(relative)
        return stem if normalized_title(stem) else product_id

    @staticmethod
    def _local_title(relative: Path) -> str:
        if len(relative.parts) <= 1:
            return chapter_title(relative)
        top = relative.parts[0]
        normalized_top = normalized_title(top)
        if normalized_top in GENERIC_COLLECTION_NAMES or re.fullmatch(r"\d{6,}(?:[ -]\d+)?", normalized_top):
            for nested in reversed(relative.parts[1:-1]):
                normalized_nested = normalized_title(nested)
                if (
                    normalized_nested
                    and normalized_nested not in GENERIC_COLLECTION_NAMES
                    and not re.fullmatch(r"\d{6,}(?:[ -]\d+)?", normalized_nested)
                ):
                    return nested
            # Keep unrelated loose recordings separate while grouping their subtitle variants.
            return chapter_title(relative)
        return top

    @staticmethod
    def _files(directory: Path) -> Iterator[Path]:
        try:
            for path in directory.rglob("*"):
                try:
                    if path.is_file() and not path.is_symlink():
                        yield path
                except OSError:
                    continue
        except OSError:
            return

    def _candidate(self, candidates: dict[str, WorkPreview], key: str, product_id: str, title: str) -> WorkPreview:
        candidate = candidates.get(key)
        if candidate is None:
            candidate = WorkPreview(
                work_id=product_id or stable_local_id(title),
                product_id=product_id,
                title=title.strip() or product_id or "Untitled",
                normalized_title=normalized_title(title),
            )
            candidates[key] = candidate
        return candidate

    def _add_directory_candidate(
        self,
        base: Path,
        directory: Path,
        candidates: dict[str, WorkPreview],
        duplicate_sources: dict[str, set[str]],
        ignored: list[dict[str, str]],
    ) -> None:
        ids = product_ids(directory.name)
        product_id = ids[0] if ids else ""
        key = product_id or stable_local_id(directory.name)
        candidate = self._candidate(candidates, key, product_id, directory.name)
        candidate.source_roots.append(str(directory))
        duplicate_sources.setdefault(key, set()).add(str(directory))
        found = False
        for path in self._files(directory):
            found = True
            candidate.assets.append(self._asset(base, path, directory, derived=False, source_work_id=product_id))
        if not found:
            ignored.append({"path": str(directory), "reason": "empty_directory"})

    def _add_top_level_file(
        self,
        base: Path,
        path: Path,
        candidates: dict[str, WorkPreview],
        duplicate_sources: dict[str, set[str]],
    ) -> None:
        product_id = normalize_product_id(path.name)
        title = chapter_title(path)
        fallback = normalized_title(title)
        key = product_id or stable_local_id(fallback or path.stem)
        candidate = self._candidate(candidates, key, product_id, title)
        source = str(path.parent)
        if source not in candidate.source_roots:
            candidate.source_roots.append(source)
        duplicate_sources.setdefault(key, set()).add(str(path.parent))
        candidate.assets.append(self._asset(base, path, base, derived=False, source_work_id=product_id))

    def _attach_derived(
        self,
        base: Path,
        assets: Sequence[tuple[Path, Path]],
        candidates: dict[str, WorkPreview],
        duplicate_sources: dict[str, set[str]],
        ignored: list[dict[str, str]],
    ) -> None:
        by_title = {work.normalized_title: work for work in candidates.values() if work.normalized_title}
        for path, derived_root in assets:
            kind = asset_kind(path)
            if kind not in {"audio", "video", "subtitle"}:
                ignored.append({"path": str(path), "reason": "derived_non_media"})
                continue
            product_id = normalize_product_id(str(path.relative_to(derived_root)))
            candidate = candidates.get(product_id) if product_id else None
            if candidate is None:
                candidate = by_title.get(normalized_title(chapter_title(path)))
            if candidate is None and product_id:
                candidate = self._candidate(candidates, product_id, product_id, product_id)
                candidate.reasons.append("derived_only")
                duplicate_sources.setdefault(product_id, set()).add(str(path.parent))
            if candidate is None:
                ignored.append({"path": str(path), "reason": "unmatched_derived_asset"})
                continue
            candidate.assets.append(
                self._asset(
                    base,
                    path,
                    derived_root,
                    derived=kind == "subtitle",
                    source_work_id=candidate.product_id,
                )
            )

    @staticmethod
    def _asset(base: Path, path: Path, work_root: Path, *, derived: bool, source_work_id: str) -> AssetPreview:
        try:
            stat = path.stat()
        except OSError:
            stat = type("Stat", (), {"st_size": 0, "st_mtime_ns": 0})()
        try:
            relative = str(path.relative_to(base))
        except ValueError:
            relative = path.name
        try:
            nested = path.relative_to(work_root).parts[:-1]
        except ValueError:
            nested = ()
        file_identity = ""
        if getattr(stat, "st_ino", 0):
            file_identity = f"{int(getattr(stat, 'st_dev', 0))}:{int(stat.st_ino)}"
        content_value = f"{max(0, int(stat.st_size))}\0{max(0, int(stat.st_mtime_ns))}"
        content_fingerprint = hashlib.sha256(content_value.encode("ascii")).hexdigest()
        return AssetPreview(
            path=str(path.resolve()),
            relative_path=relative,
            kind=asset_kind(path),
            extension=path.suffix.casefold(),
            size=max(0, int(stat.st_size)),
            modified_ns=max(0, int(stat.st_mtime_ns)),
            chapter=chapter_title(path),
            version=resource_variant(path, nested),
            derived=derived,
            source_work_id=source_work_id,
            section=resource_section(path),
            role=resource_role(path),
            group_key=resource_group_key(path),
            language=subtitle_language(path),
            variant=resource_variant(path, nested),
            sort_order=resource_sort_order(path),
            content_fingerprint=content_fingerprint,
            file_identity=file_identity,
        )

    @staticmethod
    def _classify(works: list[WorkPreview], duplicate_sources: dict[str, set[str]]) -> None:
        title_groups: dict[str, list[WorkPreview]] = {}
        for work in works:
            if work.normalized_title:
                title_groups.setdefault(work.normalized_title, []).append(work)

        for work in works:
            playable = sum(asset.kind in {"audio", "video"} for asset in work.assets)
            if work.product_id:
                work.confidence = 0.98
                work.classification = "auto"
                if "official_product_id" not in work.reasons and "unique_title_match" not in work.reasons:
                    work.reasons.append("official_product_id")
            elif playable and work.normalized_title:
                work.confidence = 0.72
                generic = work.normalized_title in GENERIC_COLLECTION_NAMES or work.normalized_title.startswith("新建文件夹")
                work.classification = "auto" if len(work.normalized_title) >= 6 and not generic else "unclassified"
                work.reasons.append("stable_folder_title")
                if generic:
                    work.reasons.append("generic_collection_name")
            else:
                work.confidence = 0.25
                work.classification = "unclassified"
                work.reasons.append("no_playable_media")

            same_title = title_groups.get(work.normalized_title, []) if work.normalized_title else []
            meaningful_title = len(work.normalized_title) >= 8 and work.normalized_title not in GENERIC_COLLECTION_NAMES
            if meaningful_title and len(same_title) > 1:
                for peer in same_title:
                    if peer is not work and peer.product_id != work.product_id:
                        work.classification = "duplicate"
                        work.confidence = min(work.confidence, 0.55)
                        work.reasons.append("same_normalized_title")
                        break

        MediaLibraryScanner._series_hints(works)

    @staticmethod
    def _series_hints(works: list[WorkPreview]) -> None:
        buckets: dict[str, list[WorkPreview]] = {}
        for work in works:
            tokens = [token for token in work.normalized_title.split() if len(token) >= 2]
            if not tokens:
                continue
            prefix = " ".join(tokens[:2])
            buckets.setdefault(prefix, []).append(work)
        for prefix, members in buckets.items():
            if len(members) < 2:
                continue
            for work in members:
                work.series_hint = prefix
                if work.classification != "duplicate":
                    work.reasons.append("possible_series")


class MediaLibraryDatabase:
    """Thread-safe SQLite catalogue stored outside all media directories."""

    def __init__(self, path: Path | str):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS library_meta(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS scan_runs(
                    id TEXT PRIMARY KEY,
                    root TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    applied_at REAL,
                    summary_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS works(
                    id TEXT PRIMARY KEY,
                    product_id TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL,
                    title_ja TEXT NOT NULL DEFAULT '',
                    title_zh TEXT NOT NULL DEFAULT '',
                    description_ja TEXT NOT NULL DEFAULT '',
                    description_zh TEXT NOT NULL DEFAULT '',
                    maker TEXT NOT NULL DEFAULT '',
                    series TEXT NOT NULL DEFAULT '',
                    rating REAL,
                    rating_count INTEGER,
                    release_date TEXT NOT NULL DEFAULT '',
                    age_rating TEXT NOT NULL DEFAULT '',
                    cover_url TEXT NOT NULL DEFAULT '',
                    manual_cover_asset_id TEXT NOT NULL DEFAULT '',
                    metadata_source TEXT NOT NULL DEFAULT 'local',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    confidence REAL NOT NULL DEFAULT 0,
                    classification TEXT NOT NULL DEFAULT 'unclassified',
                    series_hint TEXT NOT NULL DEFAULT '',
                    revision INTEGER NOT NULL DEFAULT 1,
                    updated_at REAL NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS works_product_id
                    ON works(product_id) WHERE product_id<>'';
                CREATE TABLE IF NOT EXISTS assets(
                    id TEXT PRIMARY KEY,
                    work_id TEXT NOT NULL REFERENCES works(id) ON DELETE CASCADE,
                    path TEXT NOT NULL UNIQUE,
                    relative_path TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    size INTEGER NOT NULL,
                    modified_ns INTEGER NOT NULL,
                    chapter TEXT NOT NULL,
                    version TEXT NOT NULL,
                    derived INTEGER NOT NULL DEFAULT 0,
                    fingerprint TEXT NOT NULL,
                    duration_ms INTEGER,
                    bitrate INTEGER,
                    sample_rate INTEGER,
                    channels INTEGER,
                    available INTEGER NOT NULL DEFAULT 1,
                    section TEXT NOT NULL DEFAULT 'main',
                    role TEXT NOT NULL DEFAULT 'attachment',
                    group_key TEXT NOT NULL DEFAULT '',
                    language TEXT NOT NULL DEFAULT '',
                    variant TEXT NOT NULL DEFAULT '',
                    sort_order INTEGER NOT NULL DEFAULT 1000000,
                    content_fingerprint TEXT NOT NULL DEFAULT '',
                    file_identity TEXT NOT NULL DEFAULT '',
                    mime_type TEXT NOT NULL DEFAULT 'application/octet-stream',
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS assets_work_kind ON assets(work_id, kind);
                CREATE TABLE IF NOT EXISTS work_people(
                    work_id TEXT NOT NULL REFERENCES works(id) ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    name TEXT NOT NULL,
                    PRIMARY KEY(work_id, role, name)
                );
                CREATE TABLE IF NOT EXISTS work_tags(
                    work_id TEXT NOT NULL REFERENCES works(id) ON DELETE CASCADE,
                    tag TEXT NOT NULL,
                    PRIMARY KEY(work_id, tag)
                );
                CREATE TABLE IF NOT EXISTS inbox(
                    id TEXT PRIMARY KEY,
                    work_id TEXT NOT NULL REFERENCES works(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at REAL NOT NULL,
                    UNIQUE(work_id, kind, reason)
                );
                CREATE TABLE IF NOT EXISTS metadata_cache(
                    cache_key TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    status INTEGER NOT NULL,
                    fetched_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    etag TEXT NOT NULL DEFAULT '',
                    last_modified TEXT NOT NULL DEFAULT '',
                    payload BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS organizer_plans(
                    id TEXT PRIMARY KEY,
                    root TEXT NOT NULL,
                    root_revision TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    state TEXT NOT NULL,
                    device_id TEXT NOT NULL DEFAULT '',
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS file_operations(
                    id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL REFERENCES organizer_plans(id),
                    state TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    completed_at REAL,
                    journal_json TEXT NOT NULL DEFAULT '[]',
                    error TEXT NOT NULL DEFAULT ''
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS works_fts USING fts5(
                    work_id UNINDEXED, title, title_ja, title_zh, maker, series, tags,
                    tokenize='unicode61'
                );
                """
            )
            self._ensure_schema_v2(connection)
            connection.execute(
                "INSERT INTO library_meta(key,value) VALUES('schema_version',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )

    @staticmethod
    def _ensure_schema_v2(connection: sqlite3.Connection) -> None:
        additions = {
            "works": {
                "manual_cover_asset_id": "TEXT NOT NULL DEFAULT ''",
            },
            "assets": {
                "section": "TEXT NOT NULL DEFAULT 'main'",
                "role": "TEXT NOT NULL DEFAULT 'attachment'",
                "group_key": "TEXT NOT NULL DEFAULT ''",
                "language": "TEXT NOT NULL DEFAULT ''",
                "variant": "TEXT NOT NULL DEFAULT ''",
                "sort_order": "INTEGER NOT NULL DEFAULT 1000000",
                "content_fingerprint": "TEXT NOT NULL DEFAULT ''",
                "file_identity": "TEXT NOT NULL DEFAULT ''",
                "mime_type": "TEXT NOT NULL DEFAULT 'application/octet-stream'",
            },
        }
        for table, columns in additions.items():
            existing = {str(row["name"]) for row in connection.execute(f"PRAGMA table_info({table})")}
            for name, declaration in columns.items():
                if name not in existing:
                    connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {declaration}")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS assets_work_group ON assets(work_id,group_key,sort_order)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS assets_file_identity ON assets(file_identity) WHERE file_identity<>''"
        )

    def apply_preview(self, preview: ImportPreview) -> str:
        scan_id = uuid.uuid4().hex
        now = time.time()
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "INSERT INTO scan_runs(id,root,created_at,applied_at,summary_json) VALUES(?,?,?,?,?)",
                    (scan_id, preview.root, preview.created_at, now, json.dumps(preview.summary, ensure_ascii=False)),
                )
                seen_paths: set[str] = set()
                for work in preview.works:
                    self._upsert_work(connection, work, now)
                    connection.execute("DELETE FROM inbox WHERE work_id=?", (work.work_id,))
                    for asset in work.assets:
                        seen_paths.add(_path_key(Path(asset.path)))
                        self._upsert_asset(connection, work.work_id, asset, now)
                    if work.classification != "auto":
                        reasons = work.reasons or [work.classification]
                        for reason in reasons:
                            inbox_id = hashlib.sha256(
                                f"{work.work_id}\0{work.classification}\0{reason}".encode("utf-8")
                            ).hexdigest()
                            connection.execute(
                                "INSERT OR IGNORE INTO inbox(id,work_id,kind,reason,created_at) VALUES(?,?,?,?,?)",
                                (inbox_id, work.work_id, work.classification, reason, now),
                            )
                root_prefix = str(Path(preview.root).resolve()).casefold().rstrip("\\/") + "%"
                for row in connection.execute("SELECT id,path FROM assets WHERE lower(path) LIKE ?", (root_prefix,)):
                    if _path_key(Path(row["path"])) not in seen_paths:
                        connection.execute("UPDATE assets SET available=0,updated_at=? WHERE id=?", (now, row["id"]))
                connection.execute(
                    "DELETE FROM works WHERE id LIKE 'local:%' AND metadata_source='local' "
                    "AND NOT EXISTS(SELECT 1 FROM assets a WHERE a.work_id=works.id AND a.available=1)"
                )
                connection.execute(
                    "INSERT INTO library_meta(key,value) VALUES('root_revision',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (scan_id,),
                )
                self._rebuild_fts(connection)
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return scan_id

    @staticmethod
    def _upsert_work(connection: sqlite3.Connection, work: WorkPreview, now: float) -> None:
        existing = connection.execute("SELECT revision FROM works WHERE id=?", (work.work_id,)).fetchone()
        revision = int(existing["revision"]) + 1 if existing else 1
        connection.execute(
            """
            INSERT INTO works(id,product_id,title,confidence,classification,series_hint,revision,updated_at)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                product_id=excluded.product_id,
                title=CASE WHEN works.metadata_source='local' THEN excluded.title ELSE works.title END,
                confidence=excluded.confidence,
                classification=excluded.classification,
                series_hint=excluded.series_hint,
                revision=excluded.revision,
                updated_at=excluded.updated_at
            """,
            (
                work.work_id,
                work.product_id,
                work.title,
                work.confidence,
                work.classification,
                work.series_hint,
                revision,
                now,
            ),
        )

    @staticmethod
    def _upsert_asset(connection: sqlite3.Connection, work_id: str, asset: AssetPreview, now: float) -> None:
        existing = connection.execute("SELECT id,path FROM assets WHERE path=?", (asset.path,)).fetchone()
        if existing is None and asset.file_identity:
            identity = connection.execute(
                "SELECT id,path FROM assets WHERE file_identity=? ORDER BY available DESC,updated_at DESC LIMIT 1",
                (asset.file_identity,),
            ).fetchone()
            if identity is not None and not Path(str(identity["path"])).exists():
                existing = identity
        asset_id = str(existing["id"]) if existing is not None else hashlib.sha256(
            _path_key(Path(asset.path)).encode("utf-8")
        ).hexdigest()
        if existing is not None and str(existing["path"]) != asset.path:
            connection.execute("UPDATE assets SET path=? WHERE id=?", (asset.path, asset_id))
        mime_type = mimetypes.guess_type(Path(asset.path).name)[0] or "application/octet-stream"
        connection.execute(
            """
            INSERT INTO assets(
                id,work_id,path,relative_path,kind,extension,size,modified_ns,chapter,version,
                derived,fingerprint,available,section,role,group_key,language,variant,sort_order,
                content_fingerprint,file_identity,mime_type,updated_at
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,1,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                work_id=excluded.work_id,relative_path=excluded.relative_path,kind=excluded.kind,
                extension=excluded.extension,size=excluded.size,modified_ns=excluded.modified_ns,
                chapter=excluded.chapter,version=excluded.version,derived=excluded.derived,
                fingerprint=excluded.fingerprint,available=1,section=excluded.section,role=excluded.role,
                group_key=excluded.group_key,language=excluded.language,variant=excluded.variant,
                sort_order=excluded.sort_order,content_fingerprint=excluded.content_fingerprint,
                file_identity=excluded.file_identity,mime_type=excluded.mime_type,updated_at=excluded.updated_at
            """,
            (
                asset_id,
                work_id,
                asset.path,
                asset.relative_path,
                asset.kind,
                asset.extension,
                asset.size,
                asset.modified_ns,
                asset.chapter,
                asset.version,
                int(asset.derived),
                asset.fingerprint,
                asset.section,
                asset.role,
                asset.group_key,
                asset.language,
                asset.variant,
                asset.sort_order,
                asset.content_fingerprint,
                asset.file_identity,
                mime_type,
                now,
            ),
        )

    @staticmethod
    def _rebuild_fts(connection: sqlite3.Connection) -> None:
        connection.execute("DELETE FROM works_fts")
        connection.execute(
            """
            INSERT INTO works_fts(work_id,title,title_ja,title_zh,maker,series,tags)
            SELECT w.id,w.title,w.title_ja,w.title_zh,w.maker,w.series,
                   COALESCE((SELECT group_concat(tag,' ') FROM work_tags t WHERE t.work_id=w.id),'')
            FROM works w
            """
        )

    def update_metadata(self, work_id: str, metadata: dict[str, Any]) -> None:
        now = time.time()
        people = metadata.get("people") if isinstance(metadata.get("people"), dict) else {}
        tags = [str(tag).strip() for tag in metadata.get("tags", []) if str(tag).strip()]
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    UPDATE works SET
                        title=COALESCE(NULLIF(?,''),title),
                        title_ja=CASE WHEN ? THEN ? ELSE title_ja END,
                        title_zh=CASE WHEN ? THEN ? ELSE title_zh END,
                        description_ja=CASE WHEN ? THEN ? ELSE description_ja END,
                        description_zh=CASE WHEN ? THEN ? ELSE description_zh END,
                        maker=COALESCE(NULLIF(?,''),maker), series=COALESCE(NULLIF(?,''),series),
                        rating=?, rating_count=?, release_date=COALESCE(NULLIF(?,''),release_date),
                        age_rating=COALESCE(NULLIF(?,''),age_rating),
                        cover_url=COALESCE(NULLIF(?,''),cover_url), metadata_source=?, metadata_json=?,
                        revision=revision+1,updated_at=? WHERE id=?
                    """,
                    (
                        metadata.get("title_zh") or metadata.get("title_ja") or metadata.get("title", ""),
                        "title_ja" in metadata,
                        metadata.get("title_ja") or "",
                        "title_zh" in metadata,
                        metadata.get("title_zh") or "",
                        "description_ja" in metadata,
                        metadata.get("description_ja") or "",
                        "description_zh" in metadata,
                        metadata.get("description_zh") or "",
                        metadata.get("maker", ""),
                        metadata.get("series", ""),
                        metadata.get("rating"),
                        metadata.get("rating_count"),
                        metadata.get("release_date", ""),
                        metadata.get("age_rating", ""),
                        metadata.get("cover_url", ""),
                        metadata.get("source", "dlsite"),
                        json.dumps(metadata, ensure_ascii=False),
                        now,
                        work_id,
                    ),
                )
                connection.execute("DELETE FROM work_people WHERE work_id=?", (work_id,))
                for role, names in people.items():
                    for name in names if isinstance(names, list) else [names]:
                        if str(name).strip():
                            connection.execute(
                                "INSERT OR IGNORE INTO work_people(work_id,role,name) VALUES(?,?,?)",
                                (work_id, str(role), str(name).strip()),
                            )
                connection.execute("DELETE FROM work_tags WHERE work_id=?", (work_id,))
                connection.executemany(
                    "INSERT OR IGNORE INTO work_tags(work_id,tag) VALUES(?,?)",
                    [(work_id, tag) for tag in tags],
                )
                self._rebuild_fts(connection)
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def sync(self, since: float = 0.0, limit: int = 500) -> dict[str, Any]:
        safe_limit = min(2000, max(1, int(limit)))
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * FROM works WHERE updated_at>? ORDER BY updated_at,id LIMIT ?", (float(since), safe_limit)
            ).fetchall()
            works = [self._work_row(connection, row, include_assets=True) for row in rows]
            cursor = max([float(row["updated_at"]) for row in rows], default=float(since))
            return {"cursor": cursor, "has_more": len(rows) == safe_limit, "works": works}

    def status(self) -> dict[str, Any]:
        with self._connection() as connection:
            counts = connection.execute(
                """SELECT
                       (SELECT COUNT(*) FROM works) AS works,
                       (SELECT COUNT(*) FROM assets WHERE available=1) AS assets,
                       (SELECT COUNT(*) FROM assets WHERE available=1 AND kind IN ('audio','video')) AS audio,
                       (SELECT COUNT(*) FROM assets WHERE available=1 AND kind='subtitle') AS subtitles,
                       (SELECT COUNT(*) FROM assets WHERE available=1 AND kind='image') AS images,
                       (SELECT COUNT(*) FROM assets WHERE available=1 AND kind IN ('document','archive','attachment')) AS documents,
                       (SELECT COUNT(*) FROM assets WHERE available=1 AND section='bonus') AS bonus,
                       (SELECT COUNT(*) FROM assets WHERE available=1 AND section='incomplete') AS incomplete"""
            ).fetchone()
            latest = connection.execute(
                "SELECT root,applied_at FROM scan_runs WHERE applied_at IS NOT NULL ORDER BY applied_at DESC LIMIT 1"
            ).fetchone()
            value = dict(counts)
            value["root"] = str(latest["root"]) if latest else ""
            value["applied_at"] = float(latest["applied_at"]) if latest else 0.0
            return value

    def title_aliases(self) -> dict[str, str]:
        """Known official titles used for conservative, unique filename matching."""
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id,title,title_ja,title_zh FROM works WHERE product_id<>''"
            ).fetchall()
            return {
                str(row["id"]): str(row["title_zh"] or row["title_ja"] or row["title"])
                for row in rows
                if str(row["title_zh"] or row["title_ja"] or row["title"]).strip()
            }

    def root_revision(self) -> str:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT value FROM library_meta WHERE key='root_revision'"
            ).fetchone()
            return str(row["value"]) if row else ""

    def organizer_preview(
        self,
        *,
        asset_ids: Sequence[str] = (),
        work_ids: Sequence[str] = (),
        action: str = "organize",
        destination: str = "",
        new_name: str = "",
        device_id: str = "",
    ) -> dict[str, Any]:
        if action not in {"organize", "move", "rename", "trash"}:
            raise ValueError("unsupported organizer action")
        now = time.time()
        plan_id = uuid.uuid4().hex
        with self._lock, self._connection() as connection:
            latest = connection.execute(
                "SELECT root FROM scan_runs WHERE applied_at IS NOT NULL ORDER BY applied_at DESC LIMIT 1"
            ).fetchone()
            if not latest:
                raise ValueError("media library has not been scanned")
            root = Path(str(latest["root"])).resolve(strict=True)
            revision = self.root_revision()
            conditions = ["a.available=1"]
            params: list[Any] = []
            clean_assets = [str(value) for value in asset_ids if str(value)]
            clean_works = [str(value) for value in work_ids if str(value)]
            if clean_assets:
                conditions.append(f"a.id IN ({','.join('?' for _ in clean_assets)})")
                params.extend(clean_assets)
            elif clean_works:
                conditions.append(f"a.work_id IN ({','.join('?' for _ in clean_works)})")
                params.extend(clean_works)
            else:
                raise ValueError("asset_ids or work_ids required")
            rows = connection.execute(
                "SELECT a.*,w.product_id,w.title,w.title_ja,w.title_zh FROM assets a "
                f"JOIN works w ON w.id=a.work_id WHERE {' AND '.join(conditions)} ORDER BY a.path",
                params,
            ).fetchall()
            if not rows:
                raise ValueError("no available assets selected")
            if action == "rename" and len(rows) != 1:
                raise ValueError("rename requires exactly one asset")
            operations: list[dict[str, Any]] = []
            for row in rows:
                source = Path(str(row["path"])).resolve()
                if not _inside(root, source) or source.is_symlink():
                    status, target = "outside_root", source
                else:
                    target = self._organizer_target(
                        root,
                        row,
                        action=action,
                        destination=destination,
                        new_name=new_name,
                        plan_id=plan_id,
                    )
                    status = "noop" if _path_key(source) == _path_key(target) else "ready"
                    if not _inside(root, target):
                        status = "outside_root"
                    elif target.exists() and _path_key(target) != _path_key(source):
                        status = "conflict"
                operations.append({
                    "asset_id": str(row["id"]),
                    "work_id": str(row["work_id"]),
                    "action": action,
                    "source": str(source),
                    "target": str(target),
                    "size": int(row["size"]),
                    "fingerprint": str(row["fingerprint"]),
                    "status": status,
                })
            payload = {
                "id": plan_id,
                "root": str(root),
                "root_revision": revision,
                "created_at": now,
                "expires_at": now + 15 * 60,
                "state": "pending",
                "operations": operations,
                "summary": {
                    "total": len(operations),
                    "ready": sum(item["status"] == "ready" for item in operations),
                    "noop": sum(item["status"] == "noop" for item in operations),
                    "conflicts": sum(item["status"] not in {"ready", "noop"} for item in operations),
                    "bytes": sum(item["size"] for item in operations if item["status"] == "ready"),
                },
            }
            connection.execute(
                "INSERT INTO organizer_plans(id,root,root_revision,created_at,expires_at,state,device_id,payload_json) "
                "VALUES(?,?,?,?,?,'pending',?,?)",
                (plan_id, str(root), revision, now, now + 15 * 60, device_id, json.dumps(payload, ensure_ascii=False)),
            )
            return payload

    @staticmethod
    def _organizer_target(
        root: Path,
        row: sqlite3.Row,
        *,
        action: str,
        destination: str,
        new_name: str,
        plan_id: str,
    ) -> Path:
        source = Path(str(row["path"]))
        if action == "trash":
            relative = Path(str(row["relative_path"]))
            return root / ".voicetransl_trash" / plan_id / relative
        if action == "rename":
            name = _safe_component(new_name, source.stem)
            if not Path(name).suffix:
                name += source.suffix
            return source.parent / name
        if action == "move":
            relative = Path(destination)
            if relative.is_absolute():
                return relative / source.name
            return root / relative / source.name
        title = str(row["title_zh"] or row["title_ja"] or row["title"])
        product_id = str(row["product_id"] or "")
        title_without_id = PRODUCT_ID_RE.sub("", title).strip(" []（）()_-")
        work_folder = _safe_component(
            f"{product_id} {title_without_id}".strip(),
            product_id or str(row["work_id"]),
        )
        section = str(row["section"])
        kind = str(row["kind"])
        if section == "bonus":
            category = "特典"
        else:
            category = {
                "audio": "音频", "video": "音频", "subtitle": "字幕", "image": "图片",
                "document": "文档", "archive": "压缩包",
            }.get(kind, "其他")
        return root / work_folder / category / source.name

    def organizer_plan(self, plan_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM organizer_plans WHERE id=?", (plan_id,)).fetchone()
            if not row:
                return None
            payload = json.loads(str(row["payload_json"]))
            payload["state"] = str(row["state"])
            operation = connection.execute(
                "SELECT * FROM file_operations WHERE plan_id=? ORDER BY created_at DESC LIMIT 1", (plan_id,)
            ).fetchone()
            if operation:
                payload["operation"] = dict(operation)
                payload["operation"]["journal"] = json.loads(payload["operation"].pop("journal_json"))
            return payload

    def apply_organizer_plan(self, plan_id: str) -> dict[str, Any]:
        operation_id = uuid.uuid4().hex
        now = time.time()
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT * FROM organizer_plans WHERE id=?", (plan_id,)).fetchone()
            if not row:
                raise KeyError("organizer plan not found")
            if str(row["state"]) != "pending":
                raise ValueError("organizer plan is not pending")
            if float(row["expires_at"]) < now:
                connection.execute("UPDATE organizer_plans SET state='expired' WHERE id=?", (plan_id,))
                raise ValueError("organizer plan expired")
            if str(row["root_revision"]) != self.root_revision():
                connection.execute("UPDATE organizer_plans SET state='stale' WHERE id=?", (plan_id,))
                raise ValueError("media library changed; create a new preview")
            payload = json.loads(str(row["payload_json"]))
            blocked = [item for item in payload["operations"] if item["status"] not in {"ready", "noop"}]
            if blocked:
                raise ValueError("organizer plan contains conflicts")
            root = Path(str(row["root"])).resolve(strict=True)
            connection.execute(
                "INSERT INTO file_operations(id,plan_id,state,created_at) VALUES(?,?,'running',?)",
                (operation_id, plan_id, now),
            )
            journal: list[dict[str, Any]] = []
            try:
                for item in payload["operations"]:
                    if item["status"] == "noop":
                        continue
                    source = Path(item["source"])
                    target = Path(item["target"])
                    if not source.is_file() or source.is_symlink() or not _inside(root, source) or not _inside(root, target):
                        raise ValueError(f"unsafe or missing source: {source}")
                    if _path_fingerprint(source) != item["fingerprint"]:
                        raise ValueError(f"source changed after preview: {source}")
                    if target.exists():
                        raise FileExistsError(str(target))
                    target.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(source, target)
                    available = 0 if item["action"] == "trash" else 1
                    stat = target.stat()
                    connection.execute(
                        "UPDATE assets SET path=?,relative_path=?,modified_ns=?,fingerprint=?,file_identity=?,"
                        "available=?,updated_at=? WHERE id=?",
                        (
                            str(target.resolve()), str(target.relative_to(root)), int(stat.st_mtime_ns),
                            _path_fingerprint(target), f"{int(stat.st_dev)}:{int(stat.st_ino)}" if stat.st_ino else "",
                            available, time.time(), item["asset_id"],
                        ),
                    )
                    journal.append({
                        "asset_id": item["asset_id"], "source": str(source), "target": str(target),
                        "relative_before": str(source.relative_to(root)),
                        "relative_after": str(target.relative_to(root)),
                        "available_before": 1, "available_after": available,
                    })
                completed = time.time()
                connection.execute(
                    "UPDATE file_operations SET state='succeeded',completed_at=?,journal_json=? WHERE id=?",
                    (completed, json.dumps(journal, ensure_ascii=False), operation_id),
                )
                connection.execute("UPDATE organizer_plans SET state='applied' WHERE id=?", (plan_id,))
                return {"id": operation_id, "plan_id": plan_id, "state": "succeeded", "journal": journal}
            except Exception as error:
                rollback_error = ""
                try:
                    for item in reversed(journal):
                        source, target = Path(item["source"]), Path(item["target"])
                        if target.is_file() and not source.exists():
                            source.parent.mkdir(parents=True, exist_ok=True)
                            os.replace(target, source)
                            stat = source.stat()
                            connection.execute(
                                "UPDATE assets SET path=?,relative_path=?,modified_ns=?,fingerprint=?,file_identity=?,"
                                "available=?,updated_at=? WHERE id=?",
                                (
                                    str(source.resolve()), item["relative_before"], int(stat.st_mtime_ns),
                                    _path_fingerprint(source),
                                    f"{int(stat.st_dev)}:{int(stat.st_ino)}" if stat.st_ino else "",
                                    int(item["available_before"]), time.time(), item["asset_id"],
                                ),
                            )
                except Exception as rollback_failure:
                    rollback_error = f"; rollback failed: {rollback_failure}"
                connection.execute(
                    "UPDATE file_operations SET state='failed',completed_at=?,journal_json=?,error=? WHERE id=?",
                    (time.time(), json.dumps(journal, ensure_ascii=False), str(error) + rollback_error, operation_id),
                )
                connection.execute("UPDATE organizer_plans SET state='failed' WHERE id=?", (plan_id,))
                raise

    def undo_organizer_operation(self, operation_id: str) -> dict[str, Any]:
        with self._lock, self._connection() as connection:
            row = connection.execute("SELECT * FROM file_operations WHERE id=?", (operation_id,)).fetchone()
            if not row:
                raise KeyError("organizer operation not found")
            if str(row["state"]) != "succeeded":
                raise ValueError("organizer operation cannot be undone")
            journal = json.loads(str(row["journal_json"]))
            for item in reversed(journal):
                source, target = Path(item["source"]), Path(item["target"])
                if not target.is_file() or source.exists():
                    raise ValueError(f"cannot restore {source}")
                source.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, source)
                stat = source.stat()
                connection.execute(
                    "UPDATE assets SET path=?,relative_path=?,modified_ns=?,fingerprint=?,file_identity=?,available=?,updated_at=? WHERE id=?",
                    (
                        str(source.resolve()), item["relative_before"], int(stat.st_mtime_ns), _path_fingerprint(source),
                        f"{int(stat.st_dev)}:{int(stat.st_ino)}" if stat.st_ino else "",
                        int(item["available_before"]), time.time(), item["asset_id"],
                    ),
                )
            connection.execute(
                "UPDATE file_operations SET state='undone',completed_at=? WHERE id=?", (time.time(), operation_id)
            )
            return {"id": operation_id, "state": "undone", "restored": len(journal)}

    def reassign_asset(self, asset_id: str, work_id: str) -> dict[str, Any]:
        with self._lock, self._connection() as connection:
            if connection.execute("SELECT 1 FROM works WHERE id=?", (work_id,)).fetchone() is None:
                raise KeyError("target work not found")
            if connection.execute("SELECT 1 FROM assets WHERE id=?", (asset_id,)).fetchone() is None:
                raise KeyError("asset not found")
            now = time.time()
            connection.execute("UPDATE assets SET work_id=?,updated_at=? WHERE id=?", (work_id, now, asset_id))
            connection.execute("UPDATE works SET revision=revision+1,updated_at=? WHERE id=?", (now, work_id))
            return self.asset(asset_id) or {}

    def set_manual_cover(self, work_id: str, asset_id: str = "") -> dict[str, Any]:
        with self._lock, self._connection() as connection:
            if asset_id:
                row = connection.execute(
                    "SELECT 1 FROM assets WHERE id=? AND work_id=? AND kind='image' AND available=1",
                    (asset_id, work_id),
                ).fetchone()
                if row is None:
                    raise ValueError("cover must be an available image from this work")
            now = time.time()
            connection.execute(
                "UPDATE works SET manual_cover_asset_id=?,revision=revision+1,updated_at=? WHERE id=?",
                (asset_id, now, work_id),
            )
            work = connection.execute("SELECT * FROM works WHERE id=?", (work_id,)).fetchone()
            if not work:
                raise KeyError("work not found")
            return self._work_row(connection, work, include_assets=True)

    def search(
        self,
        query: str = "",
        *,
        tag: str = "",
        maker: str = "",
        series: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        safe_limit = min(500, max(1, int(limit)))
        safe_offset = max(0, int(offset))
        conditions = ["1=1"]
        params: list[Any] = []
        join = ""
        query = query.strip()
        if query:
            join += " JOIN works_fts f ON f.work_id=w.id"
            conditions.append("works_fts MATCH ?")
            params.append('"' + query.replace('"', '""') + '"*')
        if tag:
            conditions.append("EXISTS(SELECT 1 FROM work_tags t WHERE t.work_id=w.id AND t.tag=?)")
            params.append(tag)
        if maker:
            conditions.append("w.maker LIKE ?")
            params.append(f"%{maker}%")
        if series:
            conditions.append("w.series LIKE ?")
            params.append(f"%{series}%")
        sql = f"SELECT DISTINCT w.* FROM works w{join} WHERE {' AND '.join(conditions)} ORDER BY w.title LIMIT ? OFFSET ?"
        params.extend([safe_limit, safe_offset])
        with self._connection() as connection:
            rows = connection.execute(sql, params).fetchall()
            return {
                "offset": safe_offset,
                "limit": safe_limit,
                "works": [self._work_row(connection, row, include_assets=False) for row in rows],
            }

    def work(self, work_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute("SELECT * FROM works WHERE id=?", (work_id,)).fetchone()
            return self._work_row(connection, row, include_assets=True) if row else None

    def ensure_metadata_work(self, product_id: str) -> str:
        """Return a catalogue work for a valid DLsite id, creating a metadata-only row if needed."""
        identifier = normalize_product_id(product_id)
        if not identifier:
            raise ValueError("invalid DLsite product id")
        now = time.time()
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT id FROM works WHERE id=? OR product_id=? LIMIT 1",
                (identifier, identifier),
            ).fetchone()
            if row:
                return str(row["id"])
            connection.execute(
                """INSERT INTO works(
                       id,product_id,title,confidence,classification,revision,updated_at
                   ) VALUES(?,?,?,0.25,'metadata_only',1,?)""",
                (identifier, identifier, identifier, now),
            )
            self._rebuild_fts(connection)
        return identifier

    def metadata_candidates(self, *, missing_only: bool = True) -> list[dict[str, str]]:
        conditions = ["product_id<>''"]
        if missing_only:
            conditions.append(
                "(metadata_source='local' OR title_ja='' OR cover_url='' "
                "OR NOT EXISTS(SELECT 1 FROM work_tags t WHERE t.work_id=works.id))"
            )
        with self._connection() as connection:
            return [
                {"id": str(row["id"]), "product_id": str(row["product_id"])}
                for row in connection.execute(
                    f"SELECT id,product_id FROM works WHERE {' AND '.join(conditions)} ORDER BY product_id"
                )
            ]

    def inbox(self, status: str = "open") -> list[dict[str, Any]]:
        with self._connection() as connection:
            return [
                dict(row)
                for row in connection.execute(
                    "SELECT i.*,w.product_id,w.title FROM inbox i JOIN works w ON w.id=i.work_id "
                    "WHERE i.status=? ORDER BY i.created_at,i.id",
                    (status,),
                )
            ]

    def asset(self, asset_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT a.*,w.title AS work_title,w.product_id FROM assets a JOIN works w ON w.id=a.work_id "
                "WHERE a.id=? AND a.available=1",
                (asset_id,),
            ).fetchone()
            if not row:
                return None
            value = dict(row)
            path = Path(value["path"])
            if not path.is_file():
                return None
            value["mime_type"] = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            return value

    def metadata_cache_get(self, cache_key: str) -> dict[str, Any] | None:
        now = time.time()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM metadata_cache WHERE cache_key=? AND expires_at>?",
                (cache_key, now),
            ).fetchone()
            if not row:
                return None
            value = dict(row)
            value["payload"] = bytes(value["payload"])
            return value

    def metadata_cache_put(
        self,
        cache_key: str,
        url: str,
        status: int,
        payload: bytes,
        *,
        ttl_seconds: float,
        etag: str = "",
        last_modified: str = "",
    ) -> None:
        now = time.time()
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                INSERT INTO metadata_cache(cache_key,url,status,fetched_at,expires_at,etag,last_modified,payload)
                VALUES(?,?,?,?,?,?,?,?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    url=excluded.url,status=excluded.status,fetched_at=excluded.fetched_at,
                    expires_at=excluded.expires_at,etag=excluded.etag,
                    last_modified=excluded.last_modified,payload=excluded.payload
                """,
                (
                    cache_key,
                    url,
                    int(status),
                    now,
                    now + max(60.0, float(ttl_seconds)),
                    etag,
                    last_modified,
                    sqlite3.Binary(payload),
                ),
            )

    @staticmethod
    def _work_row(connection: sqlite3.Connection, row: sqlite3.Row, include_assets: bool) -> dict[str, Any]:
        value = dict(row)
        try:
            value["metadata"] = json.loads(value.pop("metadata_json"))
        except Exception:
            value["metadata"] = {}
            value.pop("metadata_json", None)
        value["tags"] = [
            item["tag"] for item in connection.execute("SELECT tag FROM work_tags WHERE work_id=? ORDER BY tag", (row["id"],))
        ]
        value["people"] = {}
        for item in connection.execute("SELECT role,name FROM work_people WHERE work_id=? ORDER BY role,name", (row["id"],)):
            value["people"].setdefault(item["role"], []).append(item["name"])
        manual_cover = str(value.get("manual_cover_asset_id") or "")
        if manual_cover:
            cover = connection.execute(
                "SELECT id FROM assets WHERE id=? AND work_id=? AND kind='image' AND available=1",
                (manual_cover, row["id"]),
            ).fetchone()
        else:
            cover = None
        if cover is None:
            cover = connection.execute(
                "SELECT id FROM assets WHERE work_id=? AND kind='image' AND available=1 "
                "ORDER BY CASE role WHEN 'cover' THEN 0 WHEN 'illustration' THEN 1 ELSE 2 END,sort_order,relative_path LIMIT 1",
                (row["id"],),
            ).fetchone()
        value["cover_asset_id"] = str(cover["id"]) if cover else ""
        if include_assets:
            value["assets"] = [
                dict(item)
                for item in connection.execute(
                    "SELECT id,kind,relative_path,extension,mime_type,size,fingerprint,chapter,version,derived,"
                    "section,role,group_key,language,variant,sort_order,content_fingerprint,duration_ms,bitrate,"
                    "sample_rate,channels,available,updated_at FROM assets WHERE work_id=? "
                    "ORDER BY CASE section WHEN 'main' THEN 0 WHEN 'bonus' THEN 1 ELSE 2 END,sort_order,kind,relative_path",
                    (row["id"],),
                )
            ]
        return value


def write_preview(preview: ImportPreview, destination: Path | str, *, include_assets: bool = True) -> Path:
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(preview.public(include_assets=include_assets), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="VoiceTransl read-only work library scanner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preview_parser = subparsers.add_parser("preview", help="scan without changing source media")
    preview_parser.add_argument("root", type=Path)
    preview_parser.add_argument("--output", type=Path)
    preview_parser.add_argument("--summary-only", action="store_true")
    apply_parser = subparsers.add_parser("apply", help="persist a fresh scan into the independent catalogue")
    apply_parser.add_argument("root", type=Path)
    apply_parser.add_argument("--database", type=Path, required=True)
    search_parser = subparsers.add_parser("search", help="search an existing catalogue")
    search_parser.add_argument("query", nargs="?", default="")
    search_parser.add_argument("--database", type=Path, required=True)
    search_parser.add_argument("--tag", default="")
    search_parser.add_argument("--maker", default="")
    search_parser.add_argument("--series", default="")
    inbox_parser = subparsers.add_parser("inbox", help="show unclassified and duplicate candidates")
    inbox_parser.add_argument("--database", type=Path, required=True)
    enrich_parser = subparsers.add_parser("enrich", help="explicitly fetch cached DLsite metadata")
    enrich_parser.add_argument("product_ids", nargs="+")
    enrich_parser.add_argument("--database", type=Path, required=True)
    enrich_parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "preview":
        preview = MediaLibraryScanner().scan(args.root)
        if args.output:
            write_preview(preview, args.output, include_assets=not args.summary_only)
        print(json.dumps(preview.summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "apply":
        preview = MediaLibraryScanner().scan(args.root)
        scan_id = MediaLibraryDatabase(args.database).apply_preview(preview)
        print(json.dumps({"scan_id": scan_id, "summary": preview.summary}, ensure_ascii=False, indent=2))
        return 0
    database = MediaLibraryDatabase(args.database)
    if args.command == "inbox":
        print(json.dumps({"items": database.inbox()}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "enrich":
        from dlsite_metadata import DlsiteMetadataClient

        client = DlsiteMetadataClient(database)
        results = []
        for product_id in args.product_ids:
            identifier = normalize_product_id(product_id)
            work = database.work(identifier) if identifier else None
            if not work:
                results.append({"product_id": product_id, "error": "work not found"})
                continue
            try:
                metadata = client.enrich(work["id"], identifier, force=args.force)
                results.append({"product_id": identifier, "title": metadata.get("title_zh") or metadata.get("title_ja")})
            except Exception as error:
                results.append({"product_id": identifier, "error": str(error)})
        print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
        return 0
    print(
        json.dumps(
            database.search(args.query, tag=args.tag, maker=args.maker, series=args.series),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
