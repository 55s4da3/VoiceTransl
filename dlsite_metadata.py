"""Explicit, cached DLsite metadata enrichment for the work catalogue."""

from __future__ import annotations

import html as html_module
import json
import re
import threading
import time
from html.parser import HTMLParser
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from media_library import MediaLibraryDatabase, normalize_product_id


DEFAULT_CACHE_TTL = 7 * 24 * 60 * 60
USER_AGENT = "VoiceTransl-MediaLibrary/1.0 (+local personal catalogue)"


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self.parts)).strip()


def strip_html(value: Any) -> str:
    parser = _TextExtractor()
    parser.feed(str(value or ""))
    return html_module.unescape(parser.text())


def _json_ld_documents(page: str) -> Iterable[Any]:
    pattern = re.compile(
        r"<script[^>]+type=[\"']application/ld\+json[\"'][^>]*>(.*?)</script>",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(page):
        raw = html_module.unescape(match.group(1)).strip()
        try:
            yield json.loads(raw)
        except json.JSONDecodeError:
            continue


def _objects(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from _objects(item)


def _first_product(page: str) -> dict[str, Any]:
    for document in _json_ld_documents(page):
        for value in _objects(document):
            kind = value.get("@type")
            kinds = kind if isinstance(kind, list) else [kind]
            if "Product" in kinds:
                return value
    # DLsite currently publishes BreadcrumbList and WebSite JSON-LD on some
    # product pages, but no Product object. Falling back to the first object
    # with a name picks the breadcrumb category (for example "同人") as the
    # work title, which is worse than using the page's explicit work heading.
    return {}


def _meta_content(page: str, key: str) -> str:
    escaped = re.escape(key)
    patterns = [
        rf"<meta[^>]+(?:property|name)=[\"']{escaped}[\"'][^>]+content=[\"'](.*?)[\"']",
        rf"<meta[^>]+content=[\"'](.*?)[\"'][^>]+(?:property|name)=[\"']{escaped}[\"']",
    ]
    for pattern in patterns:
        match = re.search(pattern, page, re.IGNORECASE | re.DOTALL)
        if match:
            return strip_html(match.group(1))
    return ""


def _meta_itemprop_content(page: str, key: str) -> str:
    expected = key.casefold()
    for tag in re.findall(r"<meta\b[^>]*>", page, re.IGNORECASE):
        attributes = {
            name.casefold(): html_module.unescape(value)
            for name, _quote, value in re.findall(
                r"([:\w-]+)\s*=\s*([\"'])(.*?)\2",
                tag,
                re.IGNORECASE | re.DOTALL,
            )
        }
        if attributes.get("itemprop", "").casefold() == expected:
            return strip_html(attributes.get("content", ""))
    return ""


def _element_text_by_id(page: str, element_id: str) -> str:
    escaped = re.escape(element_id)
    match = re.search(
        rf"<(?P<tag>[a-z0-9]+)[^>]+id=[\"']{escaped}[\"'][^>]*>(?P<body>.*?)</(?P=tag)>",
        page,
        re.IGNORECASE | re.DOTALL,
    )
    return strip_html(match.group("body")) if match else ""


def _clean_title(value: Any) -> str:
    title = strip_html(value)
    # The Open Graph title appends "[circle] | DLsite". This fallback is
    # only used when the dedicated #work_name heading is unavailable.
    return re.sub(r"\s*(?:\[[^\]]+\])?\s*\|\s*DLsite.*$", "", title, flags=re.IGNORECASE).strip()


def _clean_description(value: Any) -> str:
    description = strip_html(value)
    # DLsite appends a localized shop advertisement to meta descriptions.
    # Keep the actual synopsis and remove only the recognisable trailing
    # boilerplate rather than truncating on arbitrary mentions of DLsite.
    return re.sub(
        r"\s*[「『\"']?DLsite(?:\s+同人\s*-\s*R18)?[」』\"']?(?:は|是)"
        r"(?:同人誌|同人志).*?$",
        "",
        description,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def _same_localized_text(left: Any, right: Any) -> bool:
    def normalise(value: Any) -> str:
        return re.sub(r"\s+", "", strip_html(value)).casefold()

    left_value = normalise(left)
    return bool(left_value) and left_value == normalise(right)


def _links_by_class(page: str, class_fragment: str) -> list[str]:
    values: list[str] = []
    pattern = re.compile(
        rf"<a[^>]+class=[\"'][^\"']*{re.escape(class_fragment)}[^\"']*[\"'][^>]*>(.*?)</a>",
        re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(page):
        text = strip_html(match.group(1))
        if text and text not in values:
            values.append(text)
    return values


def _outline_rows(page: str) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for match in re.finditer(r"<tr[^>]*>(.*?)</tr>", page, re.IGNORECASE | re.DOTALL):
        row = match.group(1)
        label_match = re.search(r"<th[^>]*>(.*?)</th>", row, re.IGNORECASE | re.DOTALL)
        value_match = re.search(r"<td[^>]*>(.*?)</td>", row, re.IGNORECASE | re.DOTALL)
        if not label_match or not value_match:
            continue
        label = strip_html(label_match.group(1)).casefold()
        links = [
            strip_html(value)
            for value in re.findall(r"<a[^>]*>(.*?)</a>", value_match.group(1), re.IGNORECASE | re.DOTALL)
        ]
        values = [value for value in links if value] or [strip_html(value_match.group(1))]
        rows[label] = [value for value in values if value]
    return rows


def parse_dlsite_page(page: str, *, locale: str, product_id: str) -> dict[str, Any]:
    product = _first_product(page)
    aggregate = product.get("aggregateRating") if isinstance(product.get("aggregateRating"), dict) else {}
    brand = product.get("brand")
    if isinstance(brand, dict):
        brand = brand.get("name", "")
    image = product.get("image", "")
    if isinstance(image, list):
        image = image[0] if image else ""
    offers = product.get("offers") if isinstance(product.get("offers"), dict) else {}
    rows = _outline_rows(page)

    def row(*labels: str) -> list[str]:
        for label, values in rows.items():
            if any(token.casefold() in label for token in labels):
                return values
        return []

    title = (
        _element_text_by_id(page, "work_name")
        or _clean_title(product.get("name"))
        or _clean_title(_meta_content(page, "og:title"))
    )
    description = _clean_description(
        product.get("description") or _meta_content(page, "og:description")
    )
    maker = strip_html(brand) or (_links_by_class(page, "maker_name") or row("サークル", "circle", "社团" ) or [""])[0]
    tags = _links_by_class(page, "main_genre") or _links_by_class(page, "genre") or row("ジャンル", "genre", "分类")
    voice = row("声優", "voice actor", "声优")
    scenario = row("シナリオ", "scenario", "剧本")
    illustrator = row("イラスト", "illustrator", "插画")
    author = row("作者", "author")
    series = (row("シリーズ", "series", "系列") or [""])[0]
    release_date = strip_html(product.get("releaseDate") or offers.get("availabilityStarts"))
    if not release_date:
        release_match = re.search(
            r"(?:販売日|贩卖日|Release date).*?"
            r"(\d{4}(?:[-/]\d{1,2}){2}|\d{4}年\d{1,2}月\d{1,2}日?)",
            page,
            re.DOTALL,
        )
        release_date = strip_html(release_match.group(1)) if release_match else ""
    age_rating = strip_html(product.get("audience", "")) or _meta_content(page, "rating")
    rating = aggregate.get("ratingValue") or _meta_itemprop_content(page, "ratingValue")
    rating_count = (
        aggregate.get("ratingCount")
        or aggregate.get("reviewCount")
        or _meta_itemprop_content(page, "ratingCount")
    )
    try:
        rating = float(rating) if rating is not None else None
    except (TypeError, ValueError):
        rating = None
    try:
        rating_count = int(str(rating_count).replace(",", "")) if rating_count is not None else None
    except (TypeError, ValueError):
        rating_count = None

    suffix = "zh" if locale.lower().startswith("zh") else "ja"
    return {
        "product_id": normalize_product_id(product_id),
        f"title_{suffix}": title,
        f"description_{suffix}": description,
        "maker": maker,
        "series": series,
        "rating": rating,
        "rating_count": rating_count,
        "release_date": release_date,
        "age_rating": age_rating,
        "cover_url": strip_html(image) or _meta_content(page, "og:image"),
        "tags": list(dict.fromkeys(tag for tag in tags if tag)),
        "people": {
            "voice_actor": voice,
            "scenario": scenario,
            "illustrator": illustrator,
            "author": author,
        },
        "source": "dlsite",
        "locale": locale,
    }


class DlsiteMetadataClient:
    def __init__(
        self,
        database: MediaLibraryDatabase,
        *,
        min_interval: float = 1.0,
        cache_ttl: float = DEFAULT_CACHE_TTL,
        opener: Callable[..., Any] = urlopen,
    ):
        self.database = database
        self.min_interval = max(0.0, float(min_interval))
        self.cache_ttl = max(60.0, float(cache_ttl))
        self.opener = opener
        self._lock = threading.Lock()
        self._last_request = 0.0

    @staticmethod
    def url(product_id: str, locale: str) -> str:
        identifier = normalize_product_id(product_id)
        if not identifier:
            raise ValueError("invalid DLsite product id")
        locale_value = "zh_CN" if locale.lower().startswith("zh") else "ja_JP"
        return f"https://www.dlsite.com/maniax/work/=/product_id/{identifier}.html/?locale={locale_value}"

    def _download(self, product_id: str, locale: str, *, force: bool) -> bytes:
        url = self.url(product_id, locale)
        cache_key = f"dlsite:{normalize_product_id(product_id)}:{locale.lower()}"
        cached = self.database.metadata_cache_get(cache_key)
        if not force and cached and int(cached["status"]) == 200:
            return cached["payload"]
        with self._lock:
            remaining = self.min_interval - (time.monotonic() - self._last_request)
            if remaining > 0:
                time.sleep(remaining)
            request = Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "zh-CN,zh;q=0.9,ja;q=0.8" if locale.startswith("zh") else "ja-JP,ja;q=0.9",
                },
            )
            try:
                response = self.opener(request, timeout=20)
                payload = response.read()
                status = int(getattr(response, "status", 200))
                headers = getattr(response, "headers", {})
                if status != 200 and cached and int(cached["status"]) == 200:
                    return cached["payload"]
                self.database.metadata_cache_put(
                    cache_key,
                    url,
                    status,
                    payload,
                    ttl_seconds=self.cache_ttl,
                    etag=headers.get("ETag", "") if headers else "",
                    last_modified=headers.get("Last-Modified", "") if headers else "",
                )
                if status != 200:
                    raise RuntimeError(f"DLsite returned HTTP {status}")
                return payload
            except HTTPError as error:
                payload = error.read() if hasattr(error, "read") else b""
                if cached and int(cached["status"]) == 200:
                    return cached["payload"]
                self.database.metadata_cache_put(
                    cache_key, url, int(error.code), payload, ttl_seconds=min(3600.0, self.cache_ttl)
                )
                raise RuntimeError(f"DLsite returned HTTP {error.code}") from error
            except URLError as error:
                if cached and int(cached["status"]) == 200:
                    return cached["payload"]
                raise RuntimeError(f"DLsite request failed: {error.reason}") from error
            finally:
                self._last_request = time.monotonic()

    def fetch(self, product_id: str, *, force: bool = False) -> dict[str, Any]:
        identifier = normalize_product_id(product_id)
        if not identifier:
            raise ValueError("invalid DLsite product id")
        merged: dict[str, Any] = {
            "product_id": identifier,
            "source": "dlsite",
            "tags": [],
            "people": {},
        }
        errors: dict[str, str] = {}
        for locale in ("ja", "zh-cn"):
            try:
                payload = self._download(identifier, locale, force=force)
                parsed = parse_dlsite_page(payload.decode("utf-8", errors="replace"), locale=locale, product_id=identifier)
            except Exception as error:
                errors[locale] = str(error)
                continue
            for key, value in parsed.items():
                if key == "tags":
                    merged["tags"] = list(dict.fromkeys([*merged["tags"], *value]))
                elif key == "people":
                    for role, names in value.items():
                        merged["people"][role] = list(
                            dict.fromkeys([*merged["people"].get(role, []), *names])
                        )
                elif value not in (None, "", [], {}):
                    merged[key] = value
        # DLsite often serves the Japanese work page for zh_CN when no
        # official Chinese localisation exists. Do not label identical
        # Japanese text as Chinese. Explicit empty values also allow a later
        # refresh to clear fields written by older, incorrect parsers.
        if _same_localized_text(merged.get("title_ja"), merged.get("title_zh")):
            merged["title_zh"] = ""
        if _same_localized_text(merged.get("description_ja"), merged.get("description_zh")):
            merged["description_zh"] = ""
        if not merged.get("title_ja") and not merged.get("title_zh"):
            raise RuntimeError("DLsite metadata unavailable: " + json.dumps(errors, ensure_ascii=False))
        if errors:
            merged["warnings"] = errors
        return merged

    def enrich(self, work_id: str, product_id: str, *, force: bool = False) -> dict[str, Any]:
        metadata = self.fetch(product_id, force=force)
        self.database.update_metadata(work_id, metadata)
        return metadata
