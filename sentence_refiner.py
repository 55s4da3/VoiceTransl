"""AI-assisted full-file subtitle resegmentation with safe timestamp mapping."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Callable, Iterable

import httpx

from GalTransl.ConfigHelper import build_httpx_sync_proxy_kwargs
from tasking import CancellationToken, TaskCancelledError


MAX_EFFECTIVE_CHARS = 42
MAX_DURATION_SECONDS = 8.0
MAX_SILENCE_GAP_SECONDS = 1.5


class RefinementError(ValueError):
    pass


def _timestamp_seconds(value, fallback: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.match(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)", value)
        if match:
            hours = int(match.group(1) or 0)
            return hours * 3600 + int(match.group(2)) * 60 + float(match.group(3))
        try:
            return float(value)
        except ValueError:
            pass
    return fallback


def crispasr_json_rows(document: dict) -> list[dict]:
    """Read common whisper.cpp/CrispASR full-JSON layouts."""
    segments = document.get("segments") or document.get("transcription") or []
    rows: list[dict] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text", "") or "").strip()
        if not text:
            continue
        timestamps = segment.get("timestamps") or {}
        offsets = segment.get("offsets") or {}
        start = _timestamp_seconds(
            segment.get("start", timestamps.get("from")),
            float(offsets.get("from", 0) or 0) / 1000.0,
        )
        end = _timestamp_seconds(
            segment.get("end", timestamps.get("to")),
            float(offsets.get("to", 0) or 0) / 1000.0,
        )
        words: list[dict] = []
        for word in segment.get("words", segment.get("tokens", [])) or []:
            if not isinstance(word, dict):
                continue
            word_text = str(word.get("word", word.get("text", "")) or "")
            word_offsets = word.get("offsets") or {}
            word_timestamps = word.get("timestamps") or {}
            word_start = _timestamp_seconds(
                word.get("start", word_timestamps.get("from")),
                float(word_offsets.get("from", 0) or 0) / 1000.0,
            )
            word_end = _timestamp_seconds(
                word.get("end", word_timestamps.get("to")),
                float(word_offsets.get("to", 0) or 0) / 1000.0,
            )
            if word_text and word_end >= word_start:
                words.append({"word": word_text, "start": word_start, "end": word_end})
        row = {"start": start, "end": max(start, end), "message": text}
        if words:
            row["words"] = words
        rows.append(row)
    return rows


def _semantic_units(text: str) -> list[str]:
    units: list[str] = []
    for char in text:
        normalized = unicodedata.normalize("NFKC", char)
        for item in normalized:
            category = unicodedata.category(item)
            if category[0] not in {"P", "Z", "C"}:
                units.append(item.casefold())
    return units


def _semantic_text(text: str) -> str:
    return "".join(_semantic_units(text))


def _joiner(left: str, right: str) -> str:
    if not left or not right:
        return ""
    return " " if left[-1].isascii() and right[0].isascii() and left[-1].isalnum() and right[0].isalnum() else ""


def _extract_jsonline_texts(content: str) -> list[str]:
    content = content.strip()
    if content.startswith("{"):
        try:
            obj = json.loads(content)
            if isinstance(obj, dict) and isinstance(obj.get("segments"), list):
                texts = [item.get("text") for item in obj["segments"] if isinstance(item, dict)]
                if texts and all(isinstance(text, str) and text.strip() for text in texts):
                    return texts
        except json.JSONDecodeError:
            pass

    texts: list[str] = []
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("```"):
            continue
        if "|{" in line and re.match(r"^[a-z0-9]{3}\|", line):
            line = line.split("|", 1)[1]
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip():
            texts.append(item["text"])
    if not texts:
        raise RefinementError("AI 未返回有效的断句 JSON")
    return texts


@dataclass
class _Timeline:
    raw_text: str
    semantic_raw_positions: list[int]
    semantic_starts: list[float]
    semantic_ends: list[float]
    mandatory_boundaries: set[int]
    preferred_boundaries: set[int]


def _word_fields(word: dict) -> tuple[str, float | None, float | None]:
    text = str(word.get("word", word.get("text", "")) or "")
    start = word.get("start")
    end = word.get("end")
    try:
        return text, float(start), float(end)
    except (TypeError, ValueError):
        return text, None, None


def _build_timeline(rows: list[dict]) -> _Timeline:
    raw_parts: list[str] = []
    semantic_raw_positions: list[int] = []
    semantic_starts: list[float] = []
    semantic_ends: list[float] = []
    mandatory: set[int] = set()
    preferred: set[int] = set()
    semantic_count = 0
    raw_offset = 0
    previous_text = ""
    previous_end: float | None = None

    for row_index, row in enumerate(rows):
        text = str(row.get("message", row.get("text", "")) or "")
        start = float(row.get("start", 0.0) or 0.0)
        end = max(start, float(row.get("end", start) or start))
        if row_index:
            separator = _joiner(previous_text, text)
            raw_parts.append(separator)
            raw_offset += len(separator)
            preferred.add(semantic_count)
            if previous_end is not None and start - previous_end > MAX_SILENCE_GAP_SECONDS:
                mandatory.add(semantic_count)

        row_raw_start = raw_offset
        raw_parts.append(text)
        raw_offset += len(text)

        per_char_units: list[tuple[int, str]] = []
        for char_index, char in enumerate(text):
            for unit in _semantic_units(char):
                per_char_units.append((row_raw_start + char_index, unit))

        row_unit_count = len(per_char_units)
        word_timings = []
        for word in row.get("words", []) or []:
            word_text, word_start, word_end = _word_fields(word)
            count = len(_semantic_units(word_text))
            if count and word_start is not None and word_end is not None:
                word_timings.extend(
                    (
                        word_start + (word_end - word_start) * i / count,
                        word_start + (word_end - word_start) * (i + 1) / count,
                    )
                    for i in range(count)
                )

        precise = len(word_timings) == row_unit_count
        for unit_index, (position, _unit) in enumerate(per_char_units):
            if precise:
                unit_start, unit_end = word_timings[unit_index]
            elif row_unit_count:
                unit_start = start + (end - start) * unit_index / row_unit_count
                unit_end = start + (end - start) * (unit_index + 1) / row_unit_count
            else:
                unit_start = unit_end = start
            unit_start = max(start, min(end, unit_start))
            unit_end = max(start, min(end, unit_end))
            if (
                precise
                and semantic_ends
                and unit_start - semantic_ends[-1] > MAX_SILENCE_GAP_SECONDS
            ):
                mandatory.add(semantic_count)
            semantic_raw_positions.append(position)
            semantic_starts.append(unit_start)
            semantic_ends.append(unit_end)
            semantic_count += 1

        previous_text = text
        previous_end = end

    return _Timeline(
        raw_text="".join(raw_parts),
        semantic_raw_positions=semantic_raw_positions,
        semantic_starts=semantic_starts,
        semantic_ends=semantic_ends,
        mandatory_boundaries=mandatory,
        preferred_boundaries=preferred,
    )


def _validate_requested_boundaries(rows: list[dict], texts: list[str]) -> list[int]:
    expected = _semantic_text("".join(
        (_joiner(str(rows[i - 1].get("message", "")), str(row.get("message", ""))) if i else "")
        + str(row.get("message", ""))
        for i, row in enumerate(rows)
    ))
    actual_parts = [_semantic_text(text) for text in texts]
    if any(not part for part in actual_parts):
        raise RefinementError("AI 返回了空断句")
    if "".join(actual_parts) != expected:
        raise RefinementError("AI 断句改变、遗漏或重复了原文")
    boundaries: list[int] = []
    cursor = 0
    for part in actual_parts[:-1]:
        cursor += len(part)
        boundaries.append(cursor)
    return boundaries


def _is_punctuation_boundary(timeline: _Timeline, semantic_index: int) -> bool:
    if semantic_index <= 0 or semantic_index >= len(timeline.semantic_raw_positions):
        return False
    left = timeline.semantic_raw_positions[semantic_index - 1]
    right = timeline.semantic_raw_positions[semantic_index]
    between = timeline.raw_text[left + 1:right]
    return any(unicodedata.category(char).startswith("P") for char in between)


def _guard_boundaries(timeline: _Timeline, requested: Iterable[int]) -> list[int]:
    total = len(timeline.semantic_raw_positions)
    requested_set = {int(value) for value in requested if 0 < int(value) < total}
    boundaries = requested_set | timeline.mandatory_boundaries
    guarded: list[int] = []
    start = 0
    candidates = sorted(boundaries | {total})
    for desired_end in candidates:
        while desired_end - start > 0:
            duration = timeline.semantic_ends[desired_end - 1] - timeline.semantic_starts[start]
            if desired_end - start <= MAX_EFFECTIVE_CHARS and duration <= MAX_DURATION_SECONDS:
                break
            char_limit = min(desired_end - 1, start + MAX_EFFECTIVE_CHARS)
            time_limit = char_limit
            while time_limit > start + 1 and (
                timeline.semantic_ends[time_limit - 1] - timeline.semantic_starts[start]
                > MAX_DURATION_SECONDS
            ):
                time_limit -= 1
            limit = max(start + 1, min(char_limit, time_limit))
            options = [
                point for point in range(start + 1, limit + 1)
                if point in timeline.preferred_boundaries
                or _is_punctuation_boundary(timeline, point)
            ]
            split = options[-1] if options else limit
            guarded.append(split)
            start = split
        if desired_end < total and desired_end > start:
            guarded.append(desired_end)
            start = desired_end
    return sorted(set(guarded))


def apply_refinement(rows: list[dict], texts: list[str]) -> list[dict]:
    if not rows:
        return []
    timeline = _build_timeline(rows)
    if not timeline.semantic_raw_positions:
        return [dict(row) for row in rows]
    requested = _validate_requested_boundaries(rows, texts)
    boundaries = _guard_boundaries(timeline, requested)
    semantic_ranges: list[tuple[int, int]] = []
    start = 0
    for end in [*boundaries, len(timeline.semantic_raw_positions)]:
        if end > start:
            semantic_ranges.append((start, end))
            start = end

    output: list[dict] = []
    raw_start = 0
    for index, (unit_start, unit_end) in enumerate(semantic_ranges):
        if index + 1 < len(semantic_ranges):
            raw_end = timeline.semantic_raw_positions[unit_end]
        else:
            raw_end = len(timeline.raw_text)
        message = timeline.raw_text[raw_start:raw_end].strip()
        raw_start = raw_end
        if not message:
            continue
        output.append({
            "start": round(timeline.semantic_starts[unit_start], 3),
            "end": round(max(
                timeline.semantic_starts[unit_start],
                timeline.semantic_ends[unit_end - 1],
            ), 3),
            "message": message,
        })
    return output or [dict(row) for row in rows]


def apply_partial_refinement(rows: list[dict], texts: list[str]) -> list[dict]:
    """Use a verified AI prefix and retain original boundaries after its first error."""
    if not rows or not texts:
        raise RefinementError("AI 没有可安全采用的断句前缀")
    timeline = _build_timeline(rows)
    expected = _semantic_text(timeline.raw_text)
    cursor = 0
    verified: list[str] = []
    for text in texts:
        semantic = _semantic_text(text)
        if not semantic or not expected.startswith(semantic, cursor):
            break
        verified.append(text)
        cursor += len(semantic)
        if cursor >= len(expected):
            break
    if not verified or cursor <= 0:
        raise RefinementError("AI 没有可安全采用的断句前缀")
    if cursor >= len(expected):
        return apply_refinement(rows, verified)

    total = len(timeline.semantic_raw_positions)
    raw_start = timeline.semantic_raw_positions[cursor]
    fallback_texts: list[str] = []
    for boundary in sorted(
        {point for point in timeline.preferred_boundaries if point > cursor} | {total}
    ):
        raw_end = (
            len(timeline.raw_text)
            if boundary >= total
            else timeline.semantic_raw_positions[boundary]
        )
        text = timeline.raw_text[raw_start:raw_end].strip()
        if text:
            fallback_texts.append(text)
        raw_start = raw_end
    if not fallback_texts:
        raise RefinementError("无法恢复 AI 错误位置之后的原断句")
    return apply_refinement(rows, [*verified, *fallback_texts])


def build_prompt(rows: list[dict], corrective: bool = False) -> str:
    input_lines = []
    for index, row in enumerate(rows, start=1):
        item = {
            "id": index,
            "start": round(float(row.get("start", 0.0)), 3),
            "end": round(float(row.get("end", 0.0)), 3),
            "text": str(row.get("message", "")),
        }
        if row.get("words"):
            item["words"] = [
                {
                    "text": str(word.get("word", word.get("text", ""))),
                    "start": round(float(word.get("start", 0.0)), 3),
                    "end": round(float(word.get("end", 0.0)), 3),
                }
                for word in row["words"]
                if isinstance(word, dict)
            ]
        input_lines.append(json.dumps(item, ensure_ascii=False))
    correction = (
        "上一次输出改变或遗漏了原文。这次必须确保所有 text 去掉标点和空白后，"
        "按顺序连接起来与输入完全相同。\n"
        if corrective else ""
    )
    return f"""你是字幕断句整理器。请理解整份识别原文，只重新安排字幕边界：
1. 可以合并相邻碎句，也可以拆分一条过长字幕；不要翻译、纠错、改写、遗漏、重复或调换任何词。
2. 简短语气词（例如“嗯”“啊”）是否独立由语义决定，不要机械并入下一句。
3. 单条尽量不超过 {MAX_EFFECTIVE_CHARS} 个字符或 {MAX_DURATION_SECONDS:g} 秒，不要跨越超过 {MAX_SILENCE_GAP_SECONDS:g} 秒的静音。
4. 仅输出 JSON Lines，每行一个 {{"text":"重新分组后的原文"}}，不要输出编号、解释或 Markdown。
{correction}输入 JSON Lines：
{chr(10).join(input_lines)}"""


def _openai_base_url(endpoint: str) -> str:
    endpoint = endpoint.rstrip("/")
    endpoint = re.sub(r"/chat/completions$", "", endpoint)
    if not re.search(r"/v\d+(?:beta)?(?:/openai)?$", endpoint):
        endpoint += "/v1"
    return endpoint


def request_openai_compatible(
    prompt: str,
    *,
    endpoint: str,
    model: str,
    api_key: str,
    proxy: str = "",
    cancel_token: CancellationToken,
    thinking_enabled: bool = False,
) -> str:
    if not endpoint or not model or not api_key:
        raise RefinementError("AI 断句缺少 API 地址、模型名称或 Token")
    base_url = _openai_base_url(endpoint)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "stream": True,
        "max_tokens": 384000 if model.lower().startswith("deepseek-v4") else 65536,
    }
    model_lower = model.lower()
    if model_lower.startswith("deepseek-v4"):
        payload["thinking"] = {
            "type": "enabled" if thinking_enabled else "disabled"
        }
    elif "qwen3" in model_lower or "qwq" in model_lower:
        payload["enable_thinking"] = bool(thinking_enabled)
    elif re.search(
        r"(^|[-_/:.])(?:r1|o1|o3|o4)(?:[-_/:.]|$)|reason",
        model_lower,
    ):
        payload["reasoning_effort"] = "high" if thinking_enabled else "low"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    client_kwargs = build_httpx_sync_proxy_kwargs(proxy or None)
    client_kwargs["trust_env"] = False
    timeout = httpx.Timeout(connect=10, read=5, write=30, pool=10)
    chunks: list[str] = []
    cancel_token.raise_if_cancelled()
    with httpx.Client(timeout=timeout, **client_kwargs) as client:
        with client.stream(
            "POST",
            base_url + "/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                cancel_token.raise_if_cancelled()
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                    delta = event.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content")
                    if isinstance(content, str):
                        chunks.append(content)
                except (json.JSONDecodeError, IndexError, AttributeError):
                    continue
    return "".join(chunks)


class SentenceRefiner:
    def __init__(
        self,
        requester: Callable[[str], str],
        cancel_token: CancellationToken,
        status: Callable[[str], None] | None = None,
    ) -> None:
        self.requester = requester
        self.cancel_token = cancel_token
        self.status = status or (lambda _message: None)

    def refine(self, rows: list[dict]) -> list[dict]:
        original = [dict(row) for row in rows]
        for attempt in range(2):
            self.cancel_token.raise_if_cancelled()
            texts: list[str] = []
            try:
                response = self.requester(build_prompt(rows, corrective=bool(attempt)))
                texts = _extract_jsonline_texts(response)
                refined = apply_refinement(rows, texts)
                self.status(f"AI 断句完成：{len(rows)} 条整理为 {len(refined)} 条")
                return refined
            except TaskCancelledError:
                raise
            except Exception as error:
                self.status(f"AI 断句第 {attempt + 1} 次响应无效：{error}")
                if attempt == 1 and texts:
                    try:
                        refined = apply_partial_refinement(rows, texts)
                        self.status(
                            "AI 断句仅部分有效，已从首个错误处恢复原断句"
                        )
                        return refined
                    except RefinementError:
                        pass
        self.status("AI 断句失败，已安全保留原断句")
        return original
