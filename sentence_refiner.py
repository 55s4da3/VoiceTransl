"""AI-assisted full-file subtitle resegmentation with safe timestamp mapping."""

from __future__ import annotations

import json
import re
import time
import unicodedata
import uuid
from dataclasses import dataclass
from typing import Callable

import httpx

from GalTransl.ConfigHelper import build_httpx_sync_proxy_kwargs
from opencode_zen import is_deepseek_v4_family, opencode_zen_api_mode
from tasking import CancellationToken, TaskCancelledError


DEFAULT_MAX_OUTPUT_TOKENS = 65536
DEFAULT_MAX_OUTPUT_CHARACTERS = 262144


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


def _range_record(item: object) -> tuple[int, int] | None:
    if not isinstance(item, dict):
        return None
    try:
        start_id = int(item["start_id"])
        end_id = int(item["end_id"])
    except (KeyError, TypeError, ValueError):
        return None
    return start_id, end_id


def _extract_group_ranges(content: str, row_count: int) -> list[tuple[int, int]]:
    """Parse and strictly validate contiguous source-row groups."""
    stripped = content.strip()
    items: list[object] = []
    try:
        document = json.loads(stripped)
    except json.JSONDecodeError:
        document = None
    if isinstance(document, list):
        items = document
    elif isinstance(document, dict) and isinstance(document.get("groups"), list):
        items = document["groups"]
    else:
        for raw in stripped.splitlines():
            line = raw.strip()
            if not line or line.startswith("```"):
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    ranges = [record for item in items if (record := _range_record(item))]
    if not ranges:
        raise RefinementError("AI 未返回有效的断句范围")
    expected_start = 1
    for start_id, end_id in ranges:
        if start_id != expected_start or end_id < start_id or end_id > row_count:
            raise RefinementError("AI 返回的断句范围存在遗漏、重复或越界")
        expected_start = end_id + 1
    if expected_start != row_count + 1:
        raise RefinementError("AI 返回的断句范围没有覆盖完整原文")
    return ranges


def apply_group_ranges(
    rows: list[dict], ranges: list[tuple[int, int]]
) -> list[dict]:
    """Apply the AI's complete grouping verbatim to the original rows."""
    output: list[dict] = []
    for start_id, end_id in ranges:
        grouped_rows = rows[start_id - 1:end_id]
        grouped = ""
        for row in grouped_rows:
            text = str(row.get("message", row.get("text", "")) or "")
            grouped += _joiner(grouped, text) + text
        start = float(grouped_rows[0].get("start", 0.0) or 0.0)
        end = max(start, float(grouped_rows[-1].get("end", start) or start))
        output.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "message": grouped,
        })
    return output


def resegment_request_limits(rows: list[dict]) -> tuple[int, int]:
    """Bound a range-only response without tying it to a provider maximum."""
    row_count = max(1, len(rows))
    max_tokens = min(DEFAULT_MAX_OUTPUT_TOKENS, max(2048, row_count * 20 + 1024))
    max_characters = min(
        DEFAULT_MAX_OUTPUT_CHARACTERS,
        max(8192, row_count * 48 + 4096),
    )
    return max_tokens, max_characters


@dataclass
class _Timeline:
    raw_text: str
    semantic_raw_positions: list[int]
    semantic_starts: list[float]
    semantic_ends: list[float]


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
    raw_offset = 0
    previous_text = ""

    for row_index, row in enumerate(rows):
        text = str(row.get("message", row.get("text", "")) or "")
        start = float(row.get("start", 0.0) or 0.0)
        end = max(start, float(row.get("end", start) or start))
        if row_index:
            separator = _joiner(previous_text, text)
            raw_parts.append(separator)
            raw_offset += len(separator)

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
            semantic_raw_positions.append(position)
            semantic_starts.append(unit_start)
            semantic_ends.append(unit_end)
        previous_text = text

    return _Timeline(
        raw_text="".join(raw_parts),
        semantic_raw_positions=semantic_raw_positions,
        semantic_starts=semantic_starts,
        semantic_ends=semantic_ends,
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


def apply_refinement(rows: list[dict], texts: list[str]) -> list[dict]:
    if not rows:
        return []
    timeline = _build_timeline(rows)
    if not timeline.semantic_raw_positions:
        return [dict(row) for row in rows]
    # The AI response is already validated as an exact, contiguous grouping.
    # Preserve those boundaries verbatim; adding local duration/silence splits
    # can reintroduce the mid-word ASR boundaries this stage is meant to fix.
    boundaries = _validate_requested_boundaries(rows, texts)
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


def build_prompt(rows: list[dict], corrective: bool = False) -> str:
    input_lines = []
    for index, row in enumerate(rows, start=1):
        item = [
            index,
            round(float(row.get("start", 0.0)), 3),
            round(float(row.get("end", 0.0)), 3),
            str(row.get("message", row.get("text", "")) or ""),
        ]
        input_lines.append(json.dumps(
            item, ensure_ascii=False, separators=(",", ":")
        ))
    correction = (
        "上一次输出的范围无效。这次必须从 1 开始连续覆盖到最后一个 id，"
        "不能遗漏、重复、交叉或越界。\n"
        if corrective else ""
    )
    return f"""你是字幕断句整理器。请理解整份识别原文，自主决定哪些相邻输入行应合为一句：
1. 以语义、语气和上下文为准，把相邻碎句整理成自然、完整的字幕句子；不要翻译、纠错、改写或复述原文。
2. 简短语气词、停顿和较长句是否独立都由你根据全文语义决定，不设机械的字符数、时长或静音限制。
3. 仅输出 JSON Lines，每组一行 {{"start_id":起始行号,"end_id":结束行号}}；第一组必须从 1 开始，后一组紧接前一组，最后一组必须结束于 {len(rows)}。
4. 不要输出原文、解释、Markdown 或任何其他字段。
{correction}输入 JSON Lines，每行格式为 [id,开始秒,结束秒,原文]：
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
    output_callback: Callable[[dict], None] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_output_characters: int = DEFAULT_MAX_OUTPUT_CHARACTERS,
    expected_last_id: int | None = None,
) -> str:
    if not endpoint or not model or not api_key:
        raise RefinementError("AI 断句缺少 API 地址、模型名称或 Token")
    base_url = _openai_base_url(endpoint)
    api_mode = opencode_zen_api_mode(base_url, model)
    if api_mode == "unsupported":
        raise RefinementError(
            f"OpenCode Zen 模型 {model} 使用 VoiceTransl 尚未支持的协议"
        )
    output_limit = max(1, min(
        int(max_output_tokens),
        384000 if is_deepseek_v4_family(model) else 65536,
    ))
    if api_mode == "responses":
        payload = {
            "model": model,
            "input": prompt,
            "stream": True,
            "max_output_tokens": output_limit,
        }
    else:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "stream": True,
            "max_tokens": output_limit,
        }
    model_lower = model.lower()
    if api_mode == "chat" and is_deepseek_v4_family(model_lower):
        payload["thinking"] = {
            "type": "enabled" if thinking_enabled else "disabled"
        }
    elif api_mode == "chat" and ("qwen3" in model_lower or "qwq" in model_lower):
        payload["enable_thinking"] = bool(thinking_enabled)
    elif api_mode == "chat" and re.search(
        r"(^|[-_/:.])(?:r1|o1|o3|o4)(?:[-_/:.]|$)|reason",
        model_lower,
    ):
        payload["reasoning_effort"] = "high" if thinking_enabled else "low"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    client_kwargs = build_httpx_sync_proxy_kwargs(proxy or None)
    client_kwargs["trust_env"] = False
    timeout = httpx.Timeout(connect=10, read=5, write=30, pool=10)
    chunks: list[str] = []
    request_id = uuid.uuid4().hex
    received_characters = 0
    last_report = -1
    last_report_at = time.monotonic()
    range_line_buffer = ""
    contiguous_end_id = 0
    last_progress_id = 0
    last_progress_at = time.monotonic()

    def report(total: int, *, final=False):
        if output_callback is None:
            return
        output_callback({
            "request": request_id,
            "characters": max(0, int(total)),
            "final": bool(final),
        })

    cancel_token.raise_if_cancelled()
    try:
        with httpx.Client(timeout=timeout, **client_kwargs) as client:
            with client.stream(
                "POST",
                base_url + ("/responses" if api_mode == "responses" else "/chat/completions"),
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
                        if api_mode == "responses":
                            content = (
                                event.get("delta")
                                if event.get("type") == "response.output_text.delta"
                                else None
                            )
                        else:
                            delta = event.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content")
                        response_complete = False
                        if isinstance(content, str):
                            chunks.append(content)
                            received_characters += len(content)
                            if received_characters > max_output_characters:
                                raise RefinementError(
                                    "AI 断句输出异常增长，已提前中止"
                                )
                            if expected_last_id:
                                range_line_buffer += content
                                lines = range_line_buffer.split("\n")
                                range_line_buffer = lines.pop()
                                for completed_line in lines:
                                    try:
                                        item = json.loads(completed_line.strip())
                                    except json.JSONDecodeError:
                                        continue
                                    record = _range_record(item)
                                    if record is None:
                                        continue
                                    start_id, end_id = record
                                    if (
                                        start_id != contiguous_end_id + 1
                                        or end_id < start_id
                                        or end_id > expected_last_id
                                    ):
                                        contiguous_end_id = -1
                                        continue
                                    contiguous_end_id = end_id
                                    progress_now = time.monotonic()
                                    progress_step = max(1, expected_last_id // 100)
                                    if (
                                        progress_callback is not None
                                        and (
                                            end_id - last_progress_id >= progress_step
                                            or progress_now - last_progress_at >= 0.2
                                            or end_id == expected_last_id
                                        )
                                    ):
                                        progress_callback(end_id, expected_last_id)
                                        last_progress_id = end_id
                                        last_progress_at = progress_now
                                    if end_id == expected_last_id:
                                        response_complete = True
                                        break
                        now = time.monotonic()
                        if (
                            received_characters > 0
                            and (
                                last_report < 0
                                or received_characters - last_report >= 32
                                or now - last_report_at >= 0.2
                            )
                        ):
                            report(received_characters)
                            last_report = received_characters
                            last_report_at = now
                        if response_complete:
                            break
                    except (json.JSONDecodeError, IndexError, AttributeError):
                        continue
    finally:
        report(received_characters, final=True)
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
        self.applied = False
        self.changed = False
        self.last_error = ""

    def refine(self, rows: list[dict]) -> list[dict]:
        original = [dict(row) for row in rows]
        self.applied = False
        self.changed = False
        self.last_error = ""
        for attempt in range(2):
            self.cancel_token.raise_if_cancelled()
            try:
                response = self.requester(build_prompt(rows, corrective=bool(attempt)))
                ranges = _extract_group_ranges(response, len(rows))
                unchanged = all(
                    start_id == index and end_id == index
                    for index, (start_id, end_id) in enumerate(ranges, start=1)
                )
                refined = original if unchanged else apply_group_ranges(rows, ranges)
                self.applied = True
                self.changed = not unchanged
                self.status(f"AI 断句完成：{len(rows)} 条整理为 {len(refined)} 条")
                return refined
            except TaskCancelledError:
                raise
            except Exception as error:
                self.last_error = str(error)
                self.status(f"AI 断句第 {attempt + 1} 次响应无效：{error}")
        self.status("AI 断句未完整覆盖原文，本次结果未应用，已保留全部原断句")
        return original
