"""Received-output accounting shared by parent and child processes."""

from __future__ import annotations

import json


OUTPUT_EVENT_PREFIX = "@@VOICETRANSL_OUTPUT@@"


def encode_output_event(event: dict) -> str:
    return OUTPUT_EVENT_PREFIX + json.dumps(
        event, ensure_ascii=False, separators=(",", ":")
    )


def decode_output_event(line: str) -> dict | None:
    if not str(line).startswith(OUTPUT_EVENT_PREFIX):
        return None
    try:
        event = json.loads(str(line)[len(OUTPUT_EVENT_PREFIX):])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(event, dict) or not event.get("request"):
        return None
    try:
        event["characters"] = max(0, int(event.get("characters", 0)))
    except (TypeError, ValueError):
        return None
    event["final"] = bool(event.get("final", False))
    return event
