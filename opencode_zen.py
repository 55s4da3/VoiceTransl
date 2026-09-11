"""OpenCode Zen endpoint and protocol helpers.

Zen exposes one model catalogue but routes models through several wire
protocols. VoiceTransl supports its OpenAI-compatible Chat Completions and
Responses surfaces. Anthropic Messages and Google's model-specific surface are
hidden until those protocols are implemented here.
"""

from __future__ import annotations

from urllib.parse import urlparse


OPENCODE_ZEN_PROVIDER = "OpenCode Zen"
OPENCODE_ZEN_ENDPOINT = "https://opencode.ai/zen"
OPENCODE_GO_PROVIDER = "OpenCode Go"
OPENCODE_GO_ENDPOINT = "https://opencode.ai/zen/go"

_RESPONSES_MODEL_PREFIXES = ("gpt-", "grok-", "muse-spark-")
_UNSUPPORTED_MODEL_PREFIXES = ("claude-", "gemini-", "qwen")


def is_deepseek_v4_family(model: str) -> bool:
    """Recognize official V4 IDs and OpenCode Go's V4.1 alias."""
    model_id = str(model or "").strip().lower()
    return (
        model_id.startswith("deepseek-v4")
        or model_id == "deepseek-flash"
        or model_id.endswith("/deepseek-flash")
    )


def is_opencode_zen_endpoint(endpoint: str) -> bool:
    """Return whether *endpoint* is an official OpenCode Zen/Go API URL."""
    raw = str(endpoint or "").strip()
    if not raw:
        return False
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.hostname or "").lower()
    path = "/" + parsed.path.strip("/").lower()
    return host in {"opencode.ai", "www.opencode.ai"} and (
        path == "/zen" or path.startswith("/zen/")
    )


def opencode_zen_api_mode(endpoint: str, model: str) -> str:
    """Return ``chat``, ``responses`` or ``unsupported`` for a request."""
    if not is_opencode_zen_endpoint(endpoint):
        return "chat"
    model_id = str(model or "").strip().lower()
    if model_id.startswith(_UNSUPPORTED_MODEL_PREFIXES):
        return "unsupported"
    if model_id.startswith(_RESPONSES_MODEL_PREFIXES):
        return "responses"
    return "chat"


def filter_supported_opencode_models(endpoint: str, models: list[str]) -> list[str]:
    """Keep catalogue order while hiding protocols VoiceTransl cannot call."""
    if not is_opencode_zen_endpoint(endpoint):
        return list(models)
    return [
        model
        for model in models
        if opencode_zen_api_mode(endpoint, model) != "unsupported"
    ]


def split_responses_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """Convert Chat Completions messages to Responses instructions/input."""
    instructions: list[str] = []
    input_messages: list[dict] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if role in {"system", "developer"} and isinstance(content, str):
            if content:
                instructions.append(content)
            continue
        input_messages.append({"role": role, "content": content})
    return "\n\n".join(instructions), input_messages


def responses_stream_delta(event) -> str:
    """Extract visible text from an OpenAI Responses streaming event."""
    event_type = event.get("type", "") if isinstance(event, dict) else getattr(event, "type", "")
    if event_type != "response.output_text.delta":
        return ""
    delta = event.get("delta", "") if isinstance(event, dict) else getattr(event, "delta", "")
    return delta if isinstance(delta, str) else ""


def responses_finish_reason(event) -> str | None:
    """Map Responses completion events to a Chat Completions-like reason."""
    event_type = event.get("type", "") if isinstance(event, dict) else getattr(event, "type", "")
    if event_type == "response.completed":
        return "stop"
    if event_type != "response.incomplete":
        return None
    response = event.get("response") if isinstance(event, dict) else getattr(event, "response", None)
    details = response.get("incomplete_details") if isinstance(response, dict) else getattr(response, "incomplete_details", None)
    reason = details.get("reason") if isinstance(details, dict) else getattr(details, "reason", None)
    return str(reason or "incomplete")


def responses_output_text(response) -> str:
    """Extract the SDK convenience ``output_text`` value safely."""
    value = response.get("output_text", "") if isinstance(response, dict) else getattr(response, "output_text", "")
    return value if isinstance(value, str) else ""
