"""Online model provider presets shared by the desktop UI and workers.

Provider-specific addresses and model aliases live here; request execution
continues to use the common OpenAI-compatible client.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from opencode_zen import (
    OPENCODE_GO_ENDPOINT,
    OPENCODE_GO_PROVIDER,
    OPENCODE_ZEN_ENDPOINT,
    OPENCODE_ZEN_PROVIDER,
)


DEEPSEEK_PROVIDER = "Deepseek"
DEEPSEEK_ENDPOINT = "https://api.deepseek.com"
DEEPSEEK_V4_FLASH_MODEL = "deepseek-v4-flash"
DEEPSEEK_V4_PRO_MODEL = "deepseek-v4-pro"


@dataclass(frozen=True)
class OnlineProviderPreset:
    endpoint: str
    default_model: str = ""
    models: tuple[str, ...] = ()
    api_format: str = "openai_chat"
    models_url: str = ""


# Dict insertion order is the order shown in provider combo boxes.
ONLINE_PROVIDER_PRESETS: dict[str, OnlineProviderPreset] = {
    "Kimi": OnlineProviderPreset("https://api.moonshot.cn"),
    "Kimi (国际)": OnlineProviderPreset("https://api.moonshot.ai"),
    "GLM": OnlineProviderPreset(
        "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    ),
    "GLM (国际)": OnlineProviderPreset(
        "https://api.z.ai/api/paas/v4/chat/completions"
    ),
    DEEPSEEK_PROVIDER: OnlineProviderPreset(
        DEEPSEEK_ENDPOINT,
        default_model=DEEPSEEK_V4_FLASH_MODEL,
        models=(DEEPSEEK_V4_FLASH_MODEL, DEEPSEEK_V4_PRO_MODEL),
    ),
    "Minimax": OnlineProviderPreset("https://api.minimaxi.com"),
    "Minimax (国际)": OnlineProviderPreset("https://api.minimaxi.io"),
    "豆包": OnlineProviderPreset("https://ark.cn-beijing.volces.com/api"),
    "阿里云": OnlineProviderPreset(
        "https://dashscope.aliyuncs.com/compatible-mode"
    ),
    "Gemini": OnlineProviderPreset(
        "https://generativelanguage.googleapis.com/v1beta/openai"
    ),
    "OpenAI": OnlineProviderPreset("https://api.openai.com"),
    OPENCODE_ZEN_PROVIDER: OnlineProviderPreset(OPENCODE_ZEN_ENDPOINT),
    OPENCODE_GO_PROVIDER: OnlineProviderPreset(
        OPENCODE_GO_ENDPOINT,
        default_model="deepseek-flash",
        models=(
            "deepseek-flash",
            DEEPSEEK_V4_PRO_MODEL,
            DEEPSEEK_V4_FLASH_MODEL,
        ),
    ),
    "Ollama": OnlineProviderPreset("http://localhost:11434"),
    "llamacpp（通用本地模型）": OnlineProviderPreset("http://localhost:8989"),
}


ONLINE_TRANSLATOR_MAPPING = {
    name: preset.endpoint for name, preset in ONLINE_PROVIDER_PRESETS.items()
}


_DEEPSEEK_OFFICIAL_ALIASES = {
    # OpenCode Go uses the shorter ID for the same model family.
    "deepseek-flash": DEEPSEEK_V4_FLASH_MODEL,
    # Accept common V4.1 spellings without sending invented model IDs.
    "deepseek-v4.1-flash": DEEPSEEK_V4_FLASH_MODEL,
    "deepseek-v4-1-flash": DEEPSEEK_V4_FLASH_MODEL,
    "deepseek-v41-flash": DEEPSEEK_V4_FLASH_MODEL,
}


def provider_preset(provider: str) -> OnlineProviderPreset | None:
    return ONLINE_PROVIDER_PRESETS.get(str(provider or ""))


def provider_model_choices(provider: str) -> list[str]:
    preset = provider_preset(provider)
    return list(preset.models) if preset else []


def provider_default_model(provider: str) -> str:
    preset = provider_preset(provider)
    return preset.default_model if preset else ""


def normalize_provider_model(provider: str, model: str) -> str:
    """Return the wire model ID without rewriting arbitrary custom IDs."""
    value = str(model or "").strip()
    if provider == DEEPSEEK_PROVIDER:
        return _DEEPSEEK_OFFICIAL_ALIASES.get(value.lower(), value)
    return value


def build_models_url_candidates(
    base_url: str,
    models_url_override: str = "",
) -> list[str]:
    """Build OpenAI-compatible model-list URLs in preferred order."""
    override = str(models_url_override or "").strip().rstrip("/")
    if override:
        return [override]

    base = str(base_url or "").strip().rstrip("/")
    if not base:
        return []
    base = re.sub(
        r"/(?:chat/completions|responses|models)$",
        "",
        base,
        flags=re.IGNORECASE,
    )

    if re.search(r"/v\d+(?:beta)?(?:/openai)?$", base, flags=re.IGNORECASE):
        candidates = [base + "/models"]
        if not re.search(r"/v1(?:/openai)?$", base, flags=re.IGNORECASE):
            candidates.append(base + "/v1/models")
    else:
        candidates = [base + "/v1/models"]

    return list(dict.fromkeys(candidates))
