"""CrispASR 本地安装发现与命令构建。

这个模块只负责枚举 CrispASR 的 GGUF 模型、查询可用后端，以及把
``param.txt`` 风格的命令模板转换成可直接交给 ``subprocess`` 的参数列表。
它不会启动识别任务，也不依赖 GUI 或 ASRLabs。
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


DEFAULT_CRISPASR_DIR = (
    Path(sys.executable).resolve().parent / "crispasr"
    if hasattr(sys, "_MEIPASS")
    else Path("crispasr")
)
DEFAULT_CRISPASR_BACKEND = "qwen3-1.7b"

# 当本地 CrispASR 不支持 --list-backends-json（或尚未安装）时，仍允许 UI
# 展示一组合理的选择。顺序与 feat/qwen3-asr 分支保持一致。
CRISPASR_BACKEND_FALLBACK = (
    "whisper",
    "parakeet",
    "canary",
    "cohere",
    "qwen3",
    "qwen3-1.7b",
    "mega-asr",
    "voxtral",
    "voxtral4b",
    "granite",
)

_ALIGNER_NAME_MARKERS = ("aligner", "alignment")
_PLACEHOLDER_PATTERN = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")
_REQUIRED_COMMAND_PLACEHOLDERS = {
    "$crispasr_executable",
    "$model_file",
    "$aligner_file",
    "$output_file",
    "$input_file",
}
_SPEECH_CAPABILITIES = {
    "timestamps-native",
    "timestamps-ctc",
    "word-timestamps",
    "language-detect",
}
_NON_SPEECH_CAPABILITIES = {
    "piano",
    "separate",
    "pitch",
    "chords",
    "tab",
    "beats",
}


class CrispASRError(Exception):
    """所有 CrispASR bridge 错误的基类。"""


class CrispASRConfigurationError(CrispASRError, ValueError):
    """CrispASR 参数或命令模板无效。"""


class CrispASRFileNotFoundError(CrispASRError, FileNotFoundError):
    """CrispASR 所需的本地文件不存在。"""


class CrispASRBackendDiscoveryError(CrispASRError, RuntimeError):
    """无法读取 CrispASR 编译进可执行文件的后端。"""


def _resolve_crispasr_dir(crispasr_dir: str | os.PathLike[str]) -> Path:
    return Path(crispasr_dir).expanduser().resolve()


def get_executable_path(
    crispasr_dir: str | os.PathLike[str] = DEFAULT_CRISPASR_DIR,
) -> Path:
    """返回当前平台所使用的 CrispASR 可执行文件路径。"""

    executable_name = "crispasr.exe" if os.name == "nt" else "crispasr"
    return _resolve_crispasr_dir(crispasr_dir) / executable_name


def _is_aligner_model(path: Path) -> bool:
    name = path.name.casefold()
    return any(marker in name for marker in _ALIGNER_NAME_MARKERS)


def _list_gguf_files(crispasr_dir: str | os.PathLike[str]) -> list[Path]:
    model_dir = _resolve_crispasr_dir(crispasr_dir)
    if not model_dir.is_dir():
        return []
    try:
        return sorted(
            (
                path
                for path in model_dir.iterdir()
                if path.is_file() and path.suffix.casefold() == ".gguf"
            ),
            key=lambda path: path.name.casefold(),
        )
    except OSError:
        return []


def list_models(
    crispasr_dir: str | os.PathLike[str] = DEFAULT_CRISPASR_DIR,
) -> list[str]:
    """枚举 CrispASR 目录中的语音识别 GGUF 文件名。"""

    return [path.name for path in _list_gguf_files(crispasr_dir) if not _is_aligner_model(path)]


def list_aligners(
    crispasr_dir: str | os.PathLike[str] = DEFAULT_CRISPASR_DIR,
) -> list[str]:
    """枚举 CrispASR 目录中的强制对齐 GGUF 文件名。"""

    return [path.name for path in _list_gguf_files(crispasr_dir) if _is_aligner_model(path)]


def _backend_discovery_error(message: str, cause: BaseException | None = None) -> CrispASRBackendDiscoveryError:
    error = CrispASRBackendDiscoveryError(message)
    if cause is not None:
        error.__cause__ = cause
    return error


def _parse_backend_document(stdout: str) -> list[str]:
    try:
        document: Any = json.loads(stdout.lstrip("\ufeff"))
    except (json.JSONDecodeError, TypeError) as exc:
        raise _backend_discovery_error("CrispASR 返回的后端列表不是有效 UTF-8 JSON", exc)

    if not isinstance(document, dict) or not isinstance(document.get("backends"), list):
        raise CrispASRBackendDiscoveryError("CrispASR 后端 JSON 缺少 backends 数组")

    backends: list[str] = []
    for item in document["backends"]:
        if not isinstance(item, dict):
            continue
        raw_name = item.get("name")
        raw_caps = item.get("caps", [])
        if not isinstance(raw_name, str) or not isinstance(raw_caps, list):
            continue

        name = raw_name.strip()
        capabilities = {cap for cap in raw_caps if isinstance(cap, str)}
        is_speech_backend = bool(capabilities & _SPEECH_CAPABILITIES)
        is_task_only_backend = bool(capabilities & _NON_SPEECH_CAPABILITIES)
        if name and is_speech_backend and not is_task_only_backend and name not in backends:
            backends.append(name)

    if not backends:
        raise CrispASRBackendDiscoveryError("CrispASR 未报告任何语音识别后端")
    return backends


def list_backends(
    crispasr_dir: str | os.PathLike[str] = DEFAULT_CRISPASR_DIR,
    *,
    timeout: float = 10,
    strict: bool = False,
    fallback: Sequence[str] = CRISPASR_BACKEND_FALLBACK,
) -> list[str]:
    """查询本地 CrispASR 编译支持的语音识别后端。

    默认情况下，未安装可执行文件、旧版程序不支持查询参数或输出格式异常时，
    返回 ``fallback``，以便设置界面仍可使用。传入 ``strict=True`` 时会改为
    抛出可测试的 :class:`CrispASRError` 子类。
    """

    if timeout <= 0:
        raise CrispASRConfigurationError("CrispASR 后端查询超时必须大于 0 秒")

    executable = get_executable_path(crispasr_dir)
    if not executable.is_file():
        error: CrispASRError = CrispASRFileNotFoundError(
            f"未找到 CrispASR 可执行文件：{executable}"
        )
        if strict:
            raise error
        return list(fallback)

    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    try:
        result = subprocess.run(
            [str(executable), "--list-backends-json"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            creationflags=creationflags,
            check=False,
        )
        if result.returncode != 0:
            stderr = (result.stderr or "").strip()
            detail = f"：{stderr}" if stderr else ""
            raise CrispASRBackendDiscoveryError(
                f"CrispASR 后端查询失败（退出码 {result.returncode}）{detail}"
            )
        return _parse_backend_document(result.stdout)
    except CrispASRError:
        if strict:
            raise
    except (OSError, subprocess.SubprocessError) as exc:
        if strict:
            raise _backend_discovery_error(f"无法运行 CrispASR 后端查询：{exc}", exc)

    return list(fallback)


def split_command_template(value: str) -> list[str]:
    """按当前平台规则拆分命令模板，不经 shell 展开。"""

    if not isinstance(value, str) or not value.strip():
        raise CrispASRConfigurationError("CrispASR 命令模板为空")
    try:
        if os.name != "nt":
            return shlex.split(value)
        tokens = shlex.split(value, posix=False)
    except ValueError as exc:
        raise CrispASRConfigurationError(f"CrispASR 命令模板无法解析：{exc}") from exc

    # posix=False 会保留包围整个参数的引号；subprocess 参数列表不需要这些引号。
    return [
        token[1:-1]
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'"
        else token
        for token in tokens
    ]


def _require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise CrispASRFileNotFoundError(f"未找到 CrispASR {label}：{resolved}")
    return resolved


def _resolve_model_path(
    value: str | os.PathLike[str],
    crispasr_dir: Path,
    label: str,
) -> Path:
    if not str(value).strip():
        raise CrispASRConfigurationError(f"未选择 CrispASR {label}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = crispasr_dir / path
    return _require_file(path, label)


def build_command(
    input_file: str | os.PathLike[str],
    output_file: str | os.PathLike[str],
    model_file: str | os.PathLike[str],
    language: str | None,
    command_template: str,
    aligner_file: str | os.PathLike[str] | None = None,
    backend: str | None = None,
    *,
    crispasr_dir: str | os.PathLike[str] = DEFAULT_CRISPASR_DIR,
) -> list[str]:
    """用实际路径替换 ``param.txt`` 占位符并返回 subprocess 参数列表。

    ``output_file`` 是 CrispASR 的输出基名；当模板包含 ``--output-srt`` 时，
    CrispASR 会在其后追加 ``.srt``。模型、对齐器、输入文件和可执行文件会在
    构建阶段校验，因此配置错误不会延迟到后台子进程中才出现。
    """

    root = _resolve_crispasr_dir(crispasr_dir)
    executable = _require_file(get_executable_path(root), "可执行文件")
    model_path = _resolve_model_path(model_file, root, "识别模型")

    if aligner_file is None or not str(aligner_file).strip():
        aligners = list_aligners(root)
        if not aligners:
            raise CrispASRFileNotFoundError(f"CrispASR 目录中没有强制对齐模型：{root}")
        aligner_file = aligners[0]
    aligner_path = _resolve_model_path(aligner_file, root, "强制对齐模型")

    input_path = _require_file(Path(input_file), "输入音频")
    if not str(output_file).strip():
        raise CrispASRConfigurationError("CrispASR 输出文件路径为空")
    output_path = Path(output_file).expanduser().resolve()

    selected_backend = (backend or DEFAULT_CRISPASR_BACKEND).strip()
    if not selected_backend:
        raise CrispASRConfigurationError("CrispASR 后端为空")

    replacements = {
        "$crispasr_executable": str(executable),
        "$backend": selected_backend,
        "$model_file": str(model_path),
        "$aligner_file": str(aligner_path),
        "$language": (language or "auto").strip() or "auto",
        "$output_file": str(output_path),
        "$input_file": str(input_path),
    }

    template_placeholders = set(_PLACEHOLDER_PATTERN.findall(command_template))
    missing_placeholders = sorted(
        _REQUIRED_COMMAND_PLACEHOLDERS - template_placeholders
    )
    unknown_placeholders = sorted(template_placeholders - set(replacements))
    if unknown_placeholders:
        raise CrispASRConfigurationError(
            "CrispASR 命令模板包含未知占位符：" + "、".join(unknown_placeholders)
        )
    if missing_placeholders:
        raise CrispASRConfigurationError(
            "CrispASR 命令模板缺少必要占位符：" + "、".join(missing_placeholders)
        )

    command = split_command_template(command_template)
    for index, token in enumerate(command):
        for placeholder, replacement in replacements.items():
            token = token.replace(placeholder, replacement)
        command[index] = token

    if not command or Path(command[0]).resolve() != executable:
        raise CrispASRConfigurationError(
            "CrispASR 命令模板必须以 $crispasr_executable 开头"
        )

    # backend 参数可能在旧模板中写死；显式选择后必须以 UI 的选择为准。
    if backend is not None:
        normalized_command: list[str] = []
        backend_added = False
        index = 0
        while index < len(command):
            token = command[index]
            if token == "--backend":
                if index + 1 >= len(command):
                    raise CrispASRConfigurationError("CrispASR --backend 参数缺少值")
                if not backend_added:
                    normalized_command.extend(["--backend", selected_backend])
                    backend_added = True
                index += 2
                continue
            if token.startswith("--backend="):
                if not backend_added:
                    normalized_command.append(f"--backend={selected_backend}")
                    backend_added = True
                index += 1
                continue
            normalized_command.append(token)
            index += 1
        if not backend_added:
            normalized_command.extend(["--backend", selected_backend])
        command = normalized_command

    return command


# feat/qwen3-asr 中旧导入名的兼容入口，便于 UI/worker 分阶段迁移。
def _list_crispasr_models(
    crispasr_dir: str | os.PathLike[str] = DEFAULT_CRISPASR_DIR,
) -> list[str]:
    return list_models(crispasr_dir)


def _list_crispasr_aligners(
    crispasr_dir: str | os.PathLike[str] = DEFAULT_CRISPASR_DIR,
) -> list[str]:
    return list_aligners(crispasr_dir)


def _list_crispasr_backends(
    crispasr_dir: str | os.PathLike[str] = DEFAULT_CRISPASR_DIR,
) -> list[str]:
    return list_backends(crispasr_dir)


def _build_crispasr_command(
    input_file: str | os.PathLike[str],
    output_file: str | os.PathLike[str],
    model_file: str | os.PathLike[str],
    language: str | None,
    param_crispasr: str,
    aligner_file: str | os.PathLike[str] | None = None,
    backend: str | None = None,
    *,
    crispasr_dir: str | os.PathLike[str] = DEFAULT_CRISPASR_DIR,
) -> list[str]:
    return build_command(
        input_file,
        output_file,
        model_file,
        language,
        param_crispasr,
        aligner_file,
        backend,
        crispasr_dir=crispasr_dir,
    )


__all__ = [
    "CRISPASR_BACKEND_FALLBACK",
    "DEFAULT_CRISPASR_BACKEND",
    "DEFAULT_CRISPASR_DIR",
    "CrispASRBackendDiscoveryError",
    "CrispASRConfigurationError",
    "CrispASRError",
    "CrispASRFileNotFoundError",
    "build_command",
    "get_executable_path",
    "list_aligners",
    "list_backends",
    "list_models",
    "split_command_template",
]
