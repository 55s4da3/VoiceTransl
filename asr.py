import os
import subprocess
import json
import shlex
from pathlib import Path

from core import DEFAULT_CRISPASR_BACKEND

CRISPASR_BACKEND_FALLBACK = (
    'whisper', 'parakeet', 'canary', 'cohere', 'qwen3',
    'qwen3-1.7b', 'mega-asr', 'voxtral', 'voxtral4b', 'granite',
)


def _list_crispasr_models():
    model_dir = Path('crispasr')
    if not model_dir.is_dir():
        return []
    return sorted(
        path.name for path in model_dir.glob('*.gguf')
        if 'aligner' not in path.name.lower() and 'alignment' not in path.name.lower()
    )


def _list_crispasr_aligners():
    model_dir = Path('crispasr')
    if not model_dir.is_dir():
        return []
    return sorted(
        path.name for path in model_dir.glob('*.gguf')
        if 'aligner' in path.name.lower() or 'alignment' in path.name.lower()
    )


def _list_crispasr_backends():
    """Return speech-recognition backends compiled into the local executable."""
    crispasr_dir = Path('crispasr').resolve()
    executable_name = 'crispasr.exe' if os.name == 'nt' else 'crispasr'
    executable = crispasr_dir / executable_name
    if not executable.is_file():
        return list(CRISPASR_BACKEND_FALLBACK)

    creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    try:
        result = subprocess.run(
            [str(executable), '--list-backends-json'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=10,
            creationflags=creationflags,
            check=False,
        )
        backend_data = json.loads(result.stdout)
        speech_caps = {
            'timestamps-native', 'timestamps-ctc', 'word-timestamps',
            'language-detect',
        }
        task_only_caps = {'piano', 'separate', 'pitch', 'chords', 'tab', 'beats'}
        backends = []
        for item in backend_data.get('backends', []):
            name = item.get('name', '').strip()
            caps = set(item.get('caps', []))
            if name and caps.intersection(speech_caps) and not caps.intersection(task_only_caps):
                backends.append(name)
        if backends:
            return list(dict.fromkeys(backends))
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        pass
    return list(CRISPASR_BACKEND_FALLBACK)


def _split_command_template(value):
    if os.name != 'nt':
        return shlex.split(value)
    tokens = shlex.split(value, posix=False)
    return [
        token[1:-1] if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'" else token
        for token in tokens
    ]


def _build_crispasr_command(
    input_file, output_file, model_file, language, param_crispasr,
    aligner_file=None, backend=None,
):
    """按 param.txt 模板替换占位符，生成与旧 Whisper 相同风格的启动参数。"""
    crispasr_dir = Path('crispasr').resolve()
    executable_name = 'crispasr.exe' if os.name == 'nt' else 'crispasr'
    executable = crispasr_dir / executable_name
    aligners = _list_crispasr_aligners()
    if not executable.is_file():
        raise FileNotFoundError(f'CrispASR executable not found: {executable}')
    if not aligner_file and not aligners:
        raise FileNotFoundError(f'CrispASR aligner model not found in: {crispasr_dir}')

    model_path = Path(model_file)
    if not model_path.is_absolute():
        model_path = crispasr_dir / model_path
    aligner_path = Path(aligner_file or aligners[0])
    if not aligner_path.is_absolute():
        aligner_path = crispasr_dir / aligner_path
    if not aligner_path.is_file():
        raise FileNotFoundError(f'CrispASR aligner model not found: {aligner_path}')
    replacements = {
        '$crispasr_executable': str(executable),
        '$backend': backend or DEFAULT_CRISPASR_BACKEND,
        '$model_file': str(model_path.resolve()),
        '$aligner_file': str(aligner_path.resolve()),
        '$language': language or 'auto',
        '$output_file': str(Path(output_file).resolve()),
        '$input_file': str(Path(input_file).resolve()),
    }
    command = _split_command_template(param_crispasr)
    for index, token in enumerate(command):
        for placeholder, replacement in replacements.items():
            token = token.replace(placeholder, replacement)
        command[index] = token
    if not command:
        raise ValueError('CrispASR param.txt is empty')
    if backend:
        for index, token in enumerate(command):
            if token == '--backend' and index + 1 < len(command):
                command[index + 1] = backend
                break
            if token.startswith('--backend='):
                command[index] = f'--backend={backend}'
                break
        else:
            command.extend(['--backend', backend])
    return command
