import sys, os
import subprocess

_FROZEN = hasattr(sys, '_MEIPASS')
os.chdir(sys._MEIPASS) if _FROZEN else os.chdir(os.path.dirname(os.path.abspath(__file__)))
# PyInstaller 打包后使用独立 exe，源码运行时使用 python 脚本
_TRANSLATE_CMD = ['translate/translate'] if _FROZEN else [sys.executable, 'translate.py']
_SEPARATE_CMD = ['separate/separate'] if _FROZEN else [sys.executable, 'separate.py']

# CTranslate2 may fail to initialize on Windows when Qt loads first.  Warm the
# CUDA DLL search path and Faster-Whisper before importing the GUI framework.
_STREAMING_PRELOAD_ERROR = None
if not _FROZEN:
    try:
        from streaming_pipeline import _activate_cuda_dll_dirs
        _activate_cuda_dll_dirs()
        import faster_whisper as _faster_whisper_preloaded
    except Exception as _streaming_preload_exc:
        _STREAMING_PRELOAD_ERROR = _streaming_preload_exc
import shutil
import shlex

NO_TRANSCRIPTION = '不进行听写'
NO_TRANSLATION = '不进行翻译'
DEFAULT_CRISPASR_BACKEND = 'qwen3-1.7b'


def _resolve_ffmpeg() -> tuple[str, str]:
    """解析 ffmpeg/ffprobe 路径"""
    if os.name == 'nt':
        _ffmpeg = 'ffmpeg/ffmpeg.exe'
        _ffprobe = 'ffmpeg/ffprobe.exe'
    else:
        _ffmpeg = 'ffmpeg/ffmpeg'
        _ffprobe = 'ffmpeg/ffprobe'

    # 优先使用本地 ffmpeg 目录
    if not os.path.exists(_ffmpeg):
        # 回退检查系统 PATH
        _ffmpeg = shutil.which('ffmpeg') or _ffmpeg
        _ffprobe = shutil.which('ffprobe') or _ffprobe

    if not os.path.exists(_ffmpeg):
        raise RuntimeError(
            '未找到 ffmpeg，请将 ffmpeg.exe 放置在 ffmpeg 目录中，'
            '或安装 ffmpeg 并添加到系统 PATH 环境变量'
        )

    return _ffmpeg, _ffprobe


_FFMPEG, _FFPROBE = _resolve_ffmpeg()


def _compose_output_format(content, container, translation_enabled):
    content = content if content in ('双语', '目标') else '双语'
    container = container if container in ('SRT', 'LRC') else 'SRT'
    return f"{content if translation_enabled else '原文'}{container}"


def _format_command(command):
    return subprocess.list2cmdline(command) if os.name == 'nt' else shlex.join(command)


ONLINE_TRANSLATOR_MAPPING = {
    'Kimi': 'https://api.moonshot.cn',
    'Kimi (国际)': 'https://api.moonshot.ai',
    'GLM': 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
    'GLM (国际)': 'https://api.z.ai/api/paas/v4/chat/completions',
    'Deepseek': 'https://api.deepseek.com',
    'Minimax': 'https://api.minimaxi.com',
    'Minimax (国际)': 'https://api.minimaxi.io',
    '豆包': 'https://ark.cn-beijing.volces.com/api',
    '阿里云': 'https://dashscope.aliyuncs.com/compatible-mode',
    'Gemini': 'https://generativelanguage.googleapis.com/v1beta/openai',
    'OpenAI': 'https://api.openai.com',
    'Ollama': 'http://localhost:11434',
    "llamacpp（通用本地模型）": "http://localhost:8989",
}

TRANSLATOR_SUPPORTED = [
    "custom（自定义模型）",
    "sakura（日语本地模型）",
] + list(ONLINE_TRANSLATOR_MAPPING.keys())


# .env API Key 读写辅助函数
def _load_api_key() -> str:
    """从项目根目录 .env 文件中读取 API Key"""
    if not os.path.exists('.env'):
        return ''
    with open('.env', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('VOICETRANSL_API_KEY='):
                return line.split('=', 1)[1].strip()


def _save_api_key(api_key: str) -> None:
    """将 API Key 写入项目根目录 .env 文件"""
    lines = []
    if os.path.exists('.env'):
        with open('.env', 'r', encoding='utf-8') as f:
            lines = f.readlines()
    found = False
    with open('.env', 'w', encoding='utf-8') as f:
        for line in lines:
            if line.startswith('VOICETRANSL_API_KEY='):
                f.write(f'VOICETRANSL_API_KEY={api_key}\n')
                found = True
            else:
                f.write(line)
        if not found:
            f.write(f'VOICETRANSL_API_KEY={api_key}\n')


# redirect sys.stdout and sys.stderr to one log file
LOG_PATH = 'log.log'
sys.stdout = open(LOG_PATH, 'w', encoding='utf-8')
sys.stderr = sys.stdout
