"""Faster-Whisper 与在线翻译的增量流水线（实验）。

识别端迭代 Faster-Whisper 已确认的 segment；翻译端在独立线程中复用同一个
GalTransl API 客户端，按小批次顺序消费。这样无需切割音频或重复加载 ASR
模型，也能让网络翻译与 GPU 识别重叠执行。
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import shlex
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


StatusCallback = Callable[[str], None]
ProgressCallback = Callable[[float], None]


@dataclass
class StreamingPipelineResult:
    source_json: str
    translated_json: str
    segment_count: int
    asr_seconds: float
    first_translation_seconds: float | None
    translated_before_asr_done: bool
    total_seconds: float


def _atomic_write_json(path: str | Path, data: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temp, target)


def _activate_cuda_dll_dirs() -> None:
    """让当前 Python 进程能找到虚拟环境中的 CUDA 12 DLL。"""
    if os.name != "nt":
        return

    venv_root = Path(os.sys.executable).resolve().parent.parent
    nvidia_root = venv_root / "Lib" / "site-packages" / "nvidia"
    candidates = (
        nvidia_root / "cublas" / "bin",
        nvidia_root / "cudnn" / "bin",
        nvidia_root / "cuda_nvrtc" / "bin",
    )
    for directory in candidates:
        if not directory.is_dir():
            continue
        directory_text = str(directory)
        if directory_text not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = directory_text + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(directory_text)
        except (AttributeError, OSError):
            pass


def _parse_faster_whisper_args(extra_args: str) -> tuple[dict, list[str]]:
    """解析实验流水线支持的少量 Faster-Whisper CLI 风格参数。"""
    values: dict = {
        "beam_size": 5,
        "vad_filter": True,
        "word_timestamps": True,
        "condition_on_previous_text": True,
    }
    ignored: list[str] = []
    tokens = shlex.split(extra_args or "", posix=os.name != "nt")
    i = 0
    value_options = {
        "--beam-size": ("beam_size", int),
        "--temperature": ("temperature", float),
        "--initial-prompt": ("initial_prompt", str),
    }
    while i < len(tokens):
        token = tokens[i]
        if token in value_options and i + 1 < len(tokens):
            key, converter = value_options[token]
            try:
                values[key] = converter(tokens[i + 1])
            except (TypeError, ValueError):
                ignored.extend(tokens[i : i + 2])
            i += 2
            continue
        if token == "--no-vad":
            values["vad_filter"] = False
            i += 1
            continue
        ignored.append(token)
        i += 1
    return values, ignored


class _IncrementalTranslator:
    def __init__(
        self,
        config_dir: str,
        cache_path: str,
        translated_json: str,
        batch_size: int,
        stop_event,
        status: StatusCallback,
    ) -> None:
        self.config_dir = config_dir
        self.cache_path = cache_path
        self.translated_json = translated_json
        self.batch_size = max(1, min(32, int(batch_size)))
        self.stop_event = stop_event
        self.status = status
        self.input_queue: queue.Queue[list[dict] | None] = queue.Queue(maxsize=4)
        self.error: BaseException | None = None
        self.first_completed_at: float | None = None
        self.sentences = []
        self.metadata: list[dict] = []
        self.thread = threading.Thread(
            target=self._thread_main,
            name="streaming-translation",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def submit(self, batch: list[dict]) -> None:
        while not self.stop_event.is_set():
            if self.error:
                raise RuntimeError("流式翻译线程失败") from self.error
            try:
                self.input_queue.put(batch, timeout=0.2)
                return
            except queue.Full:
                continue

    def finish(self) -> None:
        while self.thread.is_alive():
            if self.error:
                raise RuntimeError("流式翻译线程失败") from self.error
            try:
                self.input_queue.put(None, timeout=0.2)
                break
            except queue.Full:
                if self.stop_event.is_set():
                    break
        self.thread.join()
        if self.error:
            raise RuntimeError("流式翻译线程失败") from self.error

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        api = None
        try:
            api, config = self._create_backend()
            while not self.stop_event.is_set():
                batch_data = self.input_queue.get()
                if batch_data is None:
                    break
                batch = self._append_sentences(batch_data)
                loop.run_until_complete(
                    api.batch_translate(
                        "stream",
                        self.cache_path,
                        batch,
                        self.batch_size,
                        proofread=False,
                        translist_unhit=batch,
                    )
                )
                self._write_translated_checkpoint()
                if self.first_completed_at is None:
                    self.first_completed_at = time.monotonic()
                self.status(
                    f"[STREAM] 已翻译 {len(self.sentences)} 句，ASR 与翻译并行运行中"
                )

            if (
                not self.stop_event.is_set()
                and self.sentences
                and config.getKey("gpt.enableProofRead")
            ):
                sentence_count = len(self.sentences)
                self.status(
                    f"[STREAM] ASR 已完成，正在一次性提交全部 {sentence_count} 句进行校对..."
                )
                loop.run_until_complete(
                    api.batch_translate(
                        "stream-proofread",
                        self.cache_path,
                        self.sentences,
                        sentence_count,
                        proofread=True,
                        translist_unhit=self.sentences,
                    )
                )
                self._write_translated_checkpoint()
                self.status(f"[STREAM] 已校对 {sentence_count}/{sentence_count} 句")
        except BaseException as exc:
            self.error = exc
        finally:
            if api is not None:
                try:
                    loop.run_until_complete(api.shutdown())
                except BaseException:
                    pass
            loop.close()

    def _create_backend(self):
        from GalTransl.Backend.ForGalJsonTranslate import ForGalJsonTranslate
        from GalTransl.COpenAI import COpenAITokenPool
        from GalTransl.ConfigHelper import CProjectConfig, CProxyPool

        config = CProjectConfig(self.config_dir)
        config.select_translator = "ForGal-json"
        config.stop_event = self.stop_event
        config.active_workers = 1
        config.bar = lambda *_args, **_kwargs: None
        proxy_pool = CProxyPool(config) if config.getKey("internals.enableProxy") else None
        token_pool = COpenAITokenPool(config, config.select_translator)
        config.proxyPool = proxy_pool
        config.tokenPool = token_pool
        api = ForGalJsonTranslate(
            config,
            config.select_translator,
            proxy_pool,
            token_pool,
        )
        return api, config

    def _append_sentences(self, batch_data: Iterable[dict]):
        from GalTransl.CSentense import CSentense

        batch = []
        for item in batch_data:
            sentence = CSentense(
                str(item["message"]),
                str(item.get("name", "")),
                int(item["index"]),
            )
            if self.sentences:
                sentence.prev_tran = self.sentences[-1]
                self.sentences[-1].next_tran = sentence
            self.sentences.append(sentence)
            self.metadata.append(item)
            batch.append(sentence)
        return batch

    def _write_translated_checkpoint(self) -> None:
        output = []
        for item, sentence in zip(self.metadata, self.sentences):
            translated = sentence.proofread_zh or sentence.pre_zh
            output.append(
                {
                    "start": item["start"],
                    "end": item["end"],
                    "message": translated,
                }
            )
        _atomic_write_json(self.translated_json, output)


def run_streaming_pipeline(
    *,
    audio_path: str,
    model_path: str,
    language: str,
    device: str,
    compute_type: str,
    asr_extra: str,
    config_dir: str,
    source_json: str,
    translated_json: str,
    cache_path: str,
    batch_size: int,
    stop_event,
    status: StatusCallback | None = None,
    progress: ProgressCallback | None = None,
) -> StreamingPipelineResult:
    """运行一次增量识别翻译流程。仅支持 Faster-Whisper。"""
    status = status or (lambda _message: None)
    progress = progress or (lambda _position: None)
    started = time.monotonic()
    _activate_cuda_dll_dirs()
    from faster_whisper import WhisperModel

    transcribe_kwargs, ignored = _parse_faster_whisper_args(asr_extra)
    if ignored:
        status(f"[STREAM] 流式模式暂未处理参数：{' '.join(ignored)}")

    resolved_model = str(Path(model_path).resolve()) if model_path else "base"
    selected_device = "cuda" if device == "auto" else device
    status(f"[STREAM] 正在加载 Faster-Whisper：{resolved_model}")
    model = WhisperModel(
        resolved_model,
        device=selected_device,
        compute_type=compute_type,
    )

    translator = _IncrementalTranslator(
        config_dir=config_dir,
        cache_path=cache_path,
        translated_json=translated_json,
        batch_size=batch_size,
        stop_event=stop_event,
        status=status,
    )
    translator.start()

    source_rows: list[dict] = []
    pending: list[dict] = []
    segments, info = model.transcribe(
        audio_path,
        language=None if language == "auto" else language,
        **transcribe_kwargs,
    )
    status(
        f"[STREAM] 开始增量识别，检测语言={getattr(info, 'language', language)}"
    )
    try:
        for segment in segments:
            if stop_event.is_set():
                break
            text = str(segment.text or "").strip()
            if not text:
                continue
            item = {
                "index": len(source_rows) + 1,
                "start": float(segment.start),
                "end": float(segment.end),
                "message": text,
            }
            source_rows.append(item)
            progress(item["end"])
            pending.append(item)
            _atomic_write_json(source_json, source_rows)
            status(
                f"[STREAM] 听写第 {item['index']} 句 ({item['start']:.1f}-{item['end']:.1f}s)：{text}"
            )
            if len(pending) >= max(1, batch_size):
                translator.submit(pending)
                pending = []

        if pending and not stop_event.is_set():
            translator.submit(pending)
        asr_finished_at = time.monotonic()
        asr_seconds = asr_finished_at - started
        translator.finish()
    except BaseException:
        stop_event.set()
        translator.finish()
        raise
    finally:
        del model

    return StreamingPipelineResult(
        source_json=str(Path(source_json).resolve()),
        translated_json=str(Path(translated_json).resolve()),
        segment_count=len(source_rows),
        asr_seconds=asr_seconds,
        first_translation_seconds=(
            translator.first_completed_at - started
            if translator.first_completed_at is not None
            else None
        ),
        translated_before_asr_done=(
            translator.first_completed_at is not None
            and translator.first_completed_at < asr_finished_at
        ),
        total_seconds=time.monotonic() - started,
    )
