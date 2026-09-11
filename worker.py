import os
import json
import re
import shutil
import socket
import subprocess
import tempfile
import threading
from pathlib import Path
from time import monotonic

import requests
import httpx
import yaml
from PySide6.QtCore import QObject, Signal, Slot

import asrlabs_bridge
import crispasr_bridge
from core import (
    _FFMPEG,
    _FFPROBE,
    _SEPARATE_CMD,
    _format_command,
    _load_api_key,
    ONLINE_TRANSLATOR_MAPPING,
    model_supports_thinking,
)
from i18n import _
from log import _stream_proc_to_queue
from pool import (
    ConcurrentTranslationPool,
    TranscribedFile,
    build_llama_server_command,
)
from tasking import (
    CancellationToken,
    ProcessRegistry,
    TaskCancelledError,
    TaskSnapshot,
)
from GalTransl.ConfigHelper import build_httpx_sync_proxy_kwargs
from prompt2srt import make_lrc, make_srt, merge_lrc_files
from srt2prompt import make_prompt, merge_srt_files
from sentence_refiner import (
    SentenceRefiner,
    crispasr_json_rows,
    resegment_request_limits,
    request_openai_compatible,
)
from opencode_zen import filter_supported_opencode_models
from yt_dlp import YoutubeDL
from bilibili_dl.bilibili_dl.Video import Video
from bilibili_dl.bilibili_dl.downloader import download
from bilibili_dl.bilibili_dl.utils import send_request
from bilibili_dl.bilibili_dl.constants import URL_VIDEO_INFO

CRISPASR_DIR = crispasr_bridge.DEFAULT_CRISPASR_DIR


def _find_available_local_port() -> int:
    """Choose an unused loopback port for a short-lived model self-test."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def _offline_asr_test_audio(language) -> Path:
    """Return the bundled speech sample matching the selected language."""
    language_code = str(language or 'ja').lower()
    if language_code.startswith('zh'):
        language_code = 'zh'
    sample = Path('assets') / 'offline_model_test' / f'{language_code}.mp3'
    if not sample.is_file():
        sample = Path('assets') / 'offline_model_test' / 'en.mp3'
    return sample.resolve()

def error_handler(func):
    def wrapper(self):
        try:
            func(self)
        except TaskCancelledError:
            self._task_outcome = "cancelled"
            self._emit_status(_("status_cancel_done"))
        except Exception as e:
            self._task_outcome = "error"
            self._emit_status(_("status_generic_error", error=e))
        finally:
            self._finalize_task()

    return wrapper
class MainWorker(QObject):
    finished = Signal()
    outcome = Signal(str)
    status = Signal(str)
    show_model_dialog = Signal(list)

    def __init__(self, snapshot: TaskSnapshot, msg_queue, cancel_token: CancellationToken):
        super().__init__()
        self.config = snapshot
        self.msg_queue = msg_queue
        self.cancel_token = cancel_token
        self._process_registry = ProcessRegistry(cancel_token)
        self.child_processes = []
        self._child_processes_lock = threading.Lock()
        self._proc_readers = {}
        self._translation_pool = None
        self._stop_requested = False
        self._stop_event = cancel_token.event
        self._finished_once = False
        self._task_outcome = "success"
        self._task_asr_config = dict(self.config.get('asr_config', {}))
        self._progress_lock = threading.Lock()
        self._stage_sequence = 0
        self._stage_context: dict[int, tuple[str, int]] = {}
        self._completed_file_ids: set[str] = set()
        self._failed_file_ids: set[str] = set()
        self._translation_item_stages: dict[str, int] = {}
        self._file_progress_total = 0

    @Slot()
    def execute(self):
        """Qt slot that dispatches the snapshot operation in the worker thread."""
        operation = self.config.operation
        handler = getattr(self, operation, None)
        if not callable(handler):
            self._task_outcome = "error"
            self._emit_status(_("status_generic_error", error=f"Unknown operation: {operation}"))
            self._finalize_task()
            return
        handler()

    def _finalize_task(self):
        if self._finished_once:
            return
        self._finished_once = True
        try:
            if self._translation_pool:
                self._translation_pool.stop()
        except Exception:
            pass
        self._process_registry.terminate_all()
        self.outcome.emit(self._task_outcome)
        self.finished.emit()

    def _emit_status(self, msg: str):
        """同时向统一消息队列和窗口标题发送状态消息"""
        self.msg_queue.put("status", msg)
        self.status.emit(msg)

    def _emit_file_progress(self, completed: int, total: int, *, visible=True):
        """Publish whole-file completion independently from stage progress."""
        total = max(0, int(total))
        completed = max(0, min(total, int(completed))) if total else 0
        event = {
            "kind": "files",
            "completed": completed,
            "total": total,
            "visible": bool(visible and total > 0),
        }
        self.msg_queue.put("progress", json.dumps(event, ensure_ascii=False))

    def _set_file_progress_total(self, total: int):
        with self._progress_lock:
            self._file_progress_total = max(0, int(total))
            self._completed_file_ids.clear()
            self._failed_file_ids.clear()
            completed = 0
            file_total = self._file_progress_total
        self._emit_file_progress(completed, file_total)

    def _complete_file(self, file_id: object):
        """Count a source file once, only after its complete pipeline succeeds."""
        key = str(file_id)
        with self._progress_lock:
            if key in self._completed_file_ids:
                return
            self._completed_file_ids.add(key)
            completed = len(self._completed_file_ids)
            total = self._file_progress_total
        self._emit_file_progress(completed, total)

    def _translation_task_finished(
        self, file_id: object, outcome: str, completes_file: bool
    ):
        key = str(file_id)
        if outcome != 'success':
            with self._progress_lock:
                self._failed_file_ids.add(key)
            return
        with self._progress_lock:
            failed = key in self._failed_file_ids
        if completes_file and not failed:
            self._complete_file(key)

    def _complete_file_if_translation_succeeded(self, file_id: object):
        key = str(file_id)
        with self._progress_lock:
            failed = key in self._failed_file_ids
        if not failed:
            self._complete_file(key)

    def _translation_item_progress(
        self, tf_dict: dict, current: int, total: int
    ):
        file_id = str(tf_dict.get('source_file_id', '') or '')
        if not file_id:
            return
        with self._progress_lock:
            stage_id = self._translation_item_stages.get(file_id)
        if stage_id is not None:
            self._update_stage(stage_id, current, total)

    def _begin_stage(self, phase: str, total: int = 1) -> int:
        """Create a unique stage and publish its mandatory zero starting point."""
        total = max(1, int(total))
        with self._progress_lock:
            self._stage_sequence += 1
            stage_id = self._stage_sequence
            self._stage_context[stage_id] = (str(phase), total)
        self._update_stage(stage_id, 0, total)
        return stage_id

    def _update_stage(self, stage_id: int, current: int, total: int | None = None):
        with self._progress_lock:
            context = self._stage_context.get(int(stage_id))
        if context is None:
            return
        phase, original_total = context
        total = max(1, int(total if total is not None else original_total))
        event = {
            "kind": "stage",
            "stage_id": int(stage_id),
            "current": max(0, min(total, int(current))),
            "total": total,
            "phase": str(phase),
        }
        self.msg_queue.put("progress", json.dumps(event, ensure_ascii=False))

    def _finish_stage(self, stage_id: int):
        with self._progress_lock:
            context = self._stage_context.get(int(stage_id))
        if context is not None:
            self._update_stage(stage_id, context[1], context[1])

    def _start_process(self, args, label=None):
        proc = self._process_registry.popen(
            args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        reader = threading.Thread(
            target=_stream_proc_to_queue,
            args=(proc, self.msg_queue, label),
            daemon=True,
        )
        with self._child_processes_lock:
            self.child_processes.append(proc)
            self._proc_readers[proc] = reader
        reader.start()
        self.pid = proc
        return proc

    def _cleanup_process(self, proc):
        if not proc:
            return
        try:
            if proc.poll() is None:
                self._process_registry.terminate(proc)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        finally:
            reader = None
            with self._child_processes_lock:
                if proc in self.child_processes:
                    self.child_processes.remove(proc)
                reader = self._proc_readers.pop(proc, None)
            self._process_registry.unregister(proc)
            if reader is not None:
                reader.join(timeout=2)

    def _terminate_all_children(self):
        with self._child_processes_lock:
            children = list(self.child_processes)
        for proc in children:
            self._cleanup_process(proc)

    def stop(self):
        """Compatibility shim; safe to invoke from the GUI thread."""
        self._stop_requested = True
        self.cancel_token.cancel()

    def _raise_if_cancelled(self):
        self.cancel_token.raise_if_cancelled()

    def _copy_file_with_cancel(self, source, destination, chunk_size=8 * 1024 * 1024):
        """分块复制识别输入，并在大文件复制期间响应取消。"""
        with open(source, 'rb') as source_file, open(destination, 'wb') as target_file:
            while True:
                self._raise_if_cancelled()
                chunk = source_file.read(chunk_size)
                if not chunk:
                    break
                target_file.write(chunk)
        self._raise_if_cancelled()

    def _resolve_online_profile(self, prefix: str) -> dict:
        """Resolve an auxiliary online profile from the immutable snapshot."""
        main_provider = self.config.get('translator', '')
        main_endpoint = (
            self.config.get('gpt_address', '')
            if 'custom' in main_provider.lower()
            else ONLINE_TRANSLATOR_MAPPING.get(main_provider, '')
        )
        main_token = self.config.get('gpt_token', '') or _load_api_key()
        main_model = self.config.get('gpt_model', '')

        provider = self.config.get(f'{prefix}_provider', 'follow') or 'follow'
        if provider == 'follow':
            return {
                'provider': main_provider,
                'endpoint': main_endpoint,
                'model': main_model,
                'token': main_token,
                'follows_main': True,
            }

        endpoint = (
            self.config.get(f'{prefix}_address', '')
            if provider == 'custom'
            else ONLINE_TRANSLATOR_MAPPING.get(provider, '')
        )
        return {
            'provider': provider,
            'endpoint': endpoint,
            'model': self.config.get(f'{prefix}_model', '') or main_model,
            'token': self.config.get(f'{prefix}_token', '') or main_token,
            'follows_main': False,
        }

    def _maybe_refine_json(self, json_path: str) -> bool:
        with open(json_path, 'r', encoding='utf-8') as stream:
            rows = json.load(stream)
        if not self.config.get('enable_ai_resegment', False) or not rows:
            return False

        profile = self._resolve_online_profile('ai_resegment')
        provider = str(profile['provider'])
        if (
            profile['follows_main']
            and ('sakura' in provider.lower() or 'llamacpp' in provider.lower())
        ):
            self._emit_status(_("status_ai_resegment_unsupported"))
            return False
        self._emit_status(_("status_ai_resegment_start", count=len(rows)))
        stage_id = self._begin_stage(_("progress_phase_resegment"), len(rows))
        max_output_tokens, max_output_characters = resegment_request_limits(rows)

        def requester(prompt: str) -> str:
            return request_openai_compatible(
                prompt,
                endpoint=profile['endpoint'],
                model=profile['model'],
                api_key=profile['token'],
                proxy=self.config.get('proxy_address', ''),
                cancel_token=self.cancel_token,
                thinking_enabled=bool(
                    self.config.get('ai_resegment_thinking', False)
                ),
                output_callback=lambda event: self.msg_queue.put(
                    "characters", json.dumps(event, ensure_ascii=False)
                ),
                progress_callback=lambda current, total: self._update_stage(
                    stage_id, current, total
                ),
                max_output_tokens=max_output_tokens,
                max_output_characters=max_output_characters,
                expected_last_id=len(rows),
            )

        refiner = SentenceRefiner(
            requester=requester,
            cancel_token=self.cancel_token,
            status=lambda message: self.msg_queue.put('detail', f"[AI断句] {message}"),
        )
        refined = refiner.refine(rows)
        if not refiner.applied:
            self._emit_status(_(
                "status_ai_resegment_failed",
                error=refiner.last_error or "unknown response error",
            ))
            return False
        if not refiner.changed:
            self._finish_stage(stage_id)
            self._emit_status(_("status_ai_resegment_unchanged"))
            return False
        temp_path = json_path + '.resegment.tmp'
        with open(temp_path, 'w', encoding='utf-8') as stream:
            json.dump(refined, stream, ensure_ascii=False, indent=2)
        os.replace(temp_path, json_path)
        self._finish_stage(stage_id)
        self._emit_status(_("status_ai_resegment_done", count=len(refined)))
        return True

    def _combine_transcribed_segments(
        self,
        json_paths: list[str],
        output_path: str,
        segment_duration_seconds: float,
    ) -> list[dict]:
        combined: list[dict] = []
        for index, path in enumerate(json_paths):
            self._raise_if_cancelled()
            with open(path, 'r', encoding='utf-8') as stream:
                rows = json.load(stream)
            offset = index * segment_duration_seconds
            for row in rows:
                item = dict(row)
                item['start'] = float(item.get('start', 0.0)) + offset
                item['end'] = float(item.get('end', 0.0)) + offset
                if item.get('words'):
                    item['words'] = [
                        {
                            **word,
                            'start': float(word.get('start', 0.0)) + offset,
                            'end': float(word.get('end', 0.0)) + offset,
                        }
                        for word in item['words']
                    ]
                combined.append(item)
        with open(output_path, 'w', encoding='utf-8') as stream:
            json.dump(combined, stream, ensure_ascii=False, indent=2)
        return combined

    def _check_auto_shutdown(self):
        """检查是否需要自动关机"""
        if self.config.get('auto_shutdown', False):
            self.status.emit(_("status_auto_shutdown"))
            import platform
            system = platform.system()
            try:
                if system == 'Darwin':  # macOS
                    subprocess.Popen(['osascript', '-e', 'tell application "System Events" to shut down'])
                elif system == 'Windows':
                    subprocess.Popen(['shutdown', '/s', '/t', '0'])
                else:  # Linux
                    subprocess.Popen(['shutdown', '-h', 'now'])
            except Exception as e:
                self.status.emit(_("status_auto_shutdown_error", error=e))

    def update_translation_config(self):
        self._emit_status(_("status_config_translating"))
        translator = self.config.get('translator', '')
        language = self.config.get('language', 'ja')
        gpt_token = self.config.get('gpt_token', '') or _load_api_key()
        gpt_address = self.config.get('gpt_address', '')
        gpt_model = self.config.get('gpt_model', '')
        proofread_profile = self._resolve_online_profile('proofread')
        proofread_model = proofread_profile['model'] or gpt_model
        sakura_file = self.config.get('sakura_file', '')
        proxy_address = self.config.get('proxy_address', '')

        if not gpt_token:
            gpt_token = 'sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

        try:
            with open('project/config.yaml', 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            # 首次运行：从默认模板初始化配置文件
            from GalTransl.DefaultProjectConfig import DEFAULT_PROJECT_CONFIG_YAML
            self._emit_status(_("status_first_run_init"))
            os.makedirs('project', exist_ok=True)
            cfg = yaml.safe_load(DEFAULT_PROJECT_CONFIG_YAML) or {}
            with open('project/config.yaml', 'w', encoding='utf-8') as f:
                yaml.dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        except Exception as e:
            self._emit_status(_("status_config_read_error", error=e))
            return

        # Update language setting
        if 'common' not in cfg:
            cfg['common'] = {}
        target_lang = self.config.get('target_lang', 'zh-cn')
        source_lang = language or 'ja'
        if source_lang == 'zh':
            source_lang = 'zh-cn'
        cfg['common']['language'] = f"{source_lang}2{target_lang}"
        cfg['common']['gpt.enableProofRead'] = bool(
            self.config.get('enable_proofread', False)
        )

        # Update backendSpecific configuration
        if 'backendSpecific' not in cfg:
            cfg['backendSpecific'] = {}

        # Determine which backend to use
        if 'sakura' in translator:
            # Sakura LLM configuration
            if 'SakuraLLM' not in cfg['backendSpecific']:
                cfg['backendSpecific']['SakuraLLM'] = {}
            sakura_cfg = cfg['backendSpecific']['SakuraLLM']
            sakura_cfg['endpoints'] = ['http://127.0.0.1:8989']
            sakura_cfg['rewriteModelName'] = sakura_file if sakura_file else ""
        else:
            # OpenAI-Compatible configuration
            if 'OpenAI-Compatible' not in cfg['backendSpecific']:
                cfg['backendSpecific']['OpenAI-Compatible'] = {}
            openai_cfg = cfg['backendSpecific']['OpenAI-Compatible']

            # Determine endpoint and model
            if 'custom' in translator:
                endpoint = gpt_address if gpt_address else 'https://api.openai.com'
                model = gpt_model if gpt_model else ''
            else:
                endpoint = ONLINE_TRANSLATOR_MAPPING.get(translator, 'https://api.openai.com')
                model = gpt_model
                if 'llamacpp' in translator:
                    model = sakura_file

            # Remove trailing /v1 or /v1/ from endpoint
            endpoint = endpoint.rstrip('/')
            if endpoint.endswith('/v1'):
                endpoint = endpoint[:-3]

            # Configure tokens
            openai_cfg['tokens'] = [{
                'token': gpt_token,
                'endpoint': endpoint,
                'modelName': model
            }]
            openai_cfg['proofreadModelName'] = proofread_model
            openai_cfg['proofreadEndpoint'] = (
                '' if proofread_profile['follows_main']
                else proofread_profile['endpoint']
            )
            openai_cfg['proofreadToken'] = (
                '' if proofread_profile['follows_main']
                else proofread_profile['token']
            )
            openai_cfg['tokenStrategy'] = "random"
            openai_cfg['checkAvailable'] = True
            openai_cfg['stream'] = True
            openai_cfg['apiTimeout'] = 120
            openai_cfg['apiErrorWait'] = "auto"
            if model_supports_thinking(model):
                openai_cfg['thinkingMode'] = (
                    'enabled'
                    if self.config.get('deepseek_thinking', False)
                    else 'disabled'
                )
            else:
                openai_cfg['thinkingMode'] = 'auto'
            if model_supports_thinking(proofread_model):
                openai_cfg['proofreadThinkingMode'] = (
                    'enabled'
                    if self.config.get('proofread_thinking', False)
                    else 'disabled'
                )
            else:
                openai_cfg['proofreadThinkingMode'] = 'auto'

        # Update proxy configuration
        if 'proxy' not in cfg:
            cfg['proxy'] = {}
        cfg['proxy']['enableProxy'] = bool(proxy_address)
        if proxy_address:
            cfg['proxy']['proxies'] = [{'address': proxy_address}]
        else:
            cfg['proxy']['proxies'] = []

        # Update extra prompt configuration (gpt.change_prompt and gpt.prompt_content)
        extra_prompt = self.config.get('extra_prompt', '').strip()
        change_prompt_mode = self.config.get('change_prompt_mode', '不修改')

        # Map UI mode to config values
        mode_mapping = {
            '不修改': 'no',
            '追加': 'AdditionalPrompt',
            '覆盖': 'OverwritePrompt'
        }

        if 'common' not in cfg:
            cfg['common'] = {}

        cfg['common']['gpt.change_prompt'] = mode_mapping.get(change_prompt_mode, 'no')

        if change_prompt_mode != '不修改' and extra_prompt:
            cfg['common']['gpt.prompt_content'] = extra_prompt
        elif change_prompt_mode == '不修改':
            # If mode is 'no', clear the prompt_content to use default
            if 'gpt.prompt_content' in cfg['common']:
                del cfg['common']['gpt.prompt_content']

        try:
            with open('project/config.yaml', 'w', encoding='utf-8') as f:
                yaml.dump(cfg, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        except Exception as e:
            self._emit_status(_("status_config_write_error", error=e))

    @error_handler
    def test_online_api(self):
        self._stop_requested = False
        self._emit_file_progress(0, 0, visible=False)
        translator = self.config.get('translator', '')
        gpt_token = self.config.get('gpt_token', '') or _load_api_key()
        gpt_address = self.config.get('gpt_address', '')
        gpt_model = self.config.get('gpt_model', '')
        proxy_address = self.config.get('proxy_address', '')

        base_url = None
        if 'custom' in translator and gpt_address:
            base_url = gpt_address
        else:
            base_url = ONLINE_TRANSLATOR_MAPPING.get(translator)

        if not base_url:
            self._emit_status(_("status_api_select_model"))
            return

        base_url = re.sub(
            r'/chat/completions$', '', base_url.rstrip('/'), flags=re.IGNORECASE
        )
        if re.search(r'/v\d+(?:beta)?(?:/openai)?$', base_url):
            base_url += '/models'
        else:
            base_url += '/v1/models'

        stage_id = self._begin_stage(_("progress_phase_api"))
        self._emit_status(_("status_api_testing", url=base_url))
        try:
            if proxy_address:
                os.environ['HTTP_PROXY'] = proxy_address
                os.environ['HTTPS_PROXY'] = proxy_address
            else:
                os.environ.pop('HTTP_PROXY', None)
                os.environ.pop('HTTPS_PROXY', None)

            headers = {
                'Authorization': f'Bearer {gpt_token}',
                'Content-Type': 'application/json'
            }
            client_kwargs = build_httpx_sync_proxy_kwargs(proxy_address or None)
            client_kwargs['trust_env'] = False
            with httpx.Client(
                timeout=httpx.Timeout(connect=3, read=2, write=3, pool=3),
                **client_kwargs,
            ) as client:
                resp = client.get(base_url, headers=headers)
                resp.raise_for_status()
            self._raise_if_cancelled()

            models = []
            seen_models = set()
            parse_error = False
            try:
                data = resp.json()
                if isinstance(data, dict) and 'data' in data:
                    for item in data['data']:
                        if isinstance(item, dict) and 'id' in item:
                            model_id = str(item['id']).strip()
                            if model_id and model_id not in seen_models:
                                seen_models.add(model_id)
                                models.append(model_id)
                models = filter_supported_opencode_models(base_url, models)
                if models:
                    total_models = len(models)
                    model_limit = 500
                    if total_models > model_limit:
                        models = models[:model_limit]
                        self._emit_status(_(
                            "status_api_models_truncated",
                            total=total_models,
                            limit=model_limit,
                        ))
                    self.show_model_dialog.emit(models)
                    self._emit_status(_("status_api_complete", count=len(models)))
                else:
                    parse_error = True
            except Exception:
                parse_error = True

            if parse_error:
                try:
                    body = resp.text[:500].replace('\n', ' ')
                except Exception:
                    body = str(resp)[:500].replace('\n', ' ')
                self._emit_status(_("status_api_complete_body", url=base_url, body=body))
            self._finish_stage(stage_id)
        except Exception as e:
            self._emit_status(_("status_api_error", error=e))

    @error_handler
    def test_offline_asr(self):
        """Load the selected CrispASR models and transcribe a bundled sample."""
        self._stop_requested = False
        self._emit_file_progress(0, 0, visible=False)
        started_at = monotonic()
        proc = None
        try:
            asr_config = dict(self._task_asr_config)
            if asr_config.get('provider') != 'crispasr':
                raise ValueError(_("offline_test_requires_crispasr"))
            language = self.config.get('language', 'ja')
            audio_file = _offline_asr_test_audio(language)
            if not audio_file.is_file():
                raise FileNotFoundError(_("offline_test_audio_missing"))

            model_file = str(asr_config.get('crispasr_model', '')).strip()
            aligner_file = str(asr_config.get('crispasr_aligner', '')).strip()
            backend = str(asr_config.get('crispasr_backend', '')).strip()
            command_template = str(asr_config.get('crispasr_param', '')).strip()
            if not model_file:
                raise ValueError(_("offline_test_asr_model_missing"))
            if not aligner_file:
                raise ValueError(_("offline_test_aligner_missing"))

            self._emit_status(_(
                "status_offline_asr_test_starting", model=model_file
            ))
            with tempfile.TemporaryDirectory(prefix='voicetransl_asr_test_') as temp_dir:
                output_base = Path(temp_dir) / 'transcript'
                command = crispasr_bridge.build_command(
                    audio_file,
                    output_base,
                    model_file,
                    language,
                    command_template,
                    aligner_file=aligner_file,
                    backend=backend,
                )
                self.msg_queue.put("detail", _format_command(command))
                proc = self._start_process(command, label='CrispASR test')
                try:
                    return_code = self._process_registry.wait(proc, timeout=300)
                except subprocess.TimeoutExpired as exc:
                    raise TimeoutError(_("offline_test_asr_timeout")) from exc
                self._raise_if_cancelled()
                if return_code != 0:
                    raise RuntimeError(f'CrispASR exited with code {return_code}')
                result_file = output_base.with_suffix('.srt')
                if not result_file.is_file() or result_file.stat().st_size == 0:
                    raise RuntimeError(_("offline_test_asr_no_output"))

            self._emit_status(_(
                "status_offline_asr_test_success",
                seconds=monotonic() - started_at,
            ))
        except TaskCancelledError:
            raise
        except Exception as exc:
            self._task_outcome = "error"
            self._emit_status(_("status_offline_asr_test_failed", error=exc))
        finally:
            if proc is not None:
                self._cleanup_process(proc)

    @error_handler
    def test_offline_translation(self):
        """Start the selected llama-server model and perform one local request."""
        self._stop_requested = False
        self._emit_file_progress(0, 0, visible=False)
        started_at = monotonic()
        proc = None
        try:
            model_file = str(self.config.get('sakura_file', '')).strip()
            gpu_layers = str(self.config.get('sakura_mode', '')).strip()
            command_template = str(self.config.get('param_llama', '')).strip()
            if not model_file:
                raise ValueError(_("offline_test_translation_model_missing"))

            port = _find_available_local_port()
            command = build_llama_server_command(
                model_file, gpu_layers, command_template, port
            )
            self._emit_status(_(
                "status_offline_translation_test_starting", model=model_file
            ))
            self.msg_queue.put("detail", _format_command(command))
            proc = self._start_process(command, label='llama-server test')

            session = requests.Session()
            session.trust_env = False
            deadline = monotonic() + 180
            last_error = ''
            while monotonic() < deadline:
                self._raise_if_cancelled()
                return_code = proc.poll()
                if return_code is not None:
                    raise RuntimeError(f'llama-server exited with code {return_code}')
                try:
                    response = session.post(
                        f'http://127.0.0.1:{port}/v1/chat/completions',
                        json={
                            'model': Path(model_file).name,
                            'messages': [{
                                'role': 'user',
                                'content': 'Reply with only: OK',
                            }],
                            'max_tokens': 8,
                            'temperature': 0,
                        },
                        timeout=8,
                    )
                    if response.status_code == 200:
                        payload = response.json()
                        if isinstance(payload, dict) and payload.get('choices'):
                            self._emit_status(_(
                                "status_offline_translation_test_success",
                                seconds=monotonic() - started_at,
                            ))
                            return
                    last_error = f'HTTP {response.status_code}: {response.text[:200]}'
                except requests.RequestException as exc:
                    last_error = str(exc)
                except ValueError as exc:
                    last_error = str(exc)
                self.cancel_token.event.wait(1)

            detail = last_error or _("offline_test_no_response")
            raise TimeoutError(_(
                "offline_test_translation_timeout", detail=detail
            ))
        except TaskCancelledError:
            raise
        except Exception as exc:
            self._task_outcome = "error"
            self._emit_status(_(
                "status_offline_translation_test_failed", error=exc
            ))
        finally:
            if proc is not None:
                self._cleanup_process(proc)

    @error_handler
    def vocal_split(self):
        self._stop_requested = False
        uvr_file = self.config.get('uvr_file', '')
        if not uvr_file.endswith('.onnx'):
            self._emit_status(_("status_uvr_model_error"))
            return

        input_files = self.config.get('uvr_input_files', '')
        if input_files:
            input_files = input_files.strip().split('\n')
            self._set_file_progress_total(len(input_files))
            for idx, input_file in enumerate(input_files):
                if self._stop_requested:
                    break
                if not os.path.exists(input_file):
                    self._emit_status(_("status_file_not_exist", file=input_file))
                    return

                self._emit_status(_("status_vocal_split_label", idx=idx+1, total=len(input_files)))
                stage_id = self._begin_stage(_("progress_phase_separate"))
                proc = self._start_process([*_SEPARATE_CMD, '-m', os.path.join('separate',uvr_file), input_file])
                return_code = self._process_registry.wait(proc)
                self._cleanup_process(proc)
                if return_code != 0:
                    raise RuntimeError(f"separation exited with code {return_code}")
                self._finish_stage(stage_id)
                self._complete_file(idx)

            self._emit_status(_("status_vocal_processing_done"))
    @error_handler
    def summarize(self):
        self._stop_requested = False
        # 统一刷新翻译配置，供摘要复用
        self.update_translation_config()
        input_files = self.config.get('summarize_input_files', '')
        # 使用与主程序相同的配置：从 project/config.yaml 读取 GPT 配置与代理
        try:
            with open('project/config.yaml', 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            self._emit_status(_("status_config_read_error", error=e))
            return

        backend = (cfg or {}).get('backendSpecific', {})
        openai_cfg = backend.get('OpenAI-Compatible', {})
        tokens = openai_cfg.get('tokens', []) or []
        token = tokens[0].get('token') if tokens else ''
        address = tokens[0].get('endpoint') if tokens else ''
        model = tokens[0].get('modelName') if tokens else ''

        # 代理设置同步
        proxy_cfg = (cfg or {}).get('proxy', {})
        if proxy_cfg.get('enableProxy'):
            proxies = proxy_cfg.get('proxies') or []
            if proxies and isinstance(proxies[0], dict):
                proxy_address = proxies[0].get('address')
                if proxy_address:
                    os.environ['HTTP_PROXY'] = proxy_address
                    os.environ['HTTPS_PROXY'] = proxy_address
        else:
            # 清理可能遗留的代理环境变量
            os.environ.pop('HTTP_PROXY', None)
            os.environ.pop('HTTPS_PROXY', None)

        prompt = self.config.get('summarize_prompt', '')
        if input_files:
            input_files = input_files.strip().split('\n')
            self._set_file_progress_total(len(input_files))
            for idx, input_file in enumerate(input_files):
                if not os.path.exists(input_file):
                    self._emit_status(_("status_file_not_exist", file=input_file))
                    return

                from summarize import summarize
                self._emit_status(_("status_summarize_processing", idx=idx+1, total=len(input_files)))
                stage_id = self._begin_stage(_("progress_phase_summarize"))
                summarize(
                    input_file, address, model, token, prompt,
                    cancel_token=self.cancel_token,
                )
                self._finish_stage(stage_id)
                self._complete_file(idx)
            self._emit_status(_("status_processing_done"))
    @error_handler
    def synth(self):
        self._stop_requested = False
        subtitle_font = self.config.get('subtitle_font', '').strip()
        subtitle_type = self.config.get('subtitle_type', '硬字幕') or "硬字幕"

        video_files_text = self.config.get('synth_video_files', '').strip()
        srt_files_text = self.config.get('synth_srt_files', '').strip()

        def escape_sub_path(path_str: str) -> str:
            # ffmpeg subtitles filter needs windows drive colon escaped
            return path_str.replace('\\', '/').replace(':', '\\:').replace("'", "\\'")

        def build_subtitle_filter(srt_path: str, font_value: str) -> str:
            srt_abs = escape_sub_path(str(Path(srt_path).resolve()))
            parts = [f"subtitles='{srt_abs}'"]
            if font_value:
                font_path = Path(font_value)
                if font_path.exists():
                    fonts_dir = escape_sub_path(str(font_path.parent.resolve()))
                    font_name = font_path.name.replace("'", "\\'")
                    parts.append(f"fontsdir='{fonts_dir}'")
                    parts.append(f"force_style='FontName={font_name}'")
                else:
                    font_name = font_value.replace("'", "\\'")
                    parts.append(f"force_style='FontName={font_name}'")
            return ':'.join(parts)

        if video_files_text and srt_files_text:
            video_files = video_files_text.split('\n')
            srt_files = srt_files_text.split('\n')

            if len(srt_files) != len(video_files):
                self._emit_status(_("status_synth_mismatch"))
                return

            self._set_file_progress_total(len(video_files))

            for idx, (input_file, input_srt) in enumerate(zip(video_files, srt_files)):
                if self._stop_requested:
                    break
                if not os.path.exists(input_file):
                    self._emit_status(_("status_file_not_exist", file=input_file))
                    return

                if not os.path.exists(input_srt):
                    self._emit_status(_("status_file_not_exist", file=input_srt))
                    return

                self._emit_status(_("status_synth_processing", file=input_file, idx=idx+1, total=len(video_files)))

                output_file = input_file + '_synth.mp4'
                stage_id = self._begin_stage(_("progress_phase_synth"))

                if subtitle_type == "硬字幕":
                    input_srt_cache = shutil.copy(input_srt, 'project/cache/')
                    subtitle_filter = build_subtitle_filter(input_srt_cache, subtitle_font)
                    if subtitle_font:
                        self._emit_status(_("status_synth_font", font=subtitle_font))
                    self._emit_status(_("status_synth_hard_sub"))
                    proc = self._start_process([_FFMPEG, '-y', '-i', input_file, '-vf', subtitle_filter, '-vcodec', 'libx264', '-acodec', 'aac', output_file])
                else:
                    self._emit_status(_("status_synth_soft_sub"))
                    # For soft subtitles, we just map the streams.
                    # Depending on the container and subtitle format, -c:s mov_text works for mp4.
                    proc = self._start_process([_FFMPEG, '-y', '-i', input_file, '-i', input_srt, '-c:v', 'copy', '-c:a', 'copy', '-c:s', 'mov_text', output_file])

                return_code = self._process_registry.wait(proc)
                self._cleanup_process(proc)
                if return_code != 0:
                    raise RuntimeError(f"ffmpeg exited with code {return_code}")
                self._emit_status(_("status_synth_done"))
                self._finish_stage(stage_id)
                self._complete_file(idx)

    @error_handler
    def clip(self):
        self._stop_requested = False
        input_files = self.config.get('clip_input_files', '')
        clip_start = self.config.get('clip_start', '')
        clip_end = self.config.get('clip_end', '')
        if input_files:
            input_files = input_files.strip().split('\n')
            self._set_file_progress_total(len(input_files))
            for idx, input_file in enumerate(input_files):
                if self._stop_requested:
                    break
                if not os.path.exists(input_file):
                    self._emit_status(_("status_file_not_exist", file=input_file))
                    return

                self._emit_status(_("status_processing_file", file=input_file, idx=idx+1, total=len(input_files)))
                self._emit_status(_("status_clip_processing", start=clip_start, end=clip_end))
                stage_id = self._begin_stage(_("progress_phase_clip"))
                proc = self._start_process([_FFMPEG, '-y', '-i', input_file, '-ss', clip_start, '-to', clip_end, '-vcodec', 'libx264', '-acodec', 'aac', os.path.join(*(input_file.split('.')[:-1]))+'_clip.'+input_file.split('.')[-1]])
                return_code = self._process_registry.wait(proc)
                self._cleanup_process(proc)
                if return_code != 0:
                    raise RuntimeError(f"ffmpeg exited with code {return_code}")
                self._emit_status(_("status_clip_done"))
                self._finish_stage(stage_id)
                self._complete_file(idx)
    @error_handler
    def audiosynth(self):
        self._stop_requested = False
        input_files = self.config.get('synth_audio_files', '')
        if input_files:
            input_files = input_files.strip().split('\n')
            audio_files = sorted([i for i in input_files if i.endswith('.wav') or i.endswith('.mp3') or i.endswith('.flac')])
            image_files = sorted([i for i in input_files if i.endswith('.png') or i.endswith('.jpg') or i.endswith('.jpeg')])
            if len(audio_files) != len(image_files):
                self._emit_status(_("status_audio_mismatch"))
                return

            self._set_file_progress_total(len(image_files))

            for idx, (audio_input, image_input) in enumerate(zip(audio_files, image_files)):
                if self._stop_requested:
                    break
                if not os.path.exists(audio_input):
                    self._emit_status(_("status_file_not_exist", file=audio_input))
                    return

                if not os.path.exists(image_input):
                    self._emit_status(_("status_file_not_exist", file=image_input))
                    return

                self._emit_status(_("status_processing_file", file=audio_input, idx=idx+1, total=len(image_files)))
                stage_id = self._begin_stage(_("progress_phase_synth"))
                proc = self._start_process([_FFMPEG, '-y', '-loop', '1', '-r', '1', '-f', 'image2', '-i', image_input, '-i', audio_input, '-shortest', '-vcodec', 'libx264', '-acodec', 'aac', audio_input+'_synth.mp4'], label='ffmpeg')
                return_code = self._process_registry.wait(proc)
                self._cleanup_process(proc)
                if return_code != 0:
                    raise RuntimeError(f"ffmpeg exited with code {return_code}")
                self._emit_status(_("status_synth_done"))
                self._finish_stage(stage_id)
                self._complete_file(idx)

    @error_handler
    def clean(self):
        self._emit_file_progress(0, 0, visible=False)
        stage_id = self._begin_stage(_("progress_phase_clean"), 2)
        self._emit_status(_("status_cleaning_intermediate"))
        for path in (
            'project/gt_input',
            'project/gt_output',
            'project/transl_cache',
        ):
            self._raise_if_cancelled()
            if os.path.exists(path):
                shutil.rmtree(path)
        self._update_stage(stage_id, 1, 2)
        self._emit_status(_("status_cleaning_output"))
        self._raise_if_cancelled()
        if os.path.exists('project/cache'):
            shutil.rmtree('project/cache')
        os.makedirs('project/cache', exist_ok=True)
        self._finish_stage(stage_id)

    def _process_single_audio(
        self,
        wav_file,
        language,
        json_path,
        start_named_proc,
        stop_named_proc,
        asr_config=None,
    ):
        """处理单个音频文件的听写

        根据界面选择分发到 ASRLabs 或 CrispASR，再统一生成
        GalTransl JSON，后续翻译和字幕输出不区分识别提供方。
        """
        asr_config = asr_config or self._task_asr_config
        asr_provider = asr_config['provider']
        if asr_provider == 'crispasr':
            try:
                self._process_crispasr_audio(
                    wav_file,
                    language,
                    json_path,
                    start_named_proc,
                    stop_named_proc,
                    asr_config,
                )
            except TaskCancelledError:
                raise
            except Exception as error:
                self._emit_status(_("status_crispasr_error", error=error))
                raise
            return

        base_path = wav_file[:-4]  # 去掉 .wav

        # 从 master 获取 ASRLabs 配置
        asr_engine = asr_config['asr_engine']
        if not asr_engine:
            return

        asr_model = asr_config['asr_model']
        asr_device = asr_config['asr_device']
        asr_compute_type = asr_config['asr_compute_type']
        asr_extra = asr_config['asr_extra']
        align_engine = asr_config['align_engine']
        align_model = asr_config['align_model']
        align_device = asr_config['align_device']
        align_extra = asr_config['align_extra']

        output_dir = os.path.abspath(os.path.dirname(json_path))
        output_name = os.path.basename(json_path).replace('.json', '')

        self._emit_status(_("status_asrlabs_transcribing", engine=asr_engine, audio=os.path.basename(wav_file)))

        try:
            galtransl_json = asrlabs_bridge.run_transcribe_and_align(
                audio_path=wav_file,
                engine=asr_engine,
                model_path=asr_model,
                language=language,
                device=asr_device,
                compute_type=asr_compute_type,
                aligner=align_engine,
                align_model_path=align_model,
                align_device=align_device,
                transcribe_extra=asr_extra,
                align_extra=align_extra,
                split_logic='punct',
                max_chars=40,
                output_dir=output_dir,
                output_name=output_name,
                msg_queue=self.msg_queue,
                stop_event=self._stop_event,
            )

            # 将 galtransl JSON 复制/重命名为期望的 json_path
            if galtransl_json != json_path:
                shutil.copy(galtransl_json, json_path)

            self._emit_status(_("status_asrlabs_done", output=os.path.basename(json_path)))

        except Exception as e:
            if self._stop_requested or self._stop_event.is_set():
                raise TaskCancelledError() from e
            self._emit_status(_("status_asrlabs_error", error=e))
            raise

    def _process_crispasr_audio(
        self,
        wav_file,
        language,
        json_path,
        start_named_proc,
        stop_named_proc,
        asr_config=None,
    ):
        """运行本地 CrispASR GGUF 模型并把 SRT 转成 GalTransl JSON。"""
        asr_config = asr_config or self._task_asr_config
        backend = asr_config['crispasr_backend']
        model_file = asr_config['crispasr_model']
        aligner_file = asr_config['crispasr_aligner']
        command_template = asr_config['crispasr_param']

        self._raise_if_cancelled()

        work_root = Path('project/cache/crispasr_jobs').resolve()
        work_root.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix='job_', dir=work_root))
        staged_input = work_dir / f"input{Path(wav_file).suffix.lower()}"
        output_base = work_dir / 'transcript'
        generated_srt = output_base.with_suffix('.srt')
        generated_json = output_base.with_suffix('.json')

        self._emit_status(
            _(
                "status_crispasr_transcribing",
                backend=backend,
                audio=os.path.basename(wav_file),
            )
        )
        try:
            self._copy_file_with_cancel(wav_file, staged_input)
            command = crispasr_bridge.build_command(
                input_file=staged_input,
                output_file=output_base,
                model_file=model_file,
                language=language,
                command_template=command_template,
                aligner_file=aligner_file,
                backend=backend,
                crispasr_dir=CRISPASR_DIR,
            )
            if (
                self.config.get('enable_ai_resegment', False)
                and '--output-json-full' not in command
                and '-ojf' not in command
            ):
                try:
                    input_index = command.index(str(staged_input.resolve()))
                except ValueError:
                    input_index = len(command)
                if input_index > 0 and command[input_index - 1] in ('--file', '-f'):
                    input_index -= 1
                command.insert(input_index, '--output-json-full')
            self._raise_if_cancelled()
            self.msg_queue.put(
                "detail",
                f"[CrispASR] {subprocess.list2cmdline(command)}",
            )
            asr_proc, _duplicate = start_named_proc('crispasr', command)
            if self._stop_requested or self._stop_event.is_set():
                stop_named_proc('crispasr')
                raise TaskCancelledError()
            return_code = self._process_registry.wait(asr_proc)
            stop_named_proc('crispasr')
            self._raise_if_cancelled()
            if return_code != 0:
                raise RuntimeError(f"CrispASR exited with code {return_code}")
            if not generated_srt.is_file() or generated_srt.stat().st_size == 0:
                raise RuntimeError("CrispASR did not produce a non-empty SRT file")

            Path(json_path).parent.mkdir(parents=True, exist_ok=True)
            rows = []
            if generated_json.is_file() and generated_json.stat().st_size:
                try:
                    with generated_json.open('r', encoding='utf-8') as stream:
                        rows = crispasr_json_rows(json.load(stream))
                except Exception as error:
                    self.msg_queue.put(
                        'detail', f"[CrispASR] 完整 JSON 解析失败，回退 SRT：{error}",
                    )
            if rows:
                with open(json_path, 'w', encoding='utf-8') as stream:
                    json.dump(rows, stream, ensure_ascii=False, indent=2)
            else:
                make_prompt(str(generated_srt), json_path)
            self._emit_status(
                _("status_crispasr_done", output=os.path.basename(json_path))
            )
        finally:
            stop_named_proc('crispasr')
            shutil.rmtree(work_dir, ignore_errors=True)

    def _process_streaming_audio(
        self,
        wav_file,
        language,
        json_path,
        base_path,
        output_dir,
        output_format,
        asr_config=None,
        progress_callback=None,
    ):
        """Faster-Whisper 逐段产出并与在线翻译重叠执行。"""
        from streaming_pipeline import run_streaming_pipeline

        asr_config = asr_config or self._task_asr_config
        asr_model = asr_config['asr_model']
        asr_device = asr_config['asr_device']
        asr_compute_type = asr_config['asr_compute_type']
        asr_extra = asr_config['asr_extra']

        with open('project/config.yaml', 'r', encoding='utf-8') as f:
            project_config = yaml.safe_load(f) or {}
        common_config = project_config.get('common', {})
        batch_size = int(common_config.get('streaming.batchSize', 8) or 8)

        workspace_name = re.sub(r'[^a-zA-Z0-9._-]+', '_', os.path.basename(base_path))
        workspace = os.path.abspath(os.path.join('project', 'cache', 'streaming', workspace_name))
        os.makedirs(workspace, exist_ok=True)
        translated_json = os.path.join(workspace, 'translated.json')
        cache_path = os.path.join(workspace, 'translation_cache.json')

        self._emit_status(_("status_streaming_start"))
        try:
            result = run_streaming_pipeline(
                audio_path=wav_file,
                model_path=asr_model,
                language=language,
                device=asr_device,
                compute_type=asr_compute_type,
                asr_extra=asr_extra,
                config_dir='project',
                source_json=json_path,
                translated_json=translated_json,
                cache_path=cache_path,
                batch_size=batch_size,
                stop_event=self._stop_event,
                status=self._emit_status,
                progress=progress_callback,
            )
        except Exception as error:
            if self._stop_requested or self._stop_event.is_set():
                raise TaskCancelledError() from error
            raise
        self._raise_if_cancelled()

        base_name = os.path.basename(base_path)
        if output_format in ('原文SRT', '双语SRT'):
            make_srt(json_path, os.path.join(output_dir, base_name + '.srt'))
        if output_format in ('目标SRT', '双语SRT'):
            make_srt(translated_json, os.path.join(output_dir, base_name + '.tg.srt'))
        if output_format == '双语SRT':
            merge_srt_files(
                [
                    os.path.join(output_dir, base_name + '.srt'),
                    os.path.join(output_dir, base_name + '.tg.srt'),
                ],
                os.path.join(output_dir, base_name + '.combine.srt'),
            )

        if output_format == '原文LRC':
            make_lrc(json_path, os.path.join(output_dir, base_name + '.lrc'))
        elif output_format == '目标LRC':
            make_lrc(translated_json, os.path.join(output_dir, base_name + '.lrc'))
        elif output_format == '双语LRC':
            original_lrc = os.path.join(output_dir, base_name + '.orig.lrc')
            translated_lrc = os.path.join(output_dir, base_name + '.zh.lrc')
            make_lrc(json_path, original_lrc)
            make_lrc(translated_json, translated_lrc)
            merge_lrc_files(
                [original_lrc, translated_lrc],
                os.path.join(output_dir, base_name + '.combine.lrc'),
            )

        self._emit_status(_(
            "status_streaming_done",
            count=result.segment_count,
            asr=result.asr_seconds,
            first=result.first_translation_seconds or 0.0,
            overlap=result.translated_before_asr_done,
            total=result.total_seconds,
        ))
        return result

    def _get_audio_duration(self, audio_file):
        """获取音频文件时长（秒）"""
        try:
            proc = self._process_registry.popen(
                [_FFPROBE, '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', audio_file],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            self._process_registry.wait(proc, timeout=30)
            stdout = proc.stdout.read() if proc.stdout else ''
            return float(stdout.strip())
        except TaskCancelledError:
            raise
        except Exception as e:
            self._emit_status(_("status_audio_duration_fail", error=e))
            return 0

    def _split_audio(self, audio_file, segment_duration_minutes, output_dir):
        """将音频文件切分为多个片段，返回片段路径列表"""
        segment_files = []
        segment_duration = segment_duration_minutes * 60  # 转换为秒

        total_duration = self._get_audio_duration(audio_file)
        if total_duration == 0:
            return None, 0

        num_segments = int(total_duration // segment_duration) + (1 if total_duration % segment_duration > 1 else 0)
        base_name = os.path.basename(audio_file).rsplit('.', 1)[0]

        self._emit_status(_("status_audio_duration", duration=total_duration, segments=num_segments))

        for i in range(num_segments):
            self._raise_if_cancelled()
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, total_duration)
            duration = end_time - start_time

            segment_file = os.path.join(output_dir, f"segment_{i:04d}.16k.wav")

            proc = None
            try:
                proc = self._start_process(
                    [_FFMPEG, '-hide_banner', '-loglevel', 'error', '-y',
                     '-i', audio_file, '-ss', str(start_time),
                     '-t', str(duration), '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', segment_file],
                    label=f'ffmpeg_segment_{i + 1}',
                )
                try:
                    return_code = self._process_registry.wait(proc, timeout=120)
                except subprocess.TimeoutExpired:
                    self._emit_status(_("status_segment_slice_fail", idx=i+1))
                    continue
                self._raise_if_cancelled()
                if return_code == 0 and os.path.exists(segment_file):
                    segment_files.append(segment_file)
                else:
                    self._emit_status(_("status_segment_slice_fail", idx=i+1))
            except TaskCancelledError:
                raise
            except Exception as e:
                self._emit_status(_("status_segment_slice_fail_detail", idx=i+1, error=e))
            finally:
                self._cleanup_process(proc)

        self._raise_if_cancelled()
        return segment_files, total_duration

    def _merge_segment_translations(self, segment_files, segment_tfs, original_base_path, output_json_path, final_output_dir, output_format, duration):
        """合并多个分段的翻译结果，调整时间戳并生成最终字幕文件"""
        from prompt2srt import make_srt, make_lrc, merge_lrc_files
        from srt2prompt import merge_srt_files
        import glob as glob_module

        all_data = []
        time_offset = 0
        segment_srts_orig = []
        segment_srts_zh = []
        segment_lrcs_orig = []
        segment_lrcs_zh = []

        base_name = os.path.basename(original_base_path)

        for i, segment_file in enumerate(segment_files):
            segment_name = os.path.basename(segment_file[:-4])  # 去掉 .wav，保留 .16k
            segment_dir = os.path.dirname(segment_file)

            # 收集分段的字幕文件（用于双语合并）
            if output_format in ('原文SRT', '双语SRT'):
                orig_srt = os.path.join(segment_dir, segment_name + '.srt')
                if os.path.exists(orig_srt):
                    segment_srts_orig.append(orig_srt)

            if output_format in ('目标SRT', '双语SRT'):
                zh_srt = os.path.join(segment_dir, segment_name + '.tg.srt')
                if os.path.exists(zh_srt):
                    segment_srts_zh.append(zh_srt)

            if output_format in ('原文LRC', '双语LRC'):
                original_suffix = '.orig.lrc' if output_format == '双语LRC' else '.lrc'
                orig_lrc = os.path.join(segment_dir, segment_name + original_suffix)
                if os.path.exists(orig_lrc):
                    segment_lrcs_orig.append(orig_lrc)

            if output_format in ('目标LRC', '双语LRC'):
                target_suffix = '.zh.lrc' if output_format == '双语LRC' else '.lrc'
                zh_lrc = os.path.join(segment_dir, segment_name + target_suffix)
                if os.path.exists(zh_lrc):
                    segment_lrcs_zh.append(zh_lrc)

        # 生成最终的合并字幕文件
        if output_format in ('原文SRT', '双语SRT'):
            final_srt = os.path.join(final_output_dir, base_name + '.srt')
            merge_srt_files(segment_srts_orig, final_srt, duration)

        if output_format in ('目标SRT', '双语SRT'):
            final_zh_srt = os.path.join(final_output_dir, base_name + '.tg.srt')
            merge_srt_files(segment_srts_zh, final_zh_srt, duration)

        if output_format == '双语SRT':
            final_combine_srt = os.path.join(final_output_dir, base_name + '.combine.srt')
            left = os.path.join(final_output_dir, base_name + '.srt')
            right = os.path.join(final_output_dir, base_name + '.tg.srt')
            if os.path.exists(left) and os.path.exists(right):
                merge_srt_files([left, right], final_combine_srt)

        if output_format in ('原文LRC', '双语LRC'):
            final_lrc = os.path.join(final_output_dir, base_name + '.lrc')
            if output_format == '双语LRC':
                final_lrc = os.path.join(final_output_dir, base_name + '.orig.lrc')
            merge_lrc_files(segment_lrcs_orig, final_lrc, duration)

        if output_format in ('目标LRC', '双语LRC'):
            target_suffix = '.zh.lrc' if output_format == '双语LRC' else '.lrc'
            final_zh_lrc = os.path.join(final_output_dir, base_name + target_suffix)
            merge_lrc_files(segment_lrcs_zh, final_zh_lrc, duration)

        if output_format == '双语LRC':
            final_combine_lrc = os.path.join(final_output_dir, base_name + '.combine.lrc')
            left = os.path.join(final_output_dir, base_name + '.orig.lrc')
            right = os.path.join(final_output_dir, base_name + '.zh.lrc')
            if os.path.exists(left) and os.path.exists(right):
                merge_lrc_files([left, right], final_combine_lrc)

        return all_data

    @error_handler
    def run(self):
        self._stop_requested = False
        input_files = self.config.get('input_files', '')
        asr_config = dict(self._task_asr_config)
        asr_provider = asr_config['provider']
        asr_engine = asr_config['asr_engine']
        crispasr_model = asr_config['crispasr_model']
        transcription_enabled = bool(self.config.get('enable_transcription', True)) and (
            bool(asr_engine)
            if asr_provider == 'asrlabs'
            else bool(crispasr_model)
        )
        translator = self.config.get('translator', '')
        language = self.config.get('language', 'ja')
        sakura_file = self.config.get('sakura_file', '')
        sakura_mode = self.config.get('sakura_mode', '')
        proxy_address = self.config.get('proxy_address', '')
        before_dict = self.config.get('before_dict', '')
        gpt_dict = self.config.get('gpt_dict', '')
        after_dict = self.config.get('after_dict', '')
        param_llama = self.config.get('param_llama', '')
        output_format = self.config.get('output_format', '双语SRT')
        output_dir = self.config.get('output_dir', os.path.abspath('project/cache'))
        use_input_dir = bool(self.config.get('use_input_dir', False))
        enable_segment = bool(self.config.get('enable_segment', False))
        segment_duration_minutes = int(self.config.get('segment_duration', 0)) if enable_segment else 0
        enable_streaming = bool(self.config.get('enable_streaming', False))

        with open('llama/param.txt', 'w', encoding='utf-8') as f:
            f.write(param_llama)

        self._emit_status(_("status_init_project"))
        if use_input_dir:
            self._emit_status(_("status_use_input_dir"))
        else:
            self._emit_status(_("status_output_dir", dir=output_dir))

        os.makedirs('project/cache', exist_ok=True)
        if before_dict:
            with open('project/dict_pre.txt', 'w', encoding='utf-8') as f:
                f.write(before_dict.replace(' ','\t'))
        else:
            if os.path.exists('project/dict_pre.txt'):
                os.remove('project/dict_pre.txt')
        if gpt_dict:
            with open('project/dict_gpt.txt', 'w', encoding='utf-8') as f:
                f.write(gpt_dict.replace(' ','\t'))
        else:
            if os.path.exists('project/dict_gpt.txt'):
                os.remove('project/dict_gpt.txt')
        if after_dict:
            with open('project/dict_after.txt', 'w', encoding='utf-8') as f:
                f.write(after_dict.replace(' ','\t'))
        else:
            if os.path.exists('project/dict_after.txt'):
                os.remove('project/dict_after.txt')

        self._emit_status(_("status_current_input", files=input_files))

        if input_files:
            input_files = input_files.split('\n')
        else:
            input_files = []
        file_total = len(input_files)
        self._set_file_progress_total(file_total)

        os.makedirs('project/cache', exist_ok=True)

        # 统一刷新翻译配置
        self.update_translation_config()

        target_lang = self.config.get('target_lang', 'zh-cn')
        need_translate = bool(self.config.get('enable_translation', True))
        if not need_translate:
            if translator == '不进行翻译':
                self._emit_status(_("status_no_translator_skip"))

        engine = 'ForGal-json'
        if need_translate and 'sakura' in translator:
            engine = 'sakura-v1.0'

        running_procs = {}
        proc_lock = threading.Lock()

        def start_named_proc(proc_name, args):
            with proc_lock:
                existing = running_procs.get(proc_name)
                if existing and existing.poll() is None:
                    self._emit_status(_("status_duplicate_proc", name=proc_name))
                    return existing, True
                if existing:
                    self._cleanup_process(existing)
                    running_procs.pop(proc_name, None)

                new_proc = self._start_process(args, label=proc_name)
                running_procs[proc_name] = new_proc
                return new_proc, False

        def stop_named_proc(proc_name):
            with proc_lock:
                target = running_procs.pop(proc_name, None)
                if target:
                    self._cleanup_process(target)

        # 流水线流程：听写线程 + 翻译线程并行
        transcribed_dir = os.path.join('project', 'cache', 'transcribed')
        os.makedirs(transcribed_dir, exist_ok=True)
        # 创建并发翻译线程池
        max_concurrent = int(self.config.get('max_concurrent', 1))

        # 本地模型配置
        local_model_config = None
        if 'sakura' in translator or 'llamacpp' in translator:
            local_model_config = {
                'sakura_file': sakura_file,
                'sakura_mode': sakura_mode,
                'param_llama': param_llama,
            }

        # 同步详细日志模式设置到翻译线程池
        ConcurrentTranslationPool.verbose_galtransl = bool(
            self.config.get('verbose_mode', False)
        )

        translation_stage_lock = threading.Lock()
        translation_stage_id: int | None = None
        translation_stage_base = 0
        translation_stage_total = 0

        def emit_translation_progress(current: int, total: int):
            with translation_stage_lock:
                active_stage = translation_stage_id
                completed_before_stage = translation_stage_base
                active_total = translation_stage_total
            if active_stage is not None:
                self._update_stage(
                    active_stage,
                    max(0, current - completed_before_stage),
                    active_total,
                )

        self._translation_pool = ConcurrentTranslationPool(
            project_dir='project',
            base_config_path='project/config.yaml',
            max_concurrent=max_concurrent,
            stop_event=self._stop_event,
            msg_queue=self.msg_queue,
            local_model_config=local_model_config,
            progress_callback=emit_translation_progress,
            file_completion_callback=self._translation_task_finished,
            item_progress_callback=self._translation_item_progress,
        )
        self._translation_pool.start(engine)

        def submit_translation(tf: TranscribedFile):
            """Show per-sentence progress while serial file translation blocks."""
            if not self._translation_pool.serial_mode:
                self._translation_pool.submit(tf)
                return

            source_total = 1
            try:
                with open(tf.json_src, 'r', encoding='utf-8') as stream:
                    source_rows = json.load(stream)
                source_total = max(1, sum(
                    1 for row in source_rows
                    if isinstance(row, dict)
                    and str(row.get('message', '') or '').strip()
                ))
            except Exception:
                pass
            if self.config.get('enable_proofread', False):
                source_total *= 2
            stage_id = self._begin_stage(
                _("progress_phase_translation"), source_total
            )
            file_id = str(tf.source_file_id or '')
            if file_id:
                with self._progress_lock:
                    self._translation_item_stages[file_id] = stage_id
            try:
                self._translation_pool.submit(tf)
                with self._progress_lock:
                    failed = file_id in self._failed_file_ids
                if not failed:
                    self._finish_stage(stage_id)
            finally:
                if file_id:
                    with self._progress_lock:
                        if self._translation_item_stages.get(file_id) == stage_id:
                            self._translation_item_stages.pop(file_id, None)

        # 主线程：顺序执行下载+听写，产出放入队列
        for idx, input_file in enumerate(input_files):
            if self._stop_event.is_set():
                raise TaskCancelledError()
            input_stage = self._begin_stage(_("progress_phase_input"))
            self._finish_stage(input_stage)
            if not os.path.exists(input_file):
                download_stage = self._begin_stage(_("progress_phase_download"))
                if input_file.startswith('BV'):
                    self._emit_status(_("status_downloading_video"))
                    res = send_request(URL_VIDEO_INFO, params={'bvid': input_file})
                    download([Video(
                        bvid=res['bvid'],
                        cid=res['cid'] if res['videos'] == 1 else res['pages'][0]['cid'],
                        title=res['title'] if res['videos'] == 1 else res['pages'][0]['part'],
                        up_name=res['owner']['name'],
                        cover_url=res['pic'] if res['videos'] == 1 else res['pages'][0]['pic'],
                    )], False)
                    self._emit_status(_("status_download_complete"))
                    title = res['title'] if res['videos'] == 1 else res['pages'][0]['part']
                    title = re.sub(r'[.:?/\\]', ' ', title).strip()
                    title = re.sub(r'\s+', ' ', title)
                    downloaded_file = os.path.abspath(f"{title}.mp4")
                    target_file = os.path.join(output_dir, os.path.basename(downloaded_file))
                    if os.path.exists(downloaded_file):
                        if os.path.exists(target_file):
                            os.remove(target_file)
                        input_file = shutil.move(downloaded_file, target_file)
                    else:
                        self._emit_status(_("status_download_not_found", file=downloaded_file))
                        self._stop_event.set()
                        break

                else:
                    ydl_outtmpl = os.path.join(output_dir, 'YoutubeDL_%(title)s_%(id)s.%(ext)s')
                    if proxy_address:
                        ydl_ctx = YoutubeDL({'proxy': proxy_address, 'outtmpl': ydl_outtmpl})
                    else:
                        ydl_ctx = YoutubeDL({'outtmpl': ydl_outtmpl})

                    with ydl_ctx as ydl:
                        self._emit_status(_("status_downloading_video"))
                        info = ydl.extract_info(input_file, download=True)
                        self._emit_status(_("status_download_complete"))
                        input_file = ydl.prepare_filename(info)
                        requested_downloads = info.get('requested_downloads') if isinstance(info, dict) else None
                        if requested_downloads and isinstance(requested_downloads[0], dict):
                            actual_file = requested_downloads[0].get('filepath')
                            if actual_file:
                                input_file = actual_file
                        if isinstance(info, dict) and info.get('_filename') and os.path.exists(info.get('_filename')):
                            input_file = info.get('_filename')

                    input_file = os.path.abspath(str(input_file or ''))
                    if not os.path.exists(input_file):
                        self._emit_status(_("status_download_not_found", file=input_file))
                        self._stop_event.set()
                        break
                self._finish_stage(download_stage)

            self._emit_status(_("status_processing_file", file=input_file, idx=idx+1, total=len(input_files)))
            current_output_dir = output_dir
            if use_input_dir:
                current_output_dir = os.path.dirname(os.path.abspath(input_file)) or output_dir
                self._emit_status(_("status_file_output_dir", dir=current_output_dir))

            tf: TranscribedFile | None = None

            if input_file.endswith('.srt'):
                # —— SRT 输入：直接转换 ——
                self._emit_status(_("status_srt_converting"))
                json_path = os.path.join(transcribed_dir, os.path.basename(input_file).replace('.srt', '.json'))
                make_prompt(input_file, json_path)
                resegment_changed = self._maybe_refine_json(json_path)
                self._emit_status(_("status_srt_convert_done"))
                subtitle_stage = self._begin_stage(_("progress_phase_subtitle"))
                source_base_path = os.path.join(
                    current_output_dir, os.path.basename(input_file[:-4])
                )
                if resegment_changed:
                    source_base_path += '.resegmented'
                source_srt = source_base_path + '.srt'
                if output_format in ('原文SRT', '双语SRT'):
                    if os.path.abspath(source_srt) != os.path.abspath(input_file) or resegment_changed:
                        make_srt(json_path, source_srt)
                # 原文 LRC（双语 LRC 需要）
                if output_format in ('原文LRC', '双语LRC'):
                    lrc_output = source_base_path + (
                        '.orig.lrc' if output_format == '双语LRC' else '.lrc'
                    )
                    make_lrc(json_path, lrc_output)
                base_path = source_base_path
                if need_translate:
                    tf = TranscribedFile(
                        base_path=base_path,
                        json_src=json_path,
                        output_dir=current_output_dir,
                        output_format=output_format,
                        orig_srt_path=source_srt,
                        source_file_id=str(idx),
                    )
                self._finish_stage(subtitle_stage)
            else:
                # 音视频输入：提取音频 → 听写（如果已有srt则跳过）
                if not transcription_enabled:
                    self._emit_status(_("status_no_transcribe_skip"))
                    continue

                base_path = input_file.rsplit('.', 1)[0] if '.' in input_file else input_file
                existing_srt = base_path + '.srt'
                wav_file = base_path + '.16k.wav'
                json_path = os.path.join(transcribed_dir, os.path.basename(base_path) + '.json')

                # 检测是否已有srt文件
                if os.path.exists(existing_srt):
                    self._emit_status(_("status_existing_srt_found", file=existing_srt))
                    make_prompt(existing_srt, json_path)
                    resegment_changed = self._maybe_refine_json(json_path)
                    subtitle_stage = self._begin_stage(_("progress_phase_subtitle"))

                    output_base_path = os.path.join(
                        current_output_dir, os.path.basename(base_path)
                    )
                    if resegment_changed:
                        output_base_path = os.path.join(
                            current_output_dir,
                            os.path.basename(base_path) + '.resegmented',
                        )

                    # 生成原文 SRT/LRC 输出（与正常听写流程一致）
                    if output_format == '原文SRT' or output_format == '双语SRT':
                        srt_output = output_base_path + '.srt'
                        if resegment_changed or not os.path.exists(srt_output):
                            make_srt(json_path, srt_output)

                    if output_format == '原文LRC' or output_format == '双语LRC':
                        lrc_output = output_base_path + (
                            '.orig.lrc' if output_format == '双语LRC' else '.lrc'
                        )
                        if resegment_changed or not os.path.exists(lrc_output):
                            make_lrc(json_path, lrc_output)

                    self._emit_status(_("status_asr_done_cached"))

                    if need_translate:
                        self._emit_status(_("status_submitting_translation"))
                        tf = TranscribedFile(
                            base_path=output_base_path,
                            json_src=json_path,
                            output_dir=current_output_dir,
                            output_format=output_format,
                            orig_srt_path=output_base_path + '.srt',
                            source_file_id=str(idx),
                        )
                        submit_translation(tf)
                    self._finish_stage(subtitle_stage)
                    if not need_translate:
                        self._complete_file(idx)
                    continue

                self._emit_status(_("status_extracting_audio"))
                audio_stage = self._begin_stage(_("progress_phase_audio"))
                ffmpeg_proc, _unused = start_named_proc(
                    'ffmpeg_extract',
                    [_FFMPEG, '-y', '-i', input_file, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', wav_file]
                )
                self._process_registry.wait(ffmpeg_proc)
                stop_named_proc('ffmpeg_extract')
                self._raise_if_cancelled()

                if not os.path.exists(wav_file):
                    self._emit_status(_("status_audio_extract_error"))
                    break
                self._finish_stage(audio_stage)

                # 检查是否启用分段处理
                base_path = wav_file[:-8]  # 去掉 .16k.wav
                json_path = os.path.join(transcribed_dir, os.path.basename(base_path) + '.json')

                total_duration = self._get_audio_duration(wav_file)
                self._raise_if_cancelled()
                threshold_seconds = segment_duration_minutes * 60

                align_engine = asr_config['align_engine']
                if (
                    enable_streaming
                    and need_translate
                    and asr_provider == 'asrlabs'
                    and asr_engine == 'faster-whisper'
                    and align_engine == 'none'
                ):
                    streaming_stage = self._begin_stage(
                        _("progress_phase_transcribe"), 100
                    )
                    self._process_streaming_audio(
                        wav_file,
                        language,
                        json_path,
                        base_path,
                        current_output_dir,
                        output_format,
                        asr_config,
                        progress_callback=lambda position: self._update_stage(
                            streaming_stage,
                            round(min(100.0, max(0.0, position) * 100.0 / total_duration))
                            if total_duration > 0 else 0,
                            100,
                        ),
                    )
                    self._finish_stage(streaming_stage)
                    if os.path.exists(wav_file):
                        os.remove(wav_file)
                    tf = None
                    self._complete_file(idx)
                    continue
                elif enable_streaming:
                    self._emit_status(_("status_streaming_fallback"))

                if enable_segment and segment_duration_minutes > 0 and total_duration > threshold_seconds:
                    # 需要分段处理
                    self._emit_status(_("status_segment_threshold", duration=total_duration, threshold=threshold_seconds))

                    segment_dir = os.path.join('project', 'cache', 'segments', os.path.basename(base_path))
                    os.makedirs(segment_dir, exist_ok=True)

                    # 切分音频
                    split_stage = self._begin_stage(_("progress_phase_split"))
                    segment_files, _unused = self._split_audio(wav_file, segment_duration_minutes, segment_dir)
                    self._raise_if_cancelled()

                    if not segment_files:
                        self._emit_status(_("status_segment_fail"))
                        if os.path.exists(wav_file):
                            os.remove(wav_file)
                        break
                    self._finish_stage(split_stage)

                    # 对每个片段进行听写和翻译
                    segment_tfs = []  # 存储每个分段的 TranscribedFile
                    segment_json_paths = []
                    transcribe_stage = self._begin_stage(
                        _("progress_phase_transcribe"), len(segment_files)
                    )
                    for i, segment_file in enumerate(segment_files):
                        if self._stop_event.is_set():
                            raise TaskCancelledError()
                        self._emit_status(_("status_segment_processing", idx=i+1, total=len(segment_files)))

                        segment_base = segment_file[:-4] # 去掉 .wav
                        segment_name = os.path.basename(segment_base)

                        # ASRLabs 听写+对齐
                        segment_json = os.path.join(transcribed_dir, segment_name + '.json')
                        self._process_single_audio(
                            segment_file, language,
                            segment_json, start_named_proc, stop_named_proc,
                            asr_config,
                        )
                        segment_json_paths.append(segment_json)
                        self._update_stage(
                            transcribe_stage, i + 1, len(segment_files)
                        )

                        if self.config.get('enable_ai_resegment', False):
                            continue

                        if output_format in ('原文SRT', '双语SRT'):
                            make_srt(segment_json, segment_base + '.srt')
                        if output_format in ('原文LRC', '双语LRC'):
                            original_suffix = (
                                '.orig.lrc' if output_format == '双语LRC' else '.lrc'
                            )
                            make_lrc(segment_json, segment_base + original_suffix)

                        # 立即提交该分段进行翻译
                        if need_translate:
                            self._emit_status(_("status_segment_submit_translate", idx=i+1, total=len(segment_files)))
                            segment_tf = TranscribedFile(
                                base_path=segment_base,
                                json_src=segment_json,
                                output_dir=segment_dir,  # 临时输出到分段目录
                                output_format=output_format,
                                orig_srt_path='',
                                source_file_id=str(idx),
                                completes_source_file=False,
                            )
                            submit_translation(segment_tf)
                            segment_tfs.append(segment_tf)
                    self._finish_stage(transcribe_stage)

                    if self.config.get('enable_ai_resegment', False):
                        self._emit_status(_("status_merge_segments"))
                        merge_stage = self._begin_stage(_("progress_phase_merge"))
                        self._combine_transcribed_segments(
                            segment_json_paths,
                            json_path,
                            threshold_seconds,
                        )
                        self._finish_stage(merge_stage)
                        self._maybe_refine_json(json_path)
                        subtitle_stage = self._begin_stage(
                            _("progress_phase_subtitle")
                        )
                        final_base = os.path.join(
                            current_output_dir, os.path.basename(base_path)
                        )
                        if output_format in ('原文SRT', '双语SRT'):
                            make_srt(json_path, final_base + '.srt')
                        if output_format in ('原文LRC', '双语LRC'):
                            make_lrc(
                                json_path,
                                final_base + (
                                    '.orig.lrc' if output_format == '双语LRC' else '.lrc'
                                ),
                            )
                        if need_translate:
                            submit_translation(TranscribedFile(
                                base_path=final_base,
                                json_src=json_path,
                                output_dir=current_output_dir,
                                output_format=output_format,
                                orig_srt_path=final_base + '.srt',
                                source_file_id=str(idx),
                            ))
                        else:
                            self._complete_file(idx)
                        self._finish_stage(subtitle_stage)
                        if os.path.exists(wav_file):
                            os.remove(wav_file)
                        self._emit_status(_("status_segment_done"))
                        continue

                    # 等待所有分段翻译完成
                    if need_translate and segment_tfs:
                        self._emit_status(_("status_wait_segments"))
                        segment_translation_stage = self._begin_stage(
                            _("progress_phase_translation")
                        )
                        # Only wait for work submitted so far.  The same pool is
                        # reused by later input files and must not receive its
                        # shutdown sentinels until every producer is finished.
                        self._translation_pool.wait_for_pending()
                        self._raise_if_cancelled()
                        self._finish_stage(segment_translation_stage)

                    # 合并所有片段的翻译结果
                    self._emit_status(_("status_merge_segments"))
                    merge_stage = self._begin_stage(_("progress_phase_merge"))
                    self._merge_segment_translations(segment_files, segment_tfs, base_path, json_path, current_output_dir, output_format, threshold_seconds)
                    self._finish_stage(merge_stage)
                    if need_translate:
                        self._complete_file_if_translation_succeeded(idx)
                    else:
                        self._complete_file(idx)

                    self._emit_status(_("status_segment_done"))

                    # 分段处理已完成，跳过常规流程
                    tf = None
                else:
                    # 正常流程（未启用分段）
                    self._emit_status(_("status_asr_in_progress"))
                    transcribe_stage = self._begin_stage(
                        _("progress_phase_transcribe")
                    )
                    self._process_single_audio(
                        wav_file,
                        language,
                        json_path,
                        start_named_proc,
                        stop_named_proc,
                        asr_config,
                    )
                    self._finish_stage(transcribe_stage)
                    self._maybe_refine_json(json_path)

                    # 生成原文 SRT/LRC 输出
                    subtitle_stage = self._begin_stage(
                        _("progress_phase_subtitle")
                    )
                    if output_format == '原文SRT' or output_format == '双语SRT':
                        srt_output = os.path.join(current_output_dir, os.path.basename(base_path + '.srt'))
                        make_srt(json_path, srt_output)

                    if output_format == '原文LRC' or output_format == '双语LRC':
                        lrc_name = os.path.basename(base_path + '.lrc')
                        if output_format == '双语LRC':
                            lrc_name = os.path.basename(base_path + '.orig.lrc')
                        lrc_output = os.path.join(current_output_dir, lrc_name)
                        make_lrc(json_path, lrc_output)

                    # 清理临时文件
                    if os.path.exists(wav_file):
                        os.remove(wav_file)

                    self._emit_status(_("status_asr_done"))
                    self._finish_stage(subtitle_stage)

                    if need_translate:
                        tf = TranscribedFile(
                            base_path=base_path,
                            json_src=json_path,
                            output_dir=current_output_dir,
                            output_format=output_format,
                            orig_srt_path='',
                            source_file_id=str(idx),
                        )

            if need_translate and tf is not None:
                submit_translation(tf)
            elif not need_translate:
                self._complete_file(idx)

        # 发送哨兵，等待翻译线程结束
        self._raise_if_cancelled()
        self._emit_status(_("status_all_transcribed"))
        submitted_translations = self._translation_pool.submitted_count
        completed_before_wait = self._translation_pool.progress_completed_count
        remaining_translations = max(
            0, submitted_translations - completed_before_wait
        )
        if remaining_translations:
            new_translation_stage = self._begin_stage(
                _("progress_phase_translation"), remaining_translations
            )
            with translation_stage_lock:
                translation_stage_id = new_translation_stage
                translation_stage_base = completed_before_wait
                translation_stage_total = remaining_translations
            self._update_stage(
                new_translation_stage,
                max(
                    0,
                    self._translation_pool.progress_completed_count
                    - completed_before_wait,
                ),
                remaining_translations,
            )
        self._translation_pool.done()
        self._translation_pool.wait_all()
        self._raise_if_cancelled()
        if remaining_translations:
            self._finish_stage(new_translation_stage)
        self._translation_pool.stop()

        err_count = self._translation_pool.error_count
        if err_count > 0:
            self._task_outcome = "error"
            self._emit_status(_("status_translate_fail_count", count=err_count))

        # All producers have joined; FIFO ordering keeps completion last.
        self.msg_queue.put_completion_sentinel()
        self.msg_queue.set_completion_flag()
