import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import asyncio
import yaml

from PySide6.QtCore import QObject, Signal

from core import (
    _FFMPEG,
    _FFPROBE,
    _SEPARATE_CMD,
    _format_command,
    _load_api_key,
    ONLINE_TRANSLATOR_MAPPING,
)
from asr import _build_crispasr_command
from i18n import _
from log import _stream_proc_to_queue
from pool import ConcurrentTranslationPool, TranscribedFile
from prompt2srt import make_lrc, make_srt, merge_lrc_files
from srt2prompt import make_prompt, merge_srt_files
from yt_dlp import YoutubeDL
from bilibili_dl.bilibili_dl.Video import Video
from bilibili_dl.bilibili_dl.downloader import download
from bilibili_dl.bilibili_dl.utils import send_request
from bilibili_dl.bilibili_dl.constants import URL_VIDEO_INFO


def error_handler(func):
    def wrapper(self):
        try:
            func(self)
        except Exception as e:
            self._emit_status(_("status_generic_error", error=e))
            self.finished.emit()
            # Ensure all child processes are terminated on error
            self.stop()

    return wrapper


class MainWorker(QObject):
    finished = Signal()
    show_model_dialog = Signal(list)

    def __init__(self, master):
        super().__init__()
        self.master = master
        self.status = master.status
        self.msg_queue = master.msg_queue
        self.child_processes = []
        self._child_processes_lock = threading.Lock()
        self._proc_readers = {}
        self._translation_pool = None
        self._stop_requested = False
        self._stop_event = asyncio.Event()

    def _emit_status(self, msg: str):
        """同时向统一消息队列和窗口标题发送状态消息"""
        self.msg_queue.put("status", msg)
        self.status.emit(msg)

    def _start_process(self, args, label=None):
        creationflags = 0x08000000 if os.name == 'nt' else 0
        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=creationflags,
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
                proc.terminate()
                proc.wait(timeout=3)
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
            if reader is not None:
                reader.join(timeout=2)

    def _terminate_all_children(self):
        with self._child_processes_lock:
            children = list(self.child_processes)
        for proc in children:
            self._cleanup_process(proc)

    def stop(self):
        self._stop_requested = True
        self._stop_event.set()
        self._terminate_all_children()
        if hasattr(self, '_translation_pool') and self._translation_pool:
            self._translation_pool.stop()

    def _check_auto_shutdown(self):
        """检查是否需要自动关机"""
        if hasattr(self.master, 'auto_shutdown_checkbox') and self.master.auto_shutdown_checkbox.isChecked():
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

    def save_config(self, silent: bool = False):
        self.master.save_config(silent)

    @error_handler
    def update_translation_config(self):
        self._emit_status(_("status_config_translating"))
        translator = self.master.translator_group.currentText()
        language = self.master.transcription_lang.currentText()
        gpt_token = self.master.gpt_token.text() or _load_api_key()
        gpt_address = self.master.gpt_address.text()
        gpt_model = self.master.gpt_model.text()
        sakura_file = self.master.sakura_file.currentText()
        proxy_address = self.master.proxy_address.text()

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
        target_lang = self.master.target_lang.currentData() if hasattr(self.master, 'target_lang') else 'zh-cn'
        source_lang = self.master.transcription_lang.currentText() if hasattr(self.master, 'transcription_lang') else 'ja'
        if source_lang == 'zh':
            source_lang = 'zh-cn'
        cfg['common']['language'] = f"{source_lang}2{target_lang}"

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
            openai_cfg['tokenStrategy'] = "random"
            openai_cfg['checkAvailable'] = True
            openai_cfg['stream'] = True
            openai_cfg['apiTimeout'] = 120
            openai_cfg['apiErrorWait'] = "auto"

        # Update proxy configuration
        if 'proxy' not in cfg:
            cfg['proxy'] = {}
        cfg['proxy']['enableProxy'] = bool(proxy_address)
        if proxy_address:
            cfg['proxy']['proxies'] = [{'address': proxy_address}]
        else:
            cfg['proxy']['proxies'] = []

        # Update extra prompt configuration (gpt.change_prompt and gpt.prompt_content)
        extra_prompt = self.master.extra_prompt.toPlainText().strip() if hasattr(self.master, 'extra_prompt') else ''
        change_prompt_mode = self.master.change_prompt_mode.currentData() if hasattr(self.master, 'change_prompt_mode') else '不修改'

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
        self._stop_event.clear()
        self.save_config()
        translator = self.master.translator_group.currentText()
        gpt_token = self.master.gpt_token.text() or _load_api_key()
        gpt_address = self.master.gpt_address.text()
        gpt_model = self.master.gpt_model.text()
        proxy_address = self.master.proxy_address.text()

        base_url = None
        if 'custom' in translator and gpt_address:
            base_url = gpt_address
        else:
            base_url = ONLINE_TRANSLATOR_MAPPING.get(translator)

        if not base_url:
            self._emit_status(_("status_api_select_model"))
            self.finished.emit()
            return

        base_url = base_url.rstrip('/') + '/v1/models'

        self._emit_status(_("status_api_testing", url=base_url))
        try:
            if proxy_address:
                os.environ['HTTP_PROXY'] = proxy_address
                os.environ['HTTPS_PROXY'] = proxy_address
            else:
                os.environ.pop('HTTP_PROXY', None)
                os.environ.pop('HTTPS_PROXY', None)

            import requests
            headers = {
                'Authorization': f'Bearer {gpt_token}',
                'Content-Type': 'application/json'
            }

            resp = requests.get(base_url, headers=headers, timeout=20)
            resp.raise_for_status()

            models = []
            parse_error = False
            try:
                data = resp.json()
                if isinstance(data, dict) and 'data' in data:
                    for item in data['data']:
                        if isinstance(item, dict) and 'id' in item:
                            models.append(item['id'])
                if models:
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
        except Exception as e:
            self._emit_status(_("status_api_error", error=e))

        self.finished.emit()

    @error_handler
    def vocal_split(self):
        self._stop_requested = False
        self._stop_event.clear()
        self.save_config()
        uvr_file = self.master.uvr_file.currentText()
        if not uvr_file.endswith('.onnx'):
            self._emit_status(_("status_uvr_model_error"))
            self.finished.emit()
            return

        input_files = self.master.uvr_file_list.toPlainText()
        if input_files:
            input_files = input_files.strip().split('\n')
            for idx, input_file in enumerate(input_files):
                if self._stop_requested:
                    break
                if not os.path.exists(input_file):
                    self._emit_status(_("status_file_not_exist", file=input_file))
                    self.finished.emit()

                self._emit_status(_("status_vocal_split_label", idx=idx+1, total=len(input_files)))
                proc = self._start_process([*_SEPARATE_CMD, '-m', os.path.join('separate',uvr_file), input_file])
                proc.wait()
                self._cleanup_process(proc)

            self._emit_status(_("status_vocal_processing_done"))
        self.finished.emit()

    @error_handler
    def summarize(self):
        self._stop_requested = False
        self._stop_event.clear()
        self.save_config()
        # 统一刷新翻译配置，供摘要复用
        self.update_translation_config()
        input_files = self.master.summarize_files_list.toPlainText()
        # 使用与主程序相同的配置：从 project/config.yaml 读取 GPT 配置与代理
        try:
            with open('project/config.yaml', 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
        except Exception as e:
            self._emit_status(_("status_config_read_error", error=e))
            self.finished.emit()
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

        prompt = self.master.summarize_prompt.toPlainText()
        if input_files:
            input_files = input_files.strip().split('\n')
            for idx, input_file in enumerate(input_files):
                if not os.path.exists(input_file):
                    self._emit_status(_("status_file_not_exist", file=input_file))
                    self.finished.emit()

                from summarize import summarize
                self._emit_status(_("status_summarize_processing", idx=idx+1, total=len(input_files)))
                summarize(input_file, address, model, token, prompt)
            self._emit_status(_("status_processing_done"))
        self.finished.emit()

    @error_handler
    def synth(self):
        self._stop_requested = False
        self._stop_event.clear()
        self.save_config()
        subtitle_font = self.master.subtitle_font_combo.currentText().strip()
        subtitle_type = self.master.subtitle_type_combo.currentData() or "硬字幕"
        
        video_files_text = self.master.synth_video_files_list.toPlainText().strip()
        srt_files_text = self.master.synth_srt_files_list.toPlainText().strip()
        
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
                self.finished.emit()
                return
            
            for idx, (input_file, input_srt) in enumerate(zip(video_files, srt_files)):
                if self._stop_requested:
                    break
                if not os.path.exists(input_file):
                    self._emit_status(_("status_file_not_exist", file=input_file))
                    self.finished.emit()
                    return

                if not os.path.exists(input_srt):
                    self._emit_status(_("status_file_not_exist", file=input_srt))
                    self.finished.emit()
                    return

                self._emit_status(_("status_synth_processing", file=input_file, idx=idx+1, total=len(video_files)))
                
                output_file = input_file + '_synth.mp4'

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

                proc.wait()
                self._cleanup_process(proc)
                self._emit_status(_("status_synth_done"))
            
        self.finished.emit()

    @error_handler
    def clip(self):
        self._stop_requested = False
        self._stop_event.clear()
        self.save_config()
        input_files = self.master.clip_files_list.toPlainText()
        clip_start = self.master.clip_start_time.text()
        clip_end = self.master.clip_end_time.text()
        if input_files:
            input_files = input_files.strip().split('\n')
            for idx, input_file in enumerate(input_files):
                if self._stop_requested:
                    break
                if not os.path.exists(input_file):
                    self._emit_status(_("status_file_not_exist", file=input_file))
                    self.finished.emit()

                self._emit_status(_("status_processing_file", file=input_file, idx=idx+1, total=len(input_files)))
                self._emit_status(_("status_clip_processing", start=clip_start, end=clip_end))
                proc = self._start_process([_FFMPEG, '-y', '-i', input_file, '-ss', clip_start, '-to', clip_end, '-vcodec', 'libx264', '-acodec', 'aac', os.path.join(*(input_file.split('.')[:-1]))+'_clip.'+input_file.split('.')[-1]])
                proc.wait()
                self._cleanup_process(proc)
                self._emit_status(_("status_clip_done"))
        self.finished.emit()

    @error_handler
    def audiosynth(self):
        self._stop_requested = False
        self._stop_event.clear()
        self.save_config()
        input_files = self.master.synth_audio_files_list.toPlainText()
        if input_files:
            input_files = input_files.strip().split('\n')
            audio_files = sorted([i for i in input_files if i.endswith('.wav') or i.endswith('.mp3') or i.endswith('.flac')])
            image_files = sorted([i for i in input_files if i.endswith('.png') or i.endswith('.jpg') or i.endswith('.jpeg')])
            if len(audio_files) != len(image_files):
                self._emit_status(_("status_audio_mismatch"))
                self.finished.emit()
            
            for idx, (audio_input, image_input) in enumerate(zip(audio_files, image_files)):
                if self._stop_requested:
                    break
                if not os.path.exists(audio_input):
                    self._emit_status(_("status_file_not_exist", file=audio_input))
                    self.finished.emit()

                if not os.path.exists(image_input):
                    self._emit_status(_("status_file_not_exist", file=image_input))
                    self.finished.emit()

                self._emit_status(_("status_processing_file", file=audio_input, idx=idx+1, total=len(image_files)))
                proc = self._start_process([_FFMPEG, '-y', '-loop', '1', '-r', '1', '-f', 'image2', '-i', image_input, '-i', audio_input, '-shortest', '-vcodec', 'libx264', '-acodec', 'aac', audio_input+'_synth.mp4'], label='ffmpeg')
                proc.wait()
                self._cleanup_process(proc)
                self._emit_status(_("status_synth_done"))
            
        self.finished.emit()

    def _process_single_audio(self, wav_file, asr_backend, asr_model_file, asr_aligner_file, language, param_crispasr, json_path, start_named_proc, stop_named_proc):
        """使用 CrispASR + forced aligner 处理单个音频文件。"""
        base_path = wav_file[:-4]  # 去掉 .wav
        intermediate_srt = base_path + '.srt'
        work_root = Path('project/cache/crispasr_jobs').resolve()
        work_root.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix='job_', dir=work_root))
        staged_input = work_dir / f'input{Path(wav_file).suffix.lower()}'
        output_base = work_dir / 'transcript'
        generated_srt = output_base.with_suffix('.srt')
        try:
            shutil.copyfile(wav_file, staged_input)
            command = _build_crispasr_command(
                staged_input,
                output_base,
                asr_model_file,
                language,
                param_crispasr,
                aligner_file=asr_aligner_file,
                backend=asr_backend,
            )
            self.msg_queue.put("detail", _format_command(command))
            asr_proc, _unused = start_named_proc('crispasr', command)
            return_code = asr_proc.wait()
            stop_named_proc('crispasr')
            if return_code != 0:
                raise RuntimeError(f'CrispASR exited with code {return_code}')
            if not generated_srt.is_file() or generated_srt.stat().st_size == 0:
                raise RuntimeError('CrispASR did not produce a non-empty SRT file')
            shutil.copyfile(generated_srt, intermediate_srt)
            make_prompt(intermediate_srt, json_path)
        finally:
            stop_named_proc('crispasr')
            shutil.rmtree(work_dir, ignore_errors=True)
            # 单文件流程中的 16k SRT 只是中间产物。
            if intermediate_srt.endswith('.16k.srt') and os.path.exists(intermediate_srt):
                try:
                    os.remove(intermediate_srt)
                except Exception:
                    pass

    def _get_audio_duration(self, audio_file):
        """获取音频文件时长（秒）"""
        try:
            creationflags = 0x08000000 if os.name == 'nt' else 0
            result = subprocess.run(
                [_FFPROBE, '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', audio_file],
                capture_output=True, text=True, timeout=30, creationflags=creationflags
            )
            return float(result.stdout.strip())
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
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, total_duration)
            duration = end_time - start_time

            segment_file = os.path.join(output_dir, f"segment_{i:04d}.16k.wav")

            try:
                creationflags = 0x08000000 if os.name == 'nt' else 0
                proc = subprocess.run(
                    [_FFMPEG, '-y', '-i', audio_file, '-ss', str(start_time),
                     '-t', str(duration), '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', segment_file],
                    capture_output=True, timeout=120, creationflags=creationflags
                )
                if proc.returncode == 0 and os.path.exists(segment_file):
                    segment_files.append(segment_file)
                else:
                    self._emit_status(_("status_segment_slice_fail", idx=i+1))
            except Exception as e:
                self._emit_status(_("status_segment_slice_fail_detail", idx=i+1, error=e))

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
        # Reset stop event for new run
        self._stop_requested = False
        self._stop_event.clear()
        
        self.save_config()
        input_files = self.master.input_files_list.toPlainText()
        asr_backend = self.master.asr_backend.currentText()
        asr_model_file = self.master.asr_model_file.currentText()
        asr_aligner_file = self.master.asr_aligner_file.currentText()
        translator = self.master.translator_group.currentText()
        language = self.master.transcription_lang.currentText()
        sakura_file = self.master.sakura_file.currentText()
        sakura_mode = self.master.sakura_mode.text()
        proxy_address = self.master.proxy_address.text()
        before_dict = self.master.before_dict.toPlainText()
        gpt_dict = self.master.gpt_dict.toPlainText()
        after_dict = self.master.after_dict.toPlainText()
        param_crispasr = self.master.param_crispasr.toPlainText()
        param_llama = self.master.param_llama.toPlainText()
        enable_transcription = self.master.enable_transcription_checkbox.isChecked()
        need_translate = self.master.enable_translation_checkbox.isChecked()
        output_format = self.master.selected_output_format(need_translate)
        output_dir = self.master.output_dir_edit.text().strip() or self.master.default_output_dir()
        use_input_dir = self.master.use_input_dir_checkbox.isChecked()
        enable_segment = self.master.enable_segment_checkbox.isChecked()
        segment_duration_minutes = self.master.segment_duration_spin.value() if enable_segment else 0

        with open('crispasr/param.txt', 'w', encoding='utf-8') as f:
            f.write(param_crispasr)

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

        os.makedirs('project/cache', exist_ok=True)

        # Only prepare translation configuration when that stage is enabled.
        if need_translate:
            self.update_translation_config()

        target_lang = self.master.target_lang.currentData() if hasattr(self.master, 'target_lang') else 'zh-cn'
        if not need_translate:
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
        max_concurrent = self.master.max_concurrent_spin.value()

        # 本地模型配置
        local_model_config = None
        if 'sakura' in translator or 'llamacpp' in translator:
            local_model_config = {
                'sakura_file': sakura_file,
                'sakura_mode': sakura_mode,
                'param_llama': param_llama,
            }

        # 同步详细日志模式设置到翻译线程池
        ConcurrentTranslationPool.verbose_galtransl = self.master.verbose_checkbox.isChecked()

        if need_translate:
            self._translation_pool = ConcurrentTranslationPool(
                project_dir='project',
                base_config_path='project/config.yaml',
                max_concurrent=max_concurrent,
                stop_event=self._stop_event,
                msg_queue=self.msg_queue,
                local_model_config=local_model_config,
            )
            self._translation_pool.start(engine)

        # 主线程：顺序执行下载+听写，产出放入队列
        for idx, input_file in enumerate(input_files):
            if self._stop_event.is_set():
                break
            if not os.path.exists(input_file):
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
                self._emit_status(_("status_srt_convert_done"))
                # 复制原始 SRT 到输出目录（供双语合并用）
                try:
                    orig_srt_src = os.path.abspath(input_file)
                    orig_srt_dst = os.path.join(current_output_dir, os.path.basename(orig_srt_src))
                    if os.path.exists(orig_srt_src):
                        shutil.copy(orig_srt_src, orig_srt_dst)
                except Exception:
                    pass
                # 原文 LRC（双语 LRC 需要）
                if output_format in ('原文LRC', '双语LRC'):
                    lrc_suffix = '.orig.lrc' if output_format == '双语LRC' else '.lrc'
                    lrc_output = os.path.join(
                        current_output_dir,
                        os.path.basename(input_file[:-4] + lrc_suffix),
                    )
                    make_lrc(json_path, lrc_output)
                base_path = input_file[:-4]  # 去掉 .srt
                tf = TranscribedFile(
                    base_path=base_path,
                    json_src=json_path,
                    output_dir=current_output_dir,
                    output_format=output_format,
                    orig_srt_path=os.path.abspath(input_file),
                )
            else:
                # 音视频输入：提取音频 → 听写（如果已有srt则跳过）
                if not enable_transcription:
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

                    # 生成原文 SRT/LRC 输出（与正常听写流程一致）
                    if output_format == '原文SRT' or output_format == '双语SRT':
                        srt_output = os.path.join(current_output_dir, os.path.basename(base_path + '.srt'))
                        if not os.path.exists(srt_output):
                            make_srt(json_path, srt_output)

                    if output_format == '原文LRC' or output_format == '双语LRC':
                        lrc_name = os.path.basename(base_path + '.lrc')
                        if output_format == '双语LRC':
                            lrc_name = os.path.basename(base_path + '.orig.lrc')
                        lrc_output = os.path.join(current_output_dir, lrc_name)
                        if not os.path.exists(lrc_output):
                            make_lrc(json_path, lrc_output)

                    self._emit_status(_("status_asr_done_cached"))

                    if need_translate:
                        self._emit_status(_("status_submitting_translation"))
                        tf = TranscribedFile(
                            base_path=base_path,
                            json_src=json_path,
                            output_dir=current_output_dir,
                            output_format=output_format,
                            orig_srt_path='',
                        )
                        self._translation_pool.submit(tf)
                        continue
                    continue

                self._emit_status(_("status_extracting_audio"))
                ffmpeg_proc, _unused = start_named_proc(
                    'ffmpeg_extract',
                    [_FFMPEG, '-y', '-i', input_file, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', wav_file]
                )
                ffmpeg_proc.wait()
                stop_named_proc('ffmpeg_extract')

                if not os.path.exists(wav_file):
                    self._emit_status(_("status_audio_extract_error"))
                    break

                # 检查是否启用分段处理
                base_path = wav_file[:-8]  # 去掉 .16k.wav
                json_path = os.path.join(transcribed_dir, os.path.basename(base_path) + '.json')

                total_duration = self._get_audio_duration(wav_file)
                threshold_seconds = segment_duration_minutes * 60

                if enable_segment and segment_duration_minutes > 0 and total_duration > threshold_seconds:
                    # 需要分段处理
                    self._emit_status(_("status_segment_threshold", duration=total_duration, threshold=threshold_seconds))

                    segment_dir = os.path.join('project', 'cache', 'segments', os.path.basename(base_path))
                    os.makedirs(segment_dir, exist_ok=True)

                    # 切分音频
                    segment_files, _unused = self._split_audio(wav_file, segment_duration_minutes, segment_dir)

                    if not segment_files:
                        self._emit_status(_("status_segment_fail"))
                        if os.path.exists(wav_file):
                            os.remove(wav_file)
                        break

                    # 对每个片段进行听写和翻译
                    segment_tfs = []  # 存储每个分段的 TranscribedFile
                    for i, segment_file in enumerate(segment_files):
                        if self._stop_event.is_set():
                            break
                        self._emit_status(_("status_segment_processing", idx=i+1, total=len(segment_files)))

                        segment_base = segment_file[:-4] # 去掉 .wav
                        segment_name = os.path.basename(segment_base)

                        segment_json = os.path.join(transcribed_dir, segment_name + '.json')
                        self._process_single_audio(
                            segment_file,
                            asr_backend,
                            asr_model_file,
                            asr_aligner_file,
                            language,
                            param_crispasr,
                            segment_json,
                            start_named_proc,
                            stop_named_proc,
                        )

                        if output_format in ('原文LRC', '双语LRC'):
                            lrc_suffix = '.orig.lrc' if output_format == '双语LRC' else '.lrc'
                            make_lrc(segment_json, segment_base + lrc_suffix)

                        # 立即提交该分段进行翻译
                        if need_translate:
                            self._emit_status(_("status_segment_submit_translate", idx=i+1, total=len(segment_files)))
                            segment_tf = TranscribedFile(
                                base_path=segment_base,
                                json_src=segment_json,
                                output_dir=segment_dir,  # 临时输出到分段目录
                                output_format=output_format,
                                orig_srt_path='',
                            )
                            self._translation_pool.submit(segment_tf)
                            segment_tfs.append(segment_tf)

                    # 等待所有分段翻译完成
                    if need_translate and segment_tfs:
                        self._emit_status(_("status_wait_segments"))
                        self._translation_pool.done()
                        self._translation_pool.wait_all(timeout=600)

                    # 合并所有片段的翻译结果
                    self._emit_status(_("status_merge_segments"))
                    self._merge_segment_translations(segment_files, segment_tfs, base_path, json_path, current_output_dir, output_format, threshold_seconds)

                    self._emit_status(_("status_segment_done"))

                    # 分段处理已完成，跳过常规流程
                    tf = None
                else:
                    # 正常流程（未启用分段）
                    self._emit_status(_("status_asr_in_progress"))
                    self._process_single_audio(
                        wav_file,
                        asr_backend,
                        asr_model_file,
                        asr_aligner_file,
                        language,
                        param_crispasr,
                        json_path,
                        start_named_proc,
                        stop_named_proc,
                    )

                    # 生成原文 SRT/LRC 输出
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

                    tf = TranscribedFile(
                        base_path=base_path,
                        json_src=json_path,
                        output_dir=current_output_dir,
                        output_format=output_format,
                        orig_srt_path='',
                    )

            if tf is not None and need_translate:
                self._translation_pool.submit(tf)

        # 发送哨兵，等待翻译线程结束
        self._emit_status(_("status_all_transcribed"))
        if self._translation_pool:
            self._translation_pool.done()
            self._translation_pool.wait_all(timeout=600)
            self._translation_pool.stop()

        err_count = self._translation_pool.error_count if self._translation_pool else 0
        if err_count > 0:
            self._emit_status(_("status_translate_fail_count", count=err_count))

        # 完成屏障：先排空消息队列，再放入完成哨兵
        # 确保所有翻译日志在"所有文件处理完成"之前被 GUI 消费
        self.msg_queue.drain_all(timeout=3.0)
        self.msg_queue.put_completion_sentinel()
        self.msg_queue.set_completion_flag()
        self.finished.emit()
