import os
import re
import json
import queue
import shlex
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep, time

import requests
import yaml

from core import _TRANSLATE_CMD
from i18n import _
from log import (
    UIMessageQueue,
    _TRANSLATION_LINE_RE,
    _TranslationLogParser,
    _clean_control_chars,
    _stream_proc_to_queue,
    _strip_ansi,
)
from prompt2srt import make_lrc, make_srt, merge_lrc_files
from srt2prompt import merge_srt_files
from tasking import CancellationToken, ProcessRegistry, TaskCancelledError
from output_metrics import decode_output_event


def _split_command_template(value: str) -> list[str]:
    """Split a subprocess template without invoking a shell."""
    if os.name != 'nt':
        return shlex.split(value)
    tokens = shlex.split(value, posix=False)
    return [
        token[1:-1]
        if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'"
        else token
        for token in tokens
    ]


def _set_command_option(command, option_names, preferred_option, value):
    """Set a split command option, appending it when absent."""
    for index, token in enumerate(command):
        if token in option_names:
            if index + 1 < len(command):
                command[index + 1] = value
            else:
                command.append(value)
            return
        for option_name in option_names:
            if token.startswith(option_name + '='):
                command[index] = f'{preferred_option}={value}'
                return
    command.extend([preferred_option, value])


def build_llama_server_command(model_file, gpu_layers, param_llama, port):
    """Build a validated llama-server command for normal runs and self-tests."""
    llama_dir = Path('llama').resolve()
    model_path = Path(model_file)
    if not model_path.is_absolute():
        model_path = llama_dir / model_path
    model_path = model_path.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f'Offline translation model not found: {model_path}')

    command = _split_command_template(param_llama)
    if not command:
        raise ValueError('llama/param.txt is empty')

    replacements = {
        '$num_layers': str(gpu_layers or '0'),
        '$port': str(port),
    }
    model_value = str(model_path)
    for index, token in enumerate(command):
        token = token.replace('llama/$model_file', model_value)
        token = token.replace(r'llama\$model_file', model_value)
        token = token.replace('$model_file', model_value)
        for placeholder, replacement in replacements.items():
            token = token.replace(placeholder, replacement)
        command[index] = token

    unresolved = [token for token in command if re.search(r'\$[A-Za-z_]+', token)]
    if unresolved:
        raise ValueError(f'Unresolved llama-server placeholders: {unresolved}')

    executable = Path(command[0])
    if not executable.is_absolute():
        executable = Path.cwd() / executable
    if os.name == 'nt' and not executable.is_file() and not executable.suffix:
        executable = Path(str(executable) + '.exe')
    if executable.is_file():
        command[0] = str(executable.resolve())
    elif shutil.which(command[0]) is None:
        raise FileNotFoundError(f'llama-server executable not found: {command[0]}')

    _set_command_option(command, ('--model', '-m'), '--model', model_value)
    _set_command_option(command, ('--port',), '--port', str(port))
    if str(gpu_layers).strip():
        _set_command_option(
            command, ('--n-gpu-layers', '-ngl'), '--n-gpu-layers',
            str(gpu_layers).strip(),
        )
    return command


@dataclass
class TranscribedFile:
    """已听写完成的文件上下文，传递给翻译线程"""
    base_path: str       # 文件基本路径（无扩展名），如 /path/to/file
    json_src: str        # 听写产出的 JSON 路径（在 cache/transcribed/ 下）
    output_dir: str      # 该文件的输出目录
    output_format: str   # 输出格式（如 '目标SRT', '双语SRT'）
    orig_srt_path: str   # 原始 SRT 路径（用于双语合并，空串表示无）
    source_file_id: str = ""  # 非空时，成功生成最终输出后回报源文件完成
    completes_source_file: bool = True


class ConcurrentTranslationPool:
    """并发翻译线程池：每文件一个工作线程，工作空间隔离"""

    verbose_galtransl: bool = False  # 类变量：详细模式开关，由 MainWindow 在启动翻译前设置

    @staticmethod
    def _translate_worker_thread(task_queue, result_queue, msg_queue, stop_event,
                                 project_dir, base_config_path, engine, worker_idx,
                                 process_registry, completion_callback=None,
                                 item_progress_callback=None):
        """工作线程函数：从队列取任务并执行翻译"""
        while not stop_event.is_set():
            try:
                tf_dict = task_queue.get(timeout=1)
            except queue.Empty:
                continue

            if tf_dict is None:  # 哨兵信号
                result_queue.put(('done', worker_idx))
                break

            if stop_event.is_set():
                result_queue.put(('stopped', worker_idx))
                if completion_callback:
                    completion_callback(tf_dict, 'stopped')
                continue

            # 执行翻译
            try:
                ConcurrentTranslationPool._translate_one_impl(
                    tf_dict, worker_idx, project_dir, base_config_path, engine,
                    msg_queue, stop_event, process_registry,
                    item_progress_callback)
                result_queue.put(('success', worker_idx))
                if completion_callback:
                    completion_callback(tf_dict, 'success')
            except TaskCancelledError:
                result_queue.put(('stopped', worker_idx))
                if completion_callback:
                    completion_callback(tf_dict, 'stopped')
                break
            except Exception as e:
                result_queue.put(('error', worker_idx, str(e)))
                if completion_callback:
                    completion_callback(tf_dict, 'error')

    @staticmethod
    def _translate_one_impl(tf_dict, worker_idx, project_dir, base_config_path,
                            engine, msg_queue, stop_event, process_registry,
                            item_progress_callback=None):
        """在线程中执行单个文件的翻译"""
        base_path = tf_dict['base_path']
        json_src = tf_dict['json_src']
        output_dir = tf_dict['output_dir']
        output_format = tf_dict['output_format']
        orig_srt_path = tf_dict['orig_srt_path']

        source_total = 1
        try:
            with open(json_src, 'r', encoding='utf-8') as stream:
                source_rows = json.load(stream)
            source_total = max(1, sum(
                1 for row in source_rows
                if isinstance(row, dict) and str(row.get('message', '') or '').strip()
            ))
        except Exception:
            pass
        proofread_enabled = False
        try:
            with open(base_config_path, 'r', encoding='utf-8') as stream:
                common = (yaml.safe_load(stream) or {}).get('common', {})
            proofread_enabled = bool(common.get('gpt.enableProofRead', False))
        except Exception:
            pass
        progress_total = source_total * (2 if proofread_enabled else 1)
        if item_progress_callback:
            item_progress_callback(tf_dict, 0, progress_total)

        base = os.path.basename(base_path)

        def send_status(msg):
            """向统一消息队列发送后端详细日志"""
            msg_queue.put("detail", msg)

        send_status(_("status_translating_start", idx=worker_idx, base=base))

        # 创建工作空间
        workspace = ConcurrentTranslationPool._create_workspace_impl(project_dir, worker_idx)
        json_name = os.path.basename(json_src)

        # 将听写产出的 JSON 复制到工作空间的 gt_input
        shutil.copy(json_src, os.path.join(workspace, 'gt_input', json_name))

        # 准备独立配置文件
        ConcurrentTranslationPool._prepare_config_impl(workspace, base_config_path, project_dir)

        try:
            send_status(_("status_translating_with", idx=worker_idx, engine=engine, workspace=workspace))
            # 通过环境变量控制 GalTransl 是否跳过 _ServerStatusFilter
            # 仅在详细模式时传递 env，非详细模式让子进程直接继承父进程环境
            proc_env = None
            if ConcurrentTranslationPool.verbose_galtransl:
                proc_env = os.environ.copy()
                proc_env['GALTRANSL_VERBOSE_STDOUT'] = '1'

            proc = process_registry.popen(
                [*_TRANSLATE_CMD, workspace, engine],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', bufsize=1,
                env=proc_env,
            )

            # 翻译日志解析器：将 GalTransl 三行格式转换为 JSON
            _trans_parser = _TranslationLogParser()
            line_queue: queue.Queue = queue.Queue()

            def read_stdout():
                try:
                    for line in iter(proc.stdout.readline, ''):
                        line_queue.put(line)
                finally:
                    line_queue.put(None)

            reader = threading.Thread(target=read_stdout, daemon=True)
            reader.start()
            reader_done = False
            progress_entry_id = None
            progress_occurrences: dict[int, int] = {}
            while not reader_done or not line_queue.empty():
                if stop_event.is_set():
                    process_registry.terminate(proc)
                    raise TaskCancelledError()
                try:
                    line = line_queue.get(timeout=0.1)
                except queue.Empty:
                    if proc.poll() is not None and not reader.is_alive():
                        reader_done = True
                    continue
                if line is None:
                    reader_done = True
                    continue
                cleaned = _clean_control_chars(_strip_ansi(line.rstrip('\n\r')))
                header_match = _TRANSLATION_LINE_RE.match(cleaned)
                if header_match:
                    progress_entry_id = int(header_match.group(1))
                elif cleaned.startswith('> Dst: ') and progress_entry_id is not None:
                    seen = progress_occurrences.get(progress_entry_id, 0)
                    max_occurrences = 2 if proofread_enabled else 1
                    if seen < max_occurrences:
                        progress_occurrences[progress_entry_id] = seen + 1
                        if item_progress_callback:
                            completed_units = min(
                                progress_total,
                                sum(progress_occurrences.values()),
                            )
                            # Keep 100% for the point where final subtitle files
                            # have actually been written below.
                            item_progress_callback(
                                tf_dict,
                                min(completed_units, max(0, progress_total - 1)),
                                progress_total,
                            )
                    progress_entry_id = None
                output_event = decode_output_event(cleaned)
                if output_event is not None:
                    msg_queue.put(
                        "characters",
                        json.dumps(output_event, ensure_ascii=False),
                    )
                    continue
                if cleaned:
                    for output_line in _trans_parser.feed(cleaned):
                        if output_line.strip():
                            send_status(output_line)

            # 刷新解析器缓冲区中残留的行
            for output_line in _trans_parser.flush():
                if output_line.strip():
                    send_status(output_line)

            if proc.stdout:
                proc.stdout.close()
            retcode = process_registry.wait(proc)

            # 短暂等待，确保状态队列中的日志已排空发送到 GUI
            send_status(_("status_translate_proc_ended", idx=worker_idx, retcode=retcode))
            sleep(0.1)
            if retcode != 0:
                raise subprocess.CalledProcessError(retcode, _TRANSLATE_CMD)
        except Exception as e:
            send_status(_("status_translating_error", idx=worker_idx, base=base, error=e))
            raise

        # 生成翻译后字幕
        send_status(_("status_translating_srt", idx=worker_idx, base=base))
        ConcurrentTranslationPool._generate_output_impl(
            json_src, base_path, output_dir, output_format, workspace, orig_srt_path)

        if item_progress_callback:
            item_progress_callback(tf_dict, progress_total, progress_total)

        send_status(_("status_translating_done", idx=worker_idx, base=base))

    @staticmethod
    def _create_workspace_impl(project_dir, worker_idx):
        """在线程中创建工作空间"""
        import time
        idx = int(time.time() * 1000000) + worker_idx
        workspace = os.path.join(project_dir, 'cache', f'translate_{idx}')
        for sub in ('gt_input', 'gt_output', 'transl_cache'):
            os.makedirs(os.path.join(workspace, sub), exist_ok=True)
        return workspace

    @staticmethod
    def _prepare_config_impl(workspace, base_config_path, project_dir):
        """在线程中准备配置文件"""
        with open(base_config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        abs_project_dir = os.path.abspath(project_dir).replace('\\', '/')
        content = content.replace('(project_dir)', abs_project_dir + '/')

        # 强制启用 saveLog：使 GalTransl 将完整日志写入 workspace/GalTransl.log
        content = re.sub(
            r'^(\s*)saveLog:\s*(?:true|false)\s*$',
            r'\1saveLog: true',
            content,
            flags=re.MULTILINE,
        )

        config_path = os.path.join(workspace, 'config.yaml')
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return config_path

    @staticmethod
    def _generate_output_impl(json_src, base_path, output_dir, output_format, workspace, orig_srt_path=''):
        """在线程中生成输出文件"""
        json_name = os.path.basename(json_src)
        gt_output_json = os.path.join(workspace, 'gt_output', json_name)
        base_name = os.path.basename(base_path)

        if output_format in ('目标SRT', '双语SRT'):
            zh_srt_output = os.path.join(output_dir, base_name + '.tg.srt')
            make_srt(gt_output_json, zh_srt_output)

        if output_format in ('目标LRC', '双语LRC'):
            lrc_suffix = '.zh.lrc' if output_format == '双语LRC' else '.lrc'
            lrc_output = os.path.join(output_dir, base_name + lrc_suffix)
            make_lrc(gt_output_json, lrc_output)

        if output_format == '双语SRT':
            left = os.path.join(output_dir, base_name + '.srt')
            right = os.path.join(output_dir, base_name + '.tg.srt')
            if os.path.exists(left) and os.path.exists(right):
                merge_srt_files([left, right],
                                os.path.join(output_dir, base_name + '.combine.srt'))

        if output_format == '双语LRC':
            left = os.path.join(output_dir, base_name + '.orig.lrc')
            right = os.path.join(output_dir, base_name + '.zh.lrc')
            if os.path.exists(left) and os.path.exists(right):
                merge_lrc_files([left, right],
                                os.path.join(output_dir, base_name + '.combine.lrc'))

        if output_format not in ('双语SRT', '原文SRT'):
            left = os.path.join(output_dir, base_name + '.srt')
            if os.path.exists(left):
                os.remove(left)

    def __init__(self, project_dir, base_config_path, max_concurrent, stop_event,
                 msg_queue, local_model_config=None, progress_callback=None,
                 file_completion_callback=None, item_progress_callback=None):
        """
        msg_queue: 统一消息队列（UIMessageQueue 实例）
        local_model_config: 本地模型配置，用于多线程本地模型翻译
            {
                'sakura_file': str,      # 模型文件路径
                'sakura_mode': str,      # GPU层数
                'param_llama': str,      # llama.cpp 参数
            }
        """
        self._project_dir = project_dir
        self._base_config_path = base_config_path
        self._max_concurrent = max_concurrent
        self._stop_event = stop_event
        self._msg_queue = msg_queue
        self._local_model_config = local_model_config
        self._task_queue = queue.Queue()
        self._result_queue = queue.Queue()
        self._active_threads: list[threading.Thread] = []
        self._error_count = 0
        self._error_lock = threading.Lock()
        self._submitted_count = 0
        self._completed_count = 0
        self._progress_completed_count = 0
        self._progress_lock = threading.Lock()
        self._progress_callback = progress_callback
        self._file_completion_callback = file_completion_callback
        self._item_progress_callback = item_progress_callback
        # 本地模型相关（所有进程共享一个本地模型）
        self._shared_local_model_proc = None
        self._shared_local_model_port = None
        self._local_model_lock = threading.Lock()
        # 串行模式相关
        self._serial_mode = max_concurrent <= 0
        self._serial_lock = threading.Lock()
        self._engine = None
        self._process_registry = ProcessRegistry(CancellationToken(stop_event))

    @property
    def error_count(self):
        with self._error_lock:
            return self._error_count

    @property
    def submitted_count(self):
        with self._progress_lock:
            return self._submitted_count

    @property
    def progress_completed_count(self):
        with self._progress_lock:
            return self._progress_completed_count

    @property
    def serial_mode(self):
        return self._serial_mode

    def _notify_progress_completion(self, tf_dict=None, outcome='success'):
        with self._progress_lock:
            self._progress_completed_count += 1
            current = self._progress_completed_count
            total = max(current, self._submitted_count)
        if self._progress_callback:
            self._progress_callback(current, total)
        source_file_id = str((tf_dict or {}).get('source_file_id', '') or '')
        if source_file_id and self._file_completion_callback:
            self._file_completion_callback(
                source_file_id,
                outcome,
                bool((tf_dict or {}).get('completes_source_file', True)),
            )

    def start(self, engine):
        """启动 N 个工作线程"""
        self._engine = engine

        # 串行模式：不启动工作进程
        if self._serial_mode:
            return

        # 如果配置了本地模型，启动一个共享的本地模型实例
        if self._local_model_config and self._local_model_config.get('sakura_file'):
            proc, port = self._start_local_model(0)
            if proc:
                with self._local_model_lock:
                    self._shared_local_model_proc = proc
                    self._shared_local_model_port = port
            else:
                self._msg_queue.put("status", _("status_local_model_start_fail"))

        # 创建线程事件
        self._thread_stop_event = threading.Event()

        # 并发模式：启动多个工作线程
        for i in range(self._max_concurrent):
            t = threading.Thread(
                target=ConcurrentTranslationPool._translate_worker_thread,
                args=(self._task_queue, self._result_queue, self._msg_queue,
                      self._stop_event, self._project_dir, self._base_config_path,
                      engine, i, self._process_registry,
                      self._notify_progress_completion,
                      self._item_progress_callback),
                daemon=True
            )
            self._active_threads.append(t)
            t.start()

    def submit(self, tf):
        """提交翻译任务"""
        if self._serial_mode:
            # 串行模式
            with self._serial_lock:
                if self._stop_event.is_set():
                    return

                # 启动共享本地模型
                if self._local_model_config and self._local_model_config.get('sakura_file'):
                    with self._local_model_lock:
                        if not self._shared_local_model_proc:
                            proc, port = self._start_local_model(0)
                            if proc:
                                self._shared_local_model_proc = proc
                                self._shared_local_model_port = port
                            else:
                                self._msg_queue.put("status", _("status_local_model_start_fail"))

                # 执行翻译（在调用线程中同步执行）
                tf_dict = {
                    'base_path': tf.base_path,
                    'json_src': tf.json_src,
                    'output_dir': tf.output_dir,
                    'output_format': tf.output_format,
                    'orig_srt_path': tf.orig_srt_path,
                    'source_file_id': tf.source_file_id,
                    'completes_source_file': tf.completes_source_file,
                }
                outcome = 'success'
                try:
                    ConcurrentTranslationPool._translate_one_impl(
                        tf_dict, 0, self._project_dir, self._base_config_path,
                        self._engine, self._msg_queue, self._stop_event,
                        self._process_registry, self._item_progress_callback)
                except Exception as e:
                    outcome = 'error'
                    with self._error_lock:
                        self._error_count += 1
                    self._msg_queue.put("status", _("status_translation_fail", error=e))

                with self._progress_lock:
                    self._submitted_count += 1
                self._completed_count += 1
                self._notify_progress_completion(tf_dict, outcome)

                # 停止共享本地模型
                self._stop_shared_local_model()
        else:
            # 并发模式：放入队列
            tf_dict = {
                'base_path': tf.base_path,
                'json_src': tf.json_src,
                'output_dir': tf.output_dir,
                'output_format': tf.output_format,
                'orig_srt_path': tf.orig_srt_path,
                'source_file_id': tf.source_file_id,
                'completes_source_file': tf.completes_source_file,
            }
            with self._progress_lock:
                self._submitted_count += 1
            self._task_queue.put(tf_dict)

    def _record_result(self, result):
        """Record one worker result without counting shutdown sentinels as tasks."""
        if not result or result[0] == 'done':
            return
        if result[0] in ('success', 'error', 'stopped'):
            self._completed_count += 1
        if result[0] == 'error':
            with self._error_lock:
                self._error_count += 1

    def wait_for_pending(self):
        """Wait for tasks submitted so far while keeping the pool reusable."""
        if self._serial_mode:
            return

        target_count = self._submitted_count
        while self._completed_count < target_count:
            if self._stop_event.is_set():
                raise TaskCancelledError()
            try:
                result = self._result_queue.get(timeout=0.2)
            except queue.Empty:
                if self._active_threads and not any(
                    thread.is_alive() for thread in self._active_threads
                ):
                    raise RuntimeError("translation workers exited before completing queued tasks")
                continue
            self._record_result(result)

    def done(self):
        """所有任务已提交，发送哨兵信号"""
        if self._serial_mode:
            return
        for _unused in range(self._max_concurrent):
            self._task_queue.put(None)

    def wait_all(self):
        """等待所有工作线程结束，除非用户主动取消。"""
        if self._serial_mode:
            return

        # 翻译大文件可能持续数小时，不能使用固定总超时。旧实现会把
        # 600 秒平均分配给所有线程；例如配置 16 个并发时，正在工作的
        # 线程只会被等待 37.5 秒，随后上层就会误报任务完成并停止线程池。
        # 使用短轮询既能一直等待真实任务完成，也能及时响应用户取消。
        while True:
            alive_threads = [t for t in self._active_threads if t.is_alive()]
            if not alive_threads or self._stop_event.is_set():
                break
            for t in alive_threads:
                t.join(timeout=0.2)

        # 处理结果队列中的错误
        while True:
            try:
                result = self._result_queue.get_nowait()
                self._record_result(result)
            except queue.Empty:
                break


    def stop(self):
        """停止所有工作线程和子进程"""
        # 设置停止事件
        self._stop_event.set()
        if hasattr(self, '_thread_stop_event'):
            self._thread_stop_event.set()

        self._process_registry.terminate_all()

        # 清空任务队列（丢弃未处理的任务）
        while True:
            try:
                self._task_queue.get_nowait()
            except queue.Empty:
                break

        # 等待所有工作线程结束
        deadline = monotonic() + 3.0
        for t in self._active_threads:
            t.join(timeout=max(0.0, deadline - monotonic()))
            if t.is_alive():
                self._msg_queue.put("status",
                    _("status_worker_not_exited", name=t.name))

        # 停止共享的本地模型进程
        self._stop_shared_local_model()


    def _stop_shared_local_model(self):
        """停止共享的本地模型"""
        with self._local_model_lock:
            proc = self._shared_local_model_proc
            self._shared_local_model_proc = None
            self._shared_local_model_port = None
        if proc:
            if proc.poll() is None:
                self._msg_queue.put("status", _("status_local_model_stopping"))
                self._process_registry.terminate(proc)

    def _start_local_model(self, worker_idx):
        """启动共享的本地模型服务"""
        if not self._local_model_config:
            return None, None

        cfg = self._local_model_config
        sakura_file = cfg.get('sakura_file', '')
        sakura_mode = cfg.get('sakura_mode', '100')
        param_llama = cfg.get('param_llama', '')

        if not sakura_file:
            return None, None

        port = 8989

        args = build_llama_server_command(
            sakura_file, sakura_mode, param_llama, port
        )

        self._msg_queue.put("status", _("status_local_model_starting", port=port))

        try:
            expected_model = str(Path(sakura_file).name)
            proc = self._process_registry.popen(
                args,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            threading.Thread(
                target=_stream_proc_to_queue,
                args=(proc, self._msg_queue, expected_model),
                daemon=True,
            ).start()

            start_wait = time()

            while not self._stop_event.is_set():
                try:
                    chat_resp = requests.post(
                        f"http://localhost:{port}/v1/chat/completions",
                        json={
                            "model": expected_model,
                            "messages": [{"role": "user", "content": "ping"}],
                            "max_tokens": 1,
                            "temperature": 0
                        },
                        timeout=1
                    )
                    if chat_resp.status_code == 200:
                        try:
                            body = chat_resp.json()
                            if isinstance(body, dict) and body.get("choices"):
                                self._msg_queue.put("status",
                                    _("status_local_model_ready", port=port))
                                break
                        except Exception:
                            pass
                except requests.exceptions.RequestException:
                    pass

                if time() - start_wait > 120:
                    self._msg_queue.put("status",
                        _("status_local_model_timeout"))
                    self._process_registry.terminate(proc)
                    return None, None
                self._stop_event.wait(0.5)

            return proc, port
        except Exception as e:
            self._msg_queue.put("status",
                _("status_local_model_start_error", error=e))
            return None, None
