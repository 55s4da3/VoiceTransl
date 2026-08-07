import os
import re
import queue
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from time import sleep, time

import requests

from core import _TRANSLATE_CMD
from i18n import _
from log import (
    UIMessageQueue,
    _TranslationLogParser,
    _clean_control_chars,
    _stream_proc_to_queue,
    _strip_ansi,
)
from prompt2srt import make_lrc, make_srt, merge_lrc_files
from srt2prompt import merge_srt_files


@dataclass
class TranscribedFile:
    """已听写完成的文件上下文，传递给翻译线程"""
    base_path: str       # 文件基本路径（无扩展名），如 /path/to/file
    json_src: str        # 听写产出的 JSON 路径（在 cache/transcribed/ 下）
    output_dir: str      # 该文件的输出目录
    output_format: str   # 输出格式（如 '目标SRT', '双语SRT'）
    orig_srt_path: str   # 原始 SRT 路径（用于双语合并，空串表示无）


class ConcurrentTranslationPool:
    """并发翻译线程池：每文件一个工作线程，工作空间隔离"""

    verbose_galtransl: bool = False  # 类变量：详细模式开关，由 MainWindow 在启动翻译前设置

    @staticmethod
    def _translate_worker_thread(task_queue, result_queue, msg_queue, stop_event,
                                 project_dir, base_config_path, engine, worker_idx):
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
                continue

            # 执行翻译
            try:
                ConcurrentTranslationPool._translate_one_impl(
                    tf_dict, worker_idx, project_dir, base_config_path, engine, msg_queue)
                result_queue.put(('success', worker_idx))
            except Exception as e:
                result_queue.put(('error', worker_idx, str(e)))

    @staticmethod
    def _translate_one_impl(tf_dict, worker_idx, project_dir, base_config_path,
                            engine, msg_queue):
        """在线程中执行单个文件的翻译"""
        base_path = tf_dict['base_path']
        json_src = tf_dict['json_src']
        output_dir = tf_dict['output_dir']
        output_format = tf_dict['output_format']
        orig_srt_path = tf_dict['orig_srt_path']

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
            creationflags = 0x08000000 if os.name == 'nt' else 0

            # 通过环境变量控制 GalTransl 是否跳过 _ServerStatusFilter
            # 仅在详细模式时传递 env，非详细模式让子进程直接继承父进程环境
            proc_env = None
            if ConcurrentTranslationPool.verbose_galtransl:
                proc_env = os.environ.copy()
                proc_env['GALTRANSL_VERBOSE_STDOUT'] = '1'

            proc = subprocess.Popen(
                [*_TRANSLATE_CMD, workspace, engine],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, creationflags=creationflags, bufsize=1,
                env=proc_env,
            )

            # 翻译日志解析器：将 GalTransl 三行格式转换为 JSON
            _trans_parser = _TranslationLogParser()

            for line in iter(proc.stdout.readline, ''):
                # 清除 ANSI 转义序列和控制字符
                cleaned = _clean_control_chars(_strip_ansi(line.rstrip('\n\r')))
                if not cleaned:
                    continue
                # 通过解析器转换翻译输出格式（JSON 化），逐行写入日志和发送到 GUI
                for output_line in _trans_parser.feed(cleaned):
                    if output_line.strip():
                        send_status(output_line)

            # 刷新解析器缓冲区中残留的行
            for output_line in _trans_parser.flush():
                if output_line.strip():
                    send_status(output_line)

            proc.stdout.close()
            retcode = proc.wait()

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
                 msg_queue, local_model_config=None):
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
        # 本地模型相关（所有进程共享一个本地模型）
        self._shared_local_model_proc = None
        self._shared_local_model_port = None
        self._local_model_lock = threading.Lock()
        # 串行模式相关
        self._serial_mode = max_concurrent <= 0
        self._serial_lock = threading.Lock()
        self._engine = None
        # 跟踪 GalTransl 子进程用于取消时终止
        self._active_translate_procs: list[subprocess.Popen] = []
        self._procs_lock = threading.Lock()

    @property
    def error_count(self):
        with self._error_lock:
            return self._error_count

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
                      self._thread_stop_event, self._project_dir, self._base_config_path,
                      engine, i),
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
                }
                try:
                    ConcurrentTranslationPool._translate_one_impl(
                        tf_dict, 0, self._project_dir, self._base_config_path,
                        self._engine, self._msg_queue)
                except Exception as e:
                    with self._error_lock:
                        self._error_count += 1
                    self._msg_queue.put("status", _("status_translation_fail", error=e))

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
            }
            self._task_queue.put(tf_dict)

    def done(self):
        """所有任务已提交，发送哨兵信号"""
        if self._serial_mode:
            return
        for _unused in range(self._max_concurrent):
            self._task_queue.put(None)

    def wait_all(self, timeout=600):
        """等待所有工作线程结束"""
        if self._serial_mode:
            return

        # 等待所有线程结束
        for t in self._active_threads:
            t.join(timeout=timeout / len(self._active_threads) if self._active_threads else timeout)

        # 处理结果队列中的错误
        while True:
            try:
                result = self._result_queue.get_nowait()
                if result[0] == 'error':
                    with self._error_lock:
                        self._error_count += 1
            except queue.Empty:
                break

        # 排空统一消息队列中可能残留的后端日志
        self._msg_queue.drain_all(timeout=2.0)

    def stop(self):
        """停止所有工作线程和子进程"""
        # 设置停止事件
        self._stop_event.set()
        if hasattr(self, '_thread_stop_event'):
            self._thread_stop_event.set()

        # 终止所有在途的 GalTransl 翻译子进程
        with self._procs_lock:
            for proc in self._active_translate_procs:
                try:
                    if proc.poll() is None:
                        proc.terminate()
                except Exception:
                    pass
            # 等待子进程终止
            for proc in self._active_translate_procs:
                try:
                    proc.wait(timeout=3)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            self._active_translate_procs.clear()

        # 清空任务队列（丢弃未处理的任务）
        while True:
            try:
                self._task_queue.get_nowait()
            except queue.Empty:
                break

        # 等待所有工作线程结束
        for t in self._active_threads:
            t.join(timeout=3)
            if t.is_alive():
                self._msg_queue.put("status",
                    _("status_worker_not_exited", name=t.name))

        # 停止共享的本地模型进程
        self._stop_shared_local_model()

        # 排空消息队列中所有残留
        self._msg_queue.drain_all(timeout=2.0)

    def _stop_shared_local_model(self):
        """停止共享的本地模型"""
        with self._local_model_lock:
            proc = self._shared_local_model_proc
            self._shared_local_model_proc = None
            self._shared_local_model_port = None
        if proc:
            try:
                if proc.poll() is None:
                    self._msg_queue.put("status", _("status_local_model_stopping"))
                    proc.terminate()
                    proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

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

        args = [param.replace('$model_file', sakura_file).replace('$num_layers', sakura_mode).replace('$port', str(port))
                for param in param_llama.split()]

        self._msg_queue.put("status", _("status_local_model_starting", port=port))

        try:
            creationflags = 0x08000000 if os.name == 'nt' else 0
            expected_model = str(Path(sakura_file).name)
            proc = subprocess.Popen(
                args,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                creationflags=creationflags,
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
                        timeout=8
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
                    try:
                        proc.terminate()
                        proc.wait(timeout=3)
                    except Exception:
                        pass
                    return None, None
                sleep(1)

            return proc, port
        except Exception as e:
            self._msg_queue.put("status",
                _("status_local_model_start_error", error=e))
            return None, None
