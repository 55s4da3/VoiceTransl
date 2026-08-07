import re
import json
import queue
import threading
from time import sleep, time

from core import LOG_PATH


# ANSI 转义序列正则（覆盖 CSI、OSC、前景/背景色等）
_ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


def _strip_ansi(text: str) -> str:
    """移除 ANSI 转义序列"""
    return _ANSI_ESCAPE.sub('', text)


# 统一消息队列，所有消息都通过此队列流转
# 消息格式：(target: str, text: str)
#   target="status" → 上层"实时输出信息"框
#   target="detail" → 下层"日志文件"框
#   target="__COMPLETION__" → 完成哨兵，两个框都追加完成提示

class UIMessageQueue:
    """统一消息队列：线程安全，替代分散的日志写入和信号发射"""

    _COMPLETION_TARGET = "__COMPLETION__"
    _MAX_SIZE = 10000
    _DRAIN_EMPTY_LIMIT = 3       # drain_all 连续空返回次数上限
    _DRAIN_POLL_INTERVAL = 0.1   # drain_all 轮询间隔（秒）
    _DRAIN_MAX_WAIT = 3.0        # drain_all 最长等待（秒）

    def __init__(self, log_path: str = LOG_PATH):
        self._queue: queue.Queue = queue.Queue(maxsize=self._MAX_SIZE)
        self._log_path = log_path
        self._file_lock = threading.Lock()
        self._completion_flag = threading.Event()

    def put(self, target: str, text: str) -> None:
        """线程安全地放入一条消息。target 为 'status' 或 'detail'。

        满时丢弃最旧消息（FIFO），避免 OOM。
        同时写入日志文件（自动剥离 ANSI 码）。
        """
        # 写入日志文件
        cleaned = _strip_ansi(text)
        if cleaned.strip():
            with self._file_lock:
                try:
                    with open(self._log_path, 'a', encoding='utf-8', errors='replace') as f:
                        f.write(cleaned + '\n')
                except Exception:
                    pass

        # 放入队列
        entry = (target, text)
        try:
            self._queue.put(entry, block=False)
        except queue.Full:
            # 丢弃最旧消息，放入新消息
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put(entry, block=False)
            except queue.Full:
                pass  # 极端情况：忽略

    def drain(self) -> list[tuple[str, str]]:
        """非阻塞地取出当前队列中所有消息。"""
        entries: list[tuple[str, str]] = []
        while True:
            try:
                entries.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return entries

    def drain_all(self, timeout: float | None = None) -> list[tuple[str, str]]:
        """阻塞式排空队列，直到连续 N 次空返回或超时。

        用于确保所有工作线程的消息已被取出。
        """
        if timeout is None:
            timeout = self._DRAIN_MAX_WAIT

        all_entries: list[tuple[str, str]] = []
        empty_streak = 0
        start = time()
        while empty_streak < self._DRAIN_EMPTY_LIMIT:
            batch = self.drain()
            if batch:
                all_entries.extend(batch)
                empty_streak = 0
            else:
                empty_streak += 1
                sleep(self._DRAIN_POLL_INTERVAL)
            if time() - start > timeout:
                break
        return all_entries

    def set_completion_flag(self) -> None:
        """标记翻译池已完成（原子操作）。"""
        self._completion_flag.set()

    def is_completion_ready(self) -> bool:
        """检查翻译池是否已完成。"""
        return self._completion_flag.is_set()

    def put_completion_sentinel(self) -> None:
        """放入完成哨兵消息（在 drain_all 之后调用）。"""
        self._queue.put((self._COMPLETION_TARGET, ''), block=False)

    @staticmethod
    def is_completion_entry(target: str) -> bool:
        """判断是否为完成哨兵。"""
        return target == UIMessageQueue._COMPLETION_TARGET


# 日志级别过滤辅助函数（模块级，供 read_log_file 调用）
def _line_passes_filter(line: str, filter_level: str) -> bool:
    """判断日志行是否通过级别过滤"""
    if filter_level == 'ALL':
        return True
    m = re.search(r'\[(DEBUG|INFO|WARNING|ERROR|CRITICAL)\]', line)
    if not m:
        return True  # 无级别标记的行始终显示
    ranks = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3, 'CRITICAL': 4}
    thresholds = {'INFO+': 1, 'WARNING+': 2, 'ERROR+': 3}
    return ranks.get(m.group(1), 0) >= thresholds.get(filter_level, 0)


def _clean_control_chars(text: str) -> str:
    """清理日志行中的控制字符

    - \\r 处理：取最长段落（通常是实际内容，短段为进度条碎片）
    - 移除其他控制符（保留 \\t 和 \\n）
    - 清理进度条 \\r 残留的三字符前缀（如 yg2|、a0n|、23b|）
    """
    if '\r' in text:
        parts = text.split('\r')
        # 保留最长段落：实际日志内容远长于进度条碎片（yg2| 等仅4字符）
        text = max(parts, key=len)
    # 移除 0x00-0x08, 0x0B-0x0C, 0x0E-0x1F 范围的控制字符，保留 \\t (0x09) 和 \\n (0x0A)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    # 清理进度条碎片前缀：三个小写字母/数字 + 竖线（如 yg2|、a0n|、23b|）
    text = re.sub(r'^[a-z0-9]{3}\|', '', text)
    return text


def _decode_subprocess_line(data: bytes) -> str:
    """按 UTF-8→GBK→latin-1 顺序尝试解码子进程输出字节。"""
    for enc in ('utf-8', 'gbk', 'latin-1'):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode('utf-8', errors='replace')


def _stream_proc_to_queue(proc, msg_queue, label=None):
    """后台线程目标：将子进程 stdout/stderr 流式转发到统一消息队列（target='detail'）。

    读取二进制并解码、剥离 ANSI/控制字符后入队（同时写入日志文件）。
    进程结束后自动关闭管道。
    """
    stream = proc.stdout
    if stream is None:
        return
    prefix = f"[{label}] " if label else ""
    try:
        for raw in iter(stream.readline, b''):
            if not raw:
                break
            line = _decode_subprocess_line(raw)
            line = _clean_control_chars(_strip_ansi(line.rstrip('\n\r')))
            if line.strip():
                msg_queue.put("detail", prefix + line)
    except Exception:
        pass
    finally:
        try:
            stream.close()
        except Exception:
            pass


# GalTransl 逐行翻译输出解析器：将三行格式转换为 JSON，并按批次分组
# 原始格式:  v--{id}[-{speaker}]\n> Src: {text}\n> Dst: {text}
# 目标格式:  同一批次的所有 src 先输出，再输出所有 dst（纯 JSON，无前缀）
_TRANSLATION_LINE_RE = re.compile(r'^v--(\d+)')


class _TranslationLogParser:
    """将 GalTransl 逐行翻译记录转换为结构化 JSON，src/dst 分组输出"""

    def __init__(self):
        self._line_buf: list[str] = []      # 当前翻译条目的三行缓冲
        self._batch_src: list[str] = []     # 当前批次的 src JSON 行
        self._batch_dst: list[str] = []     # 当前批次的 dst JSON 行

    def feed(self, line: str) -> list[str]:
        """输入一行，返回转换后的零行或多行（批次边界时刷新）"""
        # 匹配翻译输出头部 v--{id}[-{speaker}]
        header_m = _TRANSLATION_LINE_RE.match(line)
        if header_m:
            # 新条目开始，刷新不完整的行缓冲
            flushed = self._flush_line_buf()
            self._line_buf = [line]
            return flushed

        if self._line_buf:
            if line.startswith('> Src: '):
                self._line_buf.append(line)
                return []
            if line.startswith('> Dst: '):
                self._line_buf.append(line)
                self._add_to_batch()
                self._line_buf = []
                return []
            # 模式中断：刷新所有缓冲区
            return self._flush_all() + [line]

        # 非翻译行：如果批次中有累积的翻译，先刷新批次
        flushed = self._flush_batch()
        if flushed:
            return flushed + [line]
        return [line]

    def flush(self) -> list[str]:
        """最终刷新所有残留缓冲"""
        return self._flush_all()

    # 内部方法

    def _add_to_batch(self):
        """将三行缓冲转换为 JSON 并添加到批次"""
        if len(self._line_buf) != 3:
            return
        header, src_line, dst_line = self._line_buf
        id_m = _TRANSLATION_LINE_RE.match(header)
        if not id_m:
            return
        trans_id = id_m.group(1)
        src_text = src_line[7:]   # 去掉 "> Src: " 前缀
        dst_text = dst_line[7:]   # 去掉 "> Dst: " 前缀
        self._batch_src.append(json.dumps(
            {"id": int(trans_id), "src": src_text}, ensure_ascii=False))
        self._batch_dst.append(json.dumps(
            {"id": int(trans_id), "dst": dst_text}, ensure_ascii=False))

    def _flush_line_buf(self) -> list[str]:
        """刷新不完整的行缓冲（模式中断时保留原始文本）"""
        result = self._line_buf
        self._line_buf = []
        return result

    def _flush_batch(self) -> list[str]:
        """刷新累积的批次：只输出 dst（翻译结果）"""
        if not self._batch_dst:
            return []
        result = list(self._batch_dst)
        self._batch_src = []
        self._batch_dst = []
        return result

    def _flush_all(self) -> list[str]:
        """刷新所有缓冲"""
        return self._flush_line_buf() + self._flush_batch()
