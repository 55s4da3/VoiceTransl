import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_ORIGINAL_STDOUT = sys.stdout
_ORIGINAL_STDERR = sys.stderr
from core import _compose_output_format
from crispasr_bridge import split_command_template
from log import (
    UIMessageQueue,
    _TranslationLogParser,
    _clean_control_chars,
    _decode_subprocess_line,
    _line_passes_filter,
    _strip_ansi,
    _stream_proc_to_queue,
)
from pool import _set_command_option, build_llama_server_command
sys.stdout = _ORIGINAL_STDOUT
sys.stderr = _ORIGINAL_STDERR


class AppFormattingTests(unittest.TestCase):
    def test_output_format_round_trip(self):
        self.assertEqual(_compose_output_format("双语", "SRT", True), "双语SRT")
        self.assertEqual(_compose_output_format("目标", "LRC", False), "原文LRC")

    def test_command_template_preserves_quoted_argument(self):
        tokens = split_command_template('tool --model "a model.gguf"')
        self.assertEqual(tokens, ["tool", "--model", "a model.gguf"])

    def test_command_option_replaces_separate_and_equals_forms(self):
        command = ["tool", "-m", "old", "--port=1"]
        _set_command_option(command, ("--model", "-m"), "--model", "new")
        _set_command_option(command, ("--port",), "--port", "2")
        _set_command_option(command, ("--backend",), "--backend", "qwen")
        self.assertEqual(command, ["tool", "-m", "new", "--port=2", "--backend", "qwen"])

    def test_llama_command_uses_validated_absolute_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model.gguf"
            executable = root / ("llama-server.exe" if os.name == "nt" else "llama-server")
            model.touch()
            executable.touch()
            command = build_llama_server_command(
                model,
                "42",
                f'"{executable}" --model old --port=1',
                9123,
            )
        self.assertEqual(Path(command[0]), executable.resolve())
        self.assertEqual(command[command.index("--model") + 1], str(model.resolve()))
        self.assertIn("--port=9123", command)
        self.assertEqual(command[command.index("--n-gpu-layers") + 1], "42")

    def test_log_text_cleaning_and_filtering(self):
        self.assertEqual(_strip_ansi("\x1b[31merror\x1b[0m"), "error")
        self.assertEqual(_clean_control_chars("abc\rthis is the longest line\x00"), "this is the longest line")
        self.assertEqual(_clean_control_chars("yg2|message"), "message")
        self.assertTrue(_line_passes_filter("plain", "ERROR+"))
        self.assertFalse(_line_passes_filter("[INFO] hello", "WARNING+"))
        self.assertTrue(_line_passes_filter("[ERROR] boom", "WARNING+"))

    def test_subprocess_decoding_falls_back_to_gbk(self):
        self.assertEqual(_decode_subprocess_line("中文".encode("gbk")), "中文")
        self.assertEqual(_decode_subprocess_line(b"ascii"), "ascii")

class MessageQueueAndLogParserTests(unittest.TestCase):
    def test_message_queue_logs_clean_text_drains_and_completes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "app.log"
            queue = UIMessageQueue(str(log_path))
            queue.put("detail", "\x1b[32mhello\x1b[0m")
            self.assertEqual(queue.drain(), [("detail", "\x1b[32mhello\x1b[0m")])
            self.assertEqual(log_path.read_text(encoding="utf-8"), "hello\n")
            self.assertFalse(queue.is_completion_ready())
            queue.set_completion_flag()
            queue.put_completion_sentinel()
            target, text = queue.drain()[0]
            self.assertTrue(queue.is_completion_ready())
            self.assertTrue(queue.is_completion_entry(target))
            self.assertEqual(text, "")

    def test_translation_log_parser_batches_destinations(self):
        parser = _TranslationLogParser()
        self.assertEqual(parser.feed("v--1-[A]"), [])
        self.assertEqual(parser.feed("> Src: 原文"), [])
        self.assertEqual(parser.feed("> Dst: 译文"), [])
        output = parser.feed("ordinary")
        self.assertEqual(json.loads(output[0]), {"id": 1, "dst": "译文"})
        self.assertEqual(output[1], "ordinary")

    def test_translation_log_parser_preserves_interrupted_record(self):
        parser = _TranslationLogParser()
        parser.feed("v--2")
        self.assertEqual(parser.feed("unexpected"), ["v--2", "unexpected"])

    def test_stream_process_forwards_clean_nonempty_lines(self):
        class Proc:
            stdout = io.BytesIO(b"\x1b[31merror\x1b[0m\n\n")

        class Queue:
            def __init__(self):
                self.items = []

            def put(self, target, text):
                self.items.append((target, text))

        queue = Queue()
        _stream_proc_to_queue(Proc(), queue, label="worker")
        self.assertEqual(queue.items, [("detail", "[worker] error")])


if __name__ == "__main__":
    unittest.main()
