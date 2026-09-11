import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from sentence_refiner import (
    RefinementError,
    SentenceRefiner,
    apply_refinement,
    build_prompt,
    resegment_request_limits,
    request_openai_compatible,
)
from tasking import CancellationToken


class SentenceRefinerTests(unittest.TestCase):
    def test_range_response_resegments_without_echoing_source_text(self):
        rows = [
            {"start": 0.0, "end": 0.5, "message": "これは"},
            {"start": 0.5, "end": 1.0, "message": "テスト"},
            {"start": 1.0, "end": 1.5, "message": "です。"},
        ]

        def grouped_request(prompt):
            self.assertIn('"start_id":起始行号', prompt)
            return '{"start_id":1,"end_id":3}\n'

        refiner = SentenceRefiner(grouped_request, CancellationToken())
        output = refiner.refine(rows)
        self.assertTrue(refiner.applied)
        self.assertTrue(refiner.changed)
        self.assertEqual([item["message"] for item in output], ["これはテストです。"])

    def test_punctuation_only_ai_group_is_preserved_when_ai_keeps_it_separate(self):
        rows = [
            {"start": 0.0, "end": 1.0, "message": "本当"},
            {"start": 1.0, "end": 1.0, "message": "？"},
            {"start": 1.0, "end": 2.0, "message": "そうだよ。"},
        ]
        response = "\n".join((
            '{"start_id":1,"end_id":1}',
            '{"start_id":2,"end_id":2}',
            '{"start_id":3,"end_id":3}',
        ))
        refiner = SentenceRefiner(lambda _prompt: response, CancellationToken())
        output = refiner.refine(rows)
        self.assertTrue(refiner.applied)
        self.assertEqual(
            [item["message"] for item in output],
            ["本当", "？", "そうだよ。"],
        )

    def test_range_group_is_applied_without_local_duration_or_silence_override(self):
        rows = [
            {"start": 0.0, "end": 4.0, "message": "家にい"},
            {"start": 8.0, "end": 15.0, "message": "くよ。"},
        ]
        refiner = SentenceRefiner(
            lambda _prompt: '{"start_id":1,"end_id":2}\n',
            CancellationToken(),
        )
        output = refiner.refine(rows)
        self.assertEqual(output, [{
            "start": 0.0,
            "end": 15.0,
            "message": "家にいくよ。",
        }])

    def test_prompt_omits_redundant_word_timestamps_and_uses_bounded_output(self):
        rows = [{
            "start": 0.0,
            "end": 1.0,
            "message": "テスト",
            "words": [{"word": "テスト", "start": 0.0, "end": 1.0}],
        }] * 1691
        prompt = build_prompt(rows)
        max_tokens, max_characters = resegment_request_limits(rows)
        self.assertNotIn('"words"', prompt)
        self.assertNotIn('"start":', prompt)
        instructions = prompt.split("输入 JSON Lines", 1)[0]
        self.assertNotIn("42 个字符", instructions)
        self.assertNotIn("8 秒", instructions)
        self.assertNotIn("1.5 秒", instructions)
        self.assertIn("不设机械的字符数、时长或静音限制", prompt)
        self.assertLess(max_tokens, 40000)
        self.assertLess(max_characters, 100000)

    def test_japanese_interjection_can_remain_independent(self):
        rows = [
            {"start": 0.0, "end": 0.5, "message": "うん。"},
            {"start": 0.6, "end": 1.2, "message": "これは"},
            {"start": 1.2, "end": 2.2, "message": "テストです。"},
        ]
        output = apply_refinement(rows, ["うん。", "これはテストです。"])
        self.assertEqual(
            [item["message"] for item in output],
            ["うん。", "これはテストです。"],
        )

    def test_merge_and_split_preserve_text_and_timeline(self):
        rows = [
            {"start": 0.0, "end": 0.5, "message": "Um."},
            {"start": 0.7, "end": 1.7, "message": "This is"},
            {"start": 1.7, "end": 2.8, "message": "a test."},
            {
                "start": 3.0,
                "end": 11.0,
                "message": "This is a longer caption that should be divided into two natural captions.",
            },
        ]
        requested = [
            "Um.",
            "This is a test.",
            "This is a longer caption that should be divided",
            "into two natural captions.",
        ]
        output = apply_refinement(rows, requested)
        self.assertEqual(output[0]["message"], "Um.")
        self.assertEqual(output[1]["message"], "This is a test.")
        self.assertEqual(output[1]["start"], 0.7)
        self.assertEqual(output[1]["end"], 2.8)
        self.assertEqual("".join(item["message"] for item in output).replace(" ", ""),
                         "".join(item["message"] for item in rows).replace(" ", ""))
        self.assertTrue(all(item["end"] >= item["start"] for item in output))
        self.assertTrue(all(output[i]["end"] <= output[i + 1]["start"]
                            for i in range(len(output) - 1)))

    def test_natural_sentence_can_exceed_eight_seconds_without_word_split(self):
        rows = [
            {"start": 7.68, "end": 12.32, "message": "あ、おい、マコト"},
            {"start": 12.72, "end": 18.4, "message": "くん、こっちこっ"},
            {"start": 18.4, "end": 18.4, "message": "ち。"},
        ]
        output = apply_refinement(rows, ["あ、おい、マコトくん、こっちこっち。"])
        self.assertEqual(
            [item["message"] for item in output],
            ["あ、おい、マコトくん、こっちこっち。"],
        )

    def test_ai_group_is_not_resplit_by_local_duration_rule(self):
        rows = [
            {"start": 0.0, "end": 4.0, "message": "バスが来たから、"},
            {"start": 4.0, "end": 8.0, "message": "これに乗って"},
            {"start": 8.0, "end": 13.0, "message": "家にい"},
            {"start": 13.0, "end": 15.0, "message": "くよ。"},
        ]
        output = apply_refinement(
            rows,
            ["バスが来たから、これに乗って家にいくよ。"],
        )
        self.assertEqual(
            [item["message"] for item in output],
            ["バスが来たから、これに乗って家にいくよ。"],
        )

    def test_small_kana_suffix_is_not_isolated_by_bad_silence_timestamp(self):
        rows = [
            {"start": 0.0, "end": 5.0, "message": "気になる子いるでし"},
            {"start": 8.5, "end": 8.5, "message": "ょ？"},
        ]
        output = apply_refinement(rows, ["気になる子いるでしょ？"])
        self.assertEqual(
            [item["message"] for item in output],
            ["気になる子いるでしょ？"],
        )

    def test_invalid_ai_output_retries_then_falls_back(self):
        rows = [{"start": 0, "end": 1, "message": "original text"}]
        calls = []

        def bad_request(prompt):
            calls.append(prompt)
            return json.dumps({"text": "changed text"})

        refiner = SentenceRefiner(bad_request, CancellationToken())
        self.assertEqual(refiner.refine(rows), rows)
        self.assertFalse(refiner.applied)
        self.assertEqual(len(calls), 2)
        self.assertIn("上一次输出", calls[1])

    def test_invalid_suffix_does_not_partially_apply_ai_output(self):
        rows = [
            {"start": 0, "end": 1, "message": "这是"},
            {"start": 1, "end": 2, "message": "前半句。"},
            {"start": 2, "end": 3, "message": "保留原断句。"},
        ]

        def partial_request(_prompt):
            return '{"start_id":1,"end_id":2}\n'

        statuses = []
        refiner = SentenceRefiner(
            partial_request,
            CancellationToken(),
            status=statuses.append,
        )
        output = refiner.refine(rows)
        self.assertEqual(output, rows)
        self.assertFalse(refiner.applied)
        self.assertIn("未完整覆盖原文", statuses[-1])

    def test_ai_boundaries_use_precise_word_timestamps_without_extra_splits(self):
        rows = [{
            "start": 0.0,
            "end": 4.0,
            "message": "abcd",
            "words": [
                {"word": "a", "start": 0.0, "end": 0.2},
                {"word": "b", "start": 0.2, "end": 0.4},
                {"word": "cd", "start": 2.0, "end": 4.0},
            ],
        }]
        output = apply_refinement(rows, ["ab", "cd"])
        self.assertEqual(output[0]["end"], 0.4)
        self.assertEqual(output[1]["start"], 2.0)

        separated = [
            {"start": 0.0, "end": 1.0, "message": "first"},
            {"start": 3.0, "end": 4.0, "message": "second"},
        ]
        output = apply_refinement(separated, ["first second"])
        self.assertEqual([item["message"] for item in output], ["first second"])

    def test_streaming_openai_compatible_request(self):
        seen = {}
        output_events = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length", "0"))
                seen["body"] = json.loads(self.rfile.read(length))
                payload = json.dumps({
                    "choices": [{"delta": {
                        "reasoning_content": "hidden reasoning",
                        "content": '{"text":"hello"}\n',
                    }}]
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"data: " + payload + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")

            def log_message(self, _format, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = request_openai_compatible(
                "prompt",
                endpoint=f"http://127.0.0.1:{server.server_port}",
                model="deepseek-v4-flash",
                api_key="test-token",
                cancel_token=CancellationToken(),
                thinking_enabled=True,
                output_callback=output_events.append,
            )
            self.assertIn('"text":"hello"', result)
            self.assertTrue(seen["body"]["stream"])
            self.assertEqual(seen["body"]["max_tokens"], 65536)
            self.assertEqual(seen["body"]["thinking"], {"type": "enabled"})
            self.assertTrue(output_events[-1]["final"])
            self.assertEqual(output_events[-1]["characters"], len(result))

            request_openai_compatible(
                "prompt",
                endpoint=f"http://127.0.0.1:{server.server_port}",
                model="qwen3-235b",
                api_key="test-token",
                cancel_token=CancellationToken(),
                thinking_enabled=False,
            )
            self.assertFalse(seen["body"]["enable_thinking"])
        finally:
            server.shutdown()
            server.server_close()

    def test_streaming_opencode_responses_request(self):
        seen = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                seen["path"] = self.path
                length = int(self.headers.get("content-length", "0"))
                seen["body"] = json.loads(self.rfile.read(length))
                payload = json.dumps({
                    "type": "response.output_text.delta",
                    "delta": '{"start_id":1,"end_id":1}\n',
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(b"data: " + payload + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")

            def log_message(self, _format, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch(
                "sentence_refiner.opencode_zen_api_mode",
                return_value="responses",
            ):
                result = request_openai_compatible(
                    "prompt",
                    endpoint=f"http://127.0.0.1:{server.server_port}",
                    model="gpt-5.6-sol",
                    api_key="test-token",
                    cancel_token=CancellationToken(),
                )
            self.assertIn('"end_id":1', result)
            self.assertEqual(seen["path"], "/v1/responses")
            self.assertEqual(seen["body"]["input"], "prompt")
            self.assertEqual(seen["body"]["max_output_tokens"], 65536)
            self.assertNotIn("messages", seen["body"])
        finally:
            server.shutdown()
            server.server_close()

    def test_runaway_visible_output_is_aborted(self):
        seen = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length", "0"))
                seen["body"] = json.loads(self.rfile.read(length))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for _index in range(20):
                    payload = json.dumps({
                        "choices": [{"delta": {"content": "x" * 100}}]
                    }).encode()
                    try:
                        self.wfile.write(b"data: " + payload + b"\n\n")
                        self.wfile.flush()
                    except OSError:
                        break

            def log_message(self, _format, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with self.assertRaisesRegex(RefinementError, "异常增长"):
                request_openai_compatible(
                    "prompt",
                    endpoint=f"http://127.0.0.1:{server.server_port}",
                    model="deepseek-v4-flash",
                    api_key="test-token",
                    cancel_token=CancellationToken(),
                    max_output_tokens=2345,
                    max_output_characters=250,
                )
            self.assertEqual(seen["body"]["max_tokens"], 2345)
        finally:
            server.shutdown()
            server.server_close()

    def test_stream_closes_as_soon_as_all_source_ids_are_covered(self):
        progress_events = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length", "0"))
                self.rfile.read(length)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                parts = (
                    '{"start_id":1,"end_id":1}\n',
                    '{"start_id":2,"end_id":2}\n',
                    "unwanted trailing output" * 100,
                )
                for content in parts:
                    payload = json.dumps({
                        "choices": [{"delta": {"content": content}}]
                    }).encode()
                    try:
                        self.wfile.write(b"data: " + payload + b"\n\n")
                        self.wfile.flush()
                    except OSError:
                        break

            def log_message(self, _format, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = request_openai_compatible(
                "prompt",
                endpoint=f"http://127.0.0.1:{server.server_port}",
                model="deepseek-v4-flash",
                api_key="test-token",
                cancel_token=CancellationToken(),
                expected_last_id=2,
                progress_callback=lambda current, total: progress_events.append(
                    (current, total)
                ),
            )
            self.assertIn('"end_id":2', result)
            self.assertNotIn("unwanted trailing output", result)
            self.assertEqual(progress_events[-1], (2, 2))
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
