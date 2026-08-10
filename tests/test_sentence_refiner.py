import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from sentence_refiner import (
    SentenceRefiner,
    apply_refinement,
    request_openai_compatible,
)
from tasking import CancellationToken


class SentenceRefinerTests(unittest.TestCase):
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

    def test_invalid_ai_output_retries_then_falls_back(self):
        rows = [{"start": 0, "end": 1, "message": "original text"}]
        calls = []

        def bad_request(prompt):
            calls.append(prompt)
            return json.dumps({"text": "changed text"})

        refiner = SentenceRefiner(bad_request, CancellationToken())
        self.assertEqual(refiner.refine(rows), rows)
        self.assertEqual(len(calls), 2)
        self.assertIn("上一次输出", calls[1])

    def test_invalid_suffix_keeps_verified_prefix_only(self):
        rows = [
            {"start": 0, "end": 1, "message": "这是"},
            {"start": 1, "end": 2, "message": "前半句。"},
            {"start": 2, "end": 3, "message": "保留原断句。"},
        ]

        def partial_request(_prompt):
            return '\n'.join((
                json.dumps({"text": "这是前半句。"}, ensure_ascii=False),
                json.dumps({"text": "这里被错误改写"}, ensure_ascii=False),
            ))

        output = SentenceRefiner(partial_request, CancellationToken()).refine(rows)
        self.assertEqual(
            [item["message"] for item in output],
            ["这是前半句。", "保留原断句。"],
        )

    def test_word_timestamps_and_hard_silence_are_respected(self):
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
        # Even if the AI asks to merge the whole row, the 1.6-second
        # word-level silence remains a mandatory caption boundary.
        output = apply_refinement(rows, ["abcd"])
        self.assertEqual(output[0]["end"], 0.4)
        self.assertEqual(output[1]["start"], 2.0)

        separated = [
            {"start": 0.0, "end": 1.0, "message": "first"},
            {"start": 3.0, "end": 4.0, "message": "second"},
        ]
        output = apply_refinement(separated, ["first second"])
        self.assertEqual([item["message"] for item in output], ["first", "second"])

    def test_streaming_openai_compatible_request(self):
        seen = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length", "0"))
                seen["body"] = json.loads(self.rfile.read(length))
                payload = json.dumps({
                    "choices": [{"delta": {"content": '{"text":"hello"}\n'}}]
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
            )
            self.assertIn('"text":"hello"', result)
            self.assertTrue(seen["body"]["stream"])
            self.assertEqual(seen["body"]["thinking"], {"type": "enabled"})

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


if __name__ == "__main__":
    unittest.main()
