import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from GalTransl.COpenAI import COpenAIToken, COpenAITokenPool

from opencode_zen import (
    OPENCODE_ZEN_ENDPOINT,
    filter_supported_opencode_models,
    is_opencode_zen_endpoint,
    opencode_zen_api_mode,
    responses_finish_reason,
    responses_output_text,
    responses_stream_delta,
    split_responses_messages,
)


class OpenCodeZenTests(unittest.TestCase):
    def test_endpoint_detection_accepts_zen_and_go(self):
        self.assertTrue(is_opencode_zen_endpoint(OPENCODE_ZEN_ENDPOINT))
        self.assertTrue(is_opencode_zen_endpoint("https://opencode.ai/zen/v1"))
        self.assertTrue(is_opencode_zen_endpoint("https://opencode.ai/zen/go/v1"))
        self.assertFalse(is_opencode_zen_endpoint("https://api.openai.com/v1"))

    def test_model_protocol_routing(self):
        endpoint = "https://opencode.ai/zen/v1"
        self.assertEqual(opencode_zen_api_mode(endpoint, "gpt-5.6-sol"), "responses")
        self.assertEqual(opencode_zen_api_mode(endpoint, "muse-spark-1.3"), "responses")
        self.assertEqual(opencode_zen_api_mode(endpoint, "deepseek-v4-flash"), "chat")
        self.assertEqual(opencode_zen_api_mode(endpoint, "kimi-k2.6"), "chat")
        self.assertEqual(opencode_zen_api_mode(endpoint, "claude-sonnet-5"), "unsupported")
        self.assertEqual(opencode_zen_api_mode(endpoint, "gemini-3.7-flash"), "unsupported")
        self.assertEqual(opencode_zen_api_mode(endpoint, "qwen3.7-plus"), "unsupported")

    def test_catalog_filter_preserves_supported_order(self):
        models = [
            "claude-sonnet-5",
            "gpt-5.6-sol",
            "deepseek-v4-flash",
            "qwen3.7-plus",
        ]
        self.assertEqual(
            filter_supported_opencode_models(OPENCODE_ZEN_ENDPOINT, models),
            ["gpt-5.6-sol", "deepseek-v4-flash"],
        )

    def test_responses_helpers(self):
        instructions, values = split_responses_messages([
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
        ])
        self.assertEqual(instructions, "system")
        self.assertEqual(values, [{"role": "user", "content": "hello"}])
        self.assertEqual(
            responses_stream_delta({
                "type": "response.output_text.delta",
                "delta": "hi",
            }),
            "hi",
        )
        self.assertEqual(responses_finish_reason({"type": "response.completed"}), "stop")
        self.assertEqual(responses_output_text({"output_text": "done"}), "done")

    def test_token_check_uses_responses_api_for_gpt_models(self):
        client = MagicMock()
        client.responses.create.return_value = SimpleNamespace(id="response-1")
        pool = COpenAITokenPool.__new__(COpenAITokenPool)
        pool.timeout = 10
        token = COpenAIToken(
            "test-key",
            "https://opencode.ai/zen/v1",
            "gpt-5.6-sol",
            stream=True,
        )

        with patch("GalTransl.COpenAI.OpenAI", return_value=client):
            available, returned_token = pool._isTokenAvailable_sync(token)

        self.assertTrue(available)
        self.assertIs(returned_token, token)
        client.responses.create.assert_called_once_with(
            model="gpt-5.6-sol",
            input="1+1=",
            timeout=10,
            stream=False,
            max_output_tokens=1,
        )
        client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
