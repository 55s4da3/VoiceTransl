import unittest

from provider_presets import (
    DEEPSEEK_V4_FLASH_MODEL,
    build_models_url_candidates,
    normalize_provider_model,
    provider_default_model,
    provider_model_choices,
)


class ProviderPresetTests(unittest.TestCase):
    def test_official_deepseek_v41_aliases_use_real_wire_model_id(self):
        for alias in (
            "deepseek-v4.1-flash",
            "deepseek-v4-1-flash",
            "deepseek-v41-flash",
            "deepseek-flash",
        ):
            self.assertEqual(
                normalize_provider_model("Deepseek", alias),
                DEEPSEEK_V4_FLASH_MODEL,
            )

        self.assertEqual(
            normalize_provider_model("OpenCode Go", "deepseek-flash"),
            "deepseek-flash",
        )

    def test_deepseek_preset_has_one_click_defaults(self):
        self.assertEqual(
            provider_default_model("Deepseek"), DEEPSEEK_V4_FLASH_MODEL
        )
        self.assertEqual(
            provider_model_choices("Deepseek"),
            ["deepseek-v4-flash", "deepseek-v4-pro"],
        )

    def test_model_url_candidates_follow_versioned_endpoint_shape(self):
        self.assertEqual(
            build_models_url_candidates("https://api.deepseek.com"),
            ["https://api.deepseek.com/v1/models"],
        )
        self.assertEqual(
            build_models_url_candidates(
                "https://open.bigmodel.cn/api/paas/v4/chat/completions"
            ),
            [
                "https://open.bigmodel.cn/api/paas/v4/models",
                "https://open.bigmodel.cn/api/paas/v4/v1/models",
            ],
        )
        self.assertEqual(
            build_models_url_candidates(
                "https://example.test/v1", "https://catalog.test/models"
            ),
            ["https://catalog.test/models"],
        )


if __name__ == "__main__":
    unittest.main()
