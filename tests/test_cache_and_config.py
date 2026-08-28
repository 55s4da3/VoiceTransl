import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import orjson
import yaml

import GalTransl.AppSettings as app_settings
from GalTransl.Cache import (
    _append_cache_file_path,
    _build_cache_dict_from_snapshot,
    _build_cache_key_for_tran,
    _build_cache_obj,
    _cache_get,
    _cache_has,
    check_retran_key,
    compact_cache_append_logs,
    get_transCache_from_json,
    save_transCache_to_json,
)
from GalTransl.ConfigHelper import (
    CProblemType,
    CProjectConfig,
    build_httpx_proxy_kwargs,
    build_httpx_sync_proxy_kwargs,
    initDictList,
    initProxyList,
    loadConfigFile,
)
from GalTransl.Loader import load_transList
from GalTransl.Problem import find_problems


class AppSettingsTests(unittest.TestCase):
    def test_normalization_clamps_jobs_and_applies_defaults(self):
        self.assertEqual(
            app_settings._normalize_settings({"maxConcurrentJobs": 0}),
            {"printTranslationLogInTerminal": True, "maxConcurrentJobs": 1},
        )
        self.assertEqual(app_settings._normalize_settings(None), app_settings.DEFAULT_APP_SETTINGS)

    def test_load_missing_and_invalid_settings_returns_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_path = str(Path(temp_dir) / "settings.json")
            with patch.object(app_settings, "_SETTINGS_PATH", settings_path):
                self.assertEqual(app_settings.load_app_settings(), app_settings.DEFAULT_APP_SETTINGS)
                Path(settings_path).write_text("[]", encoding="utf-8")
                self.assertEqual(app_settings.load_app_settings(), app_settings.DEFAULT_APP_SETTINGS)
                Path(settings_path).write_text("invalid", encoding="utf-8")
                self.assertEqual(app_settings.load_app_settings(), app_settings.DEFAULT_APP_SETTINGS)

    def test_save_is_atomic_and_normalized(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_path = Path(temp_dir) / "settings.json"
            with patch.object(app_settings, "_SETTINGS_PATH", str(settings_path)):
                result = app_settings.save_app_settings({
                    "printTranslationLogInTerminal": False,
                    "maxConcurrentJobs": "6",
                    "ignored": True,
                })
            self.assertEqual(result, {"printTranslationLogInTerminal": False, "maxConcurrentJobs": 6})
            self.assertEqual(json.loads(settings_path.read_text(encoding="utf-8")), result)
            self.assertFalse(settings_path.with_suffix(".json.tmp").exists())


class ConfigHelperTests(unittest.TestCase):
    def test_proxy_kwargs_match_installed_httpx_api(self):
        async_kwargs = build_httpx_proxy_kwargs("http://127.0.0.1:8080")
        sync_kwargs = build_httpx_sync_proxy_kwargs("http://127.0.0.1:8080")
        self.assertEqual(len(async_kwargs), 1)
        self.assertEqual(len(sync_kwargs), 1)
        self.assertIn(next(iter(async_kwargs)), {"proxy", "proxies", "mounts"})
        self.assertIn(next(iter(sync_kwargs)), {"proxy", "proxies", "mounts"})
        self.assertEqual(build_httpx_proxy_kwargs(None), {})

    def test_dictionary_path_resolution(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dictionary_dir = root / "dict"
            project_dir = root / "project"
            absolute = root / "absolute.txt"
            result = initDictList(
                ["common.txt", "(project_dir)local.txt", str(absolute)],
                str(dictionary_dir),
                str(project_dir),
            )
            self.assertEqual(Path(result[0]), dictionary_dir.resolve() / "common.txt")
            self.assertEqual(Path(result[1]), project_dir.resolve() / "local.txt")
            self.assertEqual(Path(result[2]), absolute)

    def test_project_config_accessors_and_legacy_directories(self):
        config = {
            "common": {"linebreakSymbol": "\\n", "workersPerProject": 2},
            "plugin": {"textPlugins": ["normal"], "filePlugin": "json"},
            "dictionary": {"preDict": ["a.txt"]},
            "problemAnalyze": {"problemList": ["残留日文"], "arinashiDict": {"猫": "猫"}},
            "backendSpecific": {"GPT4": {"model": "x"}},
            "proxy": {"enableProxy": True, "proxies": [{"address": "http://proxy", "username": "u", "password": "p"}]},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "json_jp").mkdir()
            (root / "json_cn").mkdir()
            (root / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
            loaded = CProjectConfig(str(root))
            self.assertEqual(Path(loaded.getInputPath()), root / "json_jp")
            self.assertEqual(Path(loaded.getOutputPath()), root / "json_cn")
            self.assertEqual(loaded.getTextPluginList(), ["normal"])
            self.assertEqual(loaded.getFilePlugin(), "json")
            self.assertEqual(loaded.getlbSymbol(), r"\n")
            self.assertTrue(loaded.getKey("internals.enableProxy"))
            self.assertEqual(loaded.getProblemAnalyzeConfig("problemList"), [CProblemType.残留日文])
            self.assertEqual(initProxyList(loaded), [{"addr": "http://proxy", "username": "u", "password": "p"}])
            self.assertEqual(loadConfigFile(str(root / "config.yaml"))["common"]["workersPerProject"], 2)


class CacheTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def translated_rows():
        trans, _ = load_transList([
            {"name": "A", "message": "一"},
            {"message": "二"},
            {"message": "三"},
        ])
        for index, item in enumerate(trans):
            item.pre_zh = f"译{index}"
            item.post_zh = item.pre_zh
            item.trans_by = "test-model"
        return trans

    def test_compatibility_keys_and_retranslation_matching(self):
        legacy = {"pre_jp": "原", "post_jp": "润", "pre_zh": "译"}
        self.assertEqual(_cache_get(legacy, "pre_src"), "原")
        self.assertTrue(_cache_has(legacy, "pre_dst"))
        self.assertFalse(_cache_has(legacy, "proofread_dst"))
        self.assertTrue(check_retran_key("原", "原文"))
        self.assertTrue(check_retran_key(["", None, "文"], "原文"))
        self.assertFalse(check_retran_key(["", None], "原文"))

    def test_cache_object_skips_empty_and_includes_post_save_fields(self):
        tran = self.translated_rows()[0]
        tran.problem = "问题"
        tran.trans_conf = 0.8
        built = _build_cache_obj(tran, post_save=True)
        self.assertEqual(built["pre_src"], "一")
        self.assertEqual(built["post_dst_preview"], "译0")
        self.assertEqual(built["problem"], "问题")
        tran.pre_zh = ""
        self.assertIsNone(_build_cache_obj(tran))

    def test_context_cache_key_skips_empty_neighbors(self):
        trans = self.translated_rows()
        trans[1].post_jp = ""
        self.assertEqual(_build_cache_key_for_tran(trans[0]), "NoneA一三")

    def test_snapshot_dictionary_preserves_order_and_last_duplicate(self):
        rows = [
            {"name": "", "pre_src": "a", "pre_dst": "1"},
            {"name": "", "pre_src": "b", "pre_dst": "2"},
        ]
        mapping, order = _build_cache_dict_from_snapshot(rows)
        self.assertEqual(len(mapping), 2)
        self.assertEqual(len(order), 2)

    async def test_full_snapshot_round_trip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cache.json"
            original = self.translated_rows()
            original[1].proofread_zh = "校对"
            await save_transCache_to_json(original, str(path), post_save=True)
            fresh, _ = load_transList([
                {"name": "A", "message": "一"},
                {"message": "二"},
                {"message": "三"},
            ])
            hits, misses = await get_transCache_from_json(fresh, str(path))
            self.assertEqual(len(hits), 3)
            self.assertEqual(misses, [])
            self.assertEqual(fresh[0].post_zh, "译0")
            self.assertEqual(fresh[1].post_zh, "校对")
            self.assertEqual(fresh[2].trans_by, "test-model")

    async def test_incremental_append_and_compaction(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cache.json"
            await save_transCache_to_json(self.translated_rows(), str(path), post_save=False)
            append_path = Path(_append_cache_file_path(str(path)))
            self.assertTrue(append_path.is_file())
            self.assertFalse(path.exists())
            count = await compact_cache_append_logs(temp_dir)
            self.assertEqual(count, 1)
            self.assertTrue(path.is_file())
            self.assertFalse(append_path.exists())
            self.assertEqual(len(orjson.loads(path.read_bytes())), 3)

    async def test_retranslation_key_turns_cache_hit_into_miss(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cache.json"
            await save_transCache_to_json(self.translated_rows(), str(path), post_save=True)
            fresh, _ = load_transList([
                {"name": "A", "message": "一"},
                {"message": "二"},
                {"message": "三"},
            ])
            hits, misses = await get_transCache_from_json(fresh, str(path), retran_key="二")
            self.assertEqual([item.pre_jp for item in hits], ["一", "三"])
            self.assertEqual([item.pre_jp for item in misses], ["二"])


class ProblemAnalysisTests(unittest.TestCase):
    class Config:
        target_lang = "zh-cn"

        def __init__(self, problem_types, arinashi=None, linebreak="auto"):
            self.problem_types = problem_types
            self.arinashi = arinashi or {}
            self.linebreak = linebreak

        def getProblemAnalyzeArinashiDict(self):
            return self.arinashi

        def getProblemAnalyzeConfig(self, key):
            return self.problem_types if key == "problemList" else []

        def getlbSymbol(self):
            return self.linebreak

    def test_detects_residual_japanese_linebreak_and_failure(self):
        tran = load_transList([{"message": "かな\\n次"}])[0][0]
        tran.pre_zh = "かな翻译"
        tran.post_zh = "かな(Failed)"
        config = self.Config([
            CProblemType.残留日文,
            CProblemType.丢失换行,
        ])
        find_problems([tran], config)
        self.assertIn("残留日文", tran.problem)
        self.assertIn("丢失换行", tran.problem)
        self.assertIn("翻译失败", tran.problem)

    def test_detects_custom_presence_rule_and_monologue_pronoun(self):
        tran = load_transList([{"message": "彼女"}])[0][0]
        tran.pre_zh = "她"
        tran.post_zh = "猫和他"
        config = self.Config([CProblemType.独白男他], {"猫": "猫"})
        find_problems([tran], config)
        self.assertIn("独白男他", tran.problem)
        self.assertIn("本无 猫 译有 猫", tran.problem)


if __name__ == "__main__":
    unittest.main()
