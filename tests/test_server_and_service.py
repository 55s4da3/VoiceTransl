import asyncio
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import orjson
import yaml

from GalTransl.Backend.BaseTranslate import RequestHealthMetrics
from GalTransl.CSplitter import DictionaryCountSplitter
from GalTransl.Frontend.LLMTranslate import (
    _build_runtime_file_maps,
    _check_stop_requested,
)
from GalTransl.Service import (
    JobCancelledError,
    JobSpec,
    _append_error_log,
    _should_skip_error_log,
    create_job_state,
    run_job_async,
)
from GalTransl.TerminalOutput import NullProgressBar, should_print_translation_logs, terminal_progress
from GalTransl.server import (
    RuntimeProgressCache,
    RuntimeRegistry,
    _categorize_common_dict_file,
    _check_retran_key,
    _collect_project_dict_payload,
    _dict_category_config_key,
    _ensure_project_dict_file_configured,
    _has_newer_release,
    _is_path_within,
    _is_safe_config_filename,
    _is_safe_dict_filename,
    _list_dir_entries,
    _normalize_dict_text,
    _normalize_retran_key,
    _normalize_retran_terms,
    _parse_runtime_job_started_at_ns,
    _read_common_dict_category_map,
    _read_dict_file_payload,
    _read_yaml_file,
    _safe_project_dir,
    _trim_preview,
    _write_common_dict_category_map,
    _write_yaml_file,
    decode_project_dir,
    encode_project_dir,
)


class ServerHelperTests(unittest.TestCase):
    def test_version_comparison_handles_prefix_and_invalid_versions(self):
        self.assertTrue(_has_newer_release("v1.2.0", "V1.3.0"))
        self.assertFalse(_has_newer_release("1.2", "v1.2.0"))
        self.assertFalse(_has_newer_release("1.2", None))
        self.assertTrue(_has_newer_release("dev-a", "dev-b"))

    def test_project_directory_token_round_trip_and_validation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            token = encode_project_dir(temp_dir)
            self.assertEqual(decode_project_dir(token.rstrip("=")), temp_dir)
            self.assertEqual(_safe_project_dir(token), temp_dir)
        with self.assertRaises(ValueError):
            _safe_project_dir("not-valid-base64")

    def test_filename_and_path_security_checks(self):
        self.assertTrue(_is_safe_dict_filename("dict.txt"))
        self.assertFalse(_is_safe_dict_filename("../dict.txt"))
        self.assertFalse(_is_safe_dict_filename("folder\\dict.txt"))
        self.assertTrue(_is_safe_config_filename("config.yaml"))
        self.assertFalse(_is_safe_config_filename("config..yaml"))
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertTrue(_is_path_within(temp_dir, str(Path(temp_dir) / "child")))
            self.assertFalse(_is_path_within(temp_dir, str(Path(temp_dir).parent / "escape")))

    def test_dictionary_helpers(self):
        self.assertEqual(_normalize_dict_text("a\r\nb\rc"), "a\nb\nc")
        self.assertEqual(_dict_category_config_key("pre"), "preDict")
        self.assertEqual(_dict_category_config_key("gpt"), "gpt.dict")
        self.assertEqual(_dict_category_config_key("post"), "postDict")
        with self.assertRaises(ValueError):
            _dict_category_config_key("bad")
        self.assertEqual(_categorize_common_dict_file("my-gpt.txt"), "gpt")
        self.assertEqual(_categorize_common_dict_file("译后修正.txt"), "post")
        self.assertEqual(_categorize_common_dict_file("common.txt"), "pre")

    def test_yaml_atomic_write_and_directory_listing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "config.yaml"
            _write_yaml_file(str(config), {"中文": [1, 2]})
            self.assertEqual(_read_yaml_file(str(config)), {"中文": [1, 2]})
            (root / "rows.json").write_text('[{"a":1},{"a":2}]', encoding="utf-8")
            (root / "subdir").mkdir()
            entries = _list_dir_entries(temp_dir, count_json_entries=True)
            by_name = {entry["name"]: entry for entry in entries}
            self.assertEqual(by_name["rows.json"]["entry_count"], 2)
            self.assertFalse(by_name["subdir"]["is_file"])

    def test_dictionary_payload_counts_content_and_configures_project_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = root / "config.yaml"
            config.write_text(yaml.safe_dump({"dictionary": {"preDict": []}}), encoding="utf-8")
            dictionary = root / "local.txt"
            dictionary.write_text("// comment\n\\\\ comment\nA\tB\n\nC\tD\n", encoding="utf-8")
            _ensure_project_dict_file_configured(temp_dir, "config.yaml", "pre", "local.txt")
            _ensure_project_dict_file_configured(temp_dir, "config.yaml", "pre", "local.txt")
            payload = _collect_project_dict_payload(temp_dir, "config.yaml")
            self.assertEqual(payload["pre_dict_files"], ["(project_dir)local.txt"])
            self.assertEqual(payload["dict_contents"]["(project_dir)local.txt"]["count"], 2)
            configured = _read_yaml_file(str(config))["dictionary"]["preDict"]
            self.assertEqual(configured, ["(project_dir)local.txt"])

    def test_dictionary_file_payload_missing_and_category_map_filtering(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.assertEqual(_read_dict_file_payload(str(root / "missing.txt"))["error"], "file not found")
            _write_common_dict_category_map(temp_dir, {"a.txt": "gpt", "bad/name": "pre", "b.txt": "invalid"})
            self.assertEqual(_read_common_dict_category_map(temp_dir), {"a.txt": "gpt"})

    def test_retranslation_normalization_and_timestamp_parsing(self):
        self.assertEqual(_normalize_retran_key(["a", "", None]), ["a"])
        self.assertEqual(_normalize_retran_key(1), "")
        self.assertEqual(_normalize_retran_terms([" a ", "", None, "b"]), ["a", "b"])
        self.assertTrue(_check_retran_key(["x", "y"], "say y"))
        self.assertFalse(_check_retran_key("", "anything"))
        self.assertIsInstance(_parse_runtime_job_started_at_ns("2026-01-01T00:00:00Z"), int)
        self.assertIsNone(_parse_runtime_job_started_at_ns("bad"))
        self.assertEqual(_trim_preview(" a\n b ", 20), "a  b")
        self.assertEqual(_trim_preview("12345", 4), "123…")


class RuntimeRegistryTests(unittest.TestCase):
    def test_status_success_error_and_reset_lifecycle(self):
        registry = RuntimeRegistry()
        with tempfile.TemporaryDirectory() as project_dir:
            empty = registry.get_runtime_snapshot(project_dir)
            self.assertEqual(empty["workers_active"], 0)
            registry.update_status(
                project_dir,
                stage="翻译中",
                workers_active=-2,
                workers_configured=3,
                file_totals={"story.json": 2},
                cache_file_display_map={"story_0.json": "story.json"},
            )
            registry.append_success(
                project_dir,
                filename="story_0",
                index=1,
                speaker="A",
                source_preview="原文",
                translation_preview="译文",
                trans_by="model",
            )
            registry.append_error(project_dir, kind="retry", message="x" * 300, filename="story_0")
            snapshot = registry.get_runtime_snapshot(project_dir)
            self.assertEqual(snapshot["workers_active"], 0)
            self.assertEqual(snapshot["workers_configured"], 3)
            self.assertEqual(snapshot["recent_successes"][0]["filename"], "story.json")
            self.assertEqual(snapshot["recent_errors"][0]["filename"], "story.json")
            self.assertLessEqual(len(snapshot["recent_errors"][0]["message"]), 240)
            self.assertGreater(snapshot["translation_speed_lpm"], 0)
            registry.reset_project(project_dir)
            self.assertEqual(registry.get_runtime_snapshot(project_dir)["recent_successes"], [])

    def test_progress_cache_counts_snapshot_and_append_without_duplicates(self):
        cache = RuntimeProgressCache()
        with tempfile.TemporaryDirectory() as project_dir:
            cache_dir = Path(project_dir) / "transl_cache"
            cache_dir.mkdir()
            snapshot = [
                {"index": 1, "name": "", "pre_src": "一", "pre_dst": "译一"},
                {"index": 2, "name": "", "pre_src": "二", "pre_dst": "(Failed)", "problem": "翻译失败"},
            ]
            (cache_dir / "story.json").write_bytes(orjson.dumps(snapshot))
            append = dict(snapshot[0], __cache_key="None一二")
            (cache_dir / "story.json.append.jsonl").write_bytes(orjson.dumps(append) + b"\n")
            progress = cache.get_progress(
                project_dir,
                {"story.json": 2},
                {"story.json": "story.json"},
            )
            self.assertEqual(progress["total"], 2)
            self.assertEqual(progress["translated"], 2)
            self.assertEqual(progress["problems"], 1)
            self.assertEqual(progress["failed"], 1)

    def test_progress_cache_reads_retranslation_config(self):
        cache = RuntimeProgressCache()
        with tempfile.TemporaryDirectory() as project_dir:
            Path(project_dir, "config.yaml").write_text(
                yaml.safe_dump({"common": {"retranslKey": ["a", "b"]}}),
                encoding="utf-8",
            )
            self.assertEqual(cache.get_retran_key(project_dir), ["a", "b"])


class ServiceAndProgressTests(unittest.IsolatedAsyncioTestCase):
    def test_job_state_creation_and_validation_failure(self):
        spec = JobSpec(project_dir="", translator="engine", job_id="job-1")
        state = create_job_state(spec)
        self.assertEqual(state.status, "pending")

    async def test_invalid_job_spec_fails_without_running(self):
        spec = JobSpec(project_dir="", translator="engine", job_id="job-1")
        with patch("GalTransl.server.reset_runtime_project"), patch("GalTransl.server.update_runtime_status"):
            result = await run_job_async(spec)
        self.assertEqual(result.status, "failed")
        self.assertFalse(result.success)
        self.assertTrue(result.finished_at)

    async def test_successful_job_lifecycle_with_translation_mocked(self):
        class Config:
            def __init__(self, project_dir, config_name):
                self.project_dir = project_dir
                self.projectConfig = {"backendSpecific": {}}
                self.keyValues = {"workersPerProject": 2}

            def getCommonConfigSection(self):
                return {"loggingLevel": "info"}

            def getKey(self, key, default=None):
                return self.keyValues.get(key, default)

            def getCachePath(self):
                return str(Path(self.project_dir) / "transl_cache")

        with tempfile.TemporaryDirectory() as project_dir:
            spec = JobSpec(project_dir=project_dir, translator="engine", job_id="ok")
            with patch("GalTransl.Service.CProjectConfig", Config), \
                 patch("GalTransl.Service.load_app_settings", return_value={"printTranslationLogInTerminal": False}), \
                 patch("GalTransl.Service.run_galtransl", new=AsyncMock(return_value=True)) as run_mock, \
                 patch("GalTransl.server.reset_runtime_project"), \
                 patch("GalTransl.server.update_runtime_status"):
                result = await run_job_async(spec)
            self.assertEqual(result.status, "completed")
            self.assertTrue(result.success)
            run_mock.assert_awaited_once()

    def test_error_log_policy_and_content(self):
        self.assertTrue(_should_skip_error_log(JobCancelledError()))
        self.assertFalse(_should_skip_error_log(ValueError("bad")))
        with tempfile.TemporaryDirectory() as project_dir:
            spec = JobSpec(project_dir=project_dir, translator="engine", job_id="job")
            try:
                raise ValueError("bad value")
            except ValueError as error:
                _append_error_log(spec, error, phase="test")
            text = Path(project_dir, "error.log").read_text(encoding="utf-8")
            self.assertIn("phase=test", text)
            self.assertIn("bad value", text)

    def test_terminal_progress_disabled_is_noop(self):
        config = type("Config", (), {"non_interactive": True, "print_translation_log_in_terminal": True})()
        self.assertFalse(should_print_translation_logs(config))
        with terminal_progress(False, total=2) as bar:
            self.assertIsInstance(bar, NullProgressBar)
            bar()
            bar.title("ignored")


class RuntimeMappingAndCancellationTests(unittest.TestCase):
    def test_runtime_file_maps_count_nonempty_nonoverlap_rows(self):
        with tempfile.TemporaryDirectory() as input_dir:
            file_path = str(Path(input_dir) / "nested" / "story.json")
            rows = [{"message": "一"}, {"message": ""}, {"message": "三"}]
            chunks = DictionaryCountSplitter(2, cross_num=1).split(rows, file_path)
            totals, mapping = _build_runtime_file_maps(chunks, input_dir)
            display = os.path.join("nested", "story.json").replace(os.sep, "/")
            self.assertEqual(totals, {display: 2})
            self.assertEqual(set(mapping.values()), {display})

    def test_stop_request_raises_cancellation(self):
        event = threading.Event()
        config = type("Config", (), {"stop_event": event})()
        _check_stop_requested(config)
        event.set()
        with self.assertRaises(JobCancelledError):
            _check_stop_requested(config)

    def test_request_health_metrics_aggregates_samples(self):
        metrics = RequestHealthMetrics()
        metrics.record(1.0, False)
        metrics.record(3.0, True)
        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["total"], 2)
        self.assertEqual(snapshot["rate_limited"], 1)
        self.assertEqual(snapshot["rate_limited_ratio"], 0.5)
        self.assertEqual(snapshot["avg_latency"], 2.0)


if __name__ == "__main__":
    unittest.main()
