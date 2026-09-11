import asyncio
import os
import json
import io
import subprocess
import sys
import threading
import time
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tasking import CancellationToken, ProcessRegistry, TaskCancelledError
from tasking import TaskSnapshot
from log import UIMessageQueue
from pool import ConcurrentTranslationPool, TranscribedFile
from worker import MainWorker
from GalTransl.Backend.BaseTranslate import BaseTranslate
from openai._types import NOT_GIVEN
from output_metrics import decode_output_event, encode_output_event


class TaskRuntimeTests(unittest.TestCase):
    def test_stage_ids_reset_same_named_phase_and_file_count_is_independent(self):
        entries = []
        worker = MainWorker(
            TaskSnapshot('run', {}),
            SimpleNamespace(put=lambda target, text: entries.append((target, text))),
            CancellationToken(),
        )
        worker._set_file_progress_total(2)
        first = worker._begin_stage('AI 断句', 100)
        worker._update_stage(first, 80, 100)
        second = worker._begin_stage('AI 断句', 100)
        worker._complete_file('file-a')
        worker._complete_file('file-a')

        events = [
            json.loads(text) for target, text in entries if target == 'progress'
        ]
        stage_events = [event for event in events if event['kind'] == 'stage']
        file_events = [event for event in events if event['kind'] == 'files']
        self.assertNotEqual(first, second)
        self.assertEqual(stage_events[-1]['current'], 0)
        self.assertEqual(stage_events[-1]['stage_id'], second)
        self.assertEqual(file_events[-1]['completed'], 1)
        self.assertEqual(file_events[-1]['total'], 2)

    def test_failed_translation_never_completes_source_file(self):
        entries = []
        worker = MainWorker(
            TaskSnapshot('run', {}),
            SimpleNamespace(put=lambda target, text: entries.append((target, text))),
            CancellationToken(),
        )
        worker._set_file_progress_total(2)
        worker._translation_task_finished('bad', 'error', False)
        worker._translation_task_finished('bad', 'success', True)
        worker._translation_task_finished('good', 'success', True)
        file_events = [
            json.loads(text)
            for target, text in entries
            if target == 'progress' and json.loads(text).get('kind') == 'files'
        ]
        self.assertEqual(file_events[-1]['completed'], 1)

    def test_received_character_event_round_trip(self):
        event = {
            'request': 'request-1',
            'characters': 123,
            'final': True,
        }
        self.assertEqual(decode_output_event(encode_output_event(event)), event)

    def test_failed_resegmentation_is_reported_and_not_applied(self):
        entries = []
        snapshot = TaskSnapshot('run', {
            'enable_ai_resegment': True,
            'translator': 'OpenAI',
            'gpt_model': 'model',
            'gpt_token': 'token',
            'ai_resegment_provider': 'custom',
            'ai_resegment_address': 'http://unused.example',
            'ai_resegment_model': 'model',
            'ai_resegment_token': 'token',
        })
        worker = MainWorker(
            snapshot,
            SimpleNamespace(put=lambda target, text: entries.append((target, text))),
            CancellationToken(),
        )
        rows = [{"start": 0.0, "end": 1.0, "message": "原文"}]
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'input.json')
            with open(path, 'w', encoding='utf-8') as stream:
                import json
                json.dump(rows, stream, ensure_ascii=False)
            with patch('worker.request_openai_compatible', return_value='invalid'):
                self.assertFalse(worker._maybe_refine_json(path))
            with open(path, 'r', encoding='utf-8') as stream:
                self.assertEqual(json.load(stream), rows)
        statuses = [text for target, text in entries if target == 'status']
        self.assertTrue(any('已保留原断句' in text for text in statuses))

    def test_auxiliary_profiles_can_follow_or_override_main_translation(self):
        snapshot = TaskSnapshot('run', {
            'translator': 'OpenAI',
            'gpt_address': '',
            'gpt_model': 'gpt-main',
            'gpt_token': 'main-key',
            'ai_resegment_provider': 'follow',
            'proofread_provider': 'custom',
            'proofread_address': 'https://proofread.example/v1',
            'proofread_model': 'proofreader-v2',
            'proofread_token': 'proofread-key',
        })
        worker = MainWorker(snapshot, SimpleNamespace(put=lambda *_args: None), CancellationToken())

        resegment = worker._resolve_online_profile('ai_resegment')
        self.assertTrue(resegment['follows_main'])
        self.assertEqual(resegment['provider'], 'OpenAI')
        self.assertEqual(resegment['model'], 'gpt-main')
        self.assertEqual(resegment['token'], 'main-key')

        proofread = worker._resolve_online_profile('proofread')
        self.assertFalse(proofread['follows_main'])
        self.assertEqual(proofread['provider'], 'custom')
        self.assertEqual(proofread['endpoint'], 'https://proofread.example/v1')
        self.assertEqual(proofread['model'], 'proofreader-v2')
        self.assertEqual(proofread['token'], 'proofread-key')

    def test_proofread_model_can_override_translation_model(self):
        token = SimpleNamespace(model_name='deepseek-v4-flash')
        self.assertEqual(
            BaseTranslate._request_model_name(token, NOT_GIVEN),
            'deepseek-v4-flash',
        )
        self.assertEqual(
            BaseTranslate._request_model_name(token, 'deepseek-v4-pro'),
            'deepseek-v4-pro',
        )

    def test_opencode_responses_stream_is_normalized_for_translation(self):
        class FakeStream:
            def __init__(self):
                self.events = iter((
                    SimpleNamespace(
                        type="response.output_text.delta",
                        delta="translated text",
                    ),
                    SimpleNamespace(type="response.completed"),
                ))

            def __aiter__(self):
                return self

            async def __anext__(self):
                try:
                    return next(self.events)
                except StopIteration:
                    raise StopAsyncIteration

            async def aclose(self):
                return None

        responses_create = AsyncMock(return_value=FakeStream())
        fake_client = SimpleNamespace(
            responses=SimpleNamespace(create=responses_create),
        )
        token = SimpleNamespace(
            domain="https://opencode.ai/zen/v1",
            model_name="gpt-5.6-sol",
            stream=True,
            maskToken=lambda: "test-token",
        )
        translator = BaseTranslate.__new__(BaseTranslate)
        translator.client_list = [(fake_client, token)]
        translator.proofread_client_list = []
        translator.thinking_mode = "auto"
        translator.proofread_thinking_mode = "auto"
        translator.tokenStrategy = "random"
        translator.api_timeout = 10
        translator.apiErrorWait = 0
        translator.global_request_rpm = 0
        translator.request_health_metrics = SimpleNamespace(record=lambda *_args: None)
        translator.pj_config = SimpleNamespace(
            stop_event=threading.Event(),
            active_workers=2,
            bar=SimpleNamespace(text=lambda *_args: None),
        )

        result, used_token = asyncio.run(translator.ask_chatbot(
            prompt="source",
            system="translate",
            stream=True,
            max_tokens=2048,
        ))

        self.assertEqual(result, "translated text")
        self.assertIs(used_token, token)
        kwargs = responses_create.await_args.kwargs
        self.assertEqual(kwargs["model"], "gpt-5.6-sol")
        self.assertEqual(kwargs["instructions"], "translate")
        self.assertEqual(kwargs["input"], [{"role": "user", "content": "source"}])
        self.assertEqual(kwargs["max_output_tokens"], 2048)

    def test_translation_config_writes_independent_proofread_profile(self):
        snapshot = TaskSnapshot('run', {
            'translator': 'OpenAI',
            'language': 'ja',
            'target_lang': 'zh-cn',
            'gpt_model': 'deepseek-v4-flash',
            'gpt_token': 'main-key',
            'proofread_provider': 'custom',
            'proofread_address': 'https://proofread.example/v1',
            'proofread_model': 'deepseek-v4-pro',
            'proofread_token': 'proofread-key',
            'enable_proofread': True,
            'deepseek_thinking': False,
            'proofread_thinking': True,
        })
        messages = SimpleNamespace(put=lambda *_args: None)
        worker = MainWorker(snapshot, messages, CancellationToken())
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                worker.update_translation_config()
                import yaml
                with open('project/config.yaml', 'r', encoding='utf-8') as stream:
                    config = yaml.safe_load(stream)
            finally:
                os.chdir(original_cwd)

        backend = config['backendSpecific']['OpenAI-Compatible']
        self.assertEqual(backend['tokens'][0]['modelName'], 'deepseek-v4-flash')
        self.assertEqual(backend['proofreadModelName'], 'deepseek-v4-pro')
        self.assertEqual(
            backend['proofreadEndpoint'], 'https://proofread.example/v1'
        )
        self.assertEqual(backend['proofreadToken'], 'proofread-key')
        self.assertEqual(backend['thinkingMode'], 'disabled')
        self.assertEqual(backend['proofreadThinkingMode'], 'enabled')

    def test_worker_uses_snapshot_and_finishes_once(self):
        token = CancellationToken()
        with tempfile.TemporaryDirectory() as directory:
            worker = MainWorker(
                TaskSnapshot("test_online_api", {"translator": ""}),
                UIMessageQueue(log_path=f"{directory}/worker.log"),
                token,
            )
            finished = []
            worker.finished.connect(lambda: finished.append(True))
            worker.test_online_api()
            worker._finalize_task()
            self.assertFalse(hasattr(worker, "master"))
            self.assertEqual(finished, [True])

    def test_cancelled_process_exits_without_blocking_caller(self):
        token = CancellationToken()
        registry = ProcessRegistry(token)
        proc = registry.popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        outcome = []

        def wait_process():
            try:
                registry.wait(proc)
            except TaskCancelledError:
                outcome.append("cancelled")

        waiter = threading.Thread(target=wait_process)
        waiter.start()
        started = time.monotonic()
        token.cancel()
        waiter.join(timeout=3)
        self.assertFalse(waiter.is_alive())
        self.assertEqual(outcome, ["cancelled"])
        self.assertIsNotNone(proc.poll())
        self.assertLess(time.monotonic() - started, 3)

    def test_translation_pool_can_wait_between_input_files(self):
        stop_event = threading.Event()
        with tempfile.TemporaryDirectory() as directory:
            messages = UIMessageQueue(log_path=f"{directory}/pool.log")
            progress = []
            completions = []
            pool = ConcurrentTranslationPool(
                project_dir=directory,
                base_config_path=f"{directory}/config.yaml",
                max_concurrent=2,
                stop_event=stop_event,
                msg_queue=messages,
                progress_callback=lambda current, total: progress.append(
                    (current, total)
                ),
                file_completion_callback=lambda file_id, outcome, final:
                    completions.append((file_id, outcome, final)),
            )
            processed = []
            original = ConcurrentTranslationPool._translate_one_impl

            def fake_translate(tf_dict, *_args, **_kwargs):
                processed.append(tf_dict['base_path'])

            ConcurrentTranslationPool._translate_one_impl = staticmethod(fake_translate)
            try:
                pool.start('fake')
                first = TranscribedFile(
                    'first', 'first.json', directory, '原文SRT', '', 'source-1', False
                )
                second = TranscribedFile(
                    'second', 'second.json', directory, '原文SRT', '', 'source-2', True
                )
                pool.submit(first)
                pool.wait_for_pending()
                self.assertTrue(any(thread.is_alive() for thread in pool._active_threads))
                pool.submit(second)
                pool.wait_for_pending()
                pool.done()
                pool.wait_all()
            finally:
                ConcurrentTranslationPool._translate_one_impl = staticmethod(original)

            self.assertCountEqual(processed, ['first', 'second'])
            self.assertEqual(progress[-1], (2, 2))
            self.assertCountEqual(completions, [
                ('source-1', 'success', False),
                ('source-2', 'success', True),
            ])

    def test_translation_subprocess_reports_live_sentence_progress(self):
        class FakeProcess:
            def __init__(self):
                self.stdout = io.StringIO(
                    'v--1\n> Src: one\n> Dst: 一\n'
                    'v--2\n> Src: two\n> Dst: 二\n'
                )

            def poll(self):
                return 0

        class FakeRegistry:
            def popen(self, *_args, **_kwargs):
                return FakeProcess()

            def wait(self, _proc):
                return 0

            def terminate(self, _proc):
                return None

        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.json')
            config = os.path.join(directory, 'config.yaml')
            with open(source, 'w', encoding='utf-8') as stream:
                json.dump([
                    {'message': 'one'},
                    {'message': 'two'},
                ], stream)
            with open(config, 'w', encoding='utf-8') as stream:
                stream.write('common:\n  gpt.enableProofRead: false\n')
            progress = []
            tf_dict = {
                'base_path': os.path.join(directory, 'input'),
                'json_src': source,
                'output_dir': directory,
                'output_format': '原文SRT',
                'orig_srt_path': '',
                'source_file_id': 'source',
            }
            messages = SimpleNamespace(put=lambda *_args: None)
            with patch.object(
                ConcurrentTranslationPool,
                '_generate_output_impl',
                return_value=None,
            ):
                ConcurrentTranslationPool._translate_one_impl(
                    tf_dict,
                    0,
                    directory,
                    config,
                    'fake',
                    messages,
                    threading.Event(),
                    FakeRegistry(),
                    lambda _tf, current, total: progress.append((current, total)),
                )
            self.assertEqual(progress[0], (0, 2))
            self.assertIn((1, 2), progress)
            self.assertEqual(progress[-1], (2, 2))


if __name__ == "__main__":
    unittest.main()
