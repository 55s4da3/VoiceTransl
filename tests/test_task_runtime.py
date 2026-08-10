import os
import subprocess
import sys
import threading
import time
import tempfile
import unittest
from types import SimpleNamespace

from tasking import CancellationToken, ProcessRegistry, TaskCancelledError
from tasking import TaskSnapshot
from log import UIMessageQueue
from pool import ConcurrentTranslationPool, TranscribedFile
from worker import MainWorker
from GalTransl.Backend.BaseTranslate import BaseTranslate
from openai._types import NOT_GIVEN


class TaskRuntimeTests(unittest.TestCase):
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
            pool = ConcurrentTranslationPool(
                project_dir=directory,
                base_config_path=f"{directory}/config.yaml",
                max_concurrent=2,
                stop_event=stop_event,
                msg_queue=messages,
            )
            processed = []
            original = ConcurrentTranslationPool._translate_one_impl

            def fake_translate(tf_dict, *_args, **_kwargs):
                processed.append(tf_dict['base_path'])

            ConcurrentTranslationPool._translate_one_impl = staticmethod(fake_translate)
            try:
                pool.start('fake')
                first = TranscribedFile('first', 'first.json', directory, '原文SRT', '')
                second = TranscribedFile('second', 'second.json', directory, '原文SRT', '')
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


if __name__ == "__main__":
    unittest.main()
