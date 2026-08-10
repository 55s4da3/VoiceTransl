import subprocess
import sys
import threading
import time
import tempfile
import unittest

from tasking import CancellationToken, ProcessRegistry, TaskCancelledError
from tasking import TaskSnapshot
from log import UIMessageQueue
from pool import ConcurrentTranslationPool, TranscribedFile
from worker import MainWorker


class TaskRuntimeTests(unittest.TestCase):
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
