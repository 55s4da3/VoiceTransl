import os
import threading
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal
from PySide6.QtWidgets import QApplication

import app


class QtResponsivenessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.qt_app = QApplication.instance() or QApplication([])

    def _window(self):
        window = app.MainWindow()
        window.save_config = lambda *args, **kwargs: None
        return window

    def test_window_and_resegment_streaming_interlock(self):
        window = self._window()
        try:
            self.assertEqual(window.top_tabs.count(), 6)
            window.streaming_checkbox.setEnabled(True)
            window.streaming_checkbox.setChecked(True)
            window.ai_resegment_checkbox.setChecked(True)
            self.assertTrue(window.ai_resegment_checkbox.isChecked())
            self.assertFalse(window.streaming_checkbox.isChecked())
        finally:
            window.close()
            self.qt_app.processEvents()

    def test_large_log_burst_keeps_event_loop_alive(self):
        window = self._window()
        ticks = []
        timer = QTimer()
        timer.setInterval(20)
        timer.timeout.connect(lambda: ticks.append(time.monotonic()))
        loop = QEventLoop()
        try:
            for index in range(10000):
                window.msg_queue.put("detail", f"[INFO] line {index}")
            timer.start()
            QTimer.singleShot(500, loop.quit)
            loop.exec()
            self.assertGreaterEqual(len(ticks), 15)
            if len(ticks) > 1:
                self.assertLess(max(b - a for a, b in zip(ticks, ticks[1:])), 0.25)
        finally:
            timer.stop()
            window.close()
            self.qt_app.processEvents()

    def test_config_persistence_does_not_block_ui_thread(self):
        window = self._window()
        started = threading.Event()
        completed = threading.Event()

        def slow_writer(*_args):
            started.set()
            time.sleep(0.4)
            completed.set()

        window._write_config_snapshot = slow_writer
        try:
            start = time.monotonic()
            writer = app.MainWindow.save_config(window, silent=True)
            elapsed = time.monotonic() - start
            self.assertLess(elapsed, 0.25)
            self.assertTrue(started.wait(1))
            self.assertTrue(completed.wait(1))
            writer.join(timeout=1)
        finally:
            window.save_config = lambda *args, **kwargs: None
            window.close()
            self.qt_app.processEvents()

    def test_close_cancels_running_worker_without_blocking(self):
        class FakeWorker(QObject):
            finished = Signal()
            status = Signal(str)
            show_model_dialog = Signal(list)

            def __init__(self, _snapshot, _messages, token):
                super().__init__()
                self.token = token

            def run(self):
                while not self.token.is_cancelled():
                    self.token.event.wait(0.02)
                self.finished.emit()

        original_worker = app.MainWorker
        app.MainWorker = FakeWorker
        window = self._window()
        heartbeat = QTimer()
        heartbeat.setInterval(20)
        ticks = []
        heartbeat.timeout.connect(lambda: ticks.append(time.monotonic()))
        loop = QEventLoop()
        watcher = QTimer()
        watcher.setInterval(20)
        watcher.timeout.connect(lambda: loop.quit() if window.thread is None else None)
        try:
            window._start_worker_task('run', 'fake task')
            self.qt_app.processEvents()
            self.assertIsNotNone(window.thread)
            self.assertTrue(window.thread.isRunning())
            heartbeat.start()
            watcher.start()
            started = time.monotonic()
            window.close()
            self.assertLess(time.monotonic() - started, 0.25)
            QTimer.singleShot(1500, loop.quit)
            loop.exec()
            self.assertIsNone(window.thread)
            self.assertGreaterEqual(len(ticks), 1)
        finally:
            heartbeat.stop()
            watcher.stop()
            app.MainWorker = original_worker
            if window.thread is None:
                window.close()
            self.qt_app.processEvents()

    def test_no_unsafe_worker_ui_or_qthread_termination(self):
        worker_source = Path("worker.py").read_text(encoding="utf-8")
        app_source = Path("app.py").read_text(encoding="utf-8")
        self.assertNotIn("self.master", worker_source)
        self.assertNotIn("thread.terminate", app_source)
        self.assertNotIn("thread.wait", app_source)


if __name__ == "__main__":
    unittest.main()
