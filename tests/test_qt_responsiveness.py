import os
import threading
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QObject, QTimer, Signal, Slot
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
            self.assertEqual(window.top_tabs.count(), 7)
            self.assertEqual(
                window.top_tabs.tabText(5), app._("tab_phone")
            )
            self.assertTrue(hasattr(window, "media_library_scan_button"))
            window.streaming_checkbox.setEnabled(True)
            window.streaming_checkbox.setChecked(True)
            window.ai_resegment_checkbox.setChecked(True)
            self.assertTrue(window.ai_resegment_checkbox.isChecked())
            self.assertFalse(window.streaming_checkbox.isChecked())

            window.gpt_model.setText('deepseek-v4-flash')
            for provider in ('Deepseek', 'OpenAI', 'Kimi'):
                self.assertGreaterEqual(
                    window.ai_resegment_provider_combo.findData(provider), 0
                )
                self.assertGreaterEqual(
                    window.proofread_provider_combo.findData(provider), 0
                )
            window._set_provider_value(
                window.ai_resegment_provider_combo, 'Deepseek'
            )
            window._set_auxiliary_model_value(
                window.ai_resegment_model_combo, 'deepseek-v4-pro'
            )
            self.assertEqual(
                window.ai_resegment_provider_combo.currentData(),
                'Deepseek',
            )
            self.assertEqual(
                window.ai_resegment_model_combo.currentData(),
                'deepseek-v4-pro',
            )
            window._set_provider_value(window.ai_resegment_provider_combo, 'OpenAI')
            self.assertEqual(
                window.ai_resegment_model_combo.findData('deepseek-v4-flash'), -1
            )
            self.assertFalse(window.ai_resegment_thinking_checkbox.isEnabled())
            window._set_provider_value(window.ai_resegment_provider_combo, 'Deepseek')
            window.ai_resegment_token.setText('resegment-key')
            window._set_provider_value(
                window.proofread_provider_combo, 'custom'
            )
            window._set_auxiliary_model_value(
                window.proofread_model_combo, 'custom-proofreader'
            )
            self.assertEqual(
                window.proofread_provider_combo.currentData(), 'custom'
            )
            self.assertTrue(window.proofread_model_combo.isEditable())
            window.proofread_address.setText('https://proofread.example/v1')
            window.proofread_token.setText('proofread-key')
            window.deepseek_thinking_checkbox.setChecked(True)
            window.ai_resegment_thinking_checkbox.setChecked(False)
            window.proofread_thinking_checkbox.setChecked(True)
            snapshot = window._capture_task_snapshot('run')
            self.assertEqual(snapshot.get('gpt_model'), 'deepseek-v4-flash')
            self.assertEqual(snapshot.get('ai_resegment_provider'), 'Deepseek')
            self.assertEqual(snapshot.get('ai_resegment_model'), 'deepseek-v4-pro')
            self.assertEqual(snapshot.get('ai_resegment_token'), 'resegment-key')
            self.assertEqual(snapshot.get('proofread_provider'), 'custom')
            self.assertEqual(snapshot.get('proofread_model'), 'custom-proofreader')
            self.assertEqual(
                snapshot.get('proofread_address'),
                'https://proofread.example/v1',
            )
            self.assertEqual(snapshot.get('proofread_token'), 'proofread-key')
            self.assertTrue(snapshot.get('deepseek_thinking'))
            self.assertFalse(snapshot.get('ai_resegment_thinking'))
            self.assertTrue(snapshot.get('proofread_thinking'))
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

    def test_model_list_loading_does_not_open_hidden_modal_dialog(self):
        window = self._window()
        try:
            models = [f'model-{index}' for index in range(500)]
            window._set_provider_value(
                window.ai_resegment_provider_combo, 'custom'
            )
            started = time.monotonic()
            window._handle_model_list_loaded(models, 'resegment')
            self.assertLess(time.monotonic() - started, 0.25)
            self.assertGreaterEqual(
                window.ai_resegment_model_combo.findData('model-499'), 0
            )
            self.assertIsNone(
                getattr(window, '_model_selection_dialog', None)
            )

            started = time.monotonic()
            window.show_model_selection_dialog(['main-model'])
            self.assertLess(time.monotonic() - started, 0.25)
            dialog = window._model_selection_dialog
            self.assertIsNotNone(dialog)
            self.assertTrue(dialog.isVisible())
            dialog.close()
            self.qt_app.processEvents()
        finally:
            window.close()
            self.qt_app.processEvents()

    def test_received_character_counter_combines_and_corrects_requests(self):
        window = self._window()
        try:
            window._set_progress_context('character test')
            window._update_received_characters(
                '{"request":"a","characters":10,"final":false}'
            )
            window._update_received_characters(
                '{"request":"b","characters":5,"final":true}'
            )
            self.assertIn('15', window.shared_character_label.text())
            window._update_received_characters(
                '{"request":"a","characters":12,"final":true}'
            )
            self.assertIn('17', window.shared_character_label.text())
        finally:
            window.close()
            self.qt_app.processEvents()

    def test_progress_bar_shows_measured_phase_progress(self):
        window = self._window()
        try:
            window._set_progress_context('progress test')
            self.assertEqual(window.shared_progress_bar.maximum(), 100)
            self.assertEqual(window.shared_progress_bar.value(), 0)
            window._update_task_progress(
                '{"kind":"files","completed":1,"total":3}'
            )
            self.assertFalse(window.shared_file_label.isHidden())
            self.assertIn('1/3', window.shared_file_label.text())
            self.assertEqual(window.shared_progress_bar.value(), 0)
            window._update_task_progress(
                '{"kind":"files","completed":0,"total":3}'
            )
            self.assertIn('1/3', window.shared_file_label.text())
            window._update_task_progress(
                '{"kind":"stage","stage_id":1,"current":25,"total":100,"phase":"AI 断句"}'
            )
            self.assertEqual(window.shared_progress_bar.value(), 25)
            self.assertIn('25/100', window.shared_progress_bar.format())
            self.assertIn('AI 断句', window.shared_progress_bar.format())
            # Concurrent callbacks for one phase must not move backwards.
            window._update_task_progress(
                '{"kind":"stage","stage_id":1,"current":20,"total":100,"phase":"AI 断句"}'
            )
            self.assertEqual(window.shared_progress_bar.value(), 25)
            # The same display name on the next file has a new identity and
            # must visibly restart from zero.
            window._update_task_progress(
                '{"kind":"stage","stage_id":2,"current":0,"total":100,"phase":"AI 断句"}'
            )
            self.assertEqual(window.shared_progress_bar.value(), 0)
            # Delayed updates from stage 1 cannot overwrite stage 2.
            window._update_task_progress(
                '{"kind":"stage","stage_id":1,"current":90,"total":100,"phase":"AI 断句"}'
            )
            self.assertEqual(window.shared_progress_bar.value(), 0)
            window._update_task_progress(
                '{"kind":"stage","stage_id":3,"current":1,"total":4,"phase":"翻译与校对"}'
            )
            self.assertEqual(window.shared_progress_bar.maximum(), 4)
            self.assertEqual(window.shared_progress_bar.value(), 1)
            window._on_task_outcome('cancelled')
            window._on_task_finished()
            self.assertEqual(window.shared_progress_bar.value(), 1)
            self.assertEqual(
                window.shared_state_label.text(), app._("task_state_cancelled")
            )
        finally:
            window.close()
            self.qt_app.processEvents()

    def test_task_finish_does_not_fabricate_stage_or_file_completion(self):
        window = self._window()
        try:
            window._set_progress_context('truthful finish test')
            window._update_task_progress(
                '{"kind":"files","completed":1,"total":2}'
            )
            window._update_task_progress(
                '{"kind":"stage","stage_id":1,"current":2,"total":5,"phase":"翻译与校对"}'
            )
            window._on_task_outcome('error')
            window._on_task_finished()
            self.assertEqual(window.shared_progress_bar.value(), 2)
            self.assertIn('1/2', window.shared_file_label.text())
            self.assertEqual(
                window.shared_state_label.text(), app._("task_state_failed")
            )
        finally:
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
                self.started = threading.Event()
                self.run_thread_id = None

            @Slot()
            def execute(self):
                self.run()

            def run(self):
                self.run_thread_id = threading.get_ident()
                self.started.set()
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
            worker = window.worker
            self.assertTrue(worker.started.wait(1))
            self.assertNotEqual(worker.run_thread_id, threading.get_ident())
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
