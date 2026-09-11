import sys, os
import json
import hashlib
import re
import shutil
import socket
import threading
import yaml
from pathlib import Path

import asrlabs_bridge
import crispasr_bridge

from core import (
    DEFAULT_CRISPASR_BACKEND,
    LOG_PATH,
    NO_TRANSCRIPTION,
    NO_TRANSLATION,
    ONLINE_TRANSLATOR_MAPPING,
    model_supports_thinking,
    TRANSLATOR_SUPPORTED,
    _compose_output_format,
    _load_api_key,
    _save_api_key,
)
from asr import _list_crispasr_aligners, _list_crispasr_backends, _list_crispasr_models
from log import UIMessageQueue, _line_passes_filter
from pool import ConcurrentTranslationPool
from worker import MainWorker
from tasking import CancellationToken, TaskSnapshot
from lan_service import DEFAULT_HTTP_PORT, LanService, build_artifact
from media_library import MediaLibraryDatabase, MediaLibraryScanner, write_preview
from i18n import _, set_language, get_language
from PySide6 import QtGui, QtCore
from PySide6.QtCore import QThread, Signal, QTimer
from PySide6.QtGui import QAction, QPixmap
from PySide6.QtWidgets import (
    QApplication, QVBoxLayout, QFileDialog, QFrame, QSystemTrayIcon, QMenu,
    QHBoxLayout, QCheckBox, QDialog, QLabel, QWidget, QGridLayout,
    QScrollArea, QProgressBar, QSizePolicy, QMainWindow, QTabWidget,
    QPushButton, QTextEdit, QLineEdit, QComboBox, QPlainTextEdit, QSpinBox,
)
from qt_material import apply_stylesheet


DEFAULT_UI_THEME = 'light_blue.xml'
GUI_SETTINGS_SCHEMA_VERSION = 2
GUI_SETTINGS_PATH = Path('gui_settings.yaml')
CRISPASR_DIR = crispasr_bridge.DEFAULT_CRISPASR_DIR
DICTIONARY_PRESET_DIR = Path('project') / 'dictionary_presets'
UI_THEME_OPTIONS = (
    ('theme_light_blue', 'light_blue.xml'),
    ('theme_light_teal', 'light_teal.xml'),
    ('theme_light_purple', 'light_purple.xml'),
    ('theme_dark_blue', 'dark_blue.xml'),
    ('theme_dark_teal', 'dark_teal.xml'),
    ('theme_dark_purple', 'dark_purple.xml'),
)

MATERIAL_OVERRIDES = """
QWidget {
    font-family: "Segoe UI", "Microsoft YaHei UI", sans-serif;
}
QPushButton {
    text-transform: none;
    font-weight: 600;
    border-radius: 6px;
    min-height: 28px;
}
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox {
    border-radius: 6px;
}
QTabBar::tab {
    text-transform: none;
    font-size: 10pt;
    font-weight: 600;
}
QProgressBar {
    border-radius: 3px;
}
QToolTip {
    padding: 6px;
    border-radius: 4px;
}
"""


def open_path(path_value: str):
    target = os.path.abspath(path_value)
    QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(target))

class BodyLabel(QLabel):
    """Native Qt replacement for the former Fluent body label."""


class SubtitleLabel(QLabel):
    """Small section heading implemented with a native QLabel."""

    def __init__(self, text: str = '', parent=None):
        super().__init__(text, parent)
        font = self.font()
        font.setPointSize(max(font.pointSize() + 2, 11))
        font.setBold(True)
        self.setFont(font)


class TitleLabel(QLabel):
    """Page heading implemented with a native QLabel."""

    def __init__(self, text: str = '', parent=None):
        super().__init__(text, parent)
        font = self.font()
        font.setPointSize(max(font.pointSize() + 6, 16))
        font.setBold(True)
        self.setFont(font)


class ScaledPixmapLabel(QLabel):
    """Full-width label that scales a pixmap to fit the widget width."""

    def __init__(self, pixmap=None, parent=None):
        super().__init__(parent)
        self._source_pixmap = pixmap
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(100)

    def setPixmap(self, pixmap):
        self._source_pixmap = pixmap
        self._update_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self):
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        width = max(self.width(), 1)
        scaled = self._source_pixmap.scaled(
            width, self.height(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        super().setPixmap(scaled)


class Widget(QFrame):

    def __init__(self, text: str, parent=None):
        super().__init__(parent=parent)
        # Set the scroll area as the parent of the widget
        self.vBoxLayout = QVBoxLayout(self)

        # Must set a globally unique object name for the sub-interface
        self.setObjectName(text.replace(' ', '-'))


def _load_ui_theme() -> str:
    try:
        if os.path.exists('gui_settings.yaml'):
            with open('gui_settings.yaml', 'r', encoding='utf-8') as f:
                saved_theme = (yaml.safe_load(f) or {}).get('ui_theme')
            available = {theme for _label, theme in UI_THEME_OPTIONS}
            if saved_theme in available:
                return saved_theme
    except Exception:
        pass
    return DEFAULT_UI_THEME


def _prepare_gui_settings_schema() -> Path | None:
    """Back up pre-PySide settings once and start with the v2 defaults."""
    if not GUI_SETTINGS_PATH.is_file():
        return None
    try:
        with GUI_SETTINGS_PATH.open('r', encoding='utf-8') as stream:
            settings = yaml.safe_load(stream) or {}
    except Exception:
        settings = {}
    if settings.get('schema_version') == GUI_SETTINGS_SCHEMA_VERSION:
        return None

    backup = GUI_SETTINGS_PATH.with_name('gui_settings.legacy.yaml')
    suffix = 1
    while backup.exists():
        backup = GUI_SETTINGS_PATH.with_name(f'gui_settings.legacy.{suffix}.yaml')
        suffix += 1
    GUI_SETTINGS_PATH.replace(backup)
    return backup


def apply_material_theme(application: QApplication, theme: str) -> None:
    """Apply a compact Material theme plus app-specific readability tweaks."""
    available = {value for _label, value in UI_THEME_OPTIONS}
    selected = theme if theme in available else DEFAULT_UI_THEME
    apply_stylesheet(
        application,
        theme=selected,
        invert_secondary=selected.startswith('light_'),
        extra={
            'density_scale': '-2',
            'font_family': (
                'Microsoft YaHei UI' if os.name == 'nt' else 'Noto Sans'
            ),
            'danger': '#d32f2f',
            'warning': '#ed6c02',
            'success': '#2e7d32',
        },
    )
    application.setStyleSheet(application.styleSheet() + MATERIAL_OVERRIDES)
    application.setProperty('ui_material_theme', selected)


class MainWindow(QMainWindow):
    status = Signal(str)
    config_write_status = Signal(str)
    lan_job_ready = Signal(str)
    lan_cancel_requested = Signal(str)
    media_scan_finished = Signal(object, object)

    @staticmethod
    def default_output_dir() -> str:
        return str(Path.cwd() / 'project' / 'cache')

    def __init__(self):
        super().__init__()
        self._legacy_settings_backup = _prepare_gui_settings_schema()
        application = QApplication.instance()
        saved_theme = _load_ui_theme()
        if application and application.property('ui_material_theme') != saved_theme:
            apply_material_theme(application, saved_theme)
        self.msg_queue = UIMessageQueue(LOG_PATH)
        self.thread = None
        self.worker = None
        self.cancel_token = None
        self._pending_close = False
        self._active_task_name = _('task_none')
        self._drop_targets = {}
        self._suppress_auto_save = True
        self._config_write_lock = threading.Lock()
        self._config_write_generation = 0
        self.media_library = MediaLibraryDatabase(
            Path('project') / 'cache' / 'media_library.sqlite3'
        )
        self.lan_service = None
        self._lan_profile_lock = threading.RLock()
        self._lan_profile_cache = {"public": {}, "snapshot": {}}
        self._lan_job_queue = []
        self._active_lan_job_id = None
        self._lan_task_outcomes = {}
        self._lan_shutdown_started = False
        self._media_scan_thread = None
        self.lan_job_ready.connect(self._on_lan_job_ready)
        self.lan_cancel_requested.connect(self._on_lan_cancel_requested)
        self.media_scan_finished.connect(self._on_media_scan_finished)
        self._auto_save_timer = QTimer(self)
        self._auto_save_timer.setSingleShot(True)
        self._auto_save_timer.setInterval(200)
        self._auto_save_timer.timeout.connect(self._auto_save_config)
        self._load_ui_language()
        self.setWindowTitle(_("window_title"))
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.init_system_tray()
        self.status.connect(lambda x: self.setWindowTitle(f"{_('window_title')} - {x}"))
        self.config_write_status.connect(self._emit_status)
        self.resize(1180, 760)
        self.setMinimumSize(960, 640)
        self.show()
        self.initUI()
        self._log_level_filter = 'ALL'  # 日志级别过滤默认值
        self.setup_timer()
        self._initialize_lan_service()
        self._initialize_media_library_ui()

    def _load_ui_language(self):
        """从 gui_settings.yaml 加载已保存的界面语言，在任何 _() 调用之前执行"""
        try:
            if os.path.exists('gui_settings.yaml'):
                with open('gui_settings.yaml', 'r', encoding='utf-8') as f:
                    settings = yaml.safe_load(f) or {}
                saved_lang = settings.get('ui_language')
                if saved_lang and saved_lang in ('zh', 'en', 'ja'):
                    set_language(saved_lang)
        except Exception:
            pass

    def _emit_status(self, msg: str):
        """同时向统一消息队列和窗口标题发送状态消息"""
        self.msg_queue.put("status", msg)
        self.status.emit(msg)

    def _schedule_auto_save(self):
        """防抖自动保存：短时间内多次调用只执行最后一次"""
        if self._auto_save_timer and not self._auto_save_timer.isActive():
            self._auto_save_timer.start()

    def _auto_save_config(self):
        """执行静默自动保存"""
        try:
            self.save_config(silent=True)
            self._refresh_lan_profile_cache()
        except Exception:
            pass

    def selected_output_format(self, translation_enabled=None) -> str:
        """Compose the legacy output value consumed by the processing pipeline."""
        if translation_enabled is None:
            translation_enabled = self.enable_translation_checkbox.isChecked()
        content = self.output_content.currentData() or '双语'
        container = self.output_container.currentData() or 'SRT'
        return _compose_output_format(content, container, translation_enabled)

    def save_config(self, silent: bool = False):
        """Capture widget values now and persist them outside the Qt thread."""
        if not silent:
            self._emit_status(_("status_reading_config"))
        asr_provider = self.asr_provider_combo.currentData() or 'crispasr'
        translator = self.translator_group.currentText()
        language = self.transcription_lang.currentData() or self.transcription_lang.currentText()
        gpt_token = self.gpt_token.text()
        gpt_address = self.gpt_address.text()
        gpt_model = self.gpt_model.text()
        ai_resegment_model = self._auxiliary_model_value(
            self.ai_resegment_model_combo
        )
        proofread_model = self._auxiliary_model_value(self.proofread_model_combo)
        api_tokens = {
            'VOICETRANSL_API_KEY': gpt_token,
            'VOICETRANSL_RESEGMENT_API_KEY': self.ai_resegment_token.text(),
            'VOICETRANSL_PROOFREAD_API_KEY': self.proofread_token.text(),
        }
        sakura_file = self.sakura_file.currentText()
        sakura_mode = self.sakura_mode.text()
        proxy_address = self.proxy_address.text()
        uvr_file = self.uvr_file.currentText()
        output_content = self.output_content.currentData()
        output_container = self.output_container.currentData()
        output_format = self.selected_output_format()
        subtitle_font = self.subtitle_font_combo.currentText()
        output_dir = self.output_dir_edit.text().strip() or self.default_output_dir()
        use_input_dir = self.use_input_dir_checkbox.isChecked()
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        enable_segment = self.enable_segment_checkbox.isChecked()
        segment_duration = self.segment_duration_spin.value()
        change_prompt_mode = self.change_prompt_mode.currentData() if hasattr(self, 'change_prompt_mode') else '不修改'
        auto_shutdown = self.auto_shutdown_checkbox.isChecked() if hasattr(self, 'auto_shutdown_checkbox') else False
        target_translation_lang = self.target_lang.currentData() if hasattr(self, 'target_lang') else 'zh-cn'
        ui_theme = self.theme_selector.currentData() if hasattr(self, 'theme_selector') else _load_ui_theme()
        current_lang = get_language()

        gui_settings = {
            'schema_version': GUI_SETTINGS_SCHEMA_VERSION,
            'asr_provider': asr_provider,
            'asr_engine': self.asr_engine_combo.currentData() or '',
            'asr_model': self.asr_model_combo.currentData() or '',
            'asr_device': self.asr_device_combo.currentText(),
            'asr_compute_type': self.asr_compute_type_combo.currentText(),
            'asr_extra': self.asr_extra_edit.toPlainText(),
            'align_engine': self.align_engine_combo.currentData() or 'none',
            'align_model': self.align_model_combo.currentData() or '',
            'align_device': self.align_device_combo.currentText(),
            'align_extra': self.align_extra_edit.toPlainText(),
            'crispasr_backend': self.crispasr_backend_combo.currentText(),
            'crispasr_model': self.crispasr_model_combo.currentText(),
            'crispasr_aligner': self.crispasr_aligner_combo.currentText(),
            'translator': translator,
            'language': language,
            'gpt_address': gpt_address,
            'gpt_model': gpt_model,
            'ai_resegment_model': ai_resegment_model,
            'ai_resegment_provider': self.ai_resegment_provider_combo.currentData() or 'follow',
            'ai_resegment_address': self.ai_resegment_address.text().strip(),
            'proofread_model': proofread_model,
            'proofread_provider': self.proofread_provider_combo.currentData() or 'follow',
            'proofread_address': self.proofread_address.text().strip(),
            'deepseek_thinking': self.deepseek_thinking_checkbox.isChecked(),
            'ai_resegment_thinking': self.ai_resegment_thinking_checkbox.isChecked(),
            'proofread_thinking': self.proofread_thinking_checkbox.isChecked(),
            'sakura_file': sakura_file,
            'sakura_mode': sakura_mode,
            'proxy_address': proxy_address,
            'uvr_file': uvr_file,
            'enable_transcription': self.enable_transcription_checkbox.isChecked(),
            'enable_translation': self.enable_translation_checkbox.isChecked(),
            'output_content': output_content,
            'output_container': output_container,
            'output_format': output_format,
            'subtitle_font': subtitle_font,
            'output_dir': output_dir,
            'use_input_dir': use_input_dir,
            'max_concurrent': self.max_concurrent_spin.value(),
            'enable_segment': enable_segment,
            'segment_duration': segment_duration,
            'enable_streaming': self.streaming_checkbox.isChecked(),
            'enable_proofread': self.proofread_checkbox.isChecked(),
            'enable_ai_resegment': self.ai_resegment_checkbox.isChecked(),
            'change_prompt_mode': change_prompt_mode,
            'auto_shutdown': auto_shutdown,
            'log_level_filter': self.log_filter_combo.currentText(),
            'verbose_mode': self.verbose_checkbox.isChecked(),
            'ui_language': current_lang,
            'ui_theme': ui_theme,
            'target_translation_lang': target_translation_lang,
            'lan_enabled': bool(
                self.lan_enabled_checkbox.isChecked()
                if hasattr(self, 'lan_enabled_checkbox') else False
            ),
            'lan_auto_start': bool(
                self.lan_auto_start_checkbox.isChecked()
                if hasattr(self, 'lan_auto_start_checkbox') else False
            ),
            'lan_port': int(
                self.lan_port_spin.value()
                if hasattr(self, 'lan_port_spin') else DEFAULT_HTTP_PORT
            ),
            'lan_device_name': (
                self.lan_device_name_edit.text().strip()
                if hasattr(self, 'lan_device_name_edit') else 'VoiceTransl'
            ),
            'lan_allowed_networks': (
                self.lan_allowed_networks_edit.text().strip()
                if hasattr(self, 'lan_allowed_networks_edit') else ''
            ),
            'media_library_root': (
                self.media_library_root_edit.text().strip()
                if hasattr(self, 'media_library_root_edit') else ''
            ),
            'media_library_auto_scan': bool(
                self.media_library_auto_scan_checkbox.isChecked()
                if hasattr(self, 'media_library_auto_scan_checkbox') else False
            ),
        }
        file_contents = {
            'crispasr/param.txt': self.param_crispasr.toPlainText(),
            'llama/param.txt': self.param_llama.toPlainText(),
            'project/dict_pre.txt': self.before_dict.toPlainText(),
            'project/dict_gpt.txt': self.gpt_dict.toPlainText(),
            'project/dict_after.txt': self.after_dict.toPlainText(),
        }
        self._config_write_generation += 1
        generation = self._config_write_generation
        writer = threading.Thread(
            target=self._write_config_snapshot,
            args=(generation, gui_settings, api_tokens, file_contents, output_dir, silent),
            name=f'config-writer-{generation}',
        )
        writer.start()
        return writer

    def _write_config_snapshot(
        self,
        generation: int,
        gui_settings: dict,
        api_tokens: dict[str, str],
        file_contents: dict[str, str],
        output_dir: str,
        silent: bool,
    ):
        """Serialize configuration writes and discard superseded snapshots."""
        try:
            with self._config_write_lock:
                if generation != self._config_write_generation:
                    return
                os.makedirs(output_dir, exist_ok=True)
                settings_temp = GUI_SETTINGS_PATH.with_suffix('.yaml.tmp')
                with settings_temp.open('w', encoding='utf-8') as stream:
                    yaml.dump(
                        gui_settings,
                        stream,
                        allow_unicode=True,
                        sort_keys=False,
                        default_flow_style=False,
                    )
                os.replace(settings_temp, GUI_SETTINGS_PATH)
                for variable_name, api_key in api_tokens.items():
                    _save_api_key(api_key, variable_name)
                for path_value, content in file_contents.items():
                    path = Path(path_value)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    temp_path = path.with_name(path.name + '.tmp')
                    temp_path.write_text(content, encoding='utf-8')
                    os.replace(temp_path, path)
            if not silent:
                self.config_write_status.emit(_("status_config_saved"))
        except Exception as error:
            try:
                self.config_write_status.emit(_("status_generic_error", error=error))
            except RuntimeError:
                pass

    def eventFilter(self, obj, event):
        drop_target = self._drop_targets.get(obj)
        if drop_target is not None:
            if event.type() in (
                QtCore.QEvent.Type.DragEnter,
                QtCore.QEvent.Type.DragMove,
            ):
                if self._normalize_drop_paths(event.mimeData()):
                    event.acceptProposedAction()
                    return True
            elif event.type() == QtCore.QEvent.Type.Drop:
                paths = self._normalize_drop_paths(event.mimeData())
                if paths:
                    drop_target.setPlainText("\n".join(paths))
                    event.acceptProposedAction()
                    return True
        if event.type() == QtCore.QEvent.Type.FocusOut:
            self._schedule_auto_save()
        return super().eventFilter(obj, event)

    def _on_config_changed(self, *args):
        """控件值变更时的防抖处理"""
        if not self._suppress_auto_save:
            self._schedule_auto_save()

    def _install_auto_save_signals(self):
        """为可编辑控件连接值变更信号"""
        for widget in self.findChildren(QWidget):
            if isinstance(widget, QLineEdit):
                widget.editingFinished.connect(self._schedule_auto_save)
            elif isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self._on_config_changed)
            elif isinstance(widget, QSpinBox):
                widget.valueChanged.connect(self._on_config_changed)
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.connect(self._on_config_changed)
            elif isinstance(widget, QTextEdit):
                widget.installEventFilter(self)

    def initUI(self):
        os.makedirs('separate', exist_ok=True)
        # Build the feature sections first, then compose them into four focused
        # navigation pages.  The section methods remain separate so their worker
        # and configuration bindings stay easy to maintain.
        self.initInputOutputTab()
        self.initSettingsTab()
        self.initAdvancedSettingTab()
        self.initDictTab()
        self.initClipTab()
        self.initSynthTab()
        self.initSummarizeTab()
        self.initLogTab()
        self.initAboutTab()

        self._enhance_workflow_page()
        self._build_config_page()
        self._build_phone_page()
        self._build_dictionary_page()
        self._build_tools_page()
        self._enhance_task_page()

        workspace = QWidget(self)
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(14, 10, 14, 12)
        workspace_layout.setSpacing(10)

        self.top_tabs = QTabWidget(workspace)
        self.top_tabs.setDocumentMode(True)
        self.top_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.top_tabs.setUsesScrollButtons(True)
        self.top_tabs.setStyleSheet(
            "QTabBar::tab { min-width: 150px; min-height: 34px; padding: 4px 16px; }"
            "QTabWidget::pane { border: 0; top: -1px; }"
        )
        self.top_tabs.addTab(self.about_tab, _("tab_about"))
        self.top_tabs.addTab(self.input_output_tab, _("tab_workflow"))
        self.top_tabs.addTab(self.config_tab, _("tab_config"))
        self.top_tabs.addTab(self.dict_tab, _("tab_dict"))
        self.top_tabs.addTab(self.tools_tab, _("tab_tools"))
        self.top_tabs.addTab(self.phone_tab, _("tab_phone"))
        self.top_tabs.addTab(self.log_tab, _("tab_tasks"))
        workspace_layout.addWidget(self.top_tabs, 1)
        workspace_layout.addWidget(self._make_shared_info_panel())
        self.setCentralWidget(workspace)

        self._install_auto_save_signals()
        self.load_config()

    def _make_shared_info_panel(self):
        """Create the progress strip that remains visible below every top tab."""
        panel = QFrame(self)
        panel.setObjectName("shared-info-panel")
        panel.setFrameShape(QFrame.Shape.NoFrame)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.addWidget(SubtitleLabel(_("progress_title")))
        self.shared_task_label = BodyLabel(_("task_none"))
        header.addWidget(self.shared_task_label)
        header.addStretch()
        self.shared_file_label = BodyLabel(_("progress_files", completed="0", total="0"))
        self.shared_file_label.setVisible(False)
        header.addWidget(self.shared_file_label)
        self._output_character_requests = {}
        self.shared_character_label = BodyLabel(
            _("progress_characters", count="0")
        )
        self.shared_character_label.setToolTip(_("progress_characters_tooltip"))
        header.addWidget(self.shared_character_label)
        self.shared_state_label = BodyLabel(_("task_state_idle"))
        header.addWidget(self.shared_state_label)
        layout.addLayout(header)

        self.shared_progress_bar = QProgressBar()
        self.shared_progress_bar.setRange(0, 100)
        self.shared_progress_bar.setValue(0)
        self.shared_progress_bar.setFormat("0%")
        self.shared_progress_bar.setTextVisible(True)
        self.shared_progress_bar.setMinimumHeight(18)
        self.shared_progress_bar.setMaximumHeight(18)
        layout.addWidget(self.shared_progress_bar)

        self.shared_progress_view = QPlainTextEdit()
        self.shared_progress_view.setReadOnly(True)
        self.shared_progress_view.document().setMaximumBlockCount(1000)
        self.shared_progress_view.setPlaceholderText(_("progress_placeholder"))
        self.shared_progress_view.setMinimumHeight(48)
        self.shared_progress_view.setMaximumHeight(64)
        self.shared_progress_view.setStyleSheet(
            "font-family: Consolas, Monospace; font-size: 9pt;"
        )
        layout.addWidget(self.shared_progress_view)
        return panel

    def _style_section(self, section: QFrame, title: str):
        """Reflow a former navigation page as a borderless section."""
        section.setParent(self)
        section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        section.setFrameShape(QFrame.Shape.NoFrame)
        section.setStyleSheet(f"QFrame#{section.objectName()} {{ border: 0; }}")
        section.vBoxLayout.setContentsMargins(14, 12, 14, 14)
        section.vBoxLayout.setSpacing(7)
        section.vBoxLayout.insertWidget(0, SubtitleLabel(title))

    def _scrollable_grid(self):
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        grid = QGridLayout(content)
        grid.setContentsMargins(2, 2, 8, 8)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(12)
        scroll.setWidget(content)
        return scroll, grid

    def _sync_combo_text(self, source, target):
        text = source.currentText()
        target.blockSignals(True)
        target.setCurrentText(text)
        target.blockSignals(False)

    def _sync_line_text(self, source, target):
        target.blockSignals(True)
        target.setText(source.text())
        target.blockSignals(False)

    def _sync_check_state(self, source, target):
        target.blockSignals(True)
        target.setChecked(source.isChecked())
        target.blockSignals(False)

    def _sync_spin_value(self, source, target):
        target.blockSignals(True)
        target.setValue(source.value())
        target.blockSignals(False)

    def _bind_mirrored_pair(self, first, second, signal_name: str, sync_method):
        getattr(first, signal_name).connect(
            lambda *_args: sync_method(first, second)
        )
        getattr(second, signal_name).connect(
            lambda *_args: sync_method(second, first)
        )
        sync_method(first, second)

    def _clear_layout(self, layout):
        """Detach all widgets/layouts so an existing section can be reflowed."""
        while layout.count():
            item = layout.takeAt(0)
            child_layout = item.layout()
            if child_layout is not None:
                self._clear_layout(child_layout)
                child_layout.deleteLater()

    def _action_column(self, *buttons):
        panel = QFrame(self)
        panel.setFrameShape(QFrame.Shape.NoFrame)
        panel.setMinimumWidth(190)
        panel.setMaximumWidth(230)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        layout.addWidget(SubtitleLabel(_("actions_title")))
        for button in buttons:
            layout.addWidget(button)
        layout.addStretch()
        return panel

    def _enhance_workflow_page(self):
        layout = self.input_output_layout
        self._clear_layout(layout)
        layout.setContentsMargins(24, 18, 24, 20)
        layout.setSpacing(10)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_widget = QWidget()
        scroll.setWidget(scroll_widget)
        content = QVBoxLayout(scroll_widget)
        content.setContentsMargins(2, 2, 8, 8)
        content.setSpacing(10)

        body = QHBoxLayout()
        body.setSpacing(12)
        input_panel = QFrame(self)
        input_panel.setFrameShape(QFrame.Shape.NoFrame)
        form = QGridLayout(input_panel)
        form.setContentsMargins(14, 12, 14, 14)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)

        row = 0

        # 🌍 Language settings
        form.addWidget(SubtitleLabel(_("workflow_section_lang")), row, 0, 1, 8)
        row += 1
        form.addWidget(self.lang_selector_label, row, 0)
        form.addWidget(self.lang_selector, row, 1)
        form.addWidget(BodyLabel(_("config_theme_label")), row, 2)
        self.theme_selector = QComboBox()
        for label_key, theme_file in UI_THEME_OPTIONS:
            self.theme_selector.addItem(_(label_key), userData=theme_file)
        current_theme_index = self.theme_selector.findData(_load_ui_theme())
        if current_theme_index >= 0:
            self.theme_selector.setCurrentIndex(current_theme_index)
        self.theme_selector.currentIndexChanged.connect(self._on_theme_changed)
        form.addWidget(self.theme_selector, row, 3)
        form.addWidget(self.io_transcription_lang_label, row, 4)
        form.addWidget(self.transcription_lang, row, 5)
        form.addWidget(self.io_target_lang_label, row, 6)
        form.addWidget(self.target_lang, row, 7)
        row += 1

        # 📂 Input files
        form.addWidget(SubtitleLabel(_("workflow_section_input")), row, 0, 1, 8)
        row += 1
        form.addWidget(self.io_input_label, row, 0, 1, 8)
        row += 1
        self.input_files_list.setMinimumHeight(86)
        self.input_files_list.setMaximumHeight(130)
        form.addWidget(self.input_files_list, row, 0, 1, 8)
        row += 1

        # ⚙️ Processing options
        form.addWidget(SubtitleLabel(_("workflow_section_options")), row, 0, 1, 8)
        row += 1
        form.addWidget(self.enable_transcription_checkbox, row, 0, 1, 4)
        form.addWidget(self.enable_translation_checkbox, row, 4, 1, 4)
        row += 1
        form.addWidget(self.enable_segment_checkbox, row, 0, 1, 3)
        form.addWidget(self.io_segment_duration_label, row, 3)
        form.addWidget(self.segment_duration_spin, row, 4)
        form.addWidget(self.use_input_dir_checkbox, row, 5, 1, 2)
        form.addWidget(self.auto_shutdown_checkbox, row, 7)
        row += 1
        form.addWidget(self.streaming_checkbox, row, 0, 1, 4)
        form.addWidget(self.ai_resegment_checkbox, row, 4, 1, 4)
        row += 1
        form.addWidget(self.proofread_checkbox, row, 0, 1, 4)
        row += 1

        # 🌐 Network
        form.addWidget(SubtitleLabel(_("workflow_section_network")), row, 0, 1, 8)
        row += 1
        form.addWidget(self.io_proxy_label, row, 0)
        form.addWidget(self.proxy_address, row, 1, 1, 7)
        row += 1

        # 📁 Output
        form.addWidget(SubtitleLabel(_("workflow_section_output")), row, 0, 1, 8)
        row += 1
        form.addWidget(self.io_output_dir_label, row, 0)
        form.addWidget(self.output_dir_edit, row, 1, 1, 7)
        row += 1
        form.addWidget(self.io_format_label, row, 0)
        form.addWidget(self.output_content, row, 1)
        form.addWidget(self.io_container_label, row, 2)
        form.addWidget(self.output_container, row, 3)
        row += 1

        for column in (1, 3, 5, 7):
            form.setColumnStretch(column, 1)

        body.addWidget(input_panel, 1)
        body.addWidget(self._action_column(
            self.run_button,
            self.cancel_button,
            self.output_dir_button,
            self.open_output_button,
            self.clean_button,
        ))
        content.addLayout(body, 1)
        layout.addWidget(scroll, 1)

    def _reflow_config_sections(self):
        self._clear_layout(self.settings_layout)
        speech_row = QHBoxLayout()
        speech_panel = QFrame(self.settings_tab)
        speech_panel.setFrameShape(QFrame.Shape.NoFrame)
        speech_form = QGridLayout(speech_panel)
        speech_form.setContentsMargins(14, 12, 14, 14)
        speech_form.setHorizontalSpacing(10)
        speech_form.setVerticalSpacing(8)
        speech_form.addWidget(self.settings_asr_provider_label, 0, 0)
        speech_form.addWidget(self.asr_provider_combo, 0, 1)

        asrlabs_pairs = (
            (self.settings_asr_engine_label, self.asr_engine_combo),
            (self.settings_asr_model_label, self.asr_model_combo),
            (self.settings_asr_device_label, self.asr_device_combo),
            (self.settings_asr_compute_type_label, self.asr_compute_type_combo),
            (self.settings_asr_extra_label, self.asr_extra_edit),
            (self.settings_align_engine_label, self.align_engine_combo),
            (self.settings_align_model_label, self.align_model_combo),
            (self.settings_align_device_label, self.align_device_combo),
            (self.settings_align_extra_label, self.align_extra_edit),
        )
        crisp_pairs = (
            (self.settings_asr_backend_label, self.crispasr_backend_combo),
            (self.settings_crispasr_model_label, self.crispasr_model_combo),
            (self.settings_crispasr_aligner_label, self.crispasr_aligner_combo),
            (self.settings_asr_param_label, self.param_crispasr),
        )
        for row, (label, editor) in enumerate(asrlabs_pairs, start=1):
            speech_form.addWidget(label, row, 0)
            speech_form.addWidget(editor, row, 1)
        for row, (label, editor) in enumerate(crisp_pairs, start=1):
            speech_form.addWidget(label, row, 0)
            speech_form.addWidget(editor, row, 1)
        speech_form.setColumnStretch(1, 1)
        speech_row.addWidget(speech_panel, 1)
        speech_row.addWidget(self._action_column(
            self.open_crispasr_dir,
            self.refresh_speech_models_button,
        ))
        self.settings_layout.addLayout(speech_row)

        self._clear_layout(self.advanced_settings_layout)
        translation_row = QHBoxLayout()
        translation_panel = QFrame(self.advanced_settings_tab)
        translation_panel.setFrameShape(QFrame.Shape.NoFrame)
        translation_form = QGridLayout(translation_panel)
        translation_form.setContentsMargins(14, 12, 14, 14)
        translation_form.setHorizontalSpacing(10)
        translation_form.setVerticalSpacing(8)
        translation_form.addWidget(self.adv_translator_label, 0, 0)
        translation_form.addWidget(self.translator_group, 0, 1)
        translation_form.addWidget(self.adv_concurrency_label, 0, 4)
        translation_form.addWidget(self.max_concurrent_spin, 0, 5)
        translation_form.addWidget(self.adv_online_token_label, 1, 0)
        translation_form.addWidget(self.gpt_token, 1, 1, 1, 5)
        translation_form.addWidget(self.adv_online_model_label, 2, 0)
        translation_form.addWidget(self.gpt_model, 2, 1, 1, 5)
        translation_form.addWidget(self.adv_auxiliary_models_label, 3, 0)
        translation_form.addWidget(self.auxiliary_model_tabs, 3, 1, 1, 5)
        translation_form.addWidget(self.adv_online_address_label, 4, 0)
        translation_form.addWidget(self.gpt_address, 4, 1, 1, 3)
        translation_form.addWidget(self.deepseek_thinking_checkbox, 4, 4, 1, 2)
        translation_form.addWidget(self.adv_offline_model_label, 5, 0)
        translation_form.addWidget(self.sakura_file, 5, 1)
        translation_form.addWidget(self.adv_offline_gpu_label, 5, 2, 1, 2)
        translation_form.addWidget(self.sakura_mode, 5, 4, 1, 2)
        translation_form.addWidget(self.adv_offline_param_label, 6, 0)
        translation_form.addWidget(self.param_llama, 6, 1, 1, 5)
        for column in (1, 2, 4, 5):
            translation_form.setColumnStretch(column, 1)
        translation_row.addWidget(translation_panel, 1)
        translation_row.addWidget(self._action_column(
            self.open_model_dir,
            self.refresh_language_models_button,
            self.test_online_button,
        ))
        self.advanced_settings_layout.addLayout(translation_row)

    def _build_config_page(self):
        self.config_tab = Widget("Configuration", self)
        layout = self.config_tab.vBoxLayout
        layout.setContentsMargins(24, 18, 24, 20)
        layout.setSpacing(10)

        self._reflow_config_sections()
        self._style_section(self.settings_tab, _("config_speech_title"))
        self._style_section(self.advanced_settings_tab, _("config_translation_title"))
        self.param_crispasr.setMaximumHeight(130)
        self.param_llama.setMaximumHeight(120)
        scroll, grid = self._scrollable_grid()
        grid.addWidget(self.settings_tab, 0, 0)
        grid.addWidget(self.advanced_settings_tab, 1, 0)
        grid.setColumnStretch(0, 1)
        grid.setRowStretch(2, 1)
        layout.addWidget(scroll, 1)

    def _build_phone_page(self):
        self.phone_tab = Widget("Phone", self)
        layout = self.phone_tab.vBoxLayout
        layout.setContentsMargins(24, 18, 24, 20)
        layout.setSpacing(10)

        scroll, grid = self._scrollable_grid()
        self.lan_settings_section = self._build_lan_config_section()
        grid.addWidget(self.lan_settings_section, 0, 0)

        library_section = Widget("PhoneMediaLibrary", self)
        library_layout = library_section.vBoxLayout
        library_layout.setContentsMargins(14, 12, 14, 14)
        library_layout.setSpacing(8)
        library_layout.addWidget(SubtitleLabel(_("media_library_title")))

        library_grid = QGridLayout()
        library_grid.setHorizontalSpacing(10)
        library_grid.setVerticalSpacing(8)
        library_grid.addWidget(BodyLabel(_("media_library_root")), 0, 0)
        self.media_library_root_edit = QLineEdit()
        self.media_library_root_edit.setPlaceholderText(r"D:\音声")
        library_grid.addWidget(self.media_library_root_edit, 0, 1, 1, 4)
        self.media_library_browse_button = QPushButton(_("media_library_browse"))
        library_grid.addWidget(self.media_library_browse_button, 0, 5)

        self.media_library_auto_scan_checkbox = QCheckBox(_("media_library_auto_scan"))
        self.media_library_scan_button = QPushButton(_("media_library_scan"))
        self.media_library_status_label = BodyLabel(_("media_library_status_empty"))
        library_grid.addWidget(self.media_library_auto_scan_checkbox, 1, 0, 1, 2)
        library_grid.addWidget(self.media_library_scan_button, 1, 2)
        library_grid.addWidget(self.media_library_status_label, 1, 3, 1, 3)
        library_grid.setColumnStretch(1, 1)
        library_grid.setColumnStretch(4, 1)
        library_layout.addLayout(library_grid)

        self.phone_queue_label = BodyLabel(_("phone_queue_idle"))
        self.phone_queue_label.setWordWrap(True)
        library_layout.addWidget(self.phone_queue_label)
        library_layout.addWidget(BodyLabel(_("phone_sync_hint")))

        self.media_library_browse_button.clicked.connect(self._browse_media_library_root)
        self.media_library_scan_button.clicked.connect(self._start_media_library_scan)
        grid.addWidget(library_section, 1, 0)
        grid.setColumnStretch(0, 1)
        grid.setRowStretch(2, 1)
        layout.addWidget(scroll, 1)

    def _build_lan_config_section(self):
        section = Widget("LanService", self)
        layout = section.vBoxLayout
        layout.setContentsMargins(14, 12, 14, 14)
        layout.setSpacing(8)
        layout.addWidget(SubtitleLabel(_("lan_title")))

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)
        self.lan_enabled_checkbox = QCheckBox(_("lan_enabled"))
        self.lan_auto_start_checkbox = QCheckBox(_("lan_auto_start"))
        self.lan_port_spin = QSpinBox()
        self.lan_port_spin.setRange(1024, 65535)
        self.lan_port_spin.setValue(DEFAULT_HTTP_PORT)
        self.lan_device_name_edit = QLineEdit(socket.gethostname() or "VoiceTransl")
        self.lan_allowed_networks_edit = QLineEdit()
        self.lan_allowed_networks_edit.setPlaceholderText("100.64.0.0/10, 10.0.0.0/8")
        grid.addWidget(self.lan_enabled_checkbox, 0, 0)
        grid.addWidget(self.lan_auto_start_checkbox, 0, 1)
        grid.addWidget(BodyLabel(_("lan_device_name")), 0, 2)
        grid.addWidget(self.lan_device_name_edit, 0, 3)
        grid.addWidget(BodyLabel(_("lan_port")), 0, 4)
        grid.addWidget(self.lan_port_spin, 0, 5)

        self.lan_status_label = BodyLabel(_("lan_status_stopped"))
        self.lan_address_label = BodyLabel("")
        self.lan_pair_code_label = BodyLabel(_("lan_pair_code", code="------"))
        grid.addWidget(self.lan_status_label, 1, 0, 1, 2)
        grid.addWidget(self.lan_address_label, 1, 2, 1, 2)
        grid.addWidget(self.lan_pair_code_label, 1, 4, 1, 2)

        self.lan_start_button = QPushButton(_("lan_restart"))
        self.lan_rotate_code_button = QPushButton(_("lan_rotate_code"))
        self.lan_devices_combo = QComboBox()
        self.lan_devices_combo.setMinimumWidth(180)
        self.lan_revoke_button = QPushButton(_("lan_revoke"))
        self.lan_file_management_checkbox = QCheckBox(_("lan_file_management"))
        grid.addWidget(self.lan_start_button, 2, 0)
        grid.addWidget(self.lan_rotate_code_button, 2, 1)
        grid.addWidget(BodyLabel(_("lan_paired_devices")), 2, 2)
        grid.addWidget(self.lan_devices_combo, 2, 3, 1, 2)
        grid.addWidget(self.lan_revoke_button, 2, 5)
        grid.addWidget(self.lan_file_management_checkbox, 3, 2, 1, 4)
        grid.addWidget(BodyLabel(_("lan_allowed_networks")), 4, 0, 1, 2)
        grid.addWidget(self.lan_allowed_networks_edit, 4, 2, 1, 4)
        grid.setColumnStretch(3, 1)
        layout.addLayout(grid)

        self.lan_enabled_checkbox.toggled.connect(self._on_lan_enabled_toggled)
        self.lan_start_button.clicked.connect(self._restart_lan_service)
        self.lan_rotate_code_button.clicked.connect(self._rotate_lan_pair_code)
        self.lan_revoke_button.clicked.connect(self._revoke_lan_device)
        self.lan_devices_combo.currentIndexChanged.connect(self._refresh_lan_device_permission)
        self.lan_file_management_checkbox.toggled.connect(self._set_lan_device_file_management)
        return section

    def _build_dictionary_page(self):
        DICTIONARY_PRESET_DIR.mkdir(parents=True, exist_ok=True)
        layout = self.dict_layout
        self._clear_layout(layout)
        self.dict_tab.setObjectName("Dictionary")
        layout.setContentsMargins(24, 18, 24, 20)
        layout.setSpacing(10)

        body = QHBoxLayout()
        body.setSpacing(12)
        inputs = QFrame(self.dict_tab)
        inputs.setFrameShape(QFrame.Shape.NoFrame)
        input_layout = QVBoxLayout(inputs)
        input_layout.setContentsMargins(14, 12, 14, 14)
        input_layout.setSpacing(6)
        for label, editor in (
            (self.dict_before_label, self.before_dict),
            (self.dict_gpt_label, self.gpt_dict),
            (self.dict_after_label, self.after_dict),
            (self.dict_extra_label, self.extra_prompt),
        ):
            editor.setMinimumHeight(64)
            editor.setMaximumHeight(94)
            input_layout.addWidget(label)
            input_layout.addWidget(editor)
        prompt_row = QHBoxLayout()
        prompt_row.addWidget(self.dict_prompt_mode_label)
        prompt_row.addWidget(self.change_prompt_mode, 1)
        input_layout.addLayout(prompt_row)
        body.addWidget(inputs, 1)

        preset_panel = QFrame(self.dict_tab)
        preset_panel.setFrameShape(QFrame.Shape.NoFrame)
        preset_panel.setMinimumWidth(220)
        preset_panel.setMaximumWidth(260)
        preset_layout = QVBoxLayout(preset_panel)
        preset_layout.setContentsMargins(12, 12, 12, 12)
        preset_layout.setSpacing(8)
        preset_layout.addWidget(SubtitleLabel(_("dictionary_presets_title")))
        preset_layout.addWidget(BodyLabel(_("dictionary_preset_name_label")))
        self.dictionary_preset_combo = QComboBox()
        self.dictionary_preset_combo.setEditable(True)
        self.dictionary_preset_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        preset_layout.addWidget(self.dictionary_preset_combo)
        self.save_dictionary_preset_button = QPushButton(_("dictionary_preset_save"))
        self.load_dictionary_preset_button = QPushButton(_("dictionary_preset_load"))
        self.refresh_dictionary_presets_button = QPushButton(_("dictionary_preset_refresh"))
        self.open_dictionary_presets_button = QPushButton(_("dictionary_preset_open_dir"))
        self.save_dictionary_preset_button.clicked.connect(self.save_dictionary_preset)
        self.load_dictionary_preset_button.clicked.connect(self.load_dictionary_preset)
        self.refresh_dictionary_presets_button.clicked.connect(self.refresh_dictionary_presets)
        self.open_dictionary_presets_button.clicked.connect(
            lambda: open_path(str(DICTIONARY_PRESET_DIR))
        )
        for button in (
            self.save_dictionary_preset_button,
            self.load_dictionary_preset_button,
            self.refresh_dictionary_presets_button,
            self.open_dictionary_presets_button,
        ):
            preset_layout.addWidget(button)
        preset_layout.addStretch()
        body.addWidget(preset_panel)
        layout.addLayout(body, 1)
        self.refresh_dictionary_presets()

    @staticmethod
    def _dictionary_preset_path(name: str) -> tuple[Path, str]:
        invalid_filename_chars = set('<>:"/\\|?*')
        safe_name = ''.join(
            '_' if char in invalid_filename_chars or ord(char) < 32 else char
            for char in (name or '')
        )
        safe_name = re.sub(r'\s+', ' ', safe_name).strip(' .')[:80]
        if not safe_name:
            raise ValueError(_("dictionary_preset_name_required"))
        preset_dir = DICTIONARY_PRESET_DIR.resolve()
        preset_dir.mkdir(parents=True, exist_ok=True)
        return preset_dir / f"{safe_name}.yaml", safe_name

    def refresh_dictionary_presets(self, selected_name: str | None = None):
        DICTIONARY_PRESET_DIR.mkdir(parents=True, exist_ok=True)
        current = selected_name or self.dictionary_preset_combo.currentText().strip()
        names = sorted(
            path.stem for path in DICTIONARY_PRESET_DIR.glob('*.yaml')
            if path.is_file()
        )
        self.dictionary_preset_combo.blockSignals(True)
        self.dictionary_preset_combo.clear()
        self.dictionary_preset_combo.addItems(names)
        self.dictionary_preset_combo.setEditText(current if current else (names[0] if names else ''))
        self.dictionary_preset_combo.blockSignals(False)

    def save_dictionary_preset(self):
        try:
            path, safe_name = self._dictionary_preset_path(
                self.dictionary_preset_combo.currentText()
            )
            payload = {
                'version': 1,
                'name': safe_name,
                'before_dict': self.before_dict.toPlainText(),
                'gpt_dict': self.gpt_dict.toPlainText(),
                'after_dict': self.after_dict.toPlainText(),
                'extra_prompt': self.extra_prompt.toPlainText(),
                'change_prompt_mode': self.change_prompt_mode.currentData(),
            }
            temp_path = path.with_suffix('.yaml.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
            os.replace(temp_path, path)
            self.refresh_dictionary_presets(safe_name)
            self._emit_status(_("dictionary_preset_saved", name=safe_name))
        except Exception as error:
            self._emit_status(_("dictionary_preset_save_error", error=error))

    def load_dictionary_preset(self):
        try:
            path, safe_name = self._dictionary_preset_path(
                self.dictionary_preset_combo.currentText()
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            with open(path, 'r', encoding='utf-8') as f:
                payload = yaml.safe_load(f) or {}
            self.before_dict.setPlainText(str(payload.get('before_dict', '')))
            self.gpt_dict.setPlainText(str(payload.get('gpt_dict', '')))
            self.after_dict.setPlainText(str(payload.get('after_dict', '')))
            self.extra_prompt.setPlainText(str(payload.get('extra_prompt', '')))
            prompt_mode = payload.get('change_prompt_mode', '不修改')
            prompt_index = self.change_prompt_mode.findData(prompt_mode)
            if prompt_index >= 0:
                self.change_prompt_mode.setCurrentIndex(prompt_index)
            self._schedule_auto_save()
            self._emit_status(_("dictionary_preset_loaded", name=safe_name))
        except Exception as error:
            self._emit_status(_("dictionary_preset_load_error", error=error))

    def _build_behavior_config_section(self):
        section = Widget("OutputBehavior", self)
        layout = section.vBoxLayout
        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        grid.addWidget(BodyLabel(_("lang_selector_label")), 0, 0)
        self.config_lang_selector = QComboBox()
        for index in range(self.lang_selector.count()):
            self.config_lang_selector.addItem(self.lang_selector.itemText(index))
        grid.addWidget(self.config_lang_selector, 0, 1)
        self._bind_mirrored_pair(
            self.lang_selector, self.config_lang_selector,
            'currentTextChanged', self._sync_combo_text,
        )
        self.config_lang_selector.currentIndexChanged.connect(self._on_language_changed)

        grid.addWidget(BodyLabel(_("io_transcription_lang_label")), 0, 2)
        self.config_transcription_lang = QComboBox()
        for index in range(self.transcription_lang.count()):
            self.config_transcription_lang.addItem(self.transcription_lang.itemText(index))
        grid.addWidget(self.config_transcription_lang, 0, 3)
        self._bind_mirrored_pair(
            self.transcription_lang, self.config_transcription_lang,
            'currentTextChanged', self._sync_combo_text,
        )


        grid.addWidget(BodyLabel(_("io_target_lang_label")), 0, 4)
        self.config_target_lang = QComboBox()
        for index in range(self.target_lang.count()):
            self.config_target_lang.addItem(self.target_lang.itemText(index))
        grid.addWidget(self.config_target_lang, 0, 5)
        self._bind_mirrored_pair(
            self.target_lang, self.config_target_lang,
            'currentTextChanged', self._sync_combo_text,
        )

        grid.addWidget(BodyLabel(_("config_theme_label")), 1, 0)
        self.theme_selector = QComboBox()
        for label_key, theme_file in UI_THEME_OPTIONS:
            self.theme_selector.addItem(_(label_key), userData=theme_file)
        current_theme_index = self.theme_selector.findData(_load_ui_theme())
        if current_theme_index >= 0:
            self.theme_selector.setCurrentIndex(current_theme_index)
        self.theme_selector.currentIndexChanged.connect(self._on_theme_changed)
        grid.addWidget(self.theme_selector, 1, 1)

        grid.addWidget(BodyLabel(_("io_output_content_label")), 1, 2)
        self.config_output_content = QComboBox()
        for index in range(self.output_content.count()):
            self.config_output_content.addItem(self.output_content.itemText(index))
        grid.addWidget(self.config_output_content, 1, 3)
        self._bind_mirrored_pair(
            self.output_content, self.config_output_content,
            'currentTextChanged', self._sync_combo_text,
        )

        grid.addWidget(BodyLabel(_("synth_font_label")), 1, 4)
        self.config_subtitle_font = QComboBox()
        for index in range(self.subtitle_font_combo.count()):
            self.config_subtitle_font.addItem(self.subtitle_font_combo.itemText(index))
        grid.addWidget(self.config_subtitle_font, 1, 5)
        self._bind_mirrored_pair(
            self.subtitle_font_combo, self.config_subtitle_font,
            'currentTextChanged', self._sync_combo_text,
        )

        grid.addWidget(BodyLabel(_("io_proxy_label")), 2, 0)
        self.config_proxy_address = QLineEdit()
        self.config_proxy_address.setPlaceholderText(_("io_proxy_placeholder"))
        grid.addWidget(self.config_proxy_address, 2, 1, 1, 5)
        self._bind_mirrored_pair(
            self.proxy_address, self.config_proxy_address,
            'textChanged', self._sync_line_text,
        )

        grid.addWidget(BodyLabel(_("io_output_dir_label")), 3, 0)
        self.config_output_dir = QLineEdit()
        grid.addWidget(self.config_output_dir, 3, 1, 1, 4)
        self._bind_mirrored_pair(
            self.output_dir_edit, self.config_output_dir,
            'textChanged', self._sync_line_text,
        )
        self.config_output_dir_button = QPushButton(_("io_browse_dir_btn"))
        self.config_output_dir_button.clicked.connect(self.browse_output_dir)
        grid.addWidget(self.config_output_dir_button, 3, 5)

        options = QHBoxLayout()
        self.config_use_input_dir = QCheckBox(self.use_input_dir_checkbox.text())
        self.config_auto_shutdown = QCheckBox(self.auto_shutdown_checkbox.text())
        self.config_enable_segment = QCheckBox(self.enable_segment_checkbox.text())
        options.addWidget(self.config_use_input_dir)
        options.addWidget(self.config_auto_shutdown)
        options.addWidget(self.config_enable_segment)
        options.addWidget(BodyLabel(_("io_segment_duration_label")))
        self.config_segment_duration = QSpinBox()
        self.config_segment_duration.setRange(1, 20)
        options.addWidget(self.config_segment_duration)
        options.addStretch()
        layout.addLayout(grid)
        layout.addLayout(options)

        self._bind_mirrored_pair(
            self.use_input_dir_checkbox, self.config_use_input_dir,
            'stateChanged', self._sync_check_state,
        )
        self._bind_mirrored_pair(
            self.auto_shutdown_checkbox, self.config_auto_shutdown,
            'stateChanged', self._sync_check_state,
        )
        self._bind_mirrored_pair(
            self.enable_segment_checkbox, self.config_enable_segment,
            'stateChanged', self._sync_check_state,
        )
        self._bind_mirrored_pair(
            self.segment_duration_spin, self.config_segment_duration,
            'valueChanged', self._sync_spin_value,
        )
        self.config_use_input_dir.stateChanged.connect(self.update_output_dir_controls)
        self.config_enable_segment.stateChanged.connect(self.update_segment_controls)
        self.update_output_dir_controls()
        self.update_segment_controls()
        return section

    def _reflow_tool_sections(self):
        self._clear_layout(self.clip_layout)
        clip_row = QHBoxLayout()
        clip_panel = QFrame(self.clip_tab)
        clip_panel.setFrameShape(QFrame.Shape.NoFrame)
        clip_inputs = QVBoxLayout(clip_panel)
        clip_inputs.setContentsMargins(14, 12, 14, 14)
        clip_inputs.addWidget(self.clip_tool_label)
        clip_inputs.addWidget(self.clip_files_list)
        clip_times = QGridLayout()
        clip_times.addWidget(self.clip_start_label, 0, 0)
        clip_times.addWidget(self.clip_end_label, 0, 1)
        clip_times.addWidget(self.clip_start_time, 1, 0)
        clip_times.addWidget(self.clip_end_time, 1, 1)
        clip_inputs.addLayout(clip_times)
        clip_row.addWidget(clip_panel, 1)
        clip_row.addWidget(self._action_column(
            self.run_clip_button,
            self.clip_cancel_button,
        ))
        self.clip_layout.addLayout(clip_row)

        separator = QFrame(self.clip_tab)
        separator.setFrameShape(QFrame.Shape.HLine)
        self.clip_layout.addWidget(separator)

        vocal_row = QHBoxLayout()
        vocal_panel = QFrame(self.clip_tab)
        vocal_panel.setFrameShape(QFrame.Shape.NoFrame)
        vocal_inputs = QVBoxLayout(vocal_panel)
        vocal_inputs.setContentsMargins(14, 12, 14, 14)
        vocal_inputs.addWidget(self.clip_vocal_split_label)
        uvr_model_row = QHBoxLayout()
        uvr_model_row.addWidget(self.clip_uvr_model_label)
        uvr_model_row.addWidget(self.uvr_file)
        uvr_model_row.addStretch()
        vocal_inputs.addLayout(uvr_model_row)
        vocal_inputs.addWidget(self.uvr_file_list)
        vocal_row.addWidget(vocal_panel, 1)
        vocal_row.addWidget(self._action_column(
            self.run_uvr_button,
            self.uvr_cancel_button,
            self.open_uvr_dir,
        ))
        self.clip_layout.addLayout(vocal_row)

        self._clear_layout(self.synth_layout)
        video_row = QHBoxLayout()
        video_panel = QFrame(self.synth_tab)
        video_panel.setFrameShape(QFrame.Shape.NoFrame)
        video_inputs = QVBoxLayout(video_panel)
        video_inputs.setContentsMargins(14, 12, 14, 14)
        video_inputs.addWidget(self.synth_label)
        video_inputs.addWidget(self.synth_video_label)
        video_inputs.addWidget(self.synth_video_files_list)
        video_inputs.addWidget(self.synth_srt_label)
        video_inputs.addWidget(self.synth_srt_files_list)
        subtitle_options = QHBoxLayout()
        subtitle_options.addWidget(self.synth_subtitle_type_label)
        subtitle_options.addWidget(self.subtitle_type_combo)
        subtitle_options.addWidget(self.synth_font_label)
        subtitle_options.addWidget(self.subtitle_font_combo)
        subtitle_options.addStretch()
        video_inputs.addLayout(subtitle_options)
        video_row.addWidget(video_panel, 1)
        video_row.addWidget(self._action_column(
            self.synth_video_browse_btn,
            self.synth_srt_browse_btn,
            self.run_synth_button,
            self.synth_cancel_button,
        ))
        self.synth_layout.addLayout(video_row)

        separator = QFrame(self.synth_tab)
        separator.setFrameShape(QFrame.Shape.HLine)
        self.synth_layout.addWidget(separator)

        audio_row = QHBoxLayout()
        audio_panel = QFrame(self.synth_tab)
        audio_panel.setFrameShape(QFrame.Shape.NoFrame)
        audio_inputs = QVBoxLayout(audio_panel)
        audio_inputs.setContentsMargins(14, 12, 14, 14)
        audio_inputs.addWidget(self.synth_audio_label)
        audio_inputs.addWidget(self.synth_audio_files_list)
        audio_row.addWidget(audio_panel, 1)
        audio_row.addWidget(self._action_column(
            self.run_synth_audio_button,
            self.synth_audio_cancel_button,
        ))
        self.synth_layout.addLayout(audio_row)

        self._clear_layout(self.summarize_layout)
        summarize_row = QHBoxLayout()
        summarize_panel = QFrame(self.summarize_tab)
        summarize_panel.setFrameShape(QFrame.Shape.NoFrame)
        summarize_inputs = QVBoxLayout(summarize_panel)
        summarize_inputs.setContentsMargins(14, 12, 14, 14)
        summarize_inputs.addWidget(self.summarize_prompt_label)
        summarize_inputs.addWidget(self.summarize_prompt)
        summarize_inputs.addWidget(self.summarize_input_label)
        summarize_inputs.addWidget(self.summarize_files_list)
        summarize_row.addWidget(summarize_panel, 1)
        summarize_row.addWidget(self._action_column(
            self.run_summarize_button,
            self.summarize_cancel_button,
        ))
        self.summarize_layout.addLayout(summarize_row)

    def _build_tools_page(self):
        self.tools_tab = Widget("Tools", self)
        layout = self.tools_tab.vBoxLayout
        layout.setContentsMargins(24, 18, 24, 20)
        layout.setSpacing(10)

        self._reflow_tool_sections()
        self._style_section(self.clip_tab, _("tools_clip_title"))
        self._style_section(self.synth_tab, _("tools_synth_title"))
        self._style_section(self.summarize_tab, _("tools_summarize_title"))
        for editor in (
            self.clip_files_list, self.uvr_file_list, self.synth_video_files_list,
            self.synth_srt_files_list, self.synth_audio_files_list,
            self.summarize_prompt, self.summarize_files_list,
        ):
            editor.setMinimumHeight(72)
            editor.setMaximumHeight(108)

        scroll, grid = self._scrollable_grid()
        grid.addWidget(self.clip_tab, 0, 0)
        grid.addWidget(self.synth_tab, 1, 0)
        grid.addWidget(self.summarize_tab, 2, 0)
        grid.setColumnStretch(0, 1)
        grid.setRowStretch(3, 1)
        layout.addWidget(scroll, 1)

    def _enhance_task_page(self):
        self.log_tab.setObjectName("Tasks")
        layout = self.log_layout
        self._clear_layout(layout)
        layout.setContentsMargins(24, 18, 24, 20)
        layout.setSpacing(8)

        log_row = QHBoxLayout()
        log_inputs = QVBoxLayout()
        log_inputs.addWidget(self.log_file_label)
        filter_row = QHBoxLayout()
        filter_row.addWidget(self.log_filter_label)
        filter_row.addWidget(self.log_filter_combo)
        filter_row.addStretch()
        filter_row.addWidget(self.verbose_checkbox)
        log_inputs.addLayout(filter_row)
        log_inputs.addWidget(self.log_display, 1)
        log_row.addLayout(log_inputs, 1)

        self.clear_log_button = QPushButton(_("log_clear_btn"))
        self.clear_log_button.clicked.connect(self.clear_log)
        log_row.addWidget(self._action_column(
            self.open_log_button,
            self.clear_log_button,
        ))
        layout.addLayout(log_row, 1)

    def clear_log(self):
        self.log_display.clear()
        try:
            open(LOG_PATH, 'w', encoding='utf-8').close()
        except OSError:
            pass

    def _set_progress_context(self, task_name: str):
        self._active_task_name = task_name
        self._output_character_requests = {}
        self.shared_character_label.setText(_("progress_characters", count="0"))
        self.shared_progress_view.clear()
        self._task_progress_phase = ""
        self._task_progress_current = 0
        self._task_progress_stage_id = 0
        self._task_file_completed = 0
        self._task_file_total = 0
        self._task_outcome = "running"
        self.shared_file_label.setText(_("progress_files", completed="0", total="0"))
        self.shared_file_label.setVisible(False)
        self.shared_progress_bar.setRange(0, 100)
        self.shared_progress_bar.setValue(0)
        self.shared_progress_bar.setFormat("0%")
        self.shared_task_label.setText(task_name)
        self.shared_state_label.setText(_("task_state_running"))

    def _on_task_finished(self):
        outcome = getattr(self, '_task_outcome', 'success')
        if outcome == 'success':
            self.shared_state_label.setText(_("task_state_done"))
        elif outcome == 'cancelled':
            self.shared_state_label.setText(_("task_state_cancelled"))
        else:
            self.shared_state_label.setText(_("task_state_failed"))

    def _on_task_outcome(self, outcome: str):
        self._task_outcome = str(outcome or 'error')

    def _start_worker_task(
        self,
        operation: str,
        task_name: str,
        show_model_dialog: bool = False,
        snapshot_overrides: dict | None = None,
        model_target: str | None = None,
        prepared_snapshot: TaskSnapshot | None = None,
        message_queue=None,
        lan_job_id: str | None = None,
    ):
        if self.thread is not None and self.thread.isRunning():
            self._emit_status(_("status_task_busy"))
            return
        snapshot = prepared_snapshot or self._capture_task_snapshot(operation)
        if snapshot_overrides:
            values = dict(snapshot.values)
            values.update(snapshot_overrides)
            snapshot = TaskSnapshot(operation=operation, values=values)
        self._set_progress_context(task_name)
        self.thread = QThread()
        self.cancel_token = CancellationToken()
        self.worker = MainWorker(
            snapshot, message_queue or self.msg_queue, self.cancel_token
        )
        self.worker.moveToThread(self.thread)
        # Connect to a real QObject slot. Connecting a decorated Python method
        # directly can make PySide invoke it in the GUI thread.
        self.thread.started.connect(self.worker.execute)
        self.worker.status.connect(self._on_worker_status)
        if hasattr(self.worker, 'outcome'):
            self.worker.outcome.connect(self._on_task_outcome)
            if lan_job_id:
                self.worker.outcome.connect(
                    lambda outcome, job_id=lan_job_id:
                    self._record_lan_outcome(job_id, outcome)
                )
        if show_model_dialog:
            self.worker.show_model_dialog.connect(
                lambda models, target=model_target:
                self._handle_model_list_loaded(models, target)
            )
        self.worker.finished.connect(self._on_task_finished)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self._on_worker_thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    def _capture_task_snapshot(self, operation: str) -> TaskSnapshot:
        """Read every Qt control once, before the worker leaves the UI thread."""
        language = (
            self.transcription_lang.currentData()
            or self.transcription_lang.currentText()
        )
        target_lang = self.target_lang.currentData() or 'zh-cn'
        enable_translation = self.enable_translation_checkbox.isChecked()
        output_dir = self.output_dir_edit.text().strip() or self.default_output_dir()
        output_dir = os.path.abspath(os.path.expanduser(output_dir))
        values = {
            'input_files': self.input_files_list.toPlainText(),
            'enable_transcription': self.enable_transcription_checkbox.isChecked(),
            'enable_translation': enable_translation,
            'translator': self.translator_group.currentText(),
            'language': language,
            'target_lang': target_lang,
            'gpt_token': self.gpt_token.text() or _load_api_key(),
            'gpt_address': self.gpt_address.text(),
            'gpt_model': self.gpt_model.text(),
            'ai_resegment_model': self._auxiliary_model_value(
                self.ai_resegment_model_combo
            ),
            'ai_resegment_provider': (
                self.ai_resegment_provider_combo.currentData() or 'follow'
            ),
            'ai_resegment_address': self.ai_resegment_address.text().strip(),
            'ai_resegment_token': (
                self.ai_resegment_token.text()
                or _load_api_key('VOICETRANSL_RESEGMENT_API_KEY')
            ),
            'proofread_model': self._auxiliary_model_value(
                self.proofread_model_combo
            ),
            'proofread_provider': (
                self.proofread_provider_combo.currentData() or 'follow'
            ),
            'proofread_address': self.proofread_address.text().strip(),
            'proofread_token': (
                self.proofread_token.text()
                or _load_api_key('VOICETRANSL_PROOFREAD_API_KEY')
            ),
            'deepseek_thinking': self.deepseek_thinking_checkbox.isChecked(),
            'ai_resegment_thinking': self.ai_resegment_thinking_checkbox.isChecked(),
            'proofread_thinking': self.proofread_thinking_checkbox.isChecked(),
            'sakura_file': self.sakura_file.currentText(),
            'sakura_mode': self.sakura_mode.text(),
            'proxy_address': self.proxy_address.text(),
            'before_dict': self.before_dict.toPlainText(),
            'gpt_dict': self.gpt_dict.toPlainText(),
            'after_dict': self.after_dict.toPlainText(),
            'extra_prompt': self.extra_prompt.toPlainText(),
            'change_prompt_mode': self.change_prompt_mode.currentData() or '不修改',
            'param_llama': self.param_llama.toPlainText(),
            'param_crispasr': self.param_crispasr.toPlainText(),
            'output_format': self.selected_output_format(enable_translation),
            'output_dir': output_dir,
            'use_input_dir': self.use_input_dir_checkbox.isChecked(),
            'enable_segment': self.enable_segment_checkbox.isChecked(),
            'segment_duration': self.segment_duration_spin.value(),
            'enable_streaming': self.streaming_checkbox.isChecked(),
            'enable_proofread': self.proofread_checkbox.isChecked(),
            'enable_ai_resegment': (
                self.ai_resegment_checkbox.isChecked()
                if hasattr(self, 'ai_resegment_checkbox') else False
            ),
            'max_concurrent': self.max_concurrent_spin.value(),
            'verbose_mode': self.verbose_checkbox.isChecked(),
            'auto_shutdown': self.auto_shutdown_checkbox.isChecked(),
            'uvr_file': self.uvr_file.currentText(),
            'uvr_input_files': self.uvr_file_list.toPlainText(),
            'summarize_input_files': self.summarize_files_list.toPlainText(),
            'summarize_prompt': self.summarize_prompt.toPlainText(),
            'subtitle_font': self.subtitle_font_combo.currentText(),
            'subtitle_type': self.subtitle_type_combo.currentData() or '硬字幕',
            'synth_video_files': self.synth_video_files_list.toPlainText(),
            'synth_srt_files': self.synth_srt_files_list.toPlainText(),
            'clip_input_files': self.clip_files_list.toPlainText(),
            'clip_start': self.clip_start_time.text(),
            'clip_end': self.clip_end_time.text(),
            'synth_audio_files': self.synth_audio_files_list.toPlainText(),
            'asr_config': {
                'provider': self.asr_provider_combo.currentData() or 'crispasr',
                'asr_engine': self.asr_engine_combo.currentData() or '',
                'asr_model': self.asr_model_combo.currentData() or '',
                'asr_device': self.asr_device_combo.currentText(),
                'asr_compute_type': self.asr_compute_type_combo.currentText(),
                'asr_extra': self.asr_extra_edit.toPlainText(),
                'align_engine': self.align_engine_combo.currentData() or 'none',
                'align_model': self.align_model_combo.currentData() or '',
                'align_device': self.align_device_combo.currentText(),
                'align_extra': self.align_extra_edit.toPlainText(),
                'crispasr_backend': self.crispasr_backend_combo.currentText(),
                'crispasr_model': self.crispasr_model_combo.currentText(),
                'crispasr_aligner': self.crispasr_aligner_combo.currentText(),
                'crispasr_param': self.param_crispasr.toPlainText().strip(),
            },
        }
        return TaskSnapshot(operation=operation, values=values)

    def _on_worker_status(self, message):
        self.setWindowTitle(f"{_('window_title')} - {message}")

    def _on_worker_thread_finished(self):
        thread = self.sender()
        completed_lan_job = None
        if self.thread is thread:
            completed_lan_job = self._active_lan_job_id
            self.thread = None
            self.worker = None
            self.cancel_token = None
            self._active_lan_job_id = None
        if completed_lan_job:
            self._finalize_lan_job(completed_lan_job)
        if self._pending_close:
            QTimer.singleShot(0, self.close)
        elif self._lan_job_queue:
            QTimer.singleShot(0, self._start_next_lan_job)

    def browse_synth_video(self):
        files, _unused = QFileDialog.getOpenFileNames(self, _("dialog_select_video"), "", "Video Files (*.mp4 *.mkv *.avi *.mov *.flv);;All Files (*)")
        if files:
            current_text = self.synth_video_files_list.toPlainText().strip()
            new_text = "\n".join(files)
            if current_text:
                self.synth_video_files_list.setText(current_text + "\n" + new_text)
            else:
                self.synth_video_files_list.setText(new_text)

    def browse_synth_srt(self):
        files, _unused = QFileDialog.getOpenFileNames(self, _("dialog_select_subtitle"), "", "Subtitle Files (*.srt *.ass *.vtt);;All Files (*)")
        if files:
            current_text = self.synth_srt_files_list.toPlainText().strip()
            new_text = "\n".join(files)
            if current_text:
                self.synth_srt_files_list.setText(current_text + "\n" + new_text)
            else:
                self.synth_srt_files_list.setText(new_text)

    def browse_output_dir(self):
        current_dir = self.output_dir_edit.text().strip() or self.default_output_dir()
        selected = QFileDialog.getExistingDirectory(self, _("dialog_select_output_dir"), current_dir)
        if selected:
            self.output_dir_edit.setText(selected)

    def update_output_dir_controls(self):
        use_input_dir = self.use_input_dir_checkbox.isChecked() if hasattr(self, 'use_input_dir_checkbox') else False
        self.output_dir_edit.setEnabled(not use_input_dir)
        self.output_dir_button.setEnabled(not use_input_dir)
        if hasattr(self, 'config_output_dir'):
            self.config_output_dir.setEnabled(not use_input_dir)
            self.config_output_dir_button.setEnabled(not use_input_dir)

    def update_segment_controls(self):
        enabled = self.enable_segment_checkbox.isChecked() if hasattr(self, 'enable_segment_checkbox') else False
        self.segment_duration_spin.setEnabled(enabled)
        if hasattr(self, 'config_segment_duration'):
            self.config_segment_duration.setEnabled(enabled)

    def update_synth_font_controls(self):
        enabled = (self.subtitle_type_combo.currentData() or "硬字幕") == "硬字幕"
        self.synth_font_label.setEnabled(enabled)
        self.subtitle_font_combo.setEnabled(enabled)
        if hasattr(self, 'config_subtitle_font'):
            self.config_subtitle_font.setEnabled(enabled)

    def _normalize_drop_paths(self, mime_data):
        paths = []
        try:
            urls = mime_data.urls()
        except Exception:
            urls = []

        if urls:
            for url in urls:
                if url.isLocalFile():
                    local_path = url.toLocalFile()
                    if local_path:
                        paths.append(local_path)
            return paths

        raw_text = mime_data.text() or ""
        if not raw_text:
            return paths

        for item in raw_text.splitlines():
            item = item.strip()
            if not item:
                continue
            if item.startswith("file://"):
                url = QtCore.QUrl(item)
                local_path = url.toLocalFile()
                if local_path:
                    paths.append(local_path)
                continue
            paths.append(item)
        return paths

    def _bind_drop_event(self, text_edit):
        text_edit.setAcceptDrops(True)
        text_edit.installEventFilter(self)
        self._drop_targets[text_edit] = text_edit
        viewport = text_edit.viewport()
        viewport.setAcceptDrops(True)
        viewport.installEventFilter(self)
        self._drop_targets[viewport] = text_edit

    def collect_font_candidates(self):
        # Scan ./font and common system font dirs for ttf/ttc/otf files
        candidates = []
        exts = {'.ttf', '.ttc', '.otf'}
        search_dirs = []
        # Windows fonts
        win_font_dir = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
        search_dirs.append(win_font_dir)
        # macOS
        search_dirs.extend([Path('/Library/Fonts'), Path.home() / 'Library/Fonts'])
        # Linux common
        search_dirs.extend([Path('/usr/share/fonts'), Path('/usr/local/share/fonts'), Path.home() / '.fonts'])

        for d in search_dirs:
            if not d.exists():
                continue
            for p in d.rglob('*'):
                if p.suffix.lower() in exts:
                    candidates.append(p.stem)  # also add family name guess

        # de-duplicate while preserving order
        seen = set()
        unique = []
        for item in candidates:
            if item in seen:
                continue
            seen.add(item)
            unique.append(item)
        return unique

    def refresh_speech_model_lists(self):
        if hasattr(self, 'crispasr_backend_combo'):
            try:
                self.refresh_crispasr_lists(query_backends=True)
            except Exception as error:
                self._emit_status(_("status_crispasr_error", error=error))
        if hasattr(self, 'asr_engine_combo'):
            self.refresh_asr_engine_lists(force_refresh=True)

        if hasattr(self, 'uvr_file'):
            current_uvr = self.uvr_file.currentText()
            uvr_lst = [i for i in os.listdir('separate') if i.endswith('onnx')]
            self.uvr_file.clear()
            self.uvr_file.addItems(uvr_lst)
            if current_uvr in uvr_lst:
                self.uvr_file.setCurrentText(current_uvr)

    def refresh_crispasr_lists(self, query_backends: bool = True):
        current_backend = self.crispasr_backend_combo.currentText()
        current_model = self.crispasr_model_combo.currentText()
        current_aligner = self.crispasr_aligner_combo.currentText()
        if query_backends:
            backends = crispasr_bridge.list_backends(CRISPASR_DIR, timeout=3)
            self._crispasr_discovery_done = True
        else:
            backends = list(crispasr_bridge.CRISPASR_BACKEND_FALLBACK)
        models = crispasr_bridge.list_models(CRISPASR_DIR)
        aligners = crispasr_bridge.list_aligners(CRISPASR_DIR)

        self.crispasr_backend_combo.blockSignals(True)
        self.crispasr_backend_combo.clear()
        self.crispasr_backend_combo.addItems(backends)
        preferred_backend = current_backend or crispasr_bridge.DEFAULT_CRISPASR_BACKEND
        backend_index = self.crispasr_backend_combo.findText(preferred_backend)
        if backend_index >= 0:
            self.crispasr_backend_combo.setCurrentIndex(backend_index)
        self.crispasr_backend_combo.blockSignals(False)

        self.crispasr_model_combo.clear()
        self.crispasr_model_combo.addItems(models)
        preferred_model = current_model or 'qwen3-asr-1.7b-q4_k.gguf'
        model_index = self.crispasr_model_combo.findText(preferred_model)
        if model_index >= 0:
            self.crispasr_model_combo.setCurrentIndex(model_index)

        self.crispasr_aligner_combo.clear()
        self.crispasr_aligner_combo.addItems(aligners)
        preferred_aligner = current_aligner or 'qwen3-forced-aligner-0.6b-q4_k.gguf'
        aligner_index = self.crispasr_aligner_combo.findText(preferred_aligner)
        if aligner_index >= 0:
            self.crispasr_aligner_combo.setCurrentIndex(aligner_index)

    def on_asr_provider_changed(self, _index: int = -1):
        use_asrlabs = (self.asr_provider_combo.currentData() == 'asrlabs')
        for widget in getattr(self, '_asrlabs_widgets', []):
            widget.setVisible(use_asrlabs)
        for widget in getattr(self, '_crispasr_widgets', []):
            widget.setVisible(not use_asrlabs)
        if not use_asrlabs and not getattr(self, '_crispasr_discovery_done', False):
            try:
                self.refresh_crispasr_lists(query_backends=True)
            except Exception as error:
                self._emit_status(_("status_crispasr_error", error=error))
        self._update_streaming_availability()
        if not self._suppress_auto_save:
            self._schedule_auto_save()

    def refresh_asr_engine_lists(self, force_refresh: bool = True):
        try:
            transcribers, _aligners = asrlabs_bridge.fetch_engine_metadata(
                force_refresh=force_refresh
            )
        except Exception as error:
            if force_refresh:
                self._emit_status(_("status_asrlabs_metadata_error", error=error))
            return

        current_engine = self.asr_engine_combo.currentData() or ''
        self.asr_engine_combo.blockSignals(True)
        self.asr_engine_combo.clear()
        self.asr_engine_combo.addItem(_("workflow_enable_transcription"), userData='')
        for item in transcribers:
            self.asr_engine_combo.addItem(
                f"{item['name']} ({item['display_name']})", userData=item['name']
            )
        default_engine = current_engine or 'faster-whisper'
        engine_index = self.asr_engine_combo.findData(default_engine)
        if engine_index >= 0:
            self.asr_engine_combo.setCurrentIndex(engine_index)
        self.asr_engine_combo.blockSignals(False)

        current_model = self.asr_model_combo.currentData() or ''
        self.asr_model_combo.clear()
        for model in asrlabs_bridge.list_transcribe_models():
            path = os.path.join('models', 'transcribe', model)
            self.asr_model_combo.addItem(model, userData=path)
        model_index = self.asr_model_combo.findData(current_model)
        if model_index >= 0:
            self.asr_model_combo.setCurrentIndex(model_index)

        current_align_model = self.align_model_combo.currentData() or ''
        self.align_model_combo.clear()
        for model in asrlabs_bridge.list_align_models():
            path = os.path.join('models', 'align', model)
            self.align_model_combo.addItem(model, userData=path)
        align_model_index = self.align_model_combo.findData(current_align_model)
        if align_model_index >= 0:
            self.align_model_combo.setCurrentIndex(align_model_index)
        self.on_asr_engine_changed()

    def on_asr_engine_changed(self, _index: int = -1):
        engine_name = self.asr_engine_combo.currentData() or ''
        current_aligner = self.align_engine_combo.currentData() or 'none'
        self.align_engine_combo.blockSignals(True)
        self.align_engine_combo.clear()
        if not engine_name:
            self.align_engine_combo.addItem(_("settings_align_no_align"), userData='none')
        else:
            meta = asrlabs_bridge.get_transcriber_meta(engine_name) or {}
            try:
                _transcribers, aligners = asrlabs_bridge.fetch_engine_metadata()
            except Exception:
                aligners = []
            if meta.get('supports_timestamps', False):
                self.align_engine_combo.addItem(_("settings_align_no_align"), userData='none')
            recommended = meta.get('recommended_aligner')
            ordered = sorted(aligners, key=lambda item: item.get('name') != recommended)
            for item in ordered:
                self.align_engine_combo.addItem(
                    f"{item['name']} ({item['display_name']})", userData=item['name']
                )
            aligner_index = self.align_engine_combo.findData(current_aligner)
            if aligner_index >= 0:
                self.align_engine_combo.setCurrentIndex(aligner_index)
        self.align_engine_combo.blockSignals(False)
        self._update_streaming_availability()

    def _update_streaming_availability(self, *_args):
        if not hasattr(self, 'streaming_checkbox') or not hasattr(self, 'asr_provider_combo'):
            return
        enabled = (
            self.asr_provider_combo.currentData() == 'asrlabs'
            and self.asr_engine_combo.currentData() == 'faster-whisper'
            and (self.align_engine_combo.currentData() or 'none') == 'none'
            and not (
                self.ai_resegment_checkbox.isChecked()
                if hasattr(self, 'ai_resegment_checkbox') else False
            )
        )
        self.streaming_checkbox.setEnabled(enabled)
        if not enabled:
            self.streaming_checkbox.setChecked(False)

    def _on_streaming_toggled(self, checked):
        if checked and self.ai_resegment_checkbox.isChecked():
            self.ai_resegment_checkbox.setChecked(False)

    def _on_ai_resegment_toggled(self, checked):
        if checked and self.streaming_checkbox.isChecked():
            self.streaming_checkbox.setChecked(False)
        self._update_streaming_availability()

    def refresh_language_model_lists(self):
        if hasattr(self, 'sakura_file'):
            current_model = self.sakura_file.currentText()
            sakura_lst = [i for i in os.listdir('llama') if i.endswith('gguf')]
            self.sakura_file.clear()
            self.sakura_file.addItems(sakura_lst)
            if current_model in sakura_lst:
                self.sakura_file.setCurrentText(current_model)

    @staticmethod
    def _auxiliary_model_value(combo: QComboBox) -> str:
        if combo.isEditable():
            return combo.currentText().strip()
        data = combo.currentData()
        if isinstance(data, str):
            return data.strip()
        return combo.currentText().strip()

    def _set_auxiliary_model_value(self, combo: QComboBox, model_name: str):
        value = (model_name or '').strip()
        index = combo.findData(value)
        combo.blockSignals(True)
        if index >= 0:
            combo.setCurrentIndex(index)
        else:
            combo.setCurrentIndex(-1)
            combo.setEditText(value)
        combo.blockSignals(False)

    def _update_auxiliary_profile_controls(
        self,
        provider_combo: QComboBox,
        model_combo: QComboBox,
        token_edit: QLineEdit,
        address_edit: QLineEdit,
        thinking_checkbox: QCheckBox | None = None,
    ):
        provider = provider_combo.currentData() or 'follow'
        self._refresh_auxiliary_model_choices(provider_combo, model_combo)
        independent = provider != 'follow'
        for widget in (model_combo, token_edit, address_edit):
            widget.setEnabled(independent)
        if independent:
            endpoint = ONLINE_TRANSLATOR_MAPPING.get(provider, '')
            address_edit.setPlaceholderText(
                endpoint or _("aux_profile_custom_address_placeholder")
            )
        if thinking_checkbox is not None:
            if provider == 'follow':
                model_name = self.gpt_model.text()
            else:
                model_name = model_combo.currentText()
            thinking_checkbox.setEnabled(model_supports_thinking(model_name))
            if not thinking_checkbox.isEnabled():
                thinking_checkbox.setChecked(False)
        self._update_thinking_availability()

    def _refresh_auxiliary_model_choices(
        self, provider_combo: QComboBox, model_combo: QComboBox
    ):
        """Keep discovered/preset models scoped to the selected provider."""
        provider = provider_combo.currentData() or 'follow'
        previous_provider = getattr(model_combo, '_auxiliary_provider', None)
        cache_key = id(model_combo)
        cache = self._auxiliary_model_cache.setdefault(cache_key, {})
        previous_text = model_combo.currentText().strip()
        if previous_provider and previous_text:
            cache[previous_provider] = previous_text

        if provider == 'Deepseek':
            choices = ['deepseek-v4-flash', 'deepseek-v4-pro']
        else:
            choices = []
        choices.extend(
            self._auxiliary_discovered_models_for_provider(
                model_combo, provider
            )
        )
        saved_value = cache.get(provider, '')
        model_combo.blockSignals(True)
        model_combo.clear()
        model_combo.setEditable(True)
        for value in dict.fromkeys(choices):
            model_combo.addItem(value, userData=value)
        if saved_value:
            model_combo.setEditText(saved_value)
        elif provider == 'Deepseek':
            model_combo.setCurrentText('deepseek-v4-flash')
        else:
            model_combo.setEditText('')
        model_combo.blockSignals(False)
        model_combo._auxiliary_provider = provider

    def _auxiliary_discovered_models_for_provider(
        self, model_combo: QComboBox, provider: str
    ) -> list[str]:
        by_provider = getattr(self, '_discovered_models_by_provider', {})
        return list(by_provider.get((id(model_combo), provider), []))

    def _update_thinking_availability(self):
        if not hasattr(self, 'deepseek_thinking_checkbox'):
            return
        main_supported = model_supports_thinking(self.gpt_model.text())
        self.deepseek_thinking_checkbox.setEnabled(main_supported)
        if not main_supported:
            self.deepseek_thinking_checkbox.setChecked(False)
        for provider_combo, model_combo, checkbox in (
            (
                getattr(self, 'ai_resegment_provider_combo', None),
                getattr(self, 'ai_resegment_model_combo', None),
                getattr(self, 'ai_resegment_thinking_checkbox', None),
            ),
            (
                getattr(self, 'proofread_provider_combo', None),
                getattr(self, 'proofread_model_combo', None),
                getattr(self, 'proofread_thinking_checkbox', None),
            ),
        ):
            if provider_combo is None or model_combo is None or checkbox is None:
                continue
            provider = provider_combo.currentData() or 'follow'
            model_name = (
                self.gpt_model.text() if provider == 'follow'
                else model_combo.currentText()
            )
            checkbox.setEnabled(model_supports_thinking(model_name))
            if not checkbox.isEnabled():
                checkbox.setChecked(False)

    @staticmethod
    def _set_provider_value(combo: QComboBox, provider: str):
        index = combo.findData(provider or 'follow')
        if index >= 0:
            combo.setCurrentIndex(index)

    def _add_discovered_auxiliary_models(self, models, target=None):
        incoming = {
            str(model).strip() for model in models if str(model).strip()
        }
        if target in ('resegment', 'proofread'):
            if target == 'resegment':
                provider_combo = self.ai_resegment_provider_combo
                model_combo = self.ai_resegment_model_combo
            else:
                provider_combo = self.proofread_provider_combo
                model_combo = self.proofread_model_combo
            provider = provider_combo.currentData() or 'follow'
            if provider != 'follow':
                by_provider = getattr(self, '_discovered_models_by_provider', {})
                key = (id(model_combo), provider)
                by_provider.setdefault(key, [])
                for model_name in incoming:
                    if model_name not in by_provider[key]:
                        by_provider[key].append(model_name)
                self._discovered_models_by_provider = by_provider
                self._refresh_auxiliary_model_choices(provider_combo, model_combo)
        discovered = set(getattr(self, '_discovered_online_models', []))
        discovered.update(incoming)
        self._discovered_online_models = sorted(discovered)

    def cancel_task(self):
        if self.cancel_token and self.thread and self.thread.isRunning():
            self._emit_status(_("status_cancelling"))
            self.cancel_token.cancel()

    def _migrate_config_txt(self):
        """从旧 config.txt 迁移到 gui_settings.yaml + .env，返回 gui_settings 字典"""
        with open('config.txt', 'r', encoding='utf-8') as f:
            lines = f.readlines()

        gpt_token = lines[3].strip() if len(lines) > 3 else ''
        _save_api_key(gpt_token)

        gui_settings = {
            'asr_model_file': lines[0].strip(),
            'asr_aligner_file': '',
            'asr_backend': DEFAULT_CRISPASR_BACKEND,
            'translator': lines[1].strip(),
            'enable_transcription': lines[0].strip() != NO_TRANSCRIPTION,
            'enable_translation': lines[1].strip() != NO_TRANSLATION,
            'language': lines[2].strip(),
            'gpt_address': lines[4].strip(),
            'gpt_model': lines[5].strip(),
            'sakura_file': lines[6].strip(),
            'sakura_mode': lines[7].strip(),
            'proxy_address': lines[8].strip(),
            'uvr_file': lines[9].strip(),
            'output_format': lines[10].strip(),
            'subtitle_font': lines[11].strip() if len(lines) > 11 else "",
            'output_dir': lines[12].strip() if len(lines) > 12 else self.default_output_dir(),
            'use_input_dir': (lines[13].strip().lower() == 'true') if len(lines) > 13 else False,
            'max_concurrent': int(lines[14].strip()) if len(lines) > 14 else 1,
            'enable_segment': (lines[15].strip().lower() == 'true') if len(lines) > 15 else False,
            'segment_duration': int(lines[16].strip()) if len(lines) > 16 else 10,
            'change_prompt_mode': lines[17].strip() if len(lines) > 17 else '不修改',
        }

        with open('gui_settings.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(gui_settings, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

        return gui_settings

    def load_config(self):
        """加载 GUI 配置（优先 gui_settings.yaml，兼容旧 config.txt 自动迁移）"""
        self._suppress_auto_save = True
        gui_settings = {}

        if os.path.exists('gui_settings.yaml'):
            with open('gui_settings.yaml', 'r', encoding='utf-8') as f:
                gui_settings = yaml.safe_load(f) or {}
            if gui_settings.get('schema_version') != GUI_SETTINGS_SCHEMA_VERSION:
                gui_settings = {}
        fresh_config = not bool(gui_settings)

        if gui_settings:
            self.enable_transcription_checkbox.setChecked(
                gui_settings.get('enable_transcription', True)
            )
            provider_index = self.asr_provider_combo.findData(
                gui_settings.get('asr_provider', 'crispasr')
            )
            if provider_index >= 0:
                self.asr_provider_combo.setCurrentIndex(provider_index)
            engine_index = self.asr_engine_combo.findData(gui_settings.get('asr_engine', ''))
            if engine_index >= 0:
                self.asr_engine_combo.setCurrentIndex(engine_index)
            model_index = self.asr_model_combo.findData(gui_settings.get('asr_model', ''))
            if model_index >= 0:
                self.asr_model_combo.setCurrentIndex(model_index)
            self.asr_device_combo.setCurrentText(gui_settings.get('asr_device', 'auto'))
            self.asr_compute_type_combo.setCurrentText(
                gui_settings.get('asr_compute_type', 'float16')
            )
            self.asr_extra_edit.setPlainText(gui_settings.get('asr_extra', ''))
            aligner_index = self.align_engine_combo.findData(
                gui_settings.get('align_engine', 'none')
            )
            if aligner_index >= 0:
                self.align_engine_combo.setCurrentIndex(aligner_index)
            align_model_index = self.align_model_combo.findData(
                gui_settings.get('align_model', '')
            )
            if align_model_index >= 0:
                self.align_model_combo.setCurrentIndex(align_model_index)
            self.align_device_combo.setCurrentText(gui_settings.get('align_device', 'auto'))
            self.align_extra_edit.setPlainText(gui_settings.get('align_extra', ''))
            for combo, value in (
                (self.crispasr_backend_combo, gui_settings.get('crispasr_backend', DEFAULT_CRISPASR_BACKEND)),
                (self.crispasr_model_combo, gui_settings.get('crispasr_model', '')),
                (self.crispasr_aligner_combo, gui_settings.get('crispasr_aligner', '')),
            ):
                index = combo.findText(value)
                if index >= 0:
                    combo.setCurrentIndex(index)
            saved_translator = gui_settings.get('translator', '')
            legacy_translation_disabled = saved_translator == NO_TRANSLATION
            if (
                saved_translator
                and not legacy_translation_disabled
                and self.translator_group.findText(saved_translator) >= 0
            ):
                self.translator_group.setCurrentText(saved_translator)
            self.enable_translation_checkbox.setChecked(gui_settings.get('enable_translation', True))
            language_index = self.transcription_lang.findData(gui_settings.get('language', 'ja'))
            if language_index >= 0:
                self.transcription_lang.setCurrentIndex(language_index)
            self.gpt_address.setText(gui_settings.get('gpt_address', ''))
            self.gpt_model.setText(gui_settings.get('gpt_model', ''))
            self._set_provider_value(
                self.ai_resegment_provider_combo,
                gui_settings.get('ai_resegment_provider', 'follow'),
            )
            self.ai_resegment_address.setText(
                gui_settings.get('ai_resegment_address', '')
            )
            self.ai_resegment_token.setText(
                _load_api_key('VOICETRANSL_RESEGMENT_API_KEY')
            )
            self._update_auxiliary_profile_controls(
                self.ai_resegment_provider_combo,
                self.ai_resegment_model_combo,
                self.ai_resegment_token,
                self.ai_resegment_address,
            )
            self._set_auxiliary_model_value(
                self.ai_resegment_model_combo,
                gui_settings.get('ai_resegment_model', ''),
            )
            self._set_provider_value(
                self.proofread_provider_combo,
                gui_settings.get('proofread_provider', 'follow'),
            )
            self.proofread_address.setText(
                gui_settings.get('proofread_address', '')
            )
            self.proofread_token.setText(
                _load_api_key('VOICETRANSL_PROOFREAD_API_KEY')
            )
            self._update_auxiliary_profile_controls(
                self.proofread_provider_combo,
                self.proofread_model_combo,
                self.proofread_token,
                self.proofread_address,
            )
            self._set_auxiliary_model_value(
                self.proofread_model_combo,
                gui_settings.get('proofread_model', ''),
            )
            self.deepseek_thinking_checkbox.setChecked(
                gui_settings.get('deepseek_thinking', False)
            )
            self.ai_resegment_thinking_checkbox.setChecked(
                gui_settings.get(
                    'ai_resegment_thinking',
                    gui_settings.get('deepseek_thinking', False),
                )
            )
            self.proofread_thinking_checkbox.setChecked(
                gui_settings.get(
                    'proofread_thinking',
                    gui_settings.get('deepseek_thinking', False),
                )
            )
            if self.sakura_file:
                self.sakura_file.setCurrentText(gui_settings.get('sakura_file', ''))
            self.sakura_mode.setText(gui_settings.get('sakura_mode', ''))
            self.proxy_address.setText(gui_settings.get('proxy_address', ''))
            if self.uvr_file:
                self.uvr_file.setCurrentText(gui_settings.get('uvr_file', ''))
            _fmt_loaded = gui_settings.get('output_format', '双语SRT')
            # 迁移旧值：中文SRT/LRC → 目标SRT/LRC
            _fmt_migrate = {'中文SRT': '目标SRT', '中文LRC': '目标LRC'}
            _fmt_loaded = _fmt_migrate.get(_fmt_loaded, _fmt_loaded)
            output_content = gui_settings.get('output_content')
            if output_content not in ('双语', '目标'):
                output_content = '双语' if _fmt_loaded.startswith('双语') else '目标'
            output_container = gui_settings.get('output_container')
            if output_container not in ('SRT', 'LRC'):
                output_container = 'LRC' if _fmt_loaded.endswith('LRC') else 'SRT'
            content_index = self.output_content.findData(output_content)
            if content_index >= 0:
                self.output_content.setCurrentIndex(content_index)
            container_index = self.output_container.findData(output_container)
            if container_index >= 0:
                self.output_container.setCurrentIndex(container_index)
            subtitle_font = gui_settings.get('subtitle_font', '')
            if subtitle_font:
                self.subtitle_font_combo.setCurrentText(subtitle_font)
            output_dir = gui_settings.get('output_dir', '')
            if output_dir:
                self.output_dir_edit.setText(output_dir)
            self.use_input_dir_checkbox.setChecked(gui_settings.get('use_input_dir', False))
            self.max_concurrent_spin.setValue(gui_settings.get('max_concurrent', 1))
            self.enable_segment_checkbox.setChecked(gui_settings.get('enable_segment', False))
            self.segment_duration_spin.setValue(gui_settings.get('segment_duration', 10))
            self.streaming_checkbox.setChecked(gui_settings.get('enable_streaming', False))
            self.proofread_checkbox.setChecked(gui_settings.get('enable_proofread', False))
            self.ai_resegment_checkbox.setChecked(
                gui_settings.get('enable_ai_resegment', False)
            )
            if hasattr(self, 'auto_shutdown_checkbox'):
                self.auto_shutdown_checkbox.setChecked(gui_settings.get('auto_shutdown', False))
            change_prompt_mode = gui_settings.get('change_prompt_mode', '')
            if hasattr(self, 'change_prompt_mode') and change_prompt_mode:
                _pm_idx = self.change_prompt_mode.findData(change_prompt_mode)
                if _pm_idx >= 0:
                    self.change_prompt_mode.setCurrentIndex(_pm_idx)
            # 日志级别过滤和详细模式
            log_filter = gui_settings.get('log_level_filter', 'ALL')
            if hasattr(self, 'log_filter_combo'):
                self.log_filter_combo.setCurrentText(log_filter)
                self._log_level_filter = log_filter
            if hasattr(self, 'verbose_checkbox'):
                self.verbose_checkbox.setChecked(gui_settings.get('verbose_mode', False))
            if hasattr(self, 'theme_selector'):
                _theme_idx = self.theme_selector.findData(
                    gui_settings.get('ui_theme', DEFAULT_UI_THEME)
                )
                if _theme_idx >= 0:
                    self.theme_selector.setCurrentIndex(_theme_idx)
            if hasattr(self, 'target_lang'):
                _tl_idx = self.target_lang.findData(gui_settings.get('target_translation_lang', 'zh-cn'))
                if _tl_idx >= 0:
                    self.target_lang.setCurrentIndex(_tl_idx)
            if hasattr(self, 'lan_port_spin'):
                self.lan_port_spin.setValue(
                    int(gui_settings.get('lan_port', DEFAULT_HTTP_PORT))
                )
                self.lan_device_name_edit.setText(
                    str(gui_settings.get('lan_device_name', socket.gethostname() or 'VoiceTransl'))
                )
                self.lan_auto_start_checkbox.setChecked(
                    bool(gui_settings.get('lan_auto_start', False))
                )
                self.lan_enabled_checkbox.setChecked(
                    bool(gui_settings.get('lan_enabled', False))
                )
                self.lan_allowed_networks_edit.setText(
                    str(gui_settings.get('lan_allowed_networks', ''))
                )
            if hasattr(self, 'media_library_root_edit'):
                default_media_root = r'D:\音声' if Path(r'D:\音声').is_dir() else ''
                self.media_library_root_edit.setText(
                    str(gui_settings.get('media_library_root', default_media_root))
                )
                self.media_library_auto_scan_checkbox.setChecked(
                    bool(gui_settings.get('media_library_auto_scan', False))
                )

        # API Key 始终从 .env 加载
        api_key = _load_api_key()
        if api_key:
            self.gpt_token.setText(api_key)
        if not gui_settings:
            self.ai_resegment_token.setText(
                _load_api_key('VOICETRANSL_RESEGMENT_API_KEY')
            )
            self.proofread_token.setText(
                _load_api_key('VOICETRANSL_PROOFREAD_API_KEY')
            )

        if not self.output_dir_edit.text().strip():
            self.output_dir_edit.setText(self.default_output_dir())

        self.update_output_dir_controls()
        self.on_asr_provider_changed()
        self._update_streaming_availability()

        if os.path.exists('crispasr/param.txt'):
            with open('crispasr/param.txt', 'r', encoding='utf-8') as f:
                self.param_crispasr.setPlainText(f.read())

        if os.path.exists('llama/param.txt'):
            with open('llama/param.txt', 'r', encoding='utf-8') as f:
                self.param_llama.setPlainText(f.read())

        if os.path.exists('project/dict_pre.txt'):
            with open('project/dict_pre.txt', 'r', encoding='utf-8') as f:
                self.before_dict.setPlainText(f.read())

        if os.path.exists('project/dict_gpt.txt'):
            with open('project/dict_gpt.txt', 'r', encoding='utf-8') as f:
                self.gpt_dict.setPlainText(f.read())

        if os.path.exists('project/dict_after.txt'):
            with open('project/dict_after.txt', 'r', encoding='utf-8') as f:
                self.after_dict.setPlainText(f.read())

        # 从 config.yaml 加载 prompt 设置
        try:
            if os.path.exists('project/config.yaml'):
                with open('project/config.yaml', 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f) or {}
                common_cfg = cfg.get('common', {})

                change_prompt_val = common_cfg.get('gpt.change_prompt', 'no')
                mode_reverse_mapping = {
                    'no': '不修改',
                    'AdditionalPrompt': '追加',
                    'OverwritePrompt': '覆盖'
                }
                if hasattr(self, 'change_prompt_mode'):
                    _pm_val = mode_reverse_mapping.get(change_prompt_val, '不修改')
                    _pm_idx = self.change_prompt_mode.findData(_pm_val)
                    if _pm_idx >= 0:
                        self.change_prompt_mode.setCurrentIndex(_pm_idx)

                prompt_content = common_cfg.get('gpt.prompt_content', '')
                if hasattr(self, 'extra_prompt') and prompt_content:
                    self.extra_prompt.setPlainText(prompt_content)
        except Exception:
            pass
        finally:
            self._suppress_auto_save = False
        if fresh_config:
            self.save_config(silent=True)

    def _refresh_lan_profile_cache(self):
        if not hasattr(self, 'asr_provider_combo'):
            return
        snapshot = self._capture_task_snapshot('run')
        values = dict(snapshot.values)
        asr_config = dict(values.get('asr_config', {}))
        provider = asr_config.get('provider', 'crispasr')
        asr_model = (
            asr_config.get('crispasr_model', '')
            if provider == 'crispasr'
            else asr_config.get('asr_model', '')
        )
        ready = bool(
            values.get('enable_transcription')
            and values.get('enable_translation')
            and asr_model
            and values.get('translator') not in ('', NO_TRANSLATION)
        )
        public = {
            'api_version': 1,
            'asr_provider': provider,
            'asr_engine': (
                asr_config.get('crispasr_backend', '')
                if provider == 'crispasr'
                else asr_config.get('asr_engine', '')
            ),
            'asr_model': os.path.basename(str(asr_model)),
            'source_language': str(values.get('language', 'ja')),
            'target_language': str(values.get('target_lang', 'zh-cn')),
            'translator': str(values.get('translator', '')),
            'translation_model': str(values.get('gpt_model', '')),
            'ai_resegment': bool(values.get('enable_ai_resegment', False)),
            'proofread': bool(values.get('enable_proofread', False)),
            'streaming': bool(values.get('enable_streaming', False)),
            'output': 'bilingual_srt',
            'ready': ready,
        }
        revision_source = json.dumps(public, ensure_ascii=False, sort_keys=True)
        public['revision'] = hashlib.sha256(
            revision_source.encode('utf-8')
        ).hexdigest()[:16]
        with self._lan_profile_lock:
            self._lan_profile_cache = {
                'public': json.loads(json.dumps(public, ensure_ascii=False)),
                'snapshot': values,
            }

    def _lan_profile_provider(self):
        with self._lan_profile_lock:
            return {
                'public': dict(self._lan_profile_cache.get('public', {})),
                'snapshot': dict(self._lan_profile_cache.get('snapshot', {})),
            }

    @staticmethod
    def _local_ipv4_addresses():
        addresses = set()
        try:
            for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                address = item[4][0]
                if not address.startswith('127.'):
                    addresses.add(address)
        except OSError:
            pass
        return sorted(addresses)

    def _initialize_media_library_ui(self):
        if not hasattr(self, 'media_library_root_edit'):
            return
        try:
            status = self.media_library.status()
            if not self.media_library_root_edit.text().strip() and status.get('root'):
                self.media_library_root_edit.setText(str(status['root']))
            self._refresh_media_library_status(status)
        except Exception as error:
            self.media_library_status_label.setText(
                _("media_library_status_error", error=error)
            )
        if self.media_library_auto_scan_checkbox.isChecked():
            QTimer.singleShot(500, self._start_media_library_scan)

    def _browse_media_library_root(self):
        selected = QFileDialog.getExistingDirectory(
            self,
            _("media_library_browse"),
            self.media_library_root_edit.text().strip() or str(Path.cwd()),
        )
        if selected:
            self.media_library_root_edit.setText(selected)
            self._schedule_auto_save()

    def _start_media_library_scan(self):
        if self._media_scan_thread and self._media_scan_thread.is_alive():
            return
        root = Path(self.media_library_root_edit.text().strip()).expanduser()
        if not root.is_dir():
            self.media_library_status_label.setText(
                _("media_library_status_error", error=_("media_library_invalid_root"))
            )
            return
        self.media_library_scan_button.setEnabled(False)
        self.media_library_status_label.setText(_("media_library_status_scanning"))
        self._schedule_auto_save()

        def run_scan():
            try:
                preview = MediaLibraryScanner(
                    known_titles=self.media_library.title_aliases()
                ).scan(root)
                write_preview(
                    preview,
                    Path('project') / 'cache' / 'media_library_preview.json',
                )
                self.media_library.apply_preview(preview)
                self.media_scan_finished.emit(self.media_library.status(), None)
            except Exception as error:
                self.media_scan_finished.emit(None, str(error))

        self._media_scan_thread = threading.Thread(
            target=run_scan,
            name='media-library-scan',
            daemon=True,
        )
        self._media_scan_thread.start()

    def _on_media_scan_finished(self, status, error):
        self._media_scan_thread = None
        self.media_library_scan_button.setEnabled(True)
        if error:
            self.media_library_status_label.setText(
                _("media_library_status_error", error=error)
            )
            return
        self._refresh_media_library_status(status or {})
        self._emit_status(_("media_library_scan_completed"))

    def _refresh_media_library_status(self, status=None):
        status = status or self.media_library.status()
        works = int(status.get('works', 0) or 0)
        if works <= 0:
            self.media_library_status_label.setText(_("media_library_status_empty"))
            return
        self.media_library_status_label.setText(_(
            "media_library_status_ready",
            works=works,
            assets=int(status.get('assets', 0) or 0),
            audio=int(status.get('audio', 0) or 0),
            subtitles=int(status.get('subtitles', 0) or 0),
            images=int(status.get('images', 0) or 0),
            documents=int(status.get('documents', 0) or 0),
            bonus=int(status.get('bonus', 0) or 0),
            incomplete=int(status.get('incomplete', 0) or 0),
        ))

    def _new_lan_service(self):
        allowed_networks = [
            value.strip()
            for value in re.split(r'[,;\s]+', self.lan_allowed_networks_edit.text())
            if value.strip()
        ]
        return LanService(
            root=Path('project') / 'cache' / 'lan_jobs',
            profile_provider=self._lan_profile_provider,
            job_ready=lambda job_id: self.lan_job_ready.emit(job_id),
            cancel_job=lambda job_id: self.lan_cancel_requested.emit(job_id),
            device_name=self.lan_device_name_edit.text().strip() or 'VoiceTransl',
            port=self.lan_port_spin.value(),
            state_dir=Path('project') / 'cache' / 'lan_state',
            media_library=self.media_library,
            allowed_networks=allowed_networks,
        )

    def _initialize_lan_service(self):
        self._refresh_lan_profile_cache()
        self.lan_service = self._new_lan_service()
        should_start = bool(
            self.lan_enabled_checkbox.isChecked()
            or self.lan_auto_start_checkbox.isChecked()
        )
        if should_start:
            self._start_lan_service()
        else:
            self._refresh_lan_ui()
        self._lan_ui_timer = QTimer(self)
        self._lan_ui_timer.timeout.connect(self._refresh_lan_ui)
        self._lan_ui_timer.start(1000)

    def _start_lan_service(self):
        try:
            if self.lan_service is None:
                self.lan_service = self._new_lan_service()
            self.lan_service.start()
            self.lan_enabled_checkbox.blockSignals(True)
            self.lan_enabled_checkbox.setChecked(True)
            self.lan_enabled_checkbox.blockSignals(False)
            self._refresh_lan_ui()
            self._schedule_auto_save()
        except Exception as error:
            self.lan_enabled_checkbox.blockSignals(True)
            self.lan_enabled_checkbox.setChecked(False)
            self.lan_enabled_checkbox.blockSignals(False)
            self.lan_status_label.setText(_("lan_status_error", error=error))
            self._emit_status(_("lan_status_error", error=error))

    def _stop_lan_service(self):
        service = self.lan_service
        if service is not None:
            service.stop()
        self.lan_enabled_checkbox.blockSignals(True)
        self.lan_enabled_checkbox.setChecked(False)
        self.lan_enabled_checkbox.blockSignals(False)
        self._refresh_lan_ui()
        self._schedule_auto_save()

    def _restart_lan_service(self):
        try:
            if self.lan_service is not None:
                self.lan_service.stop()
            self.lan_service = self._new_lan_service()
            self._start_lan_service()
        except Exception as error:
            self.lan_status_label.setText(_("lan_status_error", error=error))

    def _on_lan_enabled_toggled(self, enabled):
        if self._suppress_auto_save or self.lan_service is None:
            return
        if enabled:
            self._restart_lan_service()
        else:
            self._stop_lan_service()

    def _rotate_lan_pair_code(self):
        if self.lan_service and self.lan_service.running:
            self.lan_service.rotate_pair_code()
            self._refresh_lan_ui()

    def _revoke_lan_device(self):
        if not self.lan_service:
            return
        device_id = self.lan_devices_combo.currentData()
        if device_id:
            self.lan_service.revoke(str(device_id))
            self._refresh_lan_ui()

    def _refresh_lan_device_permission(self, *_args):
        if not hasattr(self, 'lan_file_management_checkbox'):
            return
        device_id = self.lan_devices_combo.currentData()
        enabled = False
        if self.lan_service and device_id:
            enabled = self.lan_service.can_manage_files(str(device_id))
        self.lan_file_management_checkbox.blockSignals(True)
        self.lan_file_management_checkbox.setChecked(enabled)
        self.lan_file_management_checkbox.setEnabled(bool(device_id))
        self.lan_file_management_checkbox.blockSignals(False)

    def _set_lan_device_file_management(self, enabled):
        if not self.lan_service:
            return
        device_id = self.lan_devices_combo.currentData()
        if not device_id:
            return
        try:
            self.lan_service.set_file_management(str(device_id), bool(enabled))
        except KeyError:
            self._refresh_lan_ui()

    def _refresh_lan_ui(self):
        if not hasattr(self, 'lan_status_label'):
            return
        service = self.lan_service
        running = bool(service and service.running)
        self.lan_status_label.setText(
            _("lan_status_running") if running else _("lan_status_stopped")
        )
        addresses = self._local_ipv4_addresses()
        port = service.port if service else self.lan_port_spin.value()
        self.lan_address_label.setText(
            _("lan_address", address=(addresses[0] if addresses else '127.0.0.1'), port=port)
        )
        self.lan_pair_code_label.setText(
            _("lan_pair_code", code=(service.pair_code if running else '------'))
        )
        current = self.lan_devices_combo.currentData()
        devices = service.paired_devices() if service else []
        self.lan_devices_combo.blockSignals(True)
        self.lan_devices_combo.clear()
        for device in devices:
            self.lan_devices_combo.addItem(device['name'], userData=device['id'])
        if current:
            index = self.lan_devices_combo.findData(current)
            if index >= 0:
                self.lan_devices_combo.setCurrentIndex(index)
        self.lan_devices_combo.blockSignals(False)
        self.lan_revoke_button.setEnabled(bool(devices))
        self._refresh_lan_device_permission()
        if hasattr(self, 'phone_queue_label'):
            active_name = ''
            if service and self._active_lan_job_id:
                active_job = service.registry.public(self._active_lan_job_id) or {}
                active_name = str(active_job.get('filename', '') or '')
            queued = len(self._lan_job_queue)
            if active_name or queued:
                self.phone_queue_label.setText(_(
                    "phone_queue_status",
                    active=active_name or _("phone_queue_waiting"),
                    queued=queued,
                ))
            else:
                self.phone_queue_label.setText(_("phone_queue_idle"))

    def _on_lan_job_ready(self, job_id):
        if not self.lan_service:
            return
        job = self.lan_service.registry.public(job_id)
        if not job or job.get('state') != 'queued':
            return
        if job_id not in self._lan_job_queue and job_id != self._active_lan_job_id:
            self._lan_job_queue.append(job_id)
        self._start_next_lan_job()

    def _start_next_lan_job(self):
        if self.thread is not None and self.thread.isRunning():
            return
        if not self.lan_service:
            return
        while self._lan_job_queue:
            job_id = self._lan_job_queue.pop(0)
            job = self.lan_service.registry.public(job_id)
            snapshot_values = self.lan_service.registry.snapshot(job_id)
            if not job or job.get('state') != 'queued':
                continue
            if not snapshot_values:
                self.lan_service.registry.update(
                    job_id, state='failed', phase='failed', error='Task snapshot is unavailable'
                )
                continue
            input_path, output_dir = self.lan_service.registry.paths(job_id)
            snapshot_values.update({
                'input_files': str(input_path),
                'output_dir': str(output_dir),
                'use_input_dir': False,
                'output_format': '双语SRT',
                'auto_shutdown': False,
            })
            snapshot = TaskSnapshot(operation='run', values=snapshot_values)
            self._active_lan_job_id = job_id
            self._lan_task_outcomes[job_id] = 'running'
            self.lan_service.registry.update(
                job_id, state='running', phase='starting', current=0, total=1
            )
            queue = self.lan_service.forwarding_queue(self.msg_queue, job_id)
            self._start_worker_task(
                'run', _("lan_task_name", filename=job.get('filename', '')),
                prepared_snapshot=snapshot,
                message_queue=queue,
                lan_job_id=job_id,
            )
            return

    def _record_lan_outcome(self, job_id, outcome):
        self._lan_task_outcomes[job_id] = str(outcome or 'error')

    def _finalize_lan_job(self, job_id):
        service = self.lan_service
        if not service:
            return
        outcome = self._lan_task_outcomes.pop(job_id, 'error')
        if outcome != 'success':
            state = 'cancelled' if outcome == 'cancelled' else 'failed'
            service.registry.update(
                job_id, state=state, phase=state,
                error='' if state == 'cancelled' else 'VoiceTransl task failed',
            )
            return
        job = service.registry.public(job_id) or {}
        _input, output_dir = service.registry.paths(job_id)
        profile = job.get('profile', {})
        source_language = str(profile.get('source_language', ''))
        target_language = str(profile.get('target_language', ''))
        all_srt = sorted(output_dir.rglob('*.srt'), key=lambda path: path.stat().st_mtime)
        combined = [path for path in all_srt if path.name.endswith('.combine.srt')]
        translated = [path for path in all_srt if path.name.endswith('.tg.srt')]
        originals = [
            path for path in all_srt
            if not path.name.endswith('.combine.srt') and not path.name.endswith('.tg.srt')
        ]
        artifacts = []
        if combined:
            artifacts.append(build_artifact(combined[-1], 'combined'))
        if originals:
            artifacts.append(build_artifact(originals[-1], 'source', source_language))
        if translated:
            artifacts.append(build_artifact(translated[-1], 'target', target_language))
        if not artifacts:
            service.registry.update(
                job_id, state='failed', phase='failed', error='No SRT output was generated'
            )
            return
        service.registry.set_artifacts(job_id, artifacts)

    def _on_lan_cancel_requested(self, job_id):
        if not self.lan_service:
            return
        if job_id in self._lan_job_queue:
            self._lan_job_queue = [item for item in self._lan_job_queue if item != job_id]
            self.lan_service.registry.update(
                job_id, state='cancelled', phase='cancelled', error=''
            )
            return
        if self._active_lan_job_id == job_id and self.cancel_token:
            self.cancel_token.cancel()
            self.lan_service.registry.update(job_id, phase='cancelling')

    def setup_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._consume_messages)
        self.timer.start(100)

    def init_system_tray(self):
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self.tray_icon = None
            return

        self.tray_icon = QSystemTrayIcon(self.windowIcon(), self)
        self.tray_icon.setToolTip(_("tray_tooltip"))

        tray_menu = QMenu(self)
        action_restore = QAction(_("tray_show"), self)
        action_quit = QAction(_("tray_quit"), self)
        action_restore.triggered.connect(self.restore_from_tray)
        action_quit.triggered.connect(self.close)

        tray_menu.addAction(action_restore)
        tray_menu.addSeparator()
        tray_menu.addAction(action_quit)
        self.tray_icon.setContextMenu(tray_menu)
        self.tray_icon.activated.connect(self.on_tray_activated)
        self.tray_icon.show()

    def restore_from_tray(self):
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.Trigger:
            self.restore_from_tray()

    def _consume_messages(self):
        """Show status in the shared footer and keep Tasks as the full history."""
        if not hasattr(self, 'msg_queue'):
            return

        # 每次只处理有限数量，并合并写入文本框，避免大量日志到达时
        # Qt 主线程长时间逐条排版而被 Windows 标记为“未响应”。
        entries = self.msg_queue.drain(max_items=200, time_budget_ms=4)
        if not entries:
            return

        status_lines = []
        detail_lines = []
        for target, text in entries:
            if UIMessageQueue.is_completion_entry(target):
                completion_msg = _("status_all_done")
                detail_lines.append(completion_msg)
                status_lines.append(completion_msg)
            elif target == 'status':
                if not status_lines or status_lines[-1] != text:
                    status_lines.append(text)
            elif target == 'detail':
                # 应用日志级别过滤
                if self._log_level_filter != 'ALL':
                    if not _line_passes_filter(text, self._log_level_filter):
                        continue
                detail_lines.append(text)
            elif target == 'characters':
                self._update_received_characters(text)
            elif target == 'progress':
                self._update_task_progress(text)

        if status_lines:
            self.shared_progress_view.appendPlainText('\n'.join(status_lines))
        if detail_lines:
            self.log_display.appendPlainText('\n'.join(detail_lines))

        # Keep the aggregate history and the shared compact feed at the end.
        for widget in (self.log_display, self.shared_progress_view):
            scrollbar = widget.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def _update_received_characters(self, payload: str):
        try:
            event = json.loads(payload)
            request_id = str(event["request"])
            characters = max(0, int(event.get("characters", 0)))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return
        self._output_character_requests[request_id] = {
            "characters": characters,
            "final": bool(event.get("final", False)),
        }
        received = sum(
            item["characters"]
            for item in self._output_character_requests.values()
        )
        self.shared_character_label.setText(
            _("progress_characters", count=f"{received:,}")
        )

    def _update_task_progress(self, payload: str):
        try:
            event = json.loads(payload)
            kind = str(event.get("kind", "stage"))
            if kind == "files":
                total = max(0, int(event.get("total", 0)))
                completed = max(0, min(total, int(event.get("completed", 0))))
                if total == getattr(self, '_task_file_total', 0):
                    completed = max(
                        completed, getattr(self, '_task_file_completed', 0)
                    )
                self._task_file_total = total
                self._task_file_completed = completed
                visible = bool(event.get("visible", total > 0)) and total > 0
                self.shared_file_label.setText(_(
                    "progress_files",
                    completed=f"{completed:,}",
                    total=f"{total:,}",
                ))
                self.shared_file_label.setVisible(visible)
                return
            if kind != "stage":
                return
            stage_id = int(event["stage_id"])
            total = max(1, int(event["total"]))
            current = max(0, min(total, int(event.get("current", 0))))
            phase = str(event.get("phase", "")).strip()
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return

        active_stage_id = getattr(self, '_task_progress_stage_id', 0)
        if stage_id < active_stage_id:
            return
        if stage_id == active_stage_id:
            current = min(
                total,
                max(current, getattr(self, '_task_progress_current', 0)),
            )
        else:
            # A stage identity, rather than its display name, controls reset.
            # This lets the second file's "AI segmentation" start at zero.
            self._task_progress_stage_id = stage_id
        self._task_progress_phase = phase
        self._task_progress_current = current
        self.shared_progress_bar.setRange(0, total)
        self.shared_progress_bar.setValue(current)
        prefix = f"{phase}  " if phase else ""
        self.shared_progress_bar.setFormat(
            f"{prefix}{current:,}/{total:,}  (%p%)"
        )

    def closeEvent(self, event):
        """Cancel asynchronously; never wait for a worker in the Qt event loop."""
        try:
            self.save_config(silent=True)
        except Exception:
            pass
        if not self._lan_shutdown_started and self.lan_service is not None:
            self._lan_shutdown_started = True
            service = self.lan_service
            for job_id in list(self._lan_job_queue):
                service.registry.update(
                    job_id, state='cancelled', phase='cancelled', error=''
                )
            self._lan_job_queue.clear()
            threading.Thread(
                target=service.stop,
                name='voicetransl-lan-stop',
                daemon=True,
            ).start()
        if self.thread and self.thread.isRunning():
            self._pending_close = True
            self.cancel_task()
            event.ignore()
            return

        self._pending_close = False
        self.timer.stop()
        if getattr(self, 'tray_icon', None):
            self.tray_icon.hide()
        event.accept()

    def shutdown_children(self):
        """Request cooperative shutdown without blocking the UI thread."""
        if self.cancel_token:
            self.cancel_token.cancel()

    def changeEvent(self, event):
        # Hide window instead of cluttering the taskbar when minimized
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.Type.WindowStateChange and self.isMinimized():
            if getattr(self, 'tray_icon', None):
                QTimer.singleShot(0, self.hide)
                self.tray_icon.showMessage(
                    "VoiceTransl", _("tray_minimized"),
                    QSystemTrayIcon.MessageIcon.Information, 2000,
                )

        if event.type() == QtCore.QEvent.Type.ActivationChange and not self.isActiveWindow():
            self._schedule_auto_save()

    def initLogTab(self):
        self.log_tab = Widget("Log", self)
        self.log_layout = self.log_tab.vBoxLayout

        self.log_file_label = BodyLabel(_("log_file_label"))
        self.log_layout.addWidget(self.log_file_label)

        # 日志过滤工具栏
        filter_layout = QHBoxLayout()
        self.log_filter_label = QLabel(_("log_filter_label"))
        self.log_filter_combo = QComboBox()
        self.log_filter_combo.addItems(["ALL", "INFO+", "WARNING+", "ERROR+"])
        self.log_filter_combo.currentTextChanged.connect(self._on_log_filter_changed)

        # 详细日志模式复选框
        self.verbose_checkbox = QCheckBox(_("log_verbose_checkbox"))
        self.verbose_checkbox.setToolTip(_("log_verbose_tooltip"))
        self.verbose_checkbox.stateChanged.connect(self._on_verbose_changed)

        filter_layout.addWidget(self.log_filter_label)
        filter_layout.addWidget(self.log_filter_combo)
        filter_layout.addStretch()
        filter_layout.addWidget(self.verbose_checkbox)
        self.log_layout.addLayout(filter_layout)

        # log
        self.log_display = QPlainTextEdit(self)
        self.log_display.setReadOnly(True)
        self.log_display.document().setMaximumBlockCount(10000)
        self.log_display.setStyleSheet("font-family: Consolas, Monospace; font-size: 10pt;")
        self.log_layout.addWidget(self.log_display)

        # open log file button
        self.open_log_button = QPushButton(_("log_open_btn"))
        self.open_log_button.clicked.connect(lambda: open_path(LOG_PATH))
        self.log_layout.addWidget(self.open_log_button)

    def _on_log_filter_changed(self, filter_text: str):
        """级别过滤变更：仅影响后续到达的 detail 消息（已显示内容不变）"""
        self._log_level_filter = filter_text

    def _on_verbose_changed(self, state: int):
        """详细模式复选框变更"""
        ConcurrentTranslationPool.verbose_galtransl = bool(state)

    def initAboutTab(self):
        self.about_tab = Widget("About", self)
        self.about_layout = self.about_tab.vBoxLayout
        self.about_layout.setContentsMargins(24, 18, 24, 20)
        self.about_layout.setSpacing(10)

        body = QHBoxLayout()
        body.setSpacing(12)
        content_panel = QFrame(self.about_tab)
        content_panel.setFrameShape(QFrame.Shape.NoFrame)
        content = QVBoxLayout(content_panel)
        content.setContentsMargins(14, 12, 14, 14)
        content.setSpacing(8)

        # avatar image
        avatar_path = os.path.abspath(os.path.join(os.getcwd(), 'avatar.png'))
        self.avatar_label = ScaledPixmapLabel(QPixmap(avatar_path))
        self.avatar_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content.addWidget(self.avatar_label, 1)

        # welcome title
        self.about_title_label = TitleLabel(_("about_title"))
        content.addWidget(self.about_title_label)

        body.addWidget(content_panel, 1)

        def open_url(url):
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))

        # start
        self.start_button = QPushButton(_("about_start_btn"))
        self.start_button.clicked.connect(
            lambda: self.top_tabs.setCurrentWidget(self.input_output_tab)
        )

        # wiki button
        self.btn_wiki = QPushButton(_("about_wiki_btn"))
        self.btn_wiki.clicked.connect(lambda: open_url("https://github.com/shinnpuru/VoiceTransl"))

        # sponsorship buttons
        self.about_sponsor_title = SubtitleLabel(_("about_sponsor_title"))
        self.btn_afdian = QPushButton(_("about_afdian_btn"))
        self.btn_bilibili = QPushButton(_("about_bilibili_btn"))
        self.btn_kofi = QPushButton(_("about_kofi_btn"))
        self.btn_afdian.clicked.connect(lambda: open_url("https://afdian.com/a/shinnpuru"))
        self.btn_bilibili.clicked.connect(lambda: open_url("https://space.bilibili.com/36464441"))
        self.btn_kofi.clicked.connect(lambda: open_url("https://ko-fi.com/U7U018MISY"))

        action_panel = QFrame(self.about_tab)
        action_panel.setFrameShape(QFrame.Shape.NoFrame)
        action_panel.setMinimumWidth(190)
        action_panel.setMaximumWidth(230)
        action_layout = QVBoxLayout(action_panel)
        action_layout.setContentsMargins(12, 12, 12, 12)
        action_layout.setSpacing(8)
        action_layout.addWidget(SubtitleLabel(_("actions_title")))
        action_layout.addWidget(self.start_button)
        action_layout.addWidget(self.btn_wiki)
        action_layout.addWidget(self.about_sponsor_title)
        action_layout.addWidget(self.btn_afdian)
        action_layout.addWidget(self.btn_bilibili)
        action_layout.addWidget(self.btn_kofi)
        action_layout.addStretch()
        body.addWidget(action_panel)
        self.about_layout.addLayout(body, 1)

    def _on_language_changed(self, index: int):
        """界面语言变更：保存设置并提示重启后生效"""
        if self._suppress_auto_save:
            return
        lang_map = {0: "zh", 1: "en", 2: "ja"}
        lang_code = lang_map.get(index, "zh")
        set_language(lang_code)
        # 触发防抖自动保存（save_config 会写入 ui_language）
        self._schedule_auto_save()
        # 通知用户：状态栏消息 + 托盘气泡
        lang_name = self.lang_selector.currentText() if hasattr(self, 'lang_selector') else lang_code
        self._emit_status(_("status_lang_changed", lang=lang_name))
        if getattr(self, 'tray_icon', None):
            self.tray_icon.showMessage(
                _("notify_lang_changed_title"),
                _("notify_lang_changed_msg"),
                QSystemTrayIcon.MessageIcon.Information,
                3000
            )

    def _on_theme_changed(self, _index: int):
        theme = self.theme_selector.currentData()
        application = QApplication.instance()
        if application and theme:
            apply_material_theme(application, theme)
        if not self._suppress_auto_save:
            self._schedule_auto_save()

    def initInputOutputTab(self):
        self.input_output_tab = Widget("Home", self)
        self.input_output_layout = self.input_output_tab.vBoxLayout

        # Language Selector
        lang_layout = QHBoxLayout()
        self.lang_selector_label = BodyLabel(_("lang_selector_label"))
        lang_layout.addWidget(self.lang_selector_label)
        self.lang_selector = QComboBox()
        self.lang_selector.addItem(_("lang_zh"))
        self.lang_selector.addItem(_("lang_en"))
        self.lang_selector.addItem(_("lang_ja"))
        # Set current index based on saved/current language
        lang_map = {"zh": 0, "en": 1, "ja": 2}
        self.lang_selector.setCurrentIndex(lang_map.get(get_language(), 0))
        self.lang_selector.currentIndexChanged.connect(self._on_language_changed)
        lang_layout.addWidget(self.lang_selector)
        lang_layout.addStretch()

        # Transcription Language
        self.io_transcription_lang_label = BodyLabel(_("io_transcription_lang_label"))
        lang_layout.addWidget(self.io_transcription_lang_label)
        self.transcription_lang = QComboBox()
        TRANS_LANG_CODES = ['ja', 'en', 'ko', 'ru', 'fr', 'zh']
        for code in TRANS_LANG_CODES:
            self.transcription_lang.addItem(_(f"target_lang_{code.replace('-', '_')}"), userData=code)
        lang_layout.addWidget(self.transcription_lang)
        lang_layout.addStretch()

        # Target Translation Language
        self.io_target_lang_label = BodyLabel(_("io_target_lang_label"))
        lang_layout.addWidget(self.io_target_lang_label)
        self.target_lang = QComboBox()
        TARGET_LANG_CODES = ['zh-cn', 'zh-tw', 'en', 'ja', 'ko', 'ru', 'fr']
        for code in TARGET_LANG_CODES:
            self.target_lang.addItem(_(f"target_lang_{code.replace('-', '_')}"), userData=code)
        lang_layout.addWidget(self.target_lang)
        self.input_output_layout.addLayout(lang_layout)

        processing_layout = QHBoxLayout()
        self.enable_transcription_checkbox = QCheckBox(_("workflow_enable_transcription"))
        self.enable_transcription_checkbox.setChecked(True)
        processing_layout.addWidget(self.enable_transcription_checkbox)
        processing_layout.addStretch()
        self.enable_translation_checkbox = QCheckBox(_("workflow_enable_translation"))
        self.enable_translation_checkbox.setChecked(True)
        processing_layout.addWidget(self.enable_translation_checkbox)
        processing_layout.addStretch()
        self.input_output_layout.addLayout(processing_layout)

        # Input Section (local files or URLs)
        self.io_input_label = BodyLabel(_("io_input_label"))
        self.io_input_label.setToolTip(_("tip_io_input"))
        self.input_output_layout.addWidget(self.io_input_label)
        self.input_files_list = QTextEdit()
        self.input_files_list.setAcceptDrops(True)
        self._bind_drop_event(self.input_files_list)
        self.input_files_list.setPlaceholderText(_("io_input_placeholder"))
        self.input_files_list.setToolTip(_("tip_io_input"))
        self.input_output_layout.addWidget(self.input_files_list)

        # Segment Section
        segment_layout = QHBoxLayout()
        self.enable_segment_checkbox = QCheckBox(_("io_segment_checkbox"))
        self.enable_segment_checkbox.setToolTip(_("tip_io_segment"))
        self.enable_segment_checkbox.stateChanged.connect(self.update_segment_controls)
        segment_layout.addWidget(self.enable_segment_checkbox)
        self.io_segment_duration_label = BodyLabel(_("io_segment_duration_label"))
        self.io_segment_duration_label.setToolTip(_("tip_io_segment_duration"))
        segment_layout.addWidget(self.io_segment_duration_label)
        self.segment_duration_spin = QSpinBox()
        self.segment_duration_spin.setRange(1, 20)
        self.segment_duration_spin.setValue(10)
        self.segment_duration_spin.setEnabled(False)
        self.segment_duration_spin.setToolTip(_("tip_io_segment_duration"))
        segment_layout.addWidget(self.segment_duration_spin)
        self.streaming_checkbox = QCheckBox(_("io_streaming_checkbox"))
        self.streaming_checkbox.setChecked(False)
        self.streaming_checkbox.toggled.connect(self._on_streaming_toggled)
        segment_layout.addWidget(self.streaming_checkbox)
        self.ai_resegment_checkbox = QCheckBox(_("io_ai_resegment_checkbox"))
        self.ai_resegment_checkbox.setChecked(False)
        self.ai_resegment_checkbox.setToolTip(_("io_ai_resegment_tooltip"))
        self.ai_resegment_checkbox.toggled.connect(self._on_ai_resegment_toggled)
        segment_layout.addWidget(self.ai_resegment_checkbox)
        self.proofread_checkbox = QCheckBox(_("io_proofread_checkbox"))
        self.proofread_checkbox.setChecked(False)
        self.proofread_checkbox.setToolTip(_("io_proofread_tooltip"))
        segment_layout.addWidget(self.proofread_checkbox)
        segment_layout.addStretch()
        self.input_output_layout.addLayout(segment_layout)

        # Proxy Section
        self.io_proxy_label = BodyLabel(_("io_proxy_label"))
        self.io_proxy_label.setToolTip(_("tip_io_proxy"))
        self.input_output_layout.addWidget(self.io_proxy_label)
        self.proxy_address = QLineEdit()
        self.proxy_address.setPlaceholderText(_("io_proxy_placeholder"))
        self.proxy_address.setToolTip(_("tip_io_proxy"))
        self.input_output_layout.addWidget(self.proxy_address)

        # Output Directory Section
        self.io_output_dir_label = BodyLabel(_("io_output_dir_label"))
        self.io_output_dir_label.setToolTip(_("tip_io_output_dir"))
        self.input_output_layout.addWidget(self.io_output_dir_label)
        output_dir_layout = QHBoxLayout()
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText(self.default_output_dir())
        self.output_dir_edit.setText(self.default_output_dir())
        self.output_dir_edit.setToolTip(_("tip_io_output_dir"))
        output_dir_layout.addWidget(self.output_dir_edit)
        self.output_dir_button = QPushButton(_("io_browse_dir_btn"))
        self.output_dir_button.clicked.connect(self.browse_output_dir)
        output_dir_layout.addWidget(self.output_dir_button)
        self.input_output_layout.addLayout(output_dir_layout)

        selection_layout = QHBoxLayout()
        self.use_input_dir_checkbox = QCheckBox(_("io_use_input_dir_checkbox"))
        self.use_input_dir_checkbox.setToolTip(_("tip_io_use_input_dir"))
        self.use_input_dir_checkbox.stateChanged.connect(self.update_output_dir_controls)
        selection_layout.addWidget(self.use_input_dir_checkbox)
        selection_layout.addStretch()
        self.auto_shutdown_checkbox = QCheckBox(_("io_auto_shutdown_checkbox"))
        selection_layout.addWidget(self.auto_shutdown_checkbox)
        selection_layout.addStretch()
        self.input_output_layout.addLayout(selection_layout)
        
        # Subtitle content and container are independent.  With translation
        # disabled the processing pipeline automatically outputs the original.
        self.io_format_label = BodyLabel(_("io_output_content_label"))
        self.input_output_layout.addWidget(self.io_format_label)
        self.output_content = QComboBox()
        self.output_content.addItem(_("output_content_bilingual"), userData='双语')
        self.output_content.addItem(_("output_content_target"), userData='目标')
        self.input_output_layout.addWidget(self.output_content)

        self.io_container_label = BodyLabel(_("io_output_container_label"))
        self.input_output_layout.addWidget(self.io_container_label)
        self.output_container = QComboBox()
        self.output_container.addItem(_("output_container_srt"), userData='SRT')
        self.output_container.addItem(_("output_container_lrc"), userData='LRC')
        self.input_output_layout.addWidget(self.output_container)


        button_layout = QHBoxLayout()
        self.run_button = QPushButton(_("io_run_btn"))
        self.run_button.clicked.connect(self.run_worker)
        button_layout.addWidget(self.run_button)

        self.cancel_button = QPushButton(_("io_cancel_btn"))
        self.cancel_button.clicked.connect(self.cancel_task)
        button_layout.addWidget(self.cancel_button)

        self.open_output_button = QPushButton(_("io_open_output_btn"))
        self.open_output_button.clicked.connect(lambda: open_path(self.output_dir_edit.text().strip() or self.default_output_dir()))
        button_layout.addWidget(self.open_output_button)

        self.clean_button = QPushButton(_("io_clean_btn"))
        self.clean_button.clicked.connect(self.cleaner)
        button_layout.addWidget(self.clean_button)

        # Add the button row layout to the input output layout
        self.input_output_layout.addLayout(button_layout)

    def initDictTab(self):
        self.dict_tab = Widget("Dict", self)
        self.dict_layout = self.dict_tab.vBoxLayout

        self.dict_before_label = BodyLabel(_("dict_before_label"))
        self.dict_layout.addWidget(self.dict_before_label)
        self.before_dict = QTextEdit()
        self.before_dict.setPlaceholderText(_("dict_before_placeholder"))
        self.dict_layout.addWidget(self.before_dict)

        self.dict_gpt_label = BodyLabel(_("dict_gpt_label"))
        self.dict_layout.addWidget(self.dict_gpt_label)
        self.gpt_dict = QTextEdit()
        self.gpt_dict.setPlaceholderText(_("dict_gpt_placeholder"))
        self.dict_layout.addWidget(self.gpt_dict)

        self.dict_after_label = BodyLabel(_("dict_after_label"))
        self.dict_layout.addWidget(self.dict_after_label)
        self.after_dict = QTextEdit()
        self.after_dict.setPlaceholderText(_("dict_after_placeholder"))
        self.dict_layout.addWidget(self.after_dict)

        self.dict_extra_label = BodyLabel(_("dict_extra_label"))
        self.dict_layout.addWidget(self.dict_extra_label)
        self.extra_prompt = QTextEdit()
        self.extra_prompt.setPlaceholderText(_("dict_extra_placeholder"))
        self.dict_layout.addWidget(self.extra_prompt)

        self.dict_prompt_mode_label = BodyLabel(_("dict_prompt_mode_label"))
        self.dict_prompt_mode_label.setToolTip(_("tip_dict_prompt_mode"))
        self.dict_layout.addWidget(self.dict_prompt_mode_label)
        self.change_prompt_mode = QComboBox()
        self.change_prompt_mode.setToolTip(_("tip_dict_prompt_mode"))
        for _pm_val, _pm_key in (
            ('不修改', 'dict_prompt_mode_no'),
            ('追加', 'dict_prompt_mode_append'),
            ('覆盖', 'dict_prompt_mode_overwrite'),
        ):
            self.change_prompt_mode.addItem(_(_pm_key), userData=_pm_val)
        _default_pm_idx = self.change_prompt_mode.findData('不修改')
        if _default_pm_idx >= 0:
            self.change_prompt_mode.setCurrentIndex(_default_pm_idx)
        self.dict_layout.addWidget(self.change_prompt_mode)

    def initSettingsTab(self):
        self.settings_tab = Widget("Settings", self)
        self.settings_layout = self.settings_tab.vBoxLayout
        self.settings_asr_provider_label = BodyLabel(_("settings_asr_provider_label"))
        self.asr_provider_combo = QComboBox()
        self.asr_provider_combo.addItem(
            _("settings_asr_provider_crispasr"), userData='crispasr'
        )
        self.asr_provider_combo.addItem(
            _("settings_asr_provider_asrlabs"), userData='asrlabs'
        )

        self.settings_asr_engine_label = BodyLabel(_("settings_asr_engine_label"))
        self.asr_engine_combo = QComboBox()
        self.asr_engine_combo.addItem(_("workflow_enable_transcription"), userData='')
        self.settings_asr_model_label = BodyLabel(_("settings_asr_model_label"))
        self.asr_model_combo = QComboBox()
        self.settings_asr_device_label = BodyLabel(_("settings_asr_device_label"))
        self.asr_device_combo = QComboBox()
        self.asr_device_combo.addItems(['auto', 'cuda', 'cpu'])
        self.settings_asr_compute_type_label = BodyLabel(_("settings_asr_compute_type_label"))
        self.asr_compute_type_combo = QComboBox()
        self.asr_compute_type_combo.addItems(['float16', 'int8_float16', 'int8', 'float32'])
        self.settings_asr_extra_label = BodyLabel(_("settings_asr_extra_label"))
        self.asr_extra_edit = QTextEdit()
        self.asr_extra_edit.setPlaceholderText(_("settings_asr_extra_placeholder"))

        self.settings_align_engine_label = BodyLabel(_("settings_align_engine_label"))
        self.align_engine_combo = QComboBox()
        self.align_engine_combo.addItem(_("settings_align_no_align"), userData='none')
        self.settings_align_model_label = BodyLabel(_("settings_align_model_label"))
        self.align_model_combo = QComboBox()
        self.settings_align_device_label = BodyLabel(_("settings_align_device_label"))
        self.align_device_combo = QComboBox()
        self.align_device_combo.addItems(['auto', 'cuda', 'cpu'])
        self.settings_align_extra_label = BodyLabel(_("settings_align_extra_label"))
        self.align_extra_edit = QTextEdit()
        self.align_extra_edit.setPlaceholderText(_("settings_align_extra_placeholder"))

        self.settings_asr_backend_label = BodyLabel(_("settings_crispasr_backend_label"))
        self.crispasr_backend_combo = QComboBox()
        self.settings_crispasr_model_label = BodyLabel(_("settings_crispasr_model_label"))
        self.crispasr_model_combo = QComboBox()
        self.settings_crispasr_aligner_label = BodyLabel(_("settings_crispasr_aligner_label"))
        self.crispasr_aligner_combo = QComboBox()
        self.settings_asr_param_label = BodyLabel(_("settings_crispasr_param_label"))
        self.param_crispasr = QTextEdit()
        self.param_crispasr.setPlaceholderText(_("settings_crispasr_param_placeholder"))

        # Compatibility aliases used by Qwen's original configuration helpers.
        self.asr_backend = self.crispasr_backend_combo
        self.asr_model_file = self.crispasr_model_combo
        self.asr_aligner_file = self.crispasr_aligner_combo

        self._asrlabs_widgets = [
            self.settings_asr_engine_label, self.asr_engine_combo,
            self.settings_asr_model_label, self.asr_model_combo,
            self.settings_asr_device_label, self.asr_device_combo,
            self.settings_asr_compute_type_label, self.asr_compute_type_combo,
            self.settings_asr_extra_label, self.asr_extra_edit,
            self.settings_align_engine_label, self.align_engine_combo,
            self.settings_align_model_label, self.align_model_combo,
            self.settings_align_device_label, self.align_device_combo,
            self.settings_align_extra_label, self.align_extra_edit,
        ]
        self._crispasr_widgets = [
            self.settings_asr_backend_label, self.crispasr_backend_combo,
            self.settings_crispasr_model_label, self.crispasr_model_combo,
            self.settings_crispasr_aligner_label, self.crispasr_aligner_combo,
            self.settings_asr_param_label, self.param_crispasr,
        ]

        button_layout = QHBoxLayout()

        self.open_crispasr_dir = QPushButton(_("settings_open_crispasr_btn"))
        self.open_crispasr_dir.clicked.connect(lambda: open_path(os.path.join(os.getcwd(), 'crispasr')))
        button_layout.addWidget(self.open_crispasr_dir)

        self.refresh_speech_models_button = QPushButton(_("settings_refresh_speech_btn"))
        self.refresh_speech_models_button.clicked.connect(self.refresh_speech_model_lists)
        button_layout.addWidget(self.refresh_speech_models_button)

        self.test_offline_asr_button = QPushButton(_("settings_test_offline_asr_btn"))
        self.test_offline_asr_button.clicked.connect(self.run_test_offline_asr)
        button_layout.addWidget(self.test_offline_asr_button)
        self.settings_layout.addLayout(button_layout)

        self._crispasr_discovery_done = False
        self.refresh_crispasr_lists(query_backends=False)
        try:
            self.refresh_asr_engine_lists(force_refresh=False)
        except Exception:
            pass
        self.asr_engine_combo.currentIndexChanged.connect(self.on_asr_engine_changed)
        self.align_engine_combo.currentIndexChanged.connect(
            self._update_streaming_availability
        )
        self.asr_provider_combo.currentIndexChanged.connect(self.on_asr_provider_changed)
        self.on_asr_provider_changed()

    def initAdvancedSettingTab(self):
        self.advanced_settings_tab = Widget("AdvancedSettings", self)
        self.advanced_settings_layout = self.advanced_settings_tab.vBoxLayout

        # Translator Section
        model_row = QHBoxLayout()
        self.adv_translator_label = BodyLabel(_("adv_translator_label"))
        model_row.addWidget(self.adv_translator_label)
        self.translator_group = QComboBox()
        self.translator_group.addItems(TRANSLATOR_SUPPORTED)
        model_row.addWidget(self.translator_group)
        model_row.addSpacing(20)
        self.adv_concurrency_label = BodyLabel(_("adv_concurrency_label"))
        model_row.addWidget(self.adv_concurrency_label)
        self.max_concurrent_spin = QSpinBox()
        self.max_concurrent_spin.setRange(0, 20)
        self.max_concurrent_spin.setValue(0)
        model_row.addWidget(self.max_concurrent_spin)
        model_row.addStretch()
        self.advanced_settings_layout.addLayout(model_row)

        self.adv_online_token_label = BodyLabel(_("adv_online_token_label"))
        self.advanced_settings_layout.addWidget(self.adv_online_token_label)
        self.gpt_token = QLineEdit()
        self.gpt_token.setPlaceholderText(_("adv_online_token_placeholder"))
        self.advanced_settings_layout.addWidget(self.gpt_token)

        self.adv_online_model_label = BodyLabel(_("adv_online_model_label"))
        self.advanced_settings_layout.addWidget(self.adv_online_model_label)
        self.gpt_model = QLineEdit()
        self.gpt_model.setPlaceholderText(_("adv_online_model_placeholder"))
        self.advanced_settings_layout.addWidget(self.gpt_model)

        self._discovered_online_models = []
        self._auxiliary_model_cache = {}

        def make_auxiliary_profile():
            panel = QWidget()
            grid = QGridLayout(panel)
            grid.setContentsMargins(6, 6, 6, 6)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(6)

            provider_combo = QComboBox()
            provider_combo.addItem(_("aux_profile_follow_main"), userData='follow')
            provider_combo.addItem(_("aux_profile_custom"), userData='custom')
            for provider in ONLINE_TRANSLATOR_MAPPING:
                provider_combo.addItem(provider, userData=provider)
            model_combo = QComboBox()
            model_combo.setEditable(True)
            model_combo.addItem('deepseek-v4-flash', userData='deepseek-v4-flash')
            model_combo.addItem('deepseek-v4-pro', userData='deepseek-v4-pro')
            model_combo._provider_combo = provider_combo
            token_edit = QLineEdit()
            token_edit.setEchoMode(QLineEdit.EchoMode.Password)
            token_edit.setPlaceholderText(_("aux_profile_token_placeholder"))
            address_edit = QLineEdit()
            thinking_checkbox = QCheckBox(_("aux_profile_thinking"))
            thinking_checkbox.setToolTip(_("tip_aux_profile_thinking"))
            test_button = QPushButton(_("aux_profile_test_btn"))

            grid.addWidget(BodyLabel(_("aux_profile_provider_label")), 0, 0)
            grid.addWidget(provider_combo, 0, 1)
            grid.addWidget(BodyLabel(_("aux_profile_model_label")), 0, 2)
            grid.addWidget(model_combo, 0, 3)
            grid.addWidget(BodyLabel(_("aux_profile_token_label")), 1, 0)
            grid.addWidget(token_edit, 1, 1)
            grid.addWidget(BodyLabel(_("aux_profile_address_label")), 1, 2)
            grid.addWidget(address_edit, 1, 3)
            grid.addWidget(thinking_checkbox, 2, 0, 1, 2)
            grid.addWidget(test_button, 2, 2, 1, 2)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(3, 1)

            provider_combo.currentIndexChanged.connect(
                lambda _index, provider=provider_combo, model=model_combo,
                token=token_edit, address=address_edit,
                thinking=thinking_checkbox:
                    self._update_auxiliary_profile_controls(
                        provider, model, token, address, thinking
                    )
            )
            model_combo.editTextChanged.connect(
                lambda _text, combo=model_combo:
                self._update_thinking_availability()
            )
            self._update_auxiliary_profile_controls(
                provider_combo, model_combo, token_edit, address_edit,
                thinking_checkbox,
            )
            return (
                panel,
                provider_combo,
                model_combo,
                token_edit,
                address_edit,
                thinking_checkbox,
                test_button,
            )

        self.adv_auxiliary_models_label = BodyLabel(_("adv_auxiliary_models_label"))
        self.adv_auxiliary_models_label.setToolTip(_("tip_auxiliary_profiles"))
        self.auxiliary_model_tabs = QTabWidget()
        (
            resegment_panel,
            self.ai_resegment_provider_combo,
            self.ai_resegment_model_combo,
            self.ai_resegment_token,
            self.ai_resegment_address,
            self.ai_resegment_thinking_checkbox,
            self.ai_resegment_test_button,
        ) = make_auxiliary_profile()
        (
            proofread_panel,
            self.proofread_provider_combo,
            self.proofread_model_combo,
            self.proofread_token,
            self.proofread_address,
            self.proofread_thinking_checkbox,
            self.proofread_test_button,
        ) = make_auxiliary_profile()
        self.ai_resegment_test_button.clicked.connect(
            lambda: self.run_test_auxiliary_api('ai_resegment')
        )
        self.proofread_test_button.clicked.connect(
            lambda: self.run_test_auxiliary_api('proofread')
        )
        self.auxiliary_model_tabs.addTab(
            resegment_panel, _("adv_ai_resegment_model_label")
        )
        self.auxiliary_model_tabs.addTab(
            proofread_panel, _("adv_proofread_model_label")
        )
        self.advanced_settings_layout.addWidget(self.adv_auxiliary_models_label)
        self.advanced_settings_layout.addWidget(self.auxiliary_model_tabs)

        self.deepseek_thinking_checkbox = QCheckBox(_("adv_deepseek_thinking"))
        self.deepseek_thinking_checkbox.setChecked(False)
        self.deepseek_thinking_checkbox.setToolTip(_("tip_deepseek_thinking"))
        self.gpt_model.textChanged.connect(lambda _text: self._update_thinking_availability())
        self.translator_group.currentIndexChanged.connect(
            lambda _index: self._update_thinking_availability()
        )
        self.advanced_settings_layout.addWidget(self.deepseek_thinking_checkbox)

        self.adv_online_address_label = BodyLabel(_("adv_online_address_label"))
        self.advanced_settings_layout.addWidget(self.adv_online_address_label)
        self.gpt_address = QLineEdit()
        self.gpt_address.setPlaceholderText(_("adv_online_address_placeholder"))
        self.advanced_settings_layout.addWidget(self.gpt_address)

        self.adv_offline_model_label = BodyLabel(_("adv_offline_model_label"))
        self.advanced_settings_layout.addWidget(self.adv_offline_model_label)
        self.sakura_file = QComboBox()
        sakura_lst = [i for i in os.listdir('llama') if i.endswith('gguf')]
        self.sakura_file.addItems(sakura_lst)
        self.advanced_settings_layout.addWidget(self.sakura_file)

        self.adv_offline_gpu_label = BodyLabel(_("adv_offline_gpu_label"))
        self.advanced_settings_layout.addWidget(self.adv_offline_gpu_label)
        self.sakura_mode = QLineEdit()
        self.sakura_mode.setText("100")
        self.advanced_settings_layout.addWidget(self.sakura_mode)

        self.adv_offline_param_label = BodyLabel(_("adv_offline_param_label"))
        self.advanced_settings_layout.addWidget(self.adv_offline_param_label)
        self.param_llama = QTextEdit()
        self.param_llama.setPlaceholderText(_("adv_offline_param_placeholder"))
        self.advanced_settings_layout.addWidget(self.param_llama)

        button_layout = QHBoxLayout()

        self.open_model_dir = QPushButton(_("adv_open_model_btn"))
        self.open_model_dir.clicked.connect(lambda: open_path(os.path.join(os.getcwd(),'llama')))
        button_layout.addWidget(self.open_model_dir)

        self.refresh_language_models_button = QPushButton(_("adv_refresh_model_btn"))
        self.refresh_language_models_button.clicked.connect(self.refresh_language_model_lists)
        button_layout.addWidget(self.refresh_language_models_button)

        self.test_offline_translation_button = QPushButton(_("adv_test_offline_btn"))
        self.test_offline_translation_button.clicked.connect(
            self.run_test_offline_translation
        )
        button_layout.addWidget(self.test_offline_translation_button)

        self.test_online_button = QPushButton(_("adv_test_api_btn"))
        self.test_online_button.clicked.connect(self.run_test_online_api)
        button_layout.addWidget(self.test_online_button)
        self.advanced_settings_layout.addLayout(button_layout)

    def initClipTab(self):
        self.clip_tab = Widget("Clip", self)
        self.clip_layout = self.clip_tab.vBoxLayout

        # Clip Section
        self.clip_tool_label = BodyLabel(_("clip_tool_label"))
        self.clip_layout.addWidget(self.clip_tool_label)
        self.clip_files_list = QTextEdit()
        self.clip_files_list.setAcceptDrops(True)
        self._bind_drop_event(self.clip_files_list)
        self.clip_files_list.setPlaceholderText(_("clip_placeholder"))
        self.clip_layout.addWidget(self.clip_files_list)

        hbox = QHBoxLayout()
        left_v = QVBoxLayout()
        right_v = QVBoxLayout()

        self.clip_start_time = QLineEdit()
        self.clip_start_time.setPlaceholderText(_("clip_start_placeholder"))
        self.clip_start_label = BodyLabel(_("clip_start_label"))
        left_v.addWidget(self.clip_start_label)
        left_v.addWidget(self.clip_start_time)

        self.clip_end_time = QLineEdit()
        self.clip_end_time.setPlaceholderText(_("clip_end_placeholder"))
        self.clip_end_label = BodyLabel(_("clip_end_label"))
        right_v.addWidget(self.clip_end_label)
        right_v.addWidget(self.clip_end_time)

        hbox.addLayout(left_v)
        hbox.addLayout(right_v)
        self.clip_layout.addLayout(hbox)

        self.run_clip_button = QPushButton(_("clip_run_btn"))
        self.run_clip_button.clicked.connect(self.run_clip)
        self.clip_layout.addWidget(self.run_clip_button)
        self.clip_cancel_button = QPushButton(_("io_cancel_btn"))
        self.clip_cancel_button.clicked.connect(self.cancel_task)
        self.clip_layout.addWidget(self.clip_cancel_button)

        # Vocal Split
        self.clip_vocal_split_label = BodyLabel(_("clip_vocal_split_label"))
        self.clip_layout.addWidget(self.clip_vocal_split_label)
        uvr_model_row = QHBoxLayout()
        self.clip_uvr_model_label = BodyLabel(_("clip_uvr_model_label"))
        self.clip_uvr_model_label.setToolTip(_("tip_clip_uvr_model"))
        uvr_model_row.addWidget(self.clip_uvr_model_label)
        self.uvr_file = QComboBox()
        uvr_lst = [i for i in os.listdir('separate') if i.endswith('onnx')]
        self.uvr_file.addItems(uvr_lst)
        self.uvr_file.setToolTip(_("tip_clip_uvr_model"))
        uvr_model_row.addWidget(self.uvr_file)
        uvr_model_row.addStretch()
        self.clip_layout.addLayout(uvr_model_row)
        self.uvr_file_list = QTextEdit()
        self.uvr_file_list.setAcceptDrops(True)
        self._bind_drop_event(self.uvr_file_list)
        self.uvr_file_list.setPlaceholderText(_("clip_vocal_placeholder"))
        self.clip_layout.addWidget(self.uvr_file_list)

        self.run_uvr_button = QPushButton(_("clip_vocal_run_btn"))
        self.run_uvr_button.clicked.connect(self.run_vocal_split)
        self.clip_layout.addWidget(self.run_uvr_button)
        self.uvr_cancel_button = QPushButton(_("io_cancel_btn"))
        self.uvr_cancel_button.clicked.connect(self.cancel_task)
        self.clip_layout.addWidget(self.uvr_cancel_button)
        self.open_uvr_dir = QPushButton(_("clip_open_uvr_btn"))
        self.open_uvr_dir.clicked.connect(lambda: open_path(os.path.join(os.getcwd(), 'separate')))

    def initSynthTab(self):
        self.synth_tab = Widget("Synth", self)
        self.synth_layout = self.synth_tab.vBoxLayout

        # Video Synth
        self.synth_label = BodyLabel(_("synth_label"))
        self.synth_layout.addWidget(self.synth_label)

        # Video Files
        vbox_video = QHBoxLayout()
        self.synth_video_label = BodyLabel(_("synth_video_label"))
        vbox_video.addWidget(self.synth_video_label)
        self.synth_video_browse_btn = QPushButton(_("synth_browse_video_btn"))
        self.synth_video_browse_btn.clicked.connect(self.browse_synth_video)
        vbox_video.addWidget(self.synth_video_browse_btn)
        self.synth_layout.addLayout(vbox_video)
        
        self.synth_video_files_list = QTextEdit()
        self.synth_video_files_list.setAcceptDrops(True)
        self._bind_drop_event(self.synth_video_files_list)
        self.synth_video_files_list.setPlaceholderText(_("synth_video_placeholder"))
        self.synth_layout.addWidget(self.synth_video_files_list)

        # Subtitle Files
        vbox_srt = QHBoxLayout()
        self.synth_srt_label = BodyLabel(_("synth_srt_label"))
        vbox_srt.addWidget(self.synth_srt_label)
        self.synth_srt_browse_btn = QPushButton(_("synth_browse_srt_btn"))
        self.synth_srt_browse_btn.clicked.connect(self.browse_synth_srt)
        vbox_srt.addWidget(self.synth_srt_browse_btn)
        self.synth_layout.addLayout(vbox_srt)

        self.synth_srt_files_list = QTextEdit()
        self.synth_srt_files_list.setAcceptDrops(True)
        self._bind_drop_event(self.synth_srt_files_list)
        self.synth_srt_files_list.setPlaceholderText(_("synth_srt_placeholder"))
        self.synth_layout.addWidget(self.synth_srt_files_list)

        hbox = QHBoxLayout()

        self.synth_subtitle_type_label = BodyLabel(_("synth_subtitle_type_label"))
        hbox.addWidget(self.synth_subtitle_type_label)
        self.subtitle_type_combo = QComboBox()
        self.subtitle_type_combo.addItem(_("synth_sub_hard"), userData="硬字幕")
        self.subtitle_type_combo.addItem(_("synth_sub_soft"), userData="软字幕")
        self.subtitle_type_combo.currentIndexChanged.connect(self.update_synth_font_controls)
        hbox.addWidget(self.subtitle_type_combo)

        self.synth_font_label = BodyLabel(_("synth_font_label"))
        hbox.addWidget(self.synth_font_label)

        self.subtitle_font_combo = QComboBox()
        for font_item in self.collect_font_candidates():
            self.subtitle_font_combo.addItem(font_item)
        hbox.addWidget(self.subtitle_font_combo)

        self.run_synth_button = QPushButton(_("synth_run_btn"))
        self.run_synth_button.clicked.connect(self.run_synth)
        hbox.addWidget(self.run_synth_button)
        self.synth_cancel_button = QPushButton(_("io_cancel_btn"))
        self.synth_cancel_button.clicked.connect(self.cancel_task)
        hbox.addWidget(self.synth_cancel_button)
        self.synth_layout.addLayout(hbox)

        # Audio Synth
        self.synth_audio_label = BodyLabel(_("synth_audio_label"))
        self.synth_layout.addWidget(self.synth_audio_label)
        self.synth_audio_files_list = QTextEdit()
        self.synth_audio_files_list.setAcceptDrops(True)
        self._bind_drop_event(self.synth_audio_files_list)
        self.synth_audio_files_list.setPlaceholderText(_("synth_audio_placeholder"))
        self.synth_layout.addWidget(self.synth_audio_files_list)
        self.run_synth_audio_button = QPushButton(_("synth_audio_run_btn"))
        self.run_synth_audio_button.clicked.connect(self.run_synth_audio)
        self.synth_layout.addWidget(self.run_synth_audio_button)
        self.synth_audio_cancel_button = QPushButton(_("io_cancel_btn"))
        self.synth_audio_cancel_button.clicked.connect(self.cancel_task)
        self.synth_layout.addWidget(self.synth_audio_cancel_button)

    def initSummarizeTab(self):
        self.summarize_tab = Widget("Summarize", self)
        self.summarize_layout = self.summarize_tab.vBoxLayout

        self.summarize_prompt_label = BodyLabel(_("summarize_prompt_label"))
        self.summarize_layout.addWidget(self.summarize_prompt_label)
        self.summarize_prompt = QTextEdit()
        self.summarize_prompt.setPlaceholderText(_("summarize_prompt_placeholder"))
        self.summarize_layout.addWidget(self.summarize_prompt)

        self.summarize_input_label = BodyLabel(_("summarize_input_label"))
        self.summarize_layout.addWidget(self.summarize_input_label)
        self.summarize_files_list = QTextEdit()
        self.summarize_files_list.setAcceptDrops(True)
        self._bind_drop_event(self.summarize_files_list)
        self.summarize_files_list.setPlaceholderText(_("summarize_input_placeholder"))
        self.summarize_layout.addWidget(self.summarize_files_list)

        self.run_summarize_button = QPushButton(_("summarize_run_btn"))
        self.run_summarize_button.clicked.connect(self.run_summarize)
        self.summarize_layout.addWidget(self.run_summarize_button)
        self.summarize_cancel_button = QPushButton(_("io_cancel_btn"))
        self.summarize_cancel_button.clicked.connect(self.cancel_task)
        self.summarize_layout.addWidget(self.summarize_cancel_button)

    def run_worker(self):
        self._start_worker_task('run', _("task_workflow"))

    def run_clip(self):
        self._start_worker_task('clip', _("task_clip"))

    def run_synth(self):
        self._start_worker_task('synth', _("task_synth"))

    def run_synth_audio(self):
        self._start_worker_task('audiosynth', _("task_audio_synth"))

    def run_vocal_split(self):
        self._start_worker_task('vocal_split', _("task_vocal_split"))

    def run_summarize(self):
        self._start_worker_task('summarize', _("task_summarize"))

    def _handle_model_list_loaded(self, models, fixed_target=None):
        """Apply auxiliary lists directly; only the main profile needs a chooser."""
        if fixed_target:
            self._add_discovered_auxiliary_models(models, fixed_target)
            if fixed_target == 'resegment':
                model_combo = self.ai_resegment_model_combo
            else:
                model_combo = self.proofread_model_combo
            model_combo.setFocus()
            self._emit_status(
                _("status_aux_models_loaded", count=len(models))
            )
            return
        self.show_model_selection_dialog(models)

    def show_model_selection_dialog(self, models, fixed_target=None):
        self._add_discovered_auxiliary_models(models, fixed_target)
        previous_dialog = getattr(self, '_model_selection_dialog', None)
        if previous_dialog is not None:
            previous_dialog.close()
        dialog = QDialog(self)
        self._model_selection_dialog = dialog
        dialog.setModal(False)
        dialog.setWindowTitle(_("dialog_select_model_title"))
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)

        label = QLabel(_("dialog_select_model_label"))
        layout.addWidget(label)

        combo = QComboBox()
        combo.addItems(models)
        layout.addWidget(combo)

        target_label = QLabel(_("dialog_model_target_label"))
        layout.addWidget(target_label)
        target_combo = QComboBox()
        target_combo.addItem(_("dialog_model_target_translation"), userData='translation')
        target_combo.addItem(_("dialog_model_target_resegment"), userData='resegment')
        target_combo.addItem(_("dialog_model_target_proofread"), userData='proofread')
        if fixed_target:
            fixed_index = target_combo.findData(fixed_target)
            if fixed_index >= 0:
                target_combo.setCurrentIndex(fixed_index)
                target_combo.setEnabled(False)
        layout.addWidget(target_combo)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton(_("dialog_ok"))
        cancel_btn = QPushButton(_("dialog_cancel"))
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        def apply_selected_model():
            model_name = combo.currentText()
            target = target_combo.currentData()
            if target == 'resegment':
                self._set_auxiliary_model_value(
                    self.ai_resegment_model_combo, model_name
                )
            elif target == 'proofread':
                self._set_auxiliary_model_value(
                    self.proofread_model_combo, model_name
                )
            else:
                self.gpt_model.setText(model_name)
            self._schedule_auto_save()
            dialog.accept()

        ok_btn.clicked.connect(apply_selected_model)
        cancel_btn.clicked.connect(dialog.reject)
        dialog.finished.connect(
            lambda _result: setattr(self, '_model_selection_dialog', None)
        )
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def run_test_online_api(self):
        self._start_worker_task(
            'test_online_api', _("task_api_test"),
            show_model_dialog=True,
        )

    def run_test_offline_asr(self):
        self._start_worker_task(
            'test_offline_asr', _("settings_test_offline_asr_btn")
        )

    def run_test_offline_translation(self):
        self._start_worker_task(
            'test_offline_translation', _("adv_test_offline_btn")
        )

    def run_test_auxiliary_api(self, prefix: str):
        if prefix == 'ai_resegment':
            provider_combo = self.ai_resegment_provider_combo
            model_combo = self.ai_resegment_model_combo
            token_edit = self.ai_resegment_token
            address_edit = self.ai_resegment_address
            target = 'resegment'
        else:
            provider_combo = self.proofread_provider_combo
            model_combo = self.proofread_model_combo
            token_edit = self.proofread_token
            address_edit = self.proofread_address
            target = 'proofread'

        provider = provider_combo.currentData() or 'follow'
        overrides = {}
        if provider != 'follow':
            overrides = {
                'translator': provider,
                'gpt_model': self._auxiliary_model_value(model_combo),
                'gpt_token': token_edit.text() or self.gpt_token.text(),
                'gpt_address': address_edit.text().strip(),
            }
        self._start_worker_task(
            'test_online_api',
            _("task_api_test"),
            show_model_dialog=True,
            snapshot_overrides=overrides,
            model_target=target,
        )
    
    def cleaner(self):
        self._start_worker_task('clean', _("task_clean"))

if __name__ == "__main__":
    os.makedirs('project/cache', exist_ok=True)
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
