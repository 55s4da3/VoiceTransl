import sys, os
import re
import shutil
import yaml
from pathlib import Path

from core import (
    DEFAULT_CRISPASR_BACKEND,
    LOG_PATH,
    NO_TRANSCRIPTION,
    NO_TRANSLATION,
    TRANSLATOR_SUPPORTED,
    _compose_output_format,
    _load_api_key,
    _save_api_key,
)
from asr import _list_crispasr_aligners, _list_crispasr_backends, _list_crispasr_models
from log import UIMessageQueue, _line_passes_filter
from pool import ConcurrentTranslationPool
from worker import MainWorker
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

    @staticmethod
    def default_output_dir() -> str:
        return str(Path.cwd() / 'project' / 'cache')

    def __init__(self):
        super().__init__()
        application = QApplication.instance()
        saved_theme = _load_ui_theme()
        if application and application.property('ui_material_theme') != saved_theme:
            apply_material_theme(application, saved_theme)
        self.msg_queue = UIMessageQueue(LOG_PATH)
        self.thread = None
        self.worker = None
        self._active_task_name = _('task_none')
        self._drop_targets = {}
        self._suppress_auto_save = True
        self._auto_save_timer = QTimer(self)
        self._auto_save_timer.setSingleShot(True)
        self._auto_save_timer.setInterval(200)
        self._auto_save_timer.timeout.connect(self._auto_save_config)
        self._load_ui_language()
        self.setWindowTitle(_("window_title"))
        self.setWindowIcon(QtGui.QIcon('icon.png'))
        self.init_system_tray()
        self.status.connect(lambda x: self.setWindowTitle(f"{_('window_title')} - {x}"))
        self.resize(1180, 760)
        self.setMinimumSize(960, 640)
        self.show()
        self.initUI()
        self._log_level_filter = 'ALL'  # 日志级别过滤默认值
        self.setup_timer()

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
        """保存 GUI 配置到 gui_settings.yaml 及相关文件"""
        if not silent:
            self._emit_status(_("status_reading_config"))
        asr_model_file = self.asr_model_file.currentText()
        asr_aligner_file = self.asr_aligner_file.currentText()
        asr_backend = self.asr_backend.currentText()
        translator = self.translator_group.currentText()
        language = self.transcription_lang.currentText()
        gpt_token = self.gpt_token.text()
        gpt_address = self.gpt_address.text()
        gpt_model = self.gpt_model.text()
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
        os.makedirs(output_dir, exist_ok=True)
        enable_segment = self.enable_segment_checkbox.isChecked()
        segment_duration = self.segment_duration_spin.value()
        change_prompt_mode = self.change_prompt_mode.currentData() if hasattr(self, 'change_prompt_mode') else '不修改'
        auto_shutdown = self.auto_shutdown_checkbox.isChecked() if hasattr(self, 'auto_shutdown_checkbox') else False
        target_translation_lang = self.target_lang.currentData() if hasattr(self, 'target_lang') else 'zh-cn'
        ui_theme = self.theme_selector.currentData() if hasattr(self, 'theme_selector') else _load_ui_theme()
        current_lang = get_language()

        gui_settings = {
            'asr_model_file': asr_model_file,
            'asr_aligner_file': asr_aligner_file,
            'asr_backend': asr_backend,
            'translator': translator,
            'language': language,
            'gpt_address': gpt_address,
            'gpt_model': gpt_model,
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
            'change_prompt_mode': change_prompt_mode,
            'auto_shutdown': auto_shutdown,
            'log_level_filter': self.log_filter_combo.currentText(),
            'verbose_mode': self.verbose_checkbox.isChecked(),
            'ui_language': current_lang,
            'ui_theme': ui_theme,
            'target_translation_lang': target_translation_lang,
        }
        with open('gui_settings.yaml', 'w', encoding='utf-8') as f:
            yaml.dump(gui_settings, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

        _save_api_key(gpt_token)

        with open('crispasr/param.txt', 'w', encoding='utf-8') as f:
            f.write(self.param_crispasr.toPlainText())

        with open('llama/param.txt', 'w', encoding='utf-8') as f:
            f.write(self.param_llama.toPlainText())

        with open('project/dict_pre.txt', 'w', encoding='utf-8') as f:
            f.write(self.before_dict.toPlainText())

        with open('project/dict_gpt.txt', 'w', encoding='utf-8') as f:
            f.write(self.gpt_dict.toPlainText())

        with open('project/dict_after.txt', 'w', encoding='utf-8') as f:
            f.write(self.after_dict.toPlainText())

        if not silent:
            self._emit_status(_("status_config_saved"))

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
        self.shared_state_label = BodyLabel(_("task_state_idle"))
        header.addWidget(self.shared_state_label)
        layout.addLayout(header)

        self.shared_progress_bar = QProgressBar()
        self.shared_progress_bar.setRange(0, 1)
        self.shared_progress_bar.setValue(0)
        self.shared_progress_bar.setTextVisible(False)
        self.shared_progress_bar.setMaximumHeight(5)
        layout.addWidget(self.shared_progress_bar)

        self.shared_progress_view = QPlainTextEdit()
        self.shared_progress_view.setReadOnly(True)
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
        speech_form.addWidget(self.settings_asr_backend_label, 0, 0)
        speech_form.addWidget(self.asr_backend, 0, 1)
        speech_form.addWidget(self.settings_asr_model_label, 1, 0)
        speech_form.addWidget(self.asr_model_file, 1, 1)
        speech_form.addWidget(self.settings_asr_aligner_label, 2, 0)
        speech_form.addWidget(self.asr_aligner_file, 2, 1)
        speech_form.addWidget(self.settings_asr_param_label, 3, 0)
        speech_form.addWidget(self.param_crispasr, 3, 1)
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
        translation_form.addWidget(self.adv_concurrency_label, 0, 2)
        translation_form.addWidget(self.max_concurrent_spin, 0, 3)
        translation_form.addWidget(self.adv_online_token_label, 1, 0)
        translation_form.addWidget(self.gpt_token, 1, 1, 1, 3)
        translation_form.addWidget(self.adv_online_model_label, 2, 0)
        translation_form.addWidget(self.gpt_model, 2, 1, 1, 3)
        translation_form.addWidget(self.adv_online_address_label, 3, 0)
        translation_form.addWidget(self.gpt_address, 3, 1, 1, 3)
        translation_form.addWidget(self.adv_offline_model_label, 4, 0)
        translation_form.addWidget(self.sakura_file, 4, 1)
        translation_form.addWidget(self.adv_offline_gpu_label, 4, 2)
        translation_form.addWidget(self.sakura_mode, 4, 3)
        translation_form.addWidget(self.adv_offline_param_label, 5, 0)
        translation_form.addWidget(self.param_llama, 5, 1, 1, 3)
        translation_form.setColumnStretch(1, 1)
        translation_form.setColumnStretch(3, 1)
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
        self.shared_progress_view.clear()
        self.shared_progress_bar.setRange(0, 0)
        self.shared_task_label.setText(task_name)
        self.shared_state_label.setText(_("task_state_running"))

    def _on_task_finished(self):
        self.shared_progress_bar.setRange(0, 1)
        self.shared_progress_bar.setValue(1)
        self.shared_state_label.setText(_("task_state_done"))

    def _start_worker_task(self, operation: str, task_name: str,
                           show_model_dialog: bool = False):
        if self.thread is not None and self.thread.isRunning():
            self._emit_status(_("status_task_busy"))
            return
        self._set_progress_context(task_name)
        self.thread = QThread()
        self.worker = MainWorker(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(getattr(self.worker, operation))
        if show_model_dialog:
            self.worker.show_model_dialog.connect(self.show_model_selection_dialog)
        self.worker.finished.connect(self._on_task_finished)
        self.worker.finished.connect(self.thread.quit)
        self.thread.start()

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
        if hasattr(self, 'asr_backend'):
            current_backend = self.asr_backend.currentText()
            backends = _list_crispasr_backends()
            if current_backend and current_backend not in backends:
                backends.append(current_backend)
            self.asr_backend.clear()
            self.asr_backend.addItems(backends)
            if current_backend in backends:
                self.asr_backend.setCurrentText(current_backend)

        if hasattr(self, 'asr_model_file'):
            current_model = self.asr_model_file.currentText()
            asr_models = _list_crispasr_models()
            self.asr_model_file.clear()
            self.asr_model_file.addItems(asr_models)
            if current_model in asr_models:
                self.asr_model_file.setCurrentText(current_model)

        if hasattr(self, 'asr_aligner_file'):
            current_aligner = self.asr_aligner_file.currentText()
            aligners = _list_crispasr_aligners()
            self.asr_aligner_file.clear()
            self.asr_aligner_file.addItems(aligners)
            if current_aligner in aligners:
                self.asr_aligner_file.setCurrentText(current_aligner)

        if hasattr(self, 'uvr_file'):
            current_uvr = self.uvr_file.currentText()
            uvr_lst = [i for i in os.listdir('separate') if i.endswith('onnx')]
            self.uvr_file.clear()
            self.uvr_file.addItems(uvr_lst)
            if current_uvr in uvr_lst:
                self.uvr_file.setCurrentText(current_uvr)

    def refresh_language_model_lists(self):
        if hasattr(self, 'sakura_file'):
            current_model = self.sakura_file.currentText()
            sakura_lst = [i for i in os.listdir('llama') if i.endswith('gguf')]
            self.sakura_file.clear()
            self.sakura_file.addItems(sakura_lst)
            if current_model in sakura_lst:
                self.sakura_file.setCurrentText(current_model)

    def cancel_task(self):
        self._emit_status(_("status_cancelling"))
        try:
            if self.worker:
                self.worker.stop()
        except Exception as e:
            self._emit_status(_("status_cancel_worker_error", error=e))

        try:
            if self.thread and self.thread.isRunning():
                self.thread.quit()
                if not self.thread.wait(2000):
                    self.thread.terminate()
                    self.thread.wait(2000)
        except Exception as e:
            self._emit_status(_("status_cancel_thread_error", error=e))

        self._emit_status(_("status_cancel_done"))

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
        elif os.path.exists('config.txt'):
            gui_settings = self._migrate_config_txt()

        if gui_settings:
            saved_asr_model = gui_settings.get('asr_model_file') or gui_settings.get('whisper_file')
            legacy_transcription_disabled = saved_asr_model == NO_TRANSCRIPTION
            if (
                self.asr_model_file
                and saved_asr_model
                and not legacy_transcription_disabled
                and self.asr_model_file.findText(saved_asr_model) >= 0
            ):
                self.asr_model_file.setCurrentText(saved_asr_model)
            self.enable_transcription_checkbox.setChecked(
                gui_settings.get('enable_transcription', not legacy_transcription_disabled)
            )
            saved_aligner = (
                gui_settings.get('asr_aligner_file')
                or gui_settings.get('aligner_model_file')
            )
            if saved_aligner and self.asr_aligner_file.findText(saved_aligner) >= 0:
                self.asr_aligner_file.setCurrentText(saved_aligner)
            saved_backend = gui_settings.get('asr_backend', DEFAULT_CRISPASR_BACKEND)
            if self.asr_backend.findText(saved_backend) < 0:
                self.asr_backend.addItem(saved_backend)
            self.asr_backend.setCurrentText(saved_backend)
            saved_translator = gui_settings.get('translator', '')
            legacy_translation_disabled = saved_translator == NO_TRANSLATION
            if (
                saved_translator
                and not legacy_translation_disabled
                and self.translator_group.findText(saved_translator) >= 0
            ):
                self.translator_group.setCurrentText(saved_translator)
            self.enable_translation_checkbox.setChecked(
                gui_settings.get('enable_translation', not legacy_translation_disabled)
            )
            self.transcription_lang.setCurrentText(gui_settings.get('language', ''))
            self.gpt_address.setText(gui_settings.get('gpt_address', ''))
            self.gpt_model.setText(gui_settings.get('gpt_model', ''))
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

        # API Key 始终从 .env 加载
        api_key = _load_api_key()
        if api_key:
            self.gpt_token.setText(api_key)

        if not self.output_dir_edit.text().strip():
            self.output_dir_edit.setText(self.default_output_dir())

        self.update_output_dir_controls()

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

    def setup_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._consume_messages)
        self.timer.start(1000)

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
        action_quit.triggered.connect(QApplication.instance().quit)

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

        entries = self.msg_queue.drain()
        if not entries:
            return

        for target, text in entries:
            if UIMessageQueue.is_completion_entry(target):
                completion_msg = _("status_all_done")
                self.log_display.appendPlainText(completion_msg)
                self.shared_progress_view.appendPlainText(completion_msg)
            elif target == 'status':
                self.shared_progress_view.appendPlainText(text)
            elif target == 'detail':
                # 应用日志级别过滤
                if self._log_level_filter != 'ALL':
                    if not _line_passes_filter(text, self._log_level_filter):
                        continue
                self.log_display.appendPlainText(text)

        # Keep the aggregate history and the shared compact feed at the end.
        for widget in (self.log_display, self.shared_progress_view):
            scrollbar = widget.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event):
        """确保在关闭窗口时停止定时器并关闭子进程，检查本地模型是否已关闭"""
        try:
            self.save_config(silent=True)
        except Exception:
            pass
        self.timer.stop()
        self.shutdown_children()

        # 检查本地模型是否仍在运行
        local_model_running = False
        if hasattr(self, 'worker') and self.worker:
            # 检查翻译池中的共享本地模型进程
            if hasattr(self.worker, '_translation_pool') and self.worker._translation_pool:
                pool = self.worker._translation_pool
                if hasattr(pool, '_shared_local_model_proc') and pool._shared_local_model_proc:
                    proc = pool._shared_local_model_proc
                    if proc and proc.poll() is None:
                        local_model_running = True
                        # 尝试再次停止
                        pool._stop_shared_local_model()
                        # 再次检查
                        if proc.poll() is None:
                            # 强制终止
                            try:
                                proc.kill()
                                proc.wait(timeout=2)
                            except Exception:
                                pass

        if local_model_running:
            print(_("status_local_model_closed"))

        if getattr(self, 'tray_icon', None):
            self.tray_icon.hide()
        event.accept()

    def shutdown_children(self):
        """关闭后台线程和子进程"""
        try:
            if self.worker:
                self.worker.stop()
        except Exception:
            pass

        try:
            if self.thread and self.thread.isRunning():
                self.thread.quit()
                if not self.thread.wait(2000):
                    self.thread.terminate()
                    self.thread.wait(2000)
        except Exception:
            pass

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
        
        # CrispASR Section
        self.settings_asr_backend_label = BodyLabel(_("settings_asr_backend_label"))
        self.settings_asr_backend_label.setToolTip(_("tip_settings_asr_backend"))
        self.settings_layout.addWidget(self.settings_asr_backend_label)
        self.asr_backend = QComboBox()
        self.asr_backend.addItems(_list_crispasr_backends())
        self.asr_backend.setToolTip(_("tip_settings_asr_backend"))
        default_backend_index = self.asr_backend.findText(DEFAULT_CRISPASR_BACKEND)
        if default_backend_index >= 0:
            self.asr_backend.setCurrentIndex(default_backend_index)
        self.settings_layout.addWidget(self.asr_backend)

        self.settings_asr_model_label = BodyLabel(_("settings_asr_model_label"))
        self.settings_asr_model_label.setToolTip(_("tip_settings_asr_model"))
        self.settings_layout.addWidget(self.settings_asr_model_label)
        self.asr_model_file = QComboBox()
        self.asr_model_file.addItems(_list_crispasr_models())
        self.asr_model_file.setToolTip(_("tip_settings_asr_model"))
        self.settings_layout.addWidget(self.asr_model_file)

        self.settings_asr_aligner_label = BodyLabel(_("settings_asr_aligner_label"))
        self.settings_asr_aligner_label.setToolTip(_("tip_settings_asr_aligner"))
        self.settings_layout.addWidget(self.settings_asr_aligner_label)
        self.asr_aligner_file = QComboBox()
        self.asr_aligner_file.addItems(_list_crispasr_aligners())
        self.asr_aligner_file.setToolTip(_("tip_settings_asr_aligner"))
        self.settings_layout.addWidget(self.asr_aligner_file)

        self.settings_asr_param_label = BodyLabel(_("settings_asr_param_label"))
        self.settings_asr_param_label.setToolTip(_("tip_settings_asr_param"))
        self.settings_layout.addWidget(self.settings_asr_param_label)
        self.param_crispasr = QTextEdit()
        self.param_crispasr.setPlaceholderText(_("settings_asr_param_placeholder"))
        self.param_crispasr.setToolTip(_("settings_asr_param_placeholder"))
        self.settings_layout.addWidget(self.param_crispasr)

        button_layout = QHBoxLayout()

        self.open_crispasr_dir = QPushButton(_("settings_open_crispasr_btn"))
        self.open_crispasr_dir.clicked.connect(lambda: open_path(os.path.join(os.getcwd(), 'crispasr')))
        button_layout.addWidget(self.open_crispasr_dir)

        self.refresh_speech_models_button = QPushButton(_("settings_refresh_speech_btn"))
        self.refresh_speech_models_button.clicked.connect(self.refresh_speech_model_lists)
        button_layout.addWidget(self.refresh_speech_models_button)
        self.settings_layout.addLayout(button_layout)

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

    def show_model_selection_dialog(self, models):
        dialog = QDialog(self)
        dialog.setWindowTitle(_("dialog_select_model_title"))
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)

        label = QLabel(_("dialog_select_model_label"))
        layout.addWidget(label)

        combo = QComboBox()
        combo.addItems(models)
        layout.addWidget(combo)

        btn_layout = QHBoxLayout()
        ok_btn = QPushButton(_("dialog_ok"))
        cancel_btn = QPushButton(_("dialog_cancel"))
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        ok_btn.clicked.connect(lambda: (
            self.gpt_model.setText(combo.currentText()),
            dialog.accept()
        ))
        cancel_btn.clicked.connect(dialog.reject)

        dialog.exec()

    def run_test_online_api(self):
        self._start_worker_task(
            'test_online_api', _("task_api_test"),
            show_model_dialog=True,
        )
    
    def cleaner(self):
        self._set_progress_context(_("task_clean"))
        try:
            self._emit_status(_("status_cleaning_intermediate"))
            if os.path.exists('project/gt_input'):
                shutil.rmtree('project/gt_input')
            if os.path.exists('project/gt_output'):
                shutil.rmtree('project/gt_output')
            if os.path.exists('project/transl_cache'):
                shutil.rmtree('project/transl_cache')
            self._emit_status(_("status_cleaning_output"))
            if os.path.exists('project/cache'):
                shutil.rmtree('project/cache')
            os.makedirs('project/cache', exist_ok=True)
        finally:
            self._on_task_finished()

if __name__ == "__main__":
    os.makedirs('project/cache', exist_ok=True)
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
