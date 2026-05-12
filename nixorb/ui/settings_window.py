"""nixorb/ui/settings_window.py — Settings GUI with syntax-highlighted log."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from PySide6.QtCore import Signal, Slot
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QPushButton, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import BashLexer, PythonLexer

from nixorb.core.event_bus import Event, EventPayload, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

_MONO = QFont("JetBrains Mono", 10)
_MONO.setStyleHint(QFont.StyleHint.Monospace)

_STYLE = """
QDialog, QWidget   { background:#1a1a2e; color:#e0e0e0; }
QTabWidget::pane   { border:1px solid #2a2a4e; }
QTabBar::tab       { background:#16213e; padding:7px 16px; border-radius:3px 3px 0 0; }
QTabBar::tab:selected { background:#0f3460; color:#e94560; }
QGroupBox          { border:1px solid #2a2a4e; border-radius:4px;
                     margin-top:10px; padding:10px 6px 6px 6px; }
QGroupBox::title   { color:#3498db; subcontrol-origin:margin; left:8px; }
QLineEdit, QComboBox { background:#16213e; border:1px solid #2a2a4e;
                       padding:4px 6px; border-radius:3px; }
QLineEdit:focus    { border-color:#3498db; }
QPushButton        { background:#0f3460; border:none; padding:6px 14px;
                     border-radius:3px; }
QPushButton:hover  { background:#e94560; }
"""


class SyntaxLogWidget(QTextEdit):
    _html_signal = Signal(str)

    LEVEL_COLORS = {
        "info":    "#2ecc71",
        "warning": "#f39c12",
        "error":   "#e74c3c",
        "exec":    "#3498db",
        "debug":   "#95a5a6",
        "success": "#1abc9c",
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(_MONO)
        self.setStyleSheet(
            "background:#0d0d1a;color:#e0e0e0;border:1px solid #2a2a4e;"
        )
        self._fmt = HtmlFormatter(style="monokai", noclasses=True, nowrap=True)
        self._html_signal.connect(self._insert_html)

    def append_log(self, level: str, msg: str) -> None:
        color  = self.LEVEL_COLORS.get(level, "#e0e0e0")
        prefix = (
            f"<span style='color:{color};font-weight:bold'>"
            f"[{level.upper():7}]</span>&nbsp;"
        )
        if level == "exec" or msg.lstrip().startswith("$"):
            body = highlight(msg, BashLexer(), self._fmt)
        elif "Traceback" in msg or "Error:" in msg:
            body = highlight(msg, PythonLexer(), self._fmt)
        else:
            escaped = (
                msg.replace("&", "&amp;")
                   .replace("<", "&lt;")
                   .replace(">", "&gt;")
                   .replace("\n", "<br>")
            )
            body = f"<span style='color:{color}'>{escaped}</span>"
        self._html_signal.emit(f"{prefix}{body}<br>")

    @Slot(str)
    def _insert_html(self, html: str) -> None:
        self.moveCursor(QTextCursor.MoveOperation.End)
        self.insertHtml(html)
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())


class SettingsWindow(QDialog):
    _singleton:    ClassVar[SettingsWindow | None] = None
    _settings_cls: ClassVar[Settings | None]       = None

    @classmethod
    def init_settings(cls, settings: Settings) -> None:
        cls._settings_cls = settings

    @classmethod
    def show_singleton(cls) -> None:
        if cls._settings_cls is None:
            log.error("SettingsWindow.init_settings() not called")
            return
        if cls._singleton is None or not cls._singleton.isVisible():
            cls._singleton = cls(cls._settings_cls)
            cls._singleton.show()
        else:
            cls._singleton.raise_()
            cls._singleton.activateWindow()

    def __init__(self, settings: Settings, parent=None) -> None:
        super().__init__(parent)
        self._s = settings
        self.setWindowTitle("NixOrb Settings")
        self.setMinimumSize(940, 700)
        self.setStyleSheet(_STYLE)
        self._build_ui()
        self._connect_events()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(self._tab_models(),  "🤖 Models")
        tabs.addTab(self._tab_audio(),   "🎙  Audio")
        tabs.addTab(self._tab_system(),  "⚙  System")
        tabs.addTab(self._tab_plugins(), "🔌 Plugins")
        tabs.addTab(self._tab_log(),     "📋 Log")
        root.addWidget(tabs)
        root.addLayout(self._bottom_bar())

    def _bottom_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        btn_export = QPushButton("📦 Export Config")
        btn_import = QPushButton("📂 Import Config")
        btn_save   = QPushButton("💾 Save & Apply")
        bar.addWidget(btn_export)
        bar.addWidget(btn_import)
        bar.addStretch()
        bar.addWidget(btn_save)
        btn_export.clicked.connect(self._export)
        btn_import.clicked.connect(self._import_cfg)
        btn_save.clicked.connect(self._save)
        return bar

    def _tab_models(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)

        g = QGroupBox("ASR — Speech Recognition")
        f = QFormLayout(g)
        self._asr_combo = QComboBox()
        self._asr_combo.addItems([
            "large-v3 (local INT8)", "medium (local)", "openai/whisper-1"
        ])
        f.addRow("Whisper model:", self._asr_combo)
        v.addWidget(g)

        g = QGroupBox("LLM Backend")
        f = QFormLayout(g)
        self._llm_backend = QComboBox()
        self._llm_backend.addItems(["huggingface", "openai", "ollama", "local"])
        idx = self._llm_backend.findText(self._s.llm_backend)
        if idx >= 0:
            self._llm_backend.setCurrentIndex(idx)
        self._llm_model   = QLineEdit(self._s.llm_model)
        self._llm_fast    = QLineEdit(self._s.llm_fast_model)
        self._llm_key     = QLineEdit(self._s.openai_api_key)
        self._llm_key.setEchoMode(QLineEdit.EchoMode.Password)
        self._hf_token    = QLineEdit(self._s.hf_token)
        self._hf_token.setEchoMode(QLineEdit.EchoMode.Password)
        self._llm_url     = QLineEdit(self._s.llm_base_url)
        self._llm_path    = QLineEdit(self._s.local_model_path)
        f.addRow("Backend:",           self._llm_backend)
        f.addRow("Main model:",        self._llm_model)
        f.addRow("Fast model:",        self._llm_fast)
        f.addRow("OpenAI API Key:",    self._llm_key)
        f.addRow("HuggingFace token:", self._hf_token)
        f.addRow("Base URL:",          self._llm_url)
        f.addRow("Local GGUF path:",   self._llm_path)
        v.addWidget(g)

        g = QGroupBox("TTS — Text-to-Speech")
        f = QFormLayout(g)
        self._tts_backend = QComboBox()
        self._tts_backend.addItems(["huggingface", "openai", "piper"])
        idx = self._tts_backend.findText(self._s.tts_backend)
        if idx >= 0:
            self._tts_backend.setCurrentIndex(idx)
        self._tts_hf_repo = QLineEdit(self._s.tts_hf_repo)
        self._tts_voice   = QLineEdit(self._s.tts_voice)
        f.addRow("Backend:",       self._tts_backend)
        f.addRow("HF repo:",       self._tts_hf_repo)
        f.addRow("Voice/speaker:", self._tts_voice)
        v.addWidget(g)

        g = QGroupBox("Vision")
        f = QFormLayout(g)
        self._vision_model = QLineEdit(self._s.vision_model)
        self._vlm_model    = QLineEdit(self._s.vlm_model)
        self._use_vlm      = QCheckBox(
            "Use full VLM (Qwen3.5-4B) instead of CogFlorence"
        )
        self._use_vlm.setChecked(self._s.use_vlm)
        f.addRow("Caption model:", self._vision_model)
        f.addRow("VLM model:",     self._vlm_model)
        f.addRow(self._use_vlm)
        v.addWidget(g)
        v.addStretch()
        return w

    def _tab_audio(self) -> QWidget:
        import sounddevice as sd
        w = QWidget()
        f = QFormLayout(w)
        self._mic_combo = QComboBox()
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                self._mic_combo.addItem(f"[{i}] {d['name']}", userData=i)
        f.addRow("Microphone:", self._mic_combo)
        self._wake_check = QCheckBox("Enable wake-word detection")
        self._wake_check.setChecked(self._s.wake_word_enabled)
        f.addRow(self._wake_check)
        self._wake_model = QLineEdit(self._s.wake_word_model)
        f.addRow("Wake-word model:", self._wake_model)
        return w

    def _tab_system(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self._confirm_check = QCheckBox("Require confirmation before running commands")
        self._confirm_check.setChecked(self._s.require_action_confirmation)
        self._screen_check  = QCheckBox("Enable screen context (grim)")
        self._screen_check.setChecked(self._s.screen_capture_enabled)
        self._offline_check = QCheckBox("Enable offline fallback mode")
        self._offline_check.setChecked(self._s.offline_fallback_enabled)
        self._clip_check    = QCheckBox("Enable clipboard integration")
        self._clip_check.setChecked(self._s.clipboard_enabled)
        self._web_check     = QCheckBox("Enable web search")
        self._web_check.setChecked(self._s.web_search_enabled)
        self._hotkey_edit   = QLineEdit(self._s.hotkey)
        f.addRow(self._confirm_check)
        f.addRow(self._screen_check)
        f.addRow(self._offline_check)
        f.addRow(self._clip_check)
        f.addRow(self._web_check)
        f.addRow("Global hotkey:", self._hotkey_edit)
        return w

    def _tab_plugins(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel("Drop .py plugin files into the plugin directory:"))
        lbl = QLabel(self._s.plugin_dir)
        lbl.setStyleSheet("color:#888; font-size:11px;")
        v.addWidget(lbl)
        self._plugin_list = QPlainTextEdit()
        self._plugin_list.setReadOnly(True)
        self._plugin_list.setFont(_MONO)
        v.addWidget(self._plugin_list)
        bar = QHBoxLayout()
        btn_reload = QPushButton("🔄  Reload Plugins")
        btn_open   = QPushButton("📂  Open Plugin Dir")
        btn_reload.clicked.connect(self._reload_plugins)
        btn_open.clicked.connect(self._open_plugin_dir)
        bar.addWidget(btn_reload)
        bar.addWidget(btn_open)
        bar.addStretch()
        v.addLayout(bar)
        self._reload_plugins()
        return w

    def _tab_log(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.log_widget = SyntaxLogWidget()
        v.addWidget(self.log_widget)
        bar = QHBoxLayout()
        btn_clear = QPushButton("🗑  Clear")
        btn_clear.clicked.connect(self.log_widget.clear)
        bar.addStretch()
        bar.addWidget(btn_clear)
        v.addLayout(bar)
        return w

    def _connect_events(self) -> None:
        bus.subscribe(Event.LOG, self._on_log)

    async def _on_log(self, payload: EventPayload) -> None:
        data  = payload.data or {}
        level = data.get("level", "info")
        msg   = data.get("msg", "")
        if hasattr(self, "log_widget"):
            self.log_widget.append_log(level, msg)

    def _save(self) -> None:
        self._s.openai_api_key             = self._llm_key.text()
        self._s.hf_token                   = self._hf_token.text()
        self._s.llm_model                  = self._llm_model.text()
        self._s.llm_fast_model             = self._llm_fast.text()
        self._s.llm_backend                = self._llm_backend.currentText()
        self._s.llm_base_url               = self._llm_url.text()
        self._s.local_model_path           = self._llm_path.text()
        self._s.tts_backend                = self._tts_backend.currentText()
        self._s.tts_hf_repo                = self._tts_hf_repo.text()
        self._s.tts_voice                  = self._tts_voice.text()
        self._s.vision_model               = self._vision_model.text()
        self._s.vlm_model                  = self._vlm_model.text()
        self._s.use_vlm                    = self._use_vlm.isChecked()
        self._s.hotkey                     = self._hotkey_edit.text()
        self._s.wake_word_enabled          = self._wake_check.isChecked()
        self._s.wake_word_model            = self._wake_model.text()
        self._s.require_action_confirmation = self._confirm_check.isChecked()
        self._s.screen_capture_enabled     = self._screen_check.isChecked()
        self._s.offline_fallback_enabled   = self._offline_check.isChecked()
        self._s.clipboard_enabled          = self._clip_check.isChecked()
        self._s.web_search_enabled         = self._web_check.isChecked()
        self._s.save()
        log.info("Settings saved")

        # Only emit_sync if the event loop is actually running
        # (not in standalone config-gui mode)
        if bus._loop is not None and bus._loop.is_running():  # noqa: SLF001
            bus.emit_sync(Event.SETTINGS_CHANGED, source="SettingsWindow")

        if hasattr(self, "log_widget"):
            self.log_widget.append_log("success", "✅ Settings saved.")

    def _export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Config", "nixorb_backup.tar.gz.enc",
            "Encrypted archive (*.tar.gz.enc)"
        )
        if path:
            from nixorb.utils.crypto import export_config
            pwd, ok = self._ask_password()
            if ok:
                export_config(self._s, path, pwd)
                if hasattr(self, "log_widget"):
                    self.log_widget.append_log("success", f"Exported → {path}")

    def _import_cfg(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Config", "", "Encrypted archive (*.tar.gz.enc)"
        )
        if path:
            from nixorb.utils.crypto import import_config
            pwd, ok = self._ask_password()
            if ok:
                import_config(self._s, path, pwd)
                if hasattr(self, "log_widget"):
                    self.log_widget.append_log("success", f"Imported ← {path}")

    def _ask_password(self) -> tuple[str, bool]:
        from PySide6.QtWidgets import QInputDialog
        pwd, ok = QInputDialog.getText(
            self, "Encryption Password", "Password:",
            QLineEdit.EchoMode.Password,
        )
        return pwd or "nixorb", ok

    def _reload_plugins(self) -> None:
        from nixorb.plugins.loader import PluginLoader
        loader = PluginLoader(self._s.plugin_dir)
        loader.reload_all()
        names = loader.plugin_names()
        self._plugin_list.setPlainText(
            "\n".join(names) if names else "(no plugins found)"
        )

    def _open_plugin_dir(self) -> None:
        import subprocess, shutil
        d = self._s.plugin_dir
        __import__("pathlib").Path(d).mkdir(parents=True, exist_ok=True)
        for fm in ("dolphin", "nautilus", "thunar", "xdg-open"):
            if shutil.which(fm):
                subprocess.Popen([fm, d])  # noqa: S603
                break
