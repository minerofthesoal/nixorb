"""
nixorb/ui/settings_window.py

Comprehensive settings GUI:
  - ASR / LLM / TTS model selectors
  - Microphone device selector
  - Live syntax-highlighted log (Pygments via QTextEdit)
  - Config export/import (encrypted tar.gz)
  - Plugin management
  - Wake-word toggle
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import ClassVar, Optional

from PySide6.QtCore import Qt, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QTextCharFormat, QTextCursor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QPushButton, QScrollArea, QSizePolicy, QSplitter,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import BashLexer, PythonLexer, get_lexer_by_name

from nixorb.core.event_bus import Event, EventPayload, bus

log = logging.getLogger(__name__)

MONOSPACE = QFont("JetBrains Mono", 10)
MONOSPACE.setStyleHint(QFont.StyleHint.Monospace)


# ------------------------------------------------------------------ #
#  Syntax-Highlighted Log Widget                                      #
# ------------------------------------------------------------------ #
class HighlightedLogWidget(QTextEdit):
    """
    Thread-safe scrolling log that renders:
      - Bash commands with BashLexer
      - Python tracebacks with PythonLexer
      - Plain text with colored level prefixes
    """
    _append_signal = Signal(str)

    LEVEL_COLORS = {
        "info":    "#2ECC71",
        "warning": "#F39C12",
        "error":   "#E74C3C",
        "exec":    "#3498DB",
        "debug":   "#95A5A6",
    }

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(MONOSPACE)
        self.setStyleSheet(
            "background: #1a1a2e; color: #e0e0e0; border: 1px solid #333;"
        )
        self._formatter = HtmlFormatter(
            style="monokai",
            noclasses=True,
            nowrap=True,
        )
        self._append_signal.connect(self._append_html_slot)

    def append_log(self, level: str, msg: str) -> None:
        """Thread-safe append. Detects bash/python and highlights accordingly."""
        color = self.LEVEL_COLORS.get(level, "#e0e0e0")
        prefix = f"<span style='color:{color};font-weight:bold'>[{level.upper()}]</span> "

        # Detect and highlight bash commands
        if level == "exec" or msg.strip().startswith("$"):
            highlighted = highlight(msg, BashLexer(), self._formatter)
            html = f"{prefix}<br>{highlighted}<br>"
        elif "Traceback" in msg or "Error:" in msg:
            highlighted = highlight(msg, PythonLexer(), self._formatter)
            html = f"{prefix}<br>{highlighted}<br>"
        else:
            escaped = msg.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html = f"{prefix}<span style='color:{color}'>{escaped}</span><br>"

        self._append_signal.emit(html)

    @Slot(str)
    def _append_html_slot(self, html: str) -> None:
        self.moveCursor(QTextCursor.MoveOperation.End)
        self.insertHtml(html)
        self.moveCursor(QTextCursor.MoveOperation.End)
        # Auto-scroll
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())

    def clear_log(self) -> None:
        self.clear()


# ------------------------------------------------------------------ #
#  Settings Window                                                    #
# ------------------------------------------------------------------ #
class SettingsWindow(QDialog):
    _singleton: ClassVar[Optional["SettingsWindow"]] = None

    @classmethod
    def show_singleton(cls) -> None:
        if cls._singleton is None or not cls._singleton.isVisible():
            # Settings needs the running app's settings object
            # injected from main; stored on class for convenience
            if hasattr(cls, "_settings"):
                cls._singleton = cls(cls._settings)
                cls._singleton.show()
        else:
            cls._singleton.raise_()
            cls._singleton.activateWindow()

    def __init__(self, settings, parent=None) -> None:
        super().__init__(parent)
        self._settings = settings
        self.setWindowTitle("NixOrb Settings")
        self.setMinimumSize(900, 650)
        self.setStyleSheet("""
            QDialog, QWidget { background: #1a1a2e; color: #e0e0e0; }
            QTabWidget::pane { border: 1px solid #333; }
            QTabBar::tab { background: #16213e; padding: 8px 16px; }
            QTabBar::tab:selected { background: #0f3460; color: #e94560; }
            QGroupBox { border: 1px solid #333; border-radius: 4px;
                        margin-top: 8px; padding: 8px; }
            QGroupBox::title { color: #3498db; }
            QLineEdit, QComboBox { background: #16213e; border: 1px solid #333;
                                   padding: 4px; border-radius: 3px; }
            QPushButton { background: #0f3460; border: none; padding: 6px 14px;
                          border-radius: 3px; }
            QPushButton:hover { background: #e94560; }
        """)

        self._build_ui()
        self._connect_events()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self._tabs = QTabWidget()
        layout.addWidget(self._tabs)

        self._tabs.addTab(self._build_models_tab(),   "🤖 Models")
        self._tabs.addTab(self._build_audio_tab(),    "🎙 Audio")
        self._tabs.addTab(self._build_system_tab(),   "⚙ System")
        self._tabs.addTab(self._build_plugins_tab(),  "🔌 Plugins")
        self._tabs.addTab(self._build_log_tab(),      "📋 Log")

        # Bottom bar
        bar = QHBoxLayout()
        self._export_btn = QPushButton("📦 Export Config")
        self._import_btn = QPushButton("📂 Import Config")
        self._save_btn   = QPushButton("💾 Save")
        bar.addWidget(self._export_btn)
        bar.addWidget(self._import_btn)
        bar.addStretch()
        bar.addWidget(self._save_btn)
        layout.addLayout(bar)

        self._export_btn.clicked.connect(self._export_config)
        self._import_btn.clicked.connect(self._import_config)
        self._save_btn.clicked.connect(self._save_settings)

    def _build_models_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)

        # --- ASR ---
        asr_group = QGroupBox("ASR (Speech Recognition)")
        asr_form  = QFormLayout(asr_group)
        self._asr_model_combo = QComboBox()
        self._asr_model_combo.addItems([
            "faster-whisper/large-v3 (local)",
            "faster-whisper/medium (local)",
            "openai/whisper-1 (API)",
        ])
        asr_form.addRow("Model:", self._asr_model_combo)
        layout.addWidget(asr_group)

        # --- LLM ---
        llm_group = QGroupBox("LLM Backend")
        llm_form  = QFormLayout(llm_group)
        self._llm_backend_combo = QComboBox()
        self._llm_backend_combo.addItems([
            "OpenAI API", "Local (llama.cpp)", "Ollama", "Groq API", "Together AI"
        ])
        self._llm_model_edit   = QLineEdit(self._settings.llm_model or "gpt-4o-mini")
        self._llm_api_key_edit = QLineEdit(self._settings.openai_api_key or "")
        self._llm_api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._llm_base_url_edit = QLineEdit(
            self._settings.llm_base_url or "https://api.openai.com/v1"
        )
        self._llm_model_path_edit = QLineEdit(self._settings.local_model_path or "")
        llm_form.addRow("Backend:", self._llm_backend_combo)
        llm_form.addRow("Model / Path:", self._llm_model_edit)
        llm_form.addRow("API Key:", self._llm_api_key_edit)
        llm_form.addRow("Base URL:", self._llm_base_url_edit)
        llm_form.addRow("Local model path:", self._llm_model_path_edit)
        layout.addWidget(llm_group)

        # --- TTS ---
        tts_group = QGroupBox("TTS (Text-to-Speech)")
        tts_form  = QFormLayout(tts_group)
        self._tts_backend_combo = QComboBox()
        self._tts_backend_combo.addItems([
            "OpenAI TTS (API)", "HuggingFace Repo", "Piper (offline)"
        ])
        self._tts_hf_repo_edit  = QLineEdit(self._settings.tts_hf_repo or "")
        self._tts_hf_token_edit = QLineEdit(self._settings.hf_token or "")
        self._tts_hf_token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._tts_voice_edit = QLineEdit(self._settings.tts_voice or "alloy")
        tts_form.addRow("Backend:", self._tts_backend_combo)
        tts_form.addRow("HF Repo ID:", self._tts_hf_repo_edit)
        tts_form.addRow("HF Token:", self._tts_hf_token_edit)
        tts_form.addRow("Voice / Speaker:", self._tts_voice_edit)
        layout.addWidget(tts_group)

        layout.addStretch()
        return w

    def _build_audio_tab(self) -> QWidget:
        import sounddevice as sd
        w = QWidget()
        layout = QFormLayout(w)

        self._mic_combo = QComboBox()
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                self._mic_combo.addItem(f"[{i}] {d['name']}", userData=i)
        layout.addRow("Microphone:", self._mic_combo)

        self._wake_word_check = QCheckBox("Enable wake-word detection")
        self._wake_word_check.setChecked(self._settings.wake_word_enabled)
        layout.addRow(self._wake_word_check)

        self._wake_word_model_edit = QLineEdit(
            self._settings.wake_word_model or "hey_jarvis"
        )
        layout.addRow("Wake-word model:", self._wake_word_model_edit)

        return w

    def _build_system_tab(self) -> QWidget:
        w = QWidget()
        layout = QFormLayout(w)

        self._action_confirm_check = QCheckBox("Require confirmation before executing commands")
        self._action_confirm_check.setChecked(self._settings.require_action_confirmation)
        layout.addRow(self._action_confirm_check)

        self._screen_capture_check = QCheckBox("Enable screen context awareness")
        self._screen_capture_check.setChecked(self._settings.screen_capture_enabled)
        layout.addRow(self._screen_capture_check)

        self._offline_fallback_check = QCheckBox("Enable offline fallback mode")
        self._offline_fallback_check.setChecked(self._settings.offline_fallback_enabled)
        layout.addRow(self._offline_fallback_check)

        self._hotkey_edit = QLineEdit(self._settings.hotkey or "Ctrl+Alt+Space")
        layout.addRow("Global hotkey:", self._hotkey_edit)

        return w

    def _build_plugins_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.addWidget(QLabel("Drop .py plugin files into the plugins/ directory."))
        self._plugin_list = QPlainTextEdit()
        self._plugin_list.setReadOnly(True)
        layout.addWidget(self._plugin_list)
        reload_btn = QPushButton("🔄 Reload Plugins")
        reload_btn.clicked.connect(self._reload_plugins)
        layout.addWidget(reload_btn)
        return w

    def _build_log_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        self.log_widget = HighlightedLogWidget()
        layout.addWidget(self.log_widget)
        clear_btn = QPushButton("🗑 Clear Log")
        clear_btn.clicked.connect(self.log_widget.clear_log)
        layout.addWidget(clear_btn)
        return w

    # ---------------------------------------------------------------- #
    #  EventBus connection for log forwarding                           #
    # ---------------------------------------------------------------- #
    def _connect_events(self) -> None:
        bus.subscribe(Event.LOG, self._on_log_event)

    async def _on_log_event(self, payload: EventPayload) -> None:
        data  = payload.data or {}
        level = data.get("level", "info")
        msg   = data.get("msg", "")
        if hasattr(self, "log_widget"):
            QTimer.singleShot(0, lambda: self.log_widget.append_log(level, msg))

    # ---------------------------------------------------------------- #
    #  Config export / import                                           #
    # ---------------------------------------------------------------- #
    def _export_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Config", "nixorb_config.tar.gz.enc",
            "Encrypted Archive (*.tar.gz.enc)"
        )
        if path:
            from nixorb.utils.crypto import export_config
            export_config(self._settings, path)
            self.log_widget.append_log("info", f"Config exported to {path}")

    def _import_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Config", "",
            "Encrypted Archive (*.tar.gz.enc)"
        )
        if path:
            from nixorb.utils.crypto import import_config
            import_config(self._settings, path)
            self.log_widget.append_log("info", f"Config imported from {path}")

    def _save_settings(self) -> None:
        self._settings.openai_api_key   = self._llm_api_key_edit.text()
        self._settings.llm_model        = self._llm_model_edit.text()
        self._settings.llm_base_url     = self._llm_base_url_edit.text()
        self._settings.local_model_path = self._llm_model_path_edit.text()
        self._settings.hotkey           = self._hotkey_edit.text()
        self._settings.wake_word_enabled = self._wake_word_check.isChecked()
        self._settings.require_action_confirmation = self._action_confirm_check.isChecked()
        self._settings.save()
        bus.emit_sync(Event.SETTINGS_CHANGED, source="SettingsWindow")
        self.log_widget.append_log("info", "Settings saved.")

    def _reload_plugins(self) -> None:
        from nixorb.plugins.loader import plugin_loader
        plugin_loader.reload_all()
        names = ", ".join(plugin_loader.plugin_names())
        self._plugin_list.setPlainText(names or "No plugins loaded.")
