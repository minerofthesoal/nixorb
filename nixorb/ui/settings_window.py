"""NixOrb settings window — GUI configuration editor.

Allows users to configure all NixOrb settings through a tabbed interface.
Changes are saved to ~/.config/nixorb/config.toml.
"""
from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger(__name__)

# Singleton instance
_settings_window: SettingsWindow | None = None


class SettingsWindow(QDialog):
    """Settings dialog with tabbed configuration."""

    def __init__(self, settings: Any, parent: Any = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._original_settings = settings.model_copy()

        self.setWindowTitle("NixOrb Settings")
        self.setMinimumSize(500, 400)

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self) -> None:
        """Build the settings UI."""
        layout = QVBoxLayout()

        # Tab widget
        tabs = QTabWidget()
        tabs.addTab(self._build_general_tab(), "General")
        tabs.addTab(self._build_asr_tab(), "Speech Recognition")
        tabs.addTab(self._build_llm_tab(), "AI Model")
        tabs.addTab(self._build_tts_tab(), "Voice")
        tabs.addTab(self._build_features_tab(), "Features")
        layout.addWidget(tabs)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        save_btn = QPushButton("💾 Save")
        save_btn.clicked.connect(self._on_save)
        button_layout.addWidget(save_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def _build_general_tab(self) -> QWidget:
        """Build the General settings tab."""
        widget = QWidget()
        layout = QFormLayout()

        # Hotkey
        self._hotkey_edit = QLineEdit()
        layout.addRow("Hotkey:", self._hotkey_edit)

        # Orb size
        self._orb_size_spin = QSpinBox()
        self._orb_size_spin.setRange(60, 300)
        layout.addRow("Orb Size:", self._orb_size_spin)

        # Orb opacity
        opacity_layout = QHBoxLayout()
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(20, 100)
        self._opacity_label = QLabel("100%")
        self._opacity_slider.valueChanged.connect(
            lambda v: self._opacity_label.setText(f"{v}%")
        )
        opacity_layout.addWidget(self._opacity_slider)
        opacity_layout.addWidget(self._opacity_label)
        layout.addRow("Orb Opacity:", opacity_layout)

        widget.setLayout(layout)
        return widget

    def _build_asr_tab(self) -> QWidget:
        """Build the ASR settings tab."""
        widget = QWidget()
        layout = QFormLayout()

        # Model
        self._asr_model_edit = QLineEdit()
        layout.addRow("Whisper Model:", self._asr_model_edit)

        # Language
        self._asr_lang_edit = QLineEdit()
        layout.addRow("Language (empty=auto):", self._asr_lang_edit)

        # Mic sensitivity
        sens_layout = QHBoxLayout()
        self._mic_sens_slider = QSlider(Qt.Orientation.Horizontal)
        self._mic_sens_slider.setRange(1, 100)
        self._mic_sens_label = QLabel("50%")
        self._mic_sens_slider.valueChanged.connect(
            lambda v: self._mic_sens_label.setText(f"{v}%")
        )
        sens_layout.addWidget(self._mic_sens_slider)
        sens_layout.addWidget(self._mic_sens_label)
        layout.addRow("Microphone Sensitivity:", sens_layout)

        widget.setLayout(layout)
        return widget

    def _build_llm_tab(self) -> QWidget:
        """Build the LLM settings tab."""
        widget = QWidget()
        layout = QFormLayout()

        # Ollama host
        self._ollama_host_edit = QLineEdit()
        layout.addRow("Ollama Host:", self._ollama_host_edit)

        # Model
        self._llm_model_edit = QLineEdit()
        layout.addRow("Model Name:", self._llm_model_edit)

        # Max tokens
        self._max_tokens_spin = QSpinBox()
        self._max_tokens_spin.setRange(64, 4096)
        layout.addRow("Max Tokens:", self._max_tokens_spin)

        # Temperature
        self._temp_spin = QDoubleSpinBox()
        self._temp_spin.setRange(0.0, 2.0)
        self._temp_spin.setSingleStep(0.1)
        layout.addRow("Temperature:", self._temp_spin)

        # System prompt
        self._system_prompt_edit = QTextEdit()
        self._system_prompt_edit.setMaximumHeight(150)
        layout.addRow("System Prompt:", self._system_prompt_edit)

        widget.setLayout(layout)
        return widget

    def _build_tts_tab(self) -> QWidget:
        """Build the TTS settings tab."""
        widget = QWidget()
        layout = QFormLayout()

        # Backend
        self._tts_backend_combo = QComboBox()
        self._tts_backend_combo.addItems(["piper", "espeak-ng"])
        layout.addRow("TTS Backend:", self._tts_backend_combo)

        # Voice
        self._tts_voice_edit = QLineEdit()
        layout.addRow("Voice Model:", self._tts_voice_edit)

        # Speed
        self._tts_speed_spin = QDoubleSpinBox()
        self._tts_speed_spin.setRange(0.5, 3.0)
        self._tts_speed_spin.setSingleStep(0.1)
        layout.addRow("Speech Speed:", self._tts_speed_spin)

        # Volume
        vol_layout = QHBoxLayout()
        self._tts_vol_slider = QSlider(Qt.Orientation.Horizontal)
        self._tts_vol_slider.setRange(0, 100)
        self._tts_vol_label = QLabel("100%")
        self._tts_vol_slider.valueChanged.connect(
            lambda v: self._tts_vol_label.setText(f"{v}%")
        )
        vol_layout.addWidget(self._tts_vol_slider)
        vol_layout.addWidget(self._tts_vol_label)
        layout.addRow("Volume:", vol_layout)

        widget.setLayout(layout)
        return widget

    def _build_features_tab(self) -> QWidget:
        """Build the Features settings tab."""
        widget = QWidget()
        layout = QFormLayout()

        # Wake word
        self._wake_word_check = QCheckBox("Enable wake word detection")
        layout.addRow(self._wake_word_check)

        self._wake_word_edit = QLineEdit()
        layout.addRow("Wake Word Model:", self._wake_word_edit)

        # Screen capture
        self._screen_cap_check = QCheckBox("Enable screen capture")
        layout.addRow(self._screen_cap_check)

        # Web search
        self._web_search_check = QCheckBox("Enable web search")
        layout.addRow(self._web_search_check)

        # Clipboard
        self._clipboard_check = QCheckBox("Enable clipboard integration")
        layout.addRow(self._clipboard_check)

        # Confirm actions
        self._confirm_check = QCheckBox("Require confirmation for dangerous commands")
        layout.addRow(self._confirm_check)

        # Memory
        self._memory_check = QCheckBox("Enable conversation memory")
        layout.addRow(self._memory_check)

        widget.setLayout(layout)
        return widget

    def _load_settings(self) -> None:
        """Load current settings into the UI."""
        s = self._settings

        # General
        self._hotkey_edit.setText(s.hotkey)
        self._orb_size_spin.setValue(s.orb_size)
        self._opacity_slider.setValue(int(s.orb_opacity * 100))

        # ASR
        self._asr_model_edit.setText(s.asr_model)
        self._asr_lang_edit.setText(s.asr_language)
        self._mic_sens_slider.setValue(int(s.mic_sensitivity * 100))

        # LLM
        self._ollama_host_edit.setText(s.ollama_host)
        self._llm_model_edit.setText(s.llm_model)
        self._max_tokens_spin.setValue(s.llm_max_tokens)
        self._temp_spin.setValue(s.llm_temperature)
        self._system_prompt_edit.setPlainText(s.llm_system_prompt)

        # TTS
        idx = self._tts_backend_combo.findText(s.tts_backend)
        if idx >= 0:
            self._tts_backend_combo.setCurrentIndex(idx)
        self._tts_voice_edit.setText(s.tts_voice)
        self._tts_speed_spin.setValue(s.tts_speed)
        self._tts_vol_slider.setValue(int(s.tts_volume * 100))

        # Features
        self._wake_word_check.setChecked(s.wake_word_enabled)
        self._wake_word_edit.setText(s.wake_word_model)
        self._screen_cap_check.setChecked(s.screen_capture_enabled)
        self._web_search_check.setChecked(s.web_search_enabled)
        self._clipboard_check.setChecked(s.clipboard_enabled)
        self._confirm_check.setChecked(s.require_action_confirmation)
        self._memory_check.setChecked(s.memory_enabled)

    def _on_save(self) -> None:
        """Save settings from UI to config file."""
        s = self._settings

        # General
        s.hotkey = self._hotkey_edit.text()
        s.orb_size = self._orb_size_spin.value()
        s.orb_opacity = self._opacity_slider.value() / 100.0

        # ASR
        s.asr_model = self._asr_model_edit.text()
        s.asr_language = self._asr_lang_edit.text()
        s.mic_sensitivity = self._mic_sens_slider.value() / 100.0

        # LLM
        s.ollama_host = self._ollama_host_edit.text()
        s.llm_model = self._llm_model_edit.text()
        s.llm_max_tokens = self._max_tokens_spin.value()
        s.llm_temperature = self._temp_spin.value()
        s.llm_system_prompt = self._system_prompt_edit.toPlainText()

        # TTS
        s.tts_backend = self._tts_backend_combo.currentText()
        s.tts_voice = self._tts_voice_edit.text()
        s.tts_speed = self._tts_speed_spin.value()
        s.tts_volume = self._tts_vol_slider.value() / 100.0

        # Features
        s.wake_word_enabled = self._wake_word_check.isChecked()
        s.wake_word_model = self._wake_word_edit.text()
        s.screen_capture_enabled = self._screen_cap_check.isChecked()
        s.web_search_enabled = self._web_search_check.isChecked()
        s.clipboard_enabled = self._clipboard_check.isChecked()
        s.require_action_confirmation = self._confirm_check.isChecked()
        s.memory_enabled = self._memory_check.isChecked()

        s.save()
        log.info("Settings saved")
        QMessageBox.information(self, "NixOrb", "Settings saved successfully!")
        self.accept()

    def _on_cancel(self) -> None:
        """Discard changes and close."""
        self.reject()

    @classmethod
    def show_singleton(cls, settings: Any = None) -> None:
        """Show the settings window as a singleton."""
        global _settings_window

        if _settings_window is not None and _settings_window.isVisible():
            _settings_window.raise_()
            _settings_window.activateWindow()
            return

        if settings is None:
            from nixorb.settings import Settings

            settings = Settings.load()

        _settings_window = cls(settings)
        _settings_window.show()
