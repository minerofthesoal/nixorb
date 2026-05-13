"""nixorb/ui/confirm_dialog.py — Confirmation dialog for ACTION blocks."""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
)

log = logging.getLogger(__name__)

_STYLE = """
QDialog     { background:#1a1a2e; color:#e0e0e0; }
QLabel      { color:#e0e0e0; }
QPlainTextEdit { background:#0d0d1a; color:#f39c12; font-family:monospace;
                 border:1px solid #f39c12; border-radius:4px; padding:6px; }
QPushButton { padding:6px 18px; border-radius:4px; border:none; }
QPushButton[text="▶  Run"] { background:#27ae60; color:white; }
QPushButton[text="▶  Run"]:hover { background:#2ecc71; }
QPushButton[text="✕  Deny"] { background:#c0392b; color:white; }
QPushButton[text="✕  Deny"]:hover { background:#e74c3c; }
"""


class ConfirmDialog(QDialog):
    """
    Non-blocking confirmation dialog for shell commands proposed by the LLM.
    Shown before any ACTION block executes.
    """

    def __init__(self, command: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("NixOrb — Command Confirmation")
        self.setStyleSheet(_STYLE)
        self.setMinimumWidth(520)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self._build_ui(command)

    def _build_ui(self, command: str) -> None:
        v = QVBoxLayout(self)

        warn = QLabel("⚠  NixOrb wants to run this command:")
        warn.setStyleSheet("font-weight:bold; font-size:13px;")
        v.addWidget(warn)

        cmd_box = QPlainTextEdit()
        cmd_box.setReadOnly(True)
        cmd_box.setPlainText(command)
        cmd_box.setMaximumHeight(100)
        v.addWidget(cmd_box)

        hint = QLabel("Review carefully before running.")
        hint.setStyleSheet("color:#95a5a6; font-size:11px;")
        v.addWidget(hint)

        btns = QDialogButtonBox()
        run_btn  = btns.addButton("▶  Run",  QDialogButtonBox.ButtonRole.AcceptRole)
        deny_btn = btns.addButton("✕  Deny", QDialogButtonBox.ButtonRole.RejectRole)
        run_btn.clicked.connect(self.accept)
        deny_btn.clicked.connect(self.reject)
        v.addWidget(btns)

    @staticmethod
    def ask(command: str) -> bool:
        """Show dialog and return True if user approved."""
        dlg = ConfirmDialog(command)
        result = dlg.exec()
        approved = result == QDialog.DialogCode.Accepted
        log.info("Command %s by user: %s", "approved" if approved else "denied", command[:80])
        return approved
