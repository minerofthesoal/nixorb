"""NixOrb action confirmation dialog.

Shows a modal dialog when the AI wants to execute a potentially
destructive command, requiring explicit user confirmation.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from nixorb.core.event_bus import Event, EventPayload, bus

log = logging.getLogger(__name__)

# Commands that are always denied
HARD_DENYLIST = {
    "rm -rf /",
    "rm -rf /*",
    "dd if=/dev/zero of=/dev/sda",
    "mkfs.",
    ":(){ :|:& };:",
    "> /dev/sda",
}

# Commands that require confirmation
REQUIRE_CONFIRM = {
    "rm -rf",
    "rm -r",
    "dd ",
    "mkfs",
    "fdisk",
    "parted",
    "chmod -R",
    "chown -R",
    "pacman -R",
    "pacman -S",
    "systemctl stop",
    "systemctl disable",
    "kill ",
    "pkill",
    "curl",
    "wget",
    "pip install",
    "pip uninstall",
}


def _is_dangerous(command: str) -> bool:
    """Check if a command requires confirmation."""
    cmd_lower = command.lower().strip()

    # Hard denylist
    for denied in HARD_DENYLIST:
        if denied in cmd_lower:
            return True

    # Confirmation required
    for pattern in REQUIRE_CONFIRM:
        if pattern in cmd_lower:
            return True

    return False


def _should_confirm(command: str) -> bool:
    """Check if a command should show the confirmation dialog."""
    return _is_dangerous(command)


class ConfirmDialog(QDialog):
    """Modal dialog for command confirmation."""

    def __init__(self, command: str, parent: Any = None) -> None:
        super().__init__(parent)
        self._command = command
        self._result = False

        self.setWindowTitle("NixOrb — Confirm Action")
        self.setModal(True)
        self.setMinimumWidth(500)

        layout = QVBoxLayout()

        # Warning label
        warning = QLabel("⚠️ NixOrb wants to execute a command:")
        warning.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(warning)

        # Command display
        cmd_display = QTextEdit()
        cmd_display.setPlainText(command)
        cmd_display.setReadOnly(True)
        cmd_display.setMaximumHeight(100)
        cmd_display.setStyleSheet(
            "background-color: #2d2d2d; color: #f0f0f0; font-family: monospace;"
        )
        layout.addWidget(cmd_display)

        # Info label
        info = QLabel("This command may modify your system. Only approve if you trust it.")
        info.setStyleSheet("color: #888;")
        layout.addWidget(info)

        # Buttons
        button_layout = QHBoxLayout()

        deny_btn = QPushButton("❌ Deny")
        deny_btn.setStyleSheet("background-color: #c0392b; color: white;")
        deny_btn.clicked.connect(self._on_deny)
        button_layout.addWidget(deny_btn)

        button_layout.addStretch()

        approve_btn = QPushButton("✅ Approve")
        approve_btn.setStyleSheet("background-color: #27ae60; color: white;")
        approve_btn.setDefault(True)
        approve_btn.clicked.connect(self._on_approve)
        button_layout.addWidget(approve_btn)

        layout.addLayout(button_layout)
        self.setLayout(layout)

    def _on_approve(self) -> None:
        self._result = True
        self.accept()

    def _on_deny(self) -> None:
        self._result = False
        self.reject()

    def get_result(self) -> bool:
        return self._result


# Global pending confirmations
_pending_confirmations: dict[str, asyncio.Future[bool]] = {}


def register_confirmation_handler() -> None:
    """Register the event bus handler for action confirmations."""

    async def _handle_action_requested(payload: EventPayload) -> None:
        data = payload.data or {}
        command = data.get("command", "")
        request_id = data.get("request_id", "")

        if not _should_confirm(command):
            # Auto-approve safe commands
            bus.emit_sync(
                Event.ACTION_CONFIRMED,
                data={"request_id": request_id, "command": command},
                source="confirm_dialog",
            )
            return

        # Check hard denylist
        cmd_lower = command.lower().strip()
        for denied in HARD_DENYLIST:
            if denied in cmd_lower:
                log.warning("Confirm: hard-denied command: %s", command)
                bus.emit_sync(
                    Event.ACTION_DENIED,
                    data={
                        "request_id": request_id,
                        "command": command,
                        "reason": "Hard denylist",
                    },
                    source="confirm_dialog",
                )
                return

        # Show confirmation dialog on main thread
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()
        _pending_confirmations[request_id] = future

        def _show_dialog() -> None:
            try:
                dialog = ConfirmDialog(command)
                result = dialog.exec() == QDialog.DialogCode.Accepted
                if not future.done():
                    future.set_result(result)
            except Exception as exc:
                log.error("Confirm dialog error: %s", exc)
                if not future.done():
                    future.set_result(False)

        # Schedule on Qt main thread
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, _show_dialog)

        # Wait for user response
        try:
            approved = await asyncio.wait_for(future, timeout=60.0)
        except asyncio.TimeoutError:
            log.warning("Confirm: timeout waiting for user response")
            approved = False

        if request_id in _pending_confirmations:
            del _pending_confirmations[request_id]

        if approved:
            bus.emit_sync(
                Event.ACTION_CONFIRMED,
                data={"request_id": request_id, "command": command},
                source="confirm_dialog",
            )
        else:
            bus.emit_sync(
                Event.ACTION_DENIED,
                data={
                    "request_id": request_id,
                    "command": command,
                    "reason": "User denied",
                },
                source="confirm_dialog",
            )

    bus.subscribe(Event.ACTION_REQUESTED, _handle_action_requested)
    log.info("ConfirmDialog: handler registered")
