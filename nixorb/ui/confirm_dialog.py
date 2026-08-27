"""NixOrb action confirmation dialog.

Shows a modal dialog when the AI wants to execute a potentially destructive
command. This is the other half of the handshake described in
``nixorb/action/executor.py``: it answers every ACTION_REQUESTED with either
ACTION_CONFIRMED or ACTION_DENIED carrying the same ``request_id``.

Nothing used to subscribe to ACTION_REQUESTED, so the executor's wait always
timed out and every command was silently denied.
"""
from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from typing import Any

from nixorb.core.event_bus import Event, EventBus, EventPayload
from nixorb.core.event_bus import bus as default_bus

log = logging.getLogger(__name__)

AskFn = Callable[[str], Any]


class ConfirmDialog:
    """Modal dialog asking the user to approve a command."""

    def __init__(self, command: str, parent: Any = None) -> None:
        # Qt is imported lazily so this module can be imported (and the
        # handshake tested) in a headless environment.
        from PySide6.QtWidgets import (
            QDialog,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QTextEdit,
            QVBoxLayout,
        )

        self._dialog = QDialog(parent)
        self._dialog.setWindowTitle("NixOrb — Confirm Action")
        self._dialog.setModal(True)
        self._dialog.setMinimumWidth(500)

        layout = QVBoxLayout()

        from nixorb.action.executor import is_high_risk

        high_risk = is_high_risk(command)
        warning = QLabel(
            "🛑 NixOrb wants to run a HIGH-RISK command:"
            if high_risk
            else "⚠️ NixOrb wants to execute a command:"
        )
        warning.setStyleSheet(
            "font-weight: bold; font-size: 14px;"
            + (" color: #e74c3c;" if high_risk else "")
        )
        layout.addWidget(warning)

        cmd_display = QTextEdit()
        cmd_display.setPlainText(command)
        cmd_display.setReadOnly(True)
        cmd_display.setMaximumHeight(100)
        cmd_display.setStyleSheet(
            "background-color: #2d2d2d; color: #f0f0f0; font-family: monospace;"
        )
        layout.addWidget(cmd_display)

        info = QLabel(
            "This command may modify your system. Only approve if you trust it."
        )
        info.setStyleSheet("color: #888;")
        layout.addWidget(info)

        button_layout = QHBoxLayout()

        deny_btn = QPushButton("❌ Deny")
        deny_btn.setStyleSheet("background-color: #c0392b; color: white;")
        deny_btn.clicked.connect(self._dialog.reject)
        button_layout.addWidget(deny_btn)

        button_layout.addStretch()

        approve_btn = QPushButton("✅ Approve")
        approve_btn.setStyleSheet("background-color: #27ae60; color: white;")
        # Deny is the default so a stray Enter cannot approve anything.
        deny_btn.setDefault(True)
        approve_btn.clicked.connect(self._dialog.accept)
        button_layout.addWidget(approve_btn)

        layout.addLayout(button_layout)
        self._dialog.setLayout(layout)
        self._accepted_code = QDialog.DialogCode.Accepted

    def exec(self) -> bool:
        """Show the dialog and return True if the user approved."""
        return self._dialog.exec() == self._accepted_code

    @classmethod
    def ask(cls, command: str) -> bool:
        """Show a confirmation dialog for ``command``. Never raises."""
        try:
            return cls(command).exec()
        except Exception as exc:
            # No display, no QApplication, Qt blew up — fail closed.
            log.error("Confirm: could not show dialog (%s) — denying", exc)
            return False


def register_confirmation_handler(
    bus: EventBus | None = None,
    event: Event = Event.ACTION_REQUESTED,
    ask_fn: AskFn | None = None,
) -> None:
    """Wire the confirmation dialog onto the event bus.

    Args:
        bus: bus to subscribe on (defaults to the global singleton).
        event: event to listen for (defaults to ACTION_REQUESTED).
        ask_fn: callable taking the command and returning a bool. Defaults to
            the real Qt dialog; tests inject a stub.
    """
    target_bus = bus if bus is not None else default_bus

    async def _handle_action_requested(payload: EventPayload) -> None:
        data = payload.data or {}
        command = data.get("command", "")
        request_id = data.get("request_id", "")

        # Resolved late so tests can patch ConfirmDialog.ask after wiring.
        ask = ask_fn if ask_fn is not None else ConfirmDialog.ask

        try:
            answer = ask(command)
            # The dialog is modal on purpose: the user is expected to answer
            # before anything else happens.
            approved = bool(await answer if inspect.isawaitable(answer) else answer)
        except Exception as exc:
            log.error("Confirm: ask_fn failed for '%s' (%s) — denying",
                      command, exc)
            approved = False

        reply = Event.ACTION_CONFIRMED if approved else Event.ACTION_DENIED
        await target_bus.emit(
            reply,
            data={
                "request_id": request_id,
                "command": command,
                "reason": "" if approved else "User denied",
            },
            source="confirm_dialog",
            priority=1,
        )

    target_bus.subscribe(event, _handle_action_requested)
    log.info("ConfirmDialog: handler registered for %s", event.name)
