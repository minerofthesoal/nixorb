"""
nixorb/action/executor.py

Sandboxed bash/system command executor.

Security model:
  1. Hard-deny list blocks destructive patterns unconditionally.
  2. Sensitive prefixes require interactive user confirmation via EventBus.
  3. All commands run as the current user in a subprocess with timeout.
  4. Optional bubblewrap (bwrap) filesystem sandbox when available.
  5. NixOrb refuses to run as root.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nixorb.core.event_bus import Event, EventPayload, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

ACTION_PATTERN = re.compile(r"<ACTION>(.*?)</ACTION>", re.DOTALL | re.IGNORECASE)

TIMEOUT_SECONDS  = 30
USE_BUBBLEWRAP   = shutil.which("bwrap") is not None

ALWAYS_DENY: list[str] = [
    "rm -rf /",
    "rm -rf ~",
    "dd if=",
    "mkfs",
    ":(){ :|:& };:",
    "chmod -R 777 /",
    "passwd",
    "> /dev/sda",
]

REQUIRE_CONFIRM: list[str] = [
    "rm ",
    "mv ",
    "sudo ",
    "systemctl ",
    "pacman ",
    "yay ",
    "pip install",
    "curl ",
    "wget ",
    "git push",
    "chmod",
    "chown",
    "mktemp",
    "shutdown",
    "reboot",
]


@dataclass
class ActionResult:
    command:    str
    stdout:     str
    stderr:     str
    returncode: int
    timed_out:  bool = False

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def __str__(self) -> str:
        lines = [f"$ {self.command}"]
        if self.stdout.strip():
            lines.append(self.stdout.rstrip())
        if self.stderr.strip():
            lines.append(f"[stderr] {self.stderr.rstrip()}")
        lines.append(f"[exit {self.returncode}{'  TIMEOUT' if self.timed_out else ''}]")
        return "\n".join(lines)


class ActionExecutor:
    def __init__(self, settings: "Settings") -> None:
        self._settings = settings
        self._pending: dict[str, asyncio.Future[bool]] = {}
        bus.subscribe(Event.ACTION_RESULT, self._on_confirmation)

        if os.geteuid() == 0:
            raise RuntimeError(
                "NixOrb must NOT run as root. "
                "Drop privileges before starting."
            )

    async def _on_confirmation(self, payload: EventPayload) -> None:
        data     = payload.data or {}
        cmd      = data.get("command", "")
        approved = bool(data.get("approved", False))
        fut      = self._pending.pop(cmd, None)
        if fut and not fut.done():
            fut.set_result(approved)

    async def handle_llm_output(self, text: str) -> list[ActionResult]:
        matches = ACTION_PATTERN.findall(text)
        results: list[ActionResult] = []
        for raw_cmd in matches:
            cmd    = raw_cmd.strip()
            result = await self._run_action(cmd)
            results.append(result)
            await bus.emit(
                Event.LOG,
                data={"level": "exec", "msg": str(result)},
                source="ActionExecutor",
            )
        return results

    async def _run_action(self, cmd: str) -> ActionResult:
        # 1. Hard deny
        for pattern in ALWAYS_DENY:
            if pattern in cmd:
                msg = f"Command hard-denied (matched '{pattern}'): {cmd}"
                log.warning(msg)
                return ActionResult(command=cmd, stdout="", stderr=msg, returncode=-1)

        # 2. Confirmation for sensitive commands
        needs_confirm = self._settings.require_action_confirmation or any(
            (cmd.startswith(p) or p in cmd) for p in REQUIRE_CONFIRM
        )
        if needs_confirm:
            approved = await self._request_confirmation(cmd)
            if not approved:
                return ActionResult(
                    command=cmd, stdout="", stderr="User denied", returncode=-1
                )

        return await self._execute(cmd)

    async def _request_confirmation(self, cmd: str) -> bool:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        self._pending[cmd] = fut

        await bus.emit(
            Event.ACTION_REQUESTED,
            data={"command": cmd},
            source="ActionExecutor",
            priority=1,
        )
        try:
            return await asyncio.wait_for(asyncio.shield(fut), timeout=30.0)
        except asyncio.TimeoutError:
            self._pending.pop(cmd, None)
            log.warning("Confirmation timed out for: %s", cmd)
            return False

    async def _execute(self, cmd: str) -> ActionResult:
        log.info("Executing: %s", cmd)
        cmd_args = self._wrap_bwrap(cmd) if USE_BUBBLEWRAP else ["bash", "-c", cmd]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ},
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                return ActionResult(
                    command=cmd, stdout="", stderr="Execution timed out",
                    returncode=-1, timed_out=True,
                )
            return ActionResult(
                command=cmd,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                returncode=proc.returncode or 0,
            )
        except Exception as exc:
            return ActionResult(command=cmd, stdout="", stderr=str(exc), returncode=-1)

    @staticmethod
    def _wrap_bwrap(cmd: str) -> list[str]:
        home = os.path.expanduser("~")
        return [
            "bwrap",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64",
            "--ro-bind", "/etc", "/etc",
            "--bind",    home,  home,
            "--bind",    "/tmp", "/tmp",
            "--dev",     "/dev",
            "--proc",    "/proc",
            "--unshare-net",
            "--die-with-parent",
            "bash", "-c", cmd,
        ]
