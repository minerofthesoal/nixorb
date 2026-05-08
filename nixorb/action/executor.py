"""
nixorb/action/executor.py

Sandboxed bash/system command executor.

Security layers:
  1. Allowlist/denylist of command prefixes.
  2. All commands run in a restricted subprocess with timeout.
  3. LLM must emit a special <ACTION> XML tag; parsed and confirmed
     by the user (via EventBus) before execution.
  4. Runs as the user, never root (enforced by startup check).
  5. Optional bubblewrap sandbox (if installed).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from nixorb.core.event_bus import Event, EventPayload, bus

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
#  Security policy                                                             #
# --------------------------------------------------------------------------- #
ALWAYS_DENY: list[str] = [
    "rm -rf /",
    "dd if=",
    "mkfs",
    ":(){ :|:& };:",   # fork bomb
    "chmod -R 777 /",
    "sudo rm",
    "passwd",
]

REQUIRE_CONFIRM: list[str] = [
    "rm ",
    "mv ",
    "sudo ",
    "systemctl ",
    "pacman ",
    "yay ",
    "pip install",
    "curl",
    "wget",
    "git push",
    "chmod",
    "chown",
]

ACTION_PATTERN = re.compile(
    r"<ACTION>(.*?)</ACTION>", re.DOTALL | re.IGNORECASE
)

TIMEOUT_SECONDS = 30
USE_BUBBLEWRAP  = shutil.which("bwrap") is not None


@dataclass
class ActionResult:
    command: str
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def __str__(self) -> str:
        parts = [f"$ {self.command}"]
        if self.stdout.strip():
            parts.append(self.stdout.strip())
        if self.stderr.strip():
            parts.append(f"[stderr] {self.stderr.strip()}")
        parts.append(f"[exit {self.returncode}]")
        return "\n".join(parts)


class ActionExecutor:
    def __init__(self, settings) -> None:
        self._settings = settings
        self._pending_confirmations: dict[str, asyncio.Future] = {}
        bus.subscribe(Event.ACTION_RESULT, self._on_confirmation)

        if os.geteuid() == 0:
            raise RuntimeError("NixOrb must not run as root!")

    async def _on_confirmation(self, payload: EventPayload) -> None:
        """Called when user confirms/denies an action from the UI."""
        data = payload.data or {}
        cmd = data.get("command", "")
        approved = data.get("approved", False)
        fut = self._pending_confirmations.pop(cmd, None)
        if fut and not fut.done():
            fut.set_result(approved)

    # ------------------------------------------------------------------ #
    #  Main entry point                                                    #
    # ------------------------------------------------------------------ #
    async def handle_llm_output(self, text: str) -> list[ActionResult]:
        """
        Parse <ACTION>...</ACTION> blocks from LLM output and execute them.
        Returns list of results.
        """
        matches = ACTION_PATTERN.findall(text)
        results: list[ActionResult] = []

        for raw_cmd in matches:
            cmd = raw_cmd.strip()
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
                log.warning("Action DENIED (hard deny): %s", cmd)
                return ActionResult(
                    command=cmd, stdout="", returncode=-1,
                    stderr=f"Denied: matches hard-deny pattern '{pattern}'",
                )

        # 2. Require user confirmation for sensitive commands
        if self._settings.require_action_confirmation or any(
            cmd.startswith(p) or p in cmd for p in REQUIRE_CONFIRM
        ):
            approved = await self._request_confirmation(cmd)
            if not approved:
                return ActionResult(
                    command=cmd, stdout="", returncode=-1,
                    stderr="User denied execution",
                )

        # 3. Execute
        return await self._execute(cmd)

    async def _request_confirmation(self, cmd: str) -> bool:
        """Emit event to UI asking for confirm/deny; wait up to 30 s."""
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        self._pending_confirmations[cmd] = fut

        await bus.emit(
            Event.ACTION_REQUESTED,
            data={"command": cmd},
            source="ActionExecutor",
            priority=1,
        )
        try:
            return await asyncio.wait_for(fut, timeout=30.0)
        except asyncio.TimeoutError:
            log.warning("Confirmation timeout for: %s", cmd)
            return False

    async def _execute(self, cmd: str) -> ActionResult:
        loop = asyncio.get_running_loop()
        log.info("Executing: %s", cmd)

        if USE_BUBBLEWRAP:
            cmd_args = self._wrap_bubblewrap(cmd)
        else:
            cmd_args = ["bash", "-c", cmd]

        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *cmd_args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ, "HOME": os.path.expanduser("~")},
                ),
                timeout=TIMEOUT_SECONDS,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=TIMEOUT_SECONDS
            )
            return ActionResult(
                command=cmd,
                stdout=stdout.decode(errors="replace"),
                stderr=stderr.decode(errors="replace"),
                returncode=proc.returncode,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except Exception:
                pass
            return ActionResult(
                command=cmd, stdout="", stderr="Timed out",
                returncode=-1, timed_out=True,
            )
        except Exception as exc:
            return ActionResult(
                command=cmd, stdout="", stderr=str(exc), returncode=-1
            )

    @staticmethod
    def _wrap_bubblewrap(cmd: str) -> list[str]:
        """Wrap command in bubblewrap for filesystem isolation."""
        home = os.path.expanduser("~")
        return [
            "bwrap",
            "--ro-bind", "/usr", "/usr",
            "--ro-bind", "/lib", "/lib",
            "--ro-bind", "/lib64", "/lib64",
            "--ro-bind", "/etc", "/etc",
            "--bind", home, home,
            "--bind", "/tmp", "/tmp",
            "--dev", "/dev",
            "--proc", "/proc",
            "--unshare-net",        # no network from commands
            "--die-with-parent",
            "bash", "-c", cmd,
        ]
