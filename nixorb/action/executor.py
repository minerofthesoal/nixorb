"""NixOrb action executor — sandboxed bash command execution.

Parses <ACTION> tags from LLM responses and executes them with optional
bubblewrap sandboxing. Dangerous commands require user confirmation.
"""
from __future__ import annotations

import asyncio
import logging
import re
import shutil
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nixorb.core.event_bus import Event, bus

if TYPE_CHECKING:
    from nixorb.settings import Settings

log = logging.getLogger(__name__)

# Maximum output capture size
MAX_OUTPUT_BYTES = 50_000
# Command timeout
COMMAND_TIMEOUT = 30


@dataclass
class ActionResult:
    """Result of executing an action."""

    command: str
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    approved: bool = True


class ActionExecutor:
    """Executes bash commands from LLM responses."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._sandbox = shutil.which("bwrap") is not None

    def _extract_actions(self, text: str) -> list[str]:
        """Extract <ACTION> commands from LLM response."""
        pattern = r"<ACTION>(.*?)</ACTION>"
        commands = re.findall(pattern, text, re.DOTALL)
        return [cmd.strip() for cmd in commands if cmd.strip()]

    def _build_command(self, cmd: str) -> list[str]:
        """Build the command with optional sandboxing."""
        if self._sandbox and self._settings.require_action_confirmation:
            # Use bubblewrap for sandboxing
            return [
                "bwrap",
                "--unshare-net",
                "--ro-bind", "/", "/",
                "--tmpfs", "/tmp",
                "--proc", "/proc",
                "--dev", "/dev",
                "--die-with-parent",
                "bash", "-c", cmd,
            ]
        return ["bash", "-c", cmd]

    async def _execute_command(self, command: str) -> ActionResult:
        """Execute a single command with confirmation if needed."""
        request_id = str(uuid.uuid4())[:8]

        # Emit confirmation request
        future: asyncio.Future[bool] = asyncio.get_running_loop().create_future()

        bus.emit_sync(
            Event.ACTION_REQUESTED,
            data={"command": command, "request_id": request_id},
            source="ActionExecutor",
        )

        try:
            # Wait for confirmation
            approved = await asyncio.wait_for(future, timeout=60.0)
        except TimeoutError:
            log.warning("Action: confirmation timeout for '%s'", command)
            return ActionResult(command=command, approved=False)

        if not approved:
            log.info("Action: denied '%s'", command)
            return ActionResult(command=command, approved=False)

        # Execute the command
        try:
            log.info("Action: executing '%s'", command)
            proc = await asyncio.create_subprocess_exec(
                *self._build_command(command),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=MAX_OUTPUT_BYTES,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=COMMAND_TIMEOUT
            )

            result = ActionResult(
                command=command,
                stdout=stdout.decode("utf-8", errors="replace")[:5000],
                stderr=stderr.decode("utf-8", errors="replace")[:2000],
                returncode=proc.returncode or 0,
            )

            log.info(
                "Action: '%s' → rc=%d, stdout=%d chars",
                command,
                result.returncode,
                len(result.stdout),
            )
            return result

        except TimeoutError:
            log.error("Action: command timed out '%s'", command)
            proc.kill()
            return ActionResult(
                command=command, stderr="Command timed out", returncode=-1
            )
        except Exception as exc:
            log.error("Action: error executing '%s': %s", command, exc)
            return ActionResult(
                command=command, stderr=str(exc), returncode=-1
            )

    async def handle_llm_output(self, text: str) -> list[ActionResult]:
        """Extract and execute all actions from LLM response."""
        commands = self._extract_actions(text)
        if not commands:
            return []

        log.info("Action: found %d command(s) to execute", len(commands))
        results = []
        for cmd in commands:
            result = await self._execute_command(cmd)
            results.append(result)

            # Emit result
            bus.emit_sync(
                Event.ACTION_RESULT,
                data={
                    "command": result.command,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "returncode": result.returncode,
                },
                source="ActionExecutor",
            )

        return results
