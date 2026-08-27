"""NixOrb IPC — talk to the running orb from the command line.

`nixorb trigger` exists so a KDE global shortcut can activate the assistant:
KWin runs a command, and that command has to reach the already-running
process. This is that channel — a Unix domain socket in the user's runtime
directory, one line of text in, one line of text out.

    $ nixorb trigger
    ok

Deliberately not D-Bus: this needs no extra dependency, no session bus, and
works the same under X11, Wayland, and a bare TTY.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import stat
from collections.abc import Awaitable, Callable
from pathlib import Path

log = logging.getLogger(__name__)

# One line per request, one per reply. Requests are tiny; cap them so a
# confused client cannot make us buffer forever.
MAX_REQUEST_BYTES = 4096
CLIENT_TIMEOUT = 2.0

Handler = Callable[[str], Awaitable[str]]


def socket_path() -> Path:
    """Where the control socket lives.

    XDG_RUNTIME_DIR is the right home for it: it is user-private (0700),
    on tmpfs, and cleared when the session ends, so a stale socket cannot
    outlive a reboot.
    """
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    if runtime and Path(runtime).is_dir():
        return Path(runtime) / "nixorb.sock"
    return Path(f"/tmp/nixorb-{os.getuid()}.sock")  # noqa: S108


def send(command: str, timeout: float = CLIENT_TIMEOUT) -> str:
    """Send one command to the running instance and return its reply.

    Raises ConnectionError when nothing is listening — the caller decides how
    to phrase "NixOrb isn't running".
    """
    path = socket_path()
    if not path.exists():
        raise ConnectionError(f"no NixOrb control socket at {path}")

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(str(path))
            sock.sendall(command.strip().encode("utf-8") + b"\n")

            chunks: list[bytes] = []
            while b"\n" not in b"".join(chunks):
                chunk = sock.recv(1024)
                if not chunk:
                    break
                chunks.append(chunk)
    except (ConnectionRefusedError, FileNotFoundError) as exc:
        # A socket file with nobody behind it — the app died without cleaning
        # up. Say so as "not running" rather than leaking errno at the user.
        raise ConnectionError(f"nothing is listening on {path}") from exc
    except OSError as exc:
        raise ConnectionError(f"cannot talk to NixOrb at {path}: {exc}") from exc

    return b"".join(chunks).decode("utf-8", errors="replace").strip()


def is_running() -> bool:
    """True if an instance answers on the control socket."""
    try:
        return send("ping", timeout=1.0).startswith("ok")
    except ConnectionError:
        return False


class IPCServer:
    """Serves control commands for the running NixOrb instance."""

    def __init__(self, handlers: dict[str, Handler]) -> None:
        self._handlers = handlers
        self._server: asyncio.AbstractServer | None = None
        self._path = socket_path()

    @property
    def path(self) -> Path:
        return self._path

    async def _someone_is_listening(self) -> bool:
        """Async liveness probe for an existing socket file.

        Must not use the blocking send(): from inside the running loop that
        would wait on a reply this same loop has to produce.
        """
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(str(self._path)),
                timeout=CLIENT_TIMEOUT,
            )
        except (ConnectionRefusedError, FileNotFoundError, OSError, TimeoutError):
            return False

        try:
            writer.write(b"ping\n")
            await writer.drain()
            reply = await asyncio.wait_for(reader.readline(), timeout=CLIENT_TIMEOUT)
            return reply.strip().startswith(b"ok")
        except (OSError, TimeoutError):
            return False
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def start(self) -> bool:
        """Bind the control socket. False if another instance owns it."""
        if self._path.exists():
            if await self._someone_is_listening():
                log.warning("IPC: another NixOrb already owns %s", self._path)
                return False
            # Stale file from a crash: nothing answered, so reclaim it.
            log.info("IPC: removing stale socket %s", self._path)
            with contextlib.suppress(OSError):
                self._path.unlink()

        try:
            self._server = await asyncio.start_unix_server(
                self._handle_client, path=str(self._path)
            )
        except OSError as exc:
            log.error("IPC: cannot listen on %s: %s", self._path, exc)
            return False

        # Nobody else's business: the socket accepts commands that run code.
        with contextlib.suppress(OSError):
            self._path.chmod(stat.S_IRUSR | stat.S_IWUSR)

        log.info("IPC: listening on %s", self._path)
        return True

    async def stop(self) -> None:
        """Close the socket and remove the file."""
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
        with contextlib.suppress(OSError):
            self._path.unlink()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            raw = await asyncio.wait_for(
                reader.readline(), timeout=CLIENT_TIMEOUT
            )
            command = raw[:MAX_REQUEST_BYTES].decode("utf-8", errors="replace").strip()

            if not command:
                reply = "error: empty command"
            elif command == "ping":
                reply = "ok: nixorb"
            elif command in self._handlers:
                try:
                    reply = await self._handlers[command](command)
                except Exception as exc:
                    log.exception("IPC: handler for '%s' failed", command)
                    reply = f"error: {command} failed: {exc}"
            else:
                known = ", ".join(["ping", *sorted(self._handlers)])
                reply = f"error: unknown command '{command}' (known: {known})"

            writer.write(reply.encode("utf-8") + b"\n")
            await writer.drain()
        except TimeoutError:
            log.debug("IPC: client timed out before sending a command")
        except (ConnectionResetError, BrokenPipeError):
            # The client hung up before reading the reply. Normal.
            log.debug("IPC: client disconnected early")
        except Exception:
            log.exception("IPC: error handling client")
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()
