"""
============================================================
nixorb/plugins/loader.py  — Dynamic plugin loader
============================================================
"""
from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
PLUGIN_DIR = Path(__file__).parents[2] / "plugins"


class PluginLoader:
    def __init__(self) -> None:
        self._plugins: dict[str, Any] = {}

    def load_all(self) -> None:
        PLUGIN_DIR.mkdir(exist_ok=True)
        for py_file in PLUGIN_DIR.glob("*.py"):
            self._load_file(py_file)

    def reload_all(self) -> None:
        self._plugins.clear()
        self.load_all()

    def _load_file(self, path: Path) -> None:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            return
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            self._plugins[path.stem] = module
            log.info("Plugin loaded: %s", path.stem)
        except Exception:
            log.exception("Failed to load plugin: %s", path)

    def plugin_names(self) -> list[str]:
        return list(self._plugins.keys())

    def get_tool_definitions(self) -> list[dict]:
        """Collect OpenAI-style tool definitions from all plugins."""
        tools = []
        for name, module in self._plugins.items():
            if hasattr(module, "TOOL_DEFINITION"):
                tools.append(module.TOOL_DEFINITION)
        return tools

    async def dispatch(self, tool_name: str, args: dict) -> str:
        for name, module in self._plugins.items():
            fn = getattr(module, tool_name, None)
            if fn:
                import asyncio
                if asyncio.iscoroutinefunction(fn):
                    return str(await fn(**args))
                return str(fn(**args))
        return f"Tool '{tool_name}' not found in any plugin."


plugin_loader = PluginLoader()


"""
============================================================
nixorb/memory/vector_store.py  — ChromaDB long-term memory
============================================================
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
MEMORY_DIR = Path.home() / ".local" / "share" / "nixorb" / "memory"


@dataclass
class MemoryEntry:
    text: str
    metadata: dict[str, Any]
    timestamp: float


class VectorMemory:
    """
    ChromaDB-backed long-term memory.
    Stores conversation snippets, commands, and preferences.
    """

    def __init__(self) -> None:
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        import chromadb
        self._client = chromadb.PersistentClient(path=str(MEMORY_DIR))
        self._col = self._client.get_or_create_collection(
            name="nixorb_memory",
            metadata={"hnsw:space": "cosine"},
        )
        log.info("VectorMemory initialized (%d entries)", self._col.count())

    def store(self, text: str, metadata: dict | None = None) -> None:
        doc_id = hashlib.sha256(
            f"{text}{time.time()}".encode()
        ).hexdigest()[:16]
        self._col.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id],
        )

    def query(self, text: str, n_results: int = 5) -> list[str]:
        if self._col.count() == 0:
            return []
        results = self._col.query(
            query_texts=[text], n_results=min(n_results, self._col.count())
        )
        return results.get("documents", [[]])[0]

    def build_context_block(self, query: str) -> str:
        memories = self.query(query)
        if not memories:
            return ""
        joined = "\n".join(f"- {m}" for m in memories)
        return f"\n<long_term_memory>\n{joined}\n</long_term_memory>\n"


"""
============================================================
nixorb/vision/screen_capture.py  — Wayland screen capture
============================================================
"""
from __future__ import annotations

import asyncio
import base64
import logging
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


class ScreenCapture:
    """Uses `grim` to capture the Wayland display, then passes
    the base64 image to a Vision-Language Model."""

    async def capture(self) -> str | None:
        """Returns base64-encoded PNG of the current screen."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = Path(f.name)

        try:
            proc = await asyncio.create_subprocess_exec(
                "grim", str(tmp_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                log.error("grim failed: %s", stderr.decode())
                return None
            data = tmp_path.read_bytes()
            return base64.b64encode(data).decode()
        finally:
            tmp_path.unlink(missing_ok=True)

    async def describe(self, llm_backend, question: str = "What is on this screen?") -> str:
        """Capture screen and send to VLM backend."""
        b64 = await self.capture()
        if b64 is None:
            return "Failed to capture screen."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": question},
                ],
            }
        ]
        result = []
        async for chunk in llm_backend.stream(messages):
            result.append(chunk)
        return "".join(result)


"""
============================================================
nixorb/core/aur_checker.py  — AUR dependency checker
============================================================
"""
from __future__ import annotations

import logging
import subprocess

log = logging.getLogger(__name__)

REQUIRED_PACKAGES: dict[str, str] = {
    "qt6-wayland":  "pacman",
    "cuda":         "pacman",
    "cudnn":        "pacman",
    "python":       "pacman",
    "grim":         "pacman",
    "wl-clipboard": "pacman",
    "ffmpeg":       "pacman",
    "kglobalacceld": "aur",
    "piper-tts":    "aur",
}


def check_dependencies() -> list[str]:
    """Returns list of missing packages. Logs warnings for each."""
    missing = []
    for pkg, source in REQUIRED_PACKAGES.items():
        try:
            result = subprocess.run(
                ["pacman", "-Q", pkg],
                capture_output=True, timeout=5
            )
            if result.returncode != 0:
                log.warning("⚠️ Missing package [%s]: %s", source, pkg)
                missing.append(pkg)
        except Exception:
            log.warning("Could not query pacman for: %s", pkg)
    return missing


"""
============================================================
nixorb/utils/hypernix_client.py  — hypernix integration
============================================================
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class HypernixClient:
    """
    Wraps the hypernix package (https://pypi.org/project/hypernix/).
    hypernix provides a high-level Python interface for neural network
    inference and model management. Used here as an alternative
    inference backend and model-fetching layer.
    """

    def __init__(self, settings) -> None:
        try:
            import hypernix
            self._hn = hypernix
            self._settings = settings
            log.info("hypernix initialized: %s", getattr(hypernix, "__version__", "unknown"))
        except ImportError:
            log.error("hypernix not installed. Run: pip install hypernix")
            self._hn = None

    def is_available(self) -> bool:
        return self._hn is not None

    async def fetch_model(self, repo_id: str, token: str | None = None) -> str:
        """Use hypernix to download and cache a model, returning local path."""
        if not self._hn:
            raise RuntimeError("hypernix not available")
        return self._hn.fetch(repo_id, token=token)

    async def run_inference(self, model_path: str, input_data: Any) -> Any:
        """Generic inference call through hypernix."""
        if not self._hn:
            raise RuntimeError("hypernix not available")
        return self._hn.infer(model_path, input_data)


"""
============================================================
nixorb/utils/crypto.py  — Config export/import (encrypted)
============================================================
"""
from __future__ import annotations

import io
import json
import logging
import os
import tarfile
from pathlib import Path

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

log = logging.getLogger(__name__)


def _derive_key(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=32,
        salt=salt, iterations=480_000,
    )
    return base64.urlsafe_b64encode(kdf.derive(password.encode()))


def export_config(settings, output_path: str, password: str = "nixorb") -> None:
    """Serialize settings + memory into encrypted .tar.gz.enc"""
    salt = os.urandom(16)
    key  = _derive_key(password, salt)
    f    = Fernet(key)

    config_json = settings.model_dump_json(indent=2).encode()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        info = tarfile.TarInfo(name="config.json")
        info.size = len(config_json)
        tar.addfile(info, io.BytesIO(config_json))

        memory_dir = Path.home() / ".local" / "share" / "nixorb" / "memory"
        if memory_dir.exists():
            tar.add(memory_dir, arcname="memory")

    encrypted = salt + f.encrypt(buf.getvalue())
    Path(output_path).write_bytes(encrypted)
    log.info("Config exported to %s", output_path)


def import_config(settings, input_path: str, password: str = "nixorb") -> None:
    """Decrypt and restore settings from .tar.gz.enc"""
    raw  = Path(input_path).read_bytes()
    salt = raw[:16]
    data = raw[16:]
    key  = _derive_key(password, salt)
    f    = Fernet(key)
    decrypted = f.decrypt(data)

    buf = io.BytesIO(decrypted)
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        config_member = tar.getmember("config.json")
        config_json   = tar.extractfile(config_member).read()
        settings_data = json.loads(config_json)
        for k, v in settings_data.items():
            if hasattr(settings, k):
                setattr(settings, k, v)
        settings.save()

        # Restore memory if present
        memory_dir = Path.home() / ".local" / "share" / "nixorb"
        tar.extractall(path=memory_dir, members=[
            m for m in tar.getmembers() if m.name.startswith("memory/")
        ])
    log.info("Config imported from %s", input_path)
