"""NixOrb logging setup."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_dir: Path | None = None,
) -> None:
    """Configure NixOrb logging with colored console + optional file output."""
    handlers: list[logging.Handler] = []

    # Console handler with colored output
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    datefmt = "%H:%M:%S"
    console.setFormatter(logging.Formatter(fmt, datefmt))
    handlers.append(console)

    if log_to_file:
        log_dir = log_dir or (Path.home() / ".local" / "share" / "nixorb" / "logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "nixorb.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        handlers.append(fh)

    logging.basicConfig(level=level, handlers=handlers, force=True)

    # Reduce noise from third-party libraries
    for noisy in (
        "urllib3",
        "aiohttp.access",
        "PIL",
        "matplotlib",
        "chromadb",
        "httpx",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)
