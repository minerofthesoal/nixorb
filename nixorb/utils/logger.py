"""nixorb/utils/logger.py — Centralised logging setup with bus forwarding."""
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


LOG_DIR = Path.home() / ".local" / "share" / "nixorb" / "logs"
LOG_FILE = LOG_DIR / "nixorb.log"


def setup_logging(debug: bool = False, log_to_file: bool = True) -> None:
    """Configure root logger with console + optional rotating file handler."""
    level = logging.DEBUG if debug else logging.INFO
    fmt   = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    date  = "%H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_to_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
        )
        fh.setFormatter(logging.Formatter(fmt, date))
        handlers.append(fh)

    logging.basicConfig(level=level, format=fmt, datefmt=date, handlers=handlers)
    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "chromadb", "urllib3", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


class BusLogHandler(logging.Handler):
    """Forwards Python log records to the EventBus LOG event."""

    def emit(self, record: logging.LogRecord) -> None:
        from nixorb.core.event_bus import Event, bus

        level_map = {
            logging.DEBUG:    "debug",
            logging.INFO:     "info",
            logging.WARNING:  "warning",
            logging.ERROR:    "error",
            logging.CRITICAL: "error",
        }
        level = level_map.get(record.levelno, "info")
        msg   = self.format(record)
        bus.emit_sync(Event.LOG, data={"level": level, "msg": msg}, source=record.name)
