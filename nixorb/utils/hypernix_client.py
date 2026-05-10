"""nixorb/utils/hypernix_client.py — hypernix package integration."""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class HypernixClient:
    """
    Thin wrapper around the hypernix package (https://pypi.org/project/hypernix/).
    Used as an alternative model-fetch and inference layer.
    Falls back gracefully when hypernix is not installed.
    """

    def __init__(self, settings) -> None:
        self._settings = settings
        self._hn: Any  = None
        try:
            import hypernix
            self._hn = hypernix
            log.info(
                "hypernix %s initialised",
                getattr(hypernix, "__version__", "unknown"),
            )
        except ImportError:
            log.warning("hypernix not installed — pip install hypernix")

    def is_available(self) -> bool:
        return self._hn is not None

    def _require(self) -> Any:
        if self._hn is None:
            raise RuntimeError("hypernix is not installed")
        return self._hn

    async def fetch_model(self, repo_id: str, token: str | None = None) -> str:
        """Download and cache a model via hypernix; return local path."""
        import asyncio
        hn   = self._require()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: hn.fetch(repo_id, token=token)
        )

    async def run_inference(self, model_path: str, input_data: Any) -> Any:
        """Generic inference via hypernix."""
        import asyncio
        hn   = self._require()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, lambda: hn.infer(model_path, input_data)
        )
