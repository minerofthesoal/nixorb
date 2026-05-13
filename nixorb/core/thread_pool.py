"""nixorb/core/thread_pool.py — Shared ThreadPoolExecutor for blocking I/O."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

_pool: ThreadPoolExecutor | None = None


def get_pool(max_workers: int = 8) -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="nixorb")
    return _pool


async def run_in_pool(fn, *args, **kwargs):
    """Run a blocking function in the shared thread pool."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(get_pool(), lambda: fn(*args, **kwargs))


def shutdown_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.shutdown(wait=False, cancel_futures=True)
        _pool = None
