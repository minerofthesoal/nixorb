"""Shared Hugging Face plumbing — device, dtype, auth, and import errors.

Every HF-backed engine (ASR, TTS, LLM) needs the same four things, and gets
them wrong in the same four ways, so they live here once.
"""
from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

# Env vars people already have set; used when the config leaves the token blank.
_TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")


class MissingDependency(RuntimeError):
    """A backend was selected whose Python package is not installed."""


def require(module: str, extra: str = "hf") -> Any:
    """Import a module, or explain how to install it.

    A bare ImportError from four levels down tells the user nothing; this
    turns it into the pip command that fixes it.
    """
    try:
        return __import__(module)
    except ImportError as exc:
        raise MissingDependency(
            f"'{module}' is not installed, which this backend needs. "
            f"Install it with:  pip install 'nixorb[{extra}]'"
        ) from exc


def version_tuple(raw: str) -> tuple[int, ...]:
    """Parse a dotted version into comparable ints, ignoring any suffix."""
    parts: list[int] = []
    for chunk in str(raw).split("."):
        digits = ""
        for ch in chunk:
            if not ch.isdigit():
                break
            digits += ch
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def transformers_version() -> tuple[int, ...]:
    """Installed transformers version, or () if it is not installed."""
    try:
        import transformers
    except ImportError:
        return ()
    return version_tuple(getattr(transformers, "__version__", "0"))


def require_transformers(minimum: tuple[int, ...], feature: str) -> None:
    """Fail with the pip command that fixes it, not an architecture error.

    A too-old transformers resolves happily and then dies deep inside
    from_pretrained with "unrecognized model type", which tells the user
    nothing about what to do.
    """
    have = transformers_version()
    if not have:
        raise MissingDependency(
            f"{feature} needs transformers. Install it with: "
            f"pip install 'nixorb[hf]'"
        )
    if have < minimum:
        want = ".".join(str(n) for n in minimum)
        got = ".".join(str(n) for n in have)
        raise MissingDependency(
            f"{feature} needs transformers >= {want}, but {got} is "
            f"installed. Upgrade with: pip install -U 'transformers>={want}'"
        )


def resolve_device(preference: str = "auto") -> str:
    """Pick a torch device string.

    'auto' means CUDA when it is genuinely usable — importing torch is not
    the same as having a working driver, and neither is having a GPU.
    """
    preference = (preference or "auto").lower()
    if preference not in ("auto", "cuda", "cpu"):
        log.warning("Unknown device %r — falling back to auto", preference)
        preference = "auto"

    if preference == "cpu":
        return "cpu"

    try:
        import torch
    except ImportError:
        if preference == "cuda":
            log.warning("Device 'cuda' requested but torch is not installed")
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"

    if preference == "cuda":
        log.warning("Device 'cuda' requested but no CUDA device is available")
    return "cpu"


def torch_dtype(device: str) -> Any:
    """Half precision on GPU, float32 on CPU (fp16 on CPU is slower, not faster)."""
    try:
        import torch
    except ImportError:
        return None
    return torch.float16 if device == "cuda" else torch.float32


def token(settings: Any = None) -> str | None:
    """Resolve a Hugging Face token from settings, then the usual env vars.

    Only needed for gated or private repos; public models work without one.
    """
    configured = getattr(settings, "hf_token", "") if settings else ""
    if configured:
        return str(configured)
    for name in _TOKEN_ENV:
        value = os.environ.get(name)
        if value:
            return value
    return None


def load_kwargs(settings: Any, device: str | None = None) -> dict[str, Any]:
    """Common from_pretrained() arguments for any HF model."""
    device = device or resolve_device(getattr(settings, "hf_device", "auto"))
    kwargs: dict[str, Any] = {}

    tok = token(settings)
    if tok:
        kwargs["token"] = tok

    cache = getattr(settings, "hf_cache_dir", "")
    if cache:
        kwargs["cache_dir"] = str(cache)

    if getattr(settings, "hf_trust_remote_code", False):
        # Off by default: this executes code from the model repo.
        kwargs["trust_remote_code"] = True

    return kwargs


def free_cuda() -> None:
    """Release cached GPU memory after unloading a model."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
