"""Shared Hugging Face plumbing — device, dtype, auth, and import errors.

Every HF-backed engine (ASR, TTS, LLM) needs the same four things, and gets
them wrong in the same four ways, so they live here once.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

log = logging.getLogger(__name__)

# Env vars people already have set; used when the config leaves the token blank.
_TOKEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")


class MissingDependency(RuntimeError):
    """A backend was selected whose Python package is not installed."""


# A torch stack that will not load fails at the dynamic linker, not in
# Python, so the message names a .so nobody recognises and nothing about
# torch. transformers>=5 imports torchaudio from `audio_utils` when the
# package merely *exists* on disk — `is_torchaudio_available()` is a
# find_spec, never a real import — so a torchaudio that cannot load takes
# down AutoProcessor.from_pretrained, and with it the whole turn.
#
# Two shapes, two different fixes, and giving the wrong one wastes a
# reinstall:
#
#   mismatch — the halves came from different builds. The loader finds the
#   file and it demands something absent: "libcudart.so.12: cannot open
#   shared object file", "undefined symbol: ...". Reinstalling both from
#   one index fixes it.
#
#   leftovers — an earlier install's extension is still in the directory.
#   torchaudio renamed its extension to an abi3 suffix, so the new
#   release's RECORD does not list the old `_torchaudio.so` and pip never
#   deletes it; the loader then finds two candidates where it wants one and
#   gives up with "Could not load this library: …". --force-reinstall does
#   NOT fix this. The directory has to go.
_MISMATCH_MARKERS = (
    "cannot open shared object file",
    "undefined symbol",
    "libcudart",
    "libcudnn",
    "libcublas",
    "libtorch",
    "libc10",
)

_LEFTOVER_MARKERS = (
    "could not load this library",
    "expected a single file path",
)

_SHARED_OBJECT = re.compile(r"([\w./+-]*\.so[\w.]*)")

_TORCH_INSTALL = (
    "pip install torch torchaudio "
    "--index-url https://download.pytorch.org/whl/cpu   "
    "# or …/whl/cu128 for CUDA"
)

def package_directory(package: str = "torchaudio", hint: str = "") -> str:
    """Where `package` is actually installed, for a command to name.

    A path out of the error message is the most reliable source — the
    loader was there. find_spec is the fallback, and does not run the
    package's __init__, so it works when importing is what failed.
    """
    from pathlib import Path

    for parent in Path(hint).parents if hint else ():
        if parent.name == package:
            return str(parent)

    try:
        import importlib.util

        spec = importlib.util.find_spec(package)
    except Exception:
        spec = None
    locations = list(getattr(spec, "submodule_search_locations", None) or [])
    if locations:
        return str(locations[0])

    return f"<site-packages>/{package}"


def purge_advice(hint: str = "", package: str = "torchaudio") -> str:
    """The three commands that actually clear a leftover extension."""
    return (
        f"  pip uninstall -y {package}\n"
        f"  rm -rf {package_directory(package, hint)}\n"
        f"  pip install {package} "
        "--index-url https://download.pytorch.org/whl/cpu"
    )


# Worth saying out loud: nothing in NixOrb imports torchaudio, and
# transformers skips it when the package is absent. Uninstalling it and
# stopping there is a complete fix, not a workaround.
_OPTIONAL_NOTE = (
    "torchaudio is optional here — NixOrb never calls it, and transformers "
    "skips it when it is not installed. `pip uninstall -y torchaudio` is a "
    "complete fix on its own if you do not need it elsewhere."
)


def describe_native_import_error(exc: BaseException) -> str | None:
    """Explain a torch/torchaudio native-library failure, or return None.

    None means this is not one — the caller should report the original
    error rather than invent a plausible story about torch.
    """
    text = str(exc)
    lowered = text.lower()

    match = _SHARED_OBJECT.search(text)
    library = match.group(1) if match else ""

    if any(marker in lowered for marker in _LEFTOVER_MARKERS):
        named = f" ({library})" if library else ""
        return (
            f"torchaudio's compiled extension will not load{named}. Usually "
            "an earlier install left its own extension behind: the loader "
            "finds two and refuses to choose. --force-reinstall does not "
            "clear it, because pip only removes what the current release's "
            "RECORD lists. Delete the directory and install again:\n"
            f"{purge_advice(library)}\n"
            f"{_OPTIONAL_NOTE}"
        )

    if any(marker in lowered for marker in _MISMATCH_MARKERS):
        missing = f" ({library} is missing)" if library else ""
        return (
            f"torch/torchaudio cannot load its native libraries{missing}. "
            "That means the two came from different builds — a CUDA "
            "torchaudio beside a CPU or differently-versioned torch. "
            "Reinstall both together from one index:\n"
            "  pip install --force-reinstall torch torchaudio "
            "--index-url https://download.pytorch.org/whl/cpu\n"
            "or the CUDA build matching your driver (…/whl/cu126, "
            "…/whl/cu128). `nixorb check` reports which one you have."
        )

    return None


def duplicate_extensions(package: str = "torchaudio") -> list[str]:
    """Extension files an earlier install of `package` left behind.

    Returns [] unless there is more than one — a single extension is the
    normal case, and reporting it as a problem would send people chasing
    nothing. Locating the package uses find_spec, which does not run its
    __init__, so this still works when importing it is what fails.
    """
    import importlib.util
    from pathlib import Path

    try:
        spec = importlib.util.find_spec(package)
    except Exception:
        return []
    if spec is None or not spec.submodule_search_locations:
        return []

    found: list[str] = []
    stem = f"_{package}"
    for location in spec.submodule_search_locations:
        for directory in (Path(location) / "lib", Path(location)):
            try:
                found.extend(
                    str(path) for path in sorted(directory.glob(f"{stem}*.so"))
                )
            except OSError:
                continue

    return found if len(found) > 1 else []


# What actually installs each module a backend can be missing. torch is
# deliberately not one of this project's dependencies — no single build is
# right for every machine — so it must never be routed to a NixOrb extra;
# `pip install nixorb[hf]` would not put it there.
_INSTALL_COMMANDS = {
    "torch": _TORCH_INSTALL,
    "torchaudio": _TORCH_INSTALL,
    "torchvision": _TORCH_INSTALL,
    "llama_cpp": "pip install 'nixorb[llama_cpp]'",
    "bitsandbytes": "pip install 'nixorb[quant]'",
    "openwakeword": "pip install 'nixorb[wakeword]'",
    "openai": "pip install 'nixorb[openai]'",
    "faster_whisper": "pip install faster-whisper",
    "chromadb": "pip install chromadb",
}


def install_command(module: str, extra: str = "hf") -> str:
    """The command that actually installs `module`."""
    return _INSTALL_COMMANDS.get(
        module.split(".")[0], f"pip install 'nixorb[{extra}]'"
    )


def is_installed(module: str) -> bool:
    """Is the module present on disk?

    find_spec, not import: a package that exists and then explodes still
    counts as installed. That distinction is the entire point — telling
    somebody to install what they already have sends them in a circle.
    """
    import importlib.util

    try:
        return importlib.util.find_spec(module.split(".")[0]) is not None
    except (ImportError, ValueError):
        return False


def explain_import_error(
    exc: BaseException,
    expected: str,
    feature: str,
    extra: str = "hf",
) -> str:
    """Say which module actually failed, and how to get it.

    `except ImportError` around `import torch, transformers` fires when
    either is absent, and around a lone `import transformers` it fires when
    anything transformers imports is absent too. Naming `expected` in
    either case blames a package that is very often already installed —
    which is exactly what this project did, telling people to reinstall
    transformers when torch was the missing one.

    ImportError carries the module that could not be found in `.name`;
    trust that over the caller's assumption.
    """
    culprit = str(getattr(exc, "name", None) or expected).split(".")[0]

    native = describe_native_import_error(exc)
    if native:
        return (
            f"{feature} needs {culprit}, which is installed but will not "
            f"load.\n{native}"
        )

    if not is_installed(culprit):
        return (
            f"{feature} needs {culprit}, which is not installed. "
            f"Install it with: {install_command(culprit, extra)}"
        )

    # Present but unimportable. Never suggest installing it again.
    root = expected.split(".")[0]
    through = f" (reached through {root})" if culprit != root else ""
    return (
        f"{feature} could not import {culprit}{through}, though it is "
        f"installed: {type(exc).__name__}: {exc}"
    )


def require(module: str, extra: str = "hf", feature: str = "") -> Any:
    """Import a module, or explain what is really missing and how to get it."""
    try:
        return __import__(module)
    except ImportError as exc:
        raise MissingDependency(
            explain_import_error(
                exc, module, feature or "This backend", extra
            )
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
