"""Naming the module that actually failed, and an extra that exists.

Two bugs reported from the field, in one message:

    transformers is not installed — HuggingFace LLM backend is
    unavailable. Install it with: pip install 'nixorb[huggingface]'

transformers *was* installed; torch was the missing one, imported on the
same line. And `huggingface` was not a declared extra, so the suggested
command answered with a warning and installed nothing.
"""
from __future__ import annotations

import importlib.util

import pytest

from nixorb.hf import (
    MissingDependency,
    explain_import_error,
    install_command,
    is_installed,
    require,
)

TORCH_INDEX = "download.pytorch.org"


def absent(monkeypatch, *names):
    """Make `names` look uninstalled to find_spec, leaving the rest alone."""
    real = importlib.util.find_spec

    def fake(name, *args, **kwargs):
        if name.split(".")[0] in names:
            return None
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake)


class TestInstallCommand:
    def test_torch_never_routes_to_a_nixorb_extra(self):
        # torch is deliberately not a dependency of this project, so
        # `pip install nixorb[hf]` would not put it there.
        assert TORCH_INDEX in install_command("torch")
        assert "nixorb[" not in install_command("torch")

    @pytest.mark.parametrize(
        "module,expected",
        [
            ("llama_cpp", "nixorb[llama_cpp]"),
            ("bitsandbytes", "nixorb[quant]"),
            ("openwakeword", "nixorb[wakeword]"),
            ("faster_whisper", "faster-whisper"),
        ],
    )
    def test_each_module_gets_the_command_that_installs_it(self, module, expected):
        assert expected in install_command(module)

    def test_anything_else_falls_back_to_the_named_extra(self):
        assert install_command("sentencepiece") == "pip install 'nixorb[hf]'"
        assert install_command("whatever", extra="nemotron") == (
            "pip install 'nixorb[nemotron]'"
        )

    def test_a_submodule_resolves_by_its_root(self):
        assert TORCH_INDEX in install_command("torch.nn.functional")


class TestIsInstalled:
    def test_true_for_something_really_here(self):
        assert is_installed("pytest")

    def test_false_for_something_absent(self):
        assert not is_installed("a_module_that_does_not_exist_anywhere")

    def test_a_submodule_is_judged_by_its_root(self):
        assert is_installed("pytest.something.deep")


class TestExplainImportError:
    def test_blames_the_module_that_was_actually_missing(self, monkeypatch):
        # The reported bug: `import torch` and `from transformers import ...`
        # on consecutive lines under one except, torch absent.
        absent(monkeypatch, "torch")
        message = explain_import_error(
            ModuleNotFoundError("No module named 'torch'", name="torch"),
            "transformers",
            "The HuggingFace LLM backend",
        )
        assert "needs torch" in message
        assert "transformers is not installed" not in message
        assert TORCH_INDEX in message

    def test_names_the_expected_module_when_that_is_the_missing_one(
        self, monkeypatch
    ):
        absent(monkeypatch, "transformers")
        message = explain_import_error(
            ModuleNotFoundError("No module named 'transformers'", name="transformers"),
            "transformers",
            "HuggingFace ASR",
        )
        assert "needs transformers, which is not installed" in message
        assert "nixorb[hf]" in message

    def test_never_says_install_something_already_installed(self):
        # pytest is definitely here; a failure importing it is not a
        # missing-package problem and must not be reported as one.
        message = explain_import_error(
            ImportError("cannot import name 'nope' from 'pytest'", name="pytest"),
            "pytest",
            "Something",
        )
        assert "though it is installed" in message
        assert "pip install" not in message

    def test_says_which_package_pulled_the_broken_one_in(self):
        message = explain_import_error(
            ImportError("cannot import name 'x'", name="pytest"),
            "transformers",
            "HuggingFace ASR",
        )
        assert "reached through transformers" in message

    def test_a_native_library_failure_keeps_its_own_advice(self):
        message = explain_import_error(
            ImportError(
                "libcudart.so.12: cannot open shared object file", name="torchaudio"
            ),
            "transformers",
            "HuggingFace ASR",
        )
        assert "installed but will not load" in message
        assert "came from different builds" in message
        # Not "install torchaudio" — it is already there.
        assert "which is not installed" not in message

    def test_falls_back_to_expected_when_the_error_names_nothing(self, monkeypatch):
        absent(monkeypatch, "transformers")
        message = explain_import_error(
            ImportError("something went wrong"), "transformers", "HuggingFace ASR"
        )
        assert "needs transformers" in message


class TestRequire:
    def test_a_missing_module_names_itself_and_its_extra(self):
        with pytest.raises(MissingDependency) as caught:
            require("a_module_that_does_not_exist_anywhere")
        message = str(caught.value)
        assert "a_module_that_does_not_exist_anywhere" in message
        assert "nixorb[hf]" in message

    def test_the_extra_is_configurable(self):
        with pytest.raises(MissingDependency, match=r"nixorb\[nemotron\]"):
            require("a_module_that_does_not_exist_anywhere", extra="nemotron")

    def test_the_feature_name_reaches_the_message(self):
        with pytest.raises(MissingDependency, match="Streaming ASR"):
            require(
                "a_module_that_does_not_exist_anywhere", feature="Streaming ASR"
            )

    def test_it_returns_the_module_when_it_imports(self):
        assert require("json").dumps({"a": 1}) == '{"a": 1}'


class TestEveryAdvisedExtraExists:
    """No message may advise an extra the package does not declare.

    `pip install 'nixorb[huggingface]'` answered with a warning and
    installed nothing, so the advice looked actionable and did nothing.
    """

    def test_no_source_file_advises_an_undeclared_extra(self):
        import re
        import tomllib
        from pathlib import Path

        # pyproject, not installed metadata: an editable install keeps
        # serving the extras it was built with, so metadata goes stale the
        # moment someone adds one and the guard passes on a lie.
        root = Path(__file__).resolve().parent.parent
        with (root / "pyproject.toml").open("rb") as handle:
            pyproject = tomllib.load(handle)

        declared = {
            name.replace("_", "-").lower()
            for name in pyproject["project"]["optional-dependencies"]
        }
        assert declared, "pyproject declares no extras at all"

        pattern = re.compile(r"nixorb\[([a-zA-Z0-9_.-]+)\]")
        advised: dict[str, str] = {}
        for path in (root / "nixorb").rglob("*.py"):
            for extra in pattern.findall(path.read_text()):
                advised.setdefault(extra.replace("_", "-").lower(), str(path))

        unknown = {name: where for name, where in advised.items() if name not in declared}
        assert not unknown, f"undeclared extras advised to users: {unknown}"
