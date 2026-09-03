"""The dependency-conflict reporting behind `nixorb check`.

The case that matters is real and was hit in the wild: NixOrb needs
transformers>=5.13 for its default Nemotron checkpoint, LLaMA-Factory pins
transformers<=4.57.1, and installing them into one interpreter leaves pip
printing a single warning that names neither project's symptom.
"""
from __future__ import annotations

import sys

import pytest

from nixorb.envcheck import (
    Conflict,
    Unsatisfied,
    canonical,
    find_conflicts,
    find_unsatisfied,
    isolation,
    parse_requirement,
    problems,
    read_environment,
    report,
    scan,
    torch_report,
)


class TestCanonical:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("huggingface_hub", "huggingface-hub"),
            ("Huggingface.Hub", "huggingface-hub"),
            ("llama-cpp-python", "llama-cpp-python"),
            ("  PySide6  ", "pyside6"),
            ("a__b", "a-b"),
        ],
    )
    def test_normalises_like_pep_503(self, raw, expected):
        assert canonical(raw) == expected


class TestParseRequirement:
    def test_returns_name_and_specifier(self):
        assert parse_requirement("transformers>=5.13.0") == (
            "transformers",
            ">=5.13.0",
        )

    def test_normalises_the_name(self):
        name, _ = parse_requirement("huggingface_hub>=0.24.0")
        assert name == "huggingface-hub"

    def test_bare_name_has_an_empty_specifier(self):
        assert parse_requirement("chromadb") == ("chromadb", "")

    def test_drops_extras_nobody_asked_for(self):
        # `pip install nixorb` does not install [dev], so pytest's absence
        # is not a broken environment.
        assert parse_requirement('pytest>=8.2; extra == "dev"') is None

    def test_honours_environment_markers(self):
        assert parse_requirement('tomli; python_version < "3.0"') is None
        assert parse_requirement('tomli; python_version >= "3.0"') == ("tomli", "")

    def test_survives_garbage(self):
        assert parse_requirement("!!! not a requirement") is None
        assert parse_requirement("") is None


class TestFindConflicts:
    def test_reports_the_llamafactory_case(self):
        conflicts = find_conflicts(
            ours=["transformers"],
            installed={"transformers": "5.16.1", "llamafactory": "0.9.5.dev0"},
            requirements={
                "llamafactory": ["transformers<=4.57.1,>=4.51.0"],
            },
        )
        assert len(conflicts) == 1
        found = conflicts[0]
        assert found.package == "transformers"
        assert found.installed == "5.16.1"
        assert found.holder == "llamafactory"
        assert found.holder_version == "0.9.5.dev0"
        assert "llamafactory 0.9.5.dev0" in found.describe()
        assert "5.16.1" in found.describe()

    def test_silent_when_the_pin_is_satisfied(self):
        assert find_conflicts(
            ours=["transformers"],
            installed={"transformers": "5.16.1", "other": "1.0"},
            requirements={"other": ["transformers>=5.0"]},
        ) == []

    def test_ignores_packages_nixorb_does_not_use(self):
        # Two other projects disagreeing about numpy is not our business.
        assert find_conflicts(
            ours=["transformers"],
            installed={"numpy": "2.1.0", "other": "1.0"},
            requirements={"other": ["numpy<2"]},
        ) == []

    def test_ignores_requirements_on_uninstalled_packages(self):
        assert find_conflicts(
            ours=["transformers"],
            installed={"other": "1.0"},
            requirements={"other": ["transformers<=4.57.1"]},
        ) == []

    def test_does_not_report_nixorb_against_itself(self):
        assert find_conflicts(
            ours=["transformers"],
            installed={"transformers": "5.16.1", "nixorb": "0.3.3"},
            requirements={"nixorb": ["transformers>=99"]},
        ) == []

    def test_matches_across_name_spellings(self):
        conflicts = find_conflicts(
            ours=["huggingface-hub"],
            installed={"huggingface-hub": "1.29.0", "other": "1.0"},
            requirements={"other": ["huggingface_hub<1.0"]},
        )
        assert len(conflicts) == 1

    def test_a_prerelease_still_counts_as_installed(self):
        conflicts = find_conflicts(
            ours=["transformers"],
            installed={"transformers": "5.16.1rc1", "other": "1.0"},
            requirements={"other": ["transformers<5"]},
        )
        assert len(conflicts) == 1

    def test_unparseable_requirements_are_skipped_not_fatal(self):
        assert find_conflicts(
            ours=["transformers"],
            installed={"transformers": "5.16.1", "other": "1.0"},
            requirements={"other": ["!!!", "transformers>=5.0"]},
        ) == []


class TestFindUnsatisfied:
    def test_reports_a_downgraded_dependency(self):
        # The mirror image: installing a transformers<=4.57 project *after*
        # NixOrb downgrades it and breaks NixOrb instead.
        missing = find_unsatisfied(
            ["transformers>=5.13.0"], {"transformers": "4.57.1"}
        )
        assert len(missing) == 1
        assert missing[0] == Unsatisfied("transformers", ">=5.13.0", "4.57.1")
        assert "4.57.1 is installed" in missing[0].describe()

    def test_reports_a_missing_dependency(self):
        missing = find_unsatisfied(["chromadb>=0.5.3"], {})
        assert missing[0].installed is None
        assert "missing" in missing[0].describe()

    def test_silent_when_everything_fits(self):
        assert find_unsatisfied(
            ["transformers>=5.13.0", "numpy>=1.26.0"],
            {"transformers": "5.16.1", "numpy": "2.1.0"},
        ) == []

    def test_skips_extras(self):
        assert find_unsatisfied(['ruff>=0.4; extra == "dev"'], {}) == []


class TestIsolation:
    def test_detects_a_virtualenv(self, monkeypatch):
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        monkeypatch.setattr(sys, "prefix", "/home/u/.local/share/nixorb/venv")
        isolated, where = isolation()
        assert isolated
        assert "venv" in where

    def test_flags_a_shared_interpreter(self, monkeypatch):
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        monkeypatch.setattr(sys, "prefix", "/usr")
        isolated, where = isolation()
        assert not isolated
        assert "shared" in where


class TestLiveEnvironment:
    def test_reads_this_interpreter(self):
        installed, requirements = read_environment()
        assert installed, "no distributions found at all"
        assert "pytest" in installed
        assert all(key == canonical(key) for key in installed)

    def test_report_runs_and_names_the_interpreter(self):
        lines = report()
        assert any(sys.executable in line for line in lines)

    def test_report_warns_when_the_interpreter_is_shared(self, monkeypatch):
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        monkeypatch.setattr(sys, "prefix", "/usr")
        text = "\n".join(report())
        assert "transformers>=5.13" in text
        assert "venv" in text

    def test_scan_returns_both_directions(self):
        unsatisfied, conflicts = scan()
        assert isinstance(unsatisfied, list)
        assert isinstance(conflicts, list)

    def test_problems_is_quiet_when_nothing_is_wrong(self, monkeypatch):
        # `nixorb status` prints these unconditionally, so a healthy
        # environment has to produce no lines at all.
        monkeypatch.setattr("nixorb.envcheck.scan", lambda: ([], []))
        assert problems() == []

    def test_problems_names_the_collision_and_the_way_out(self, monkeypatch):
        monkeypatch.setattr(sys, "base_prefix", "/usr")
        monkeypatch.setattr(sys, "prefix", "/usr")
        monkeypatch.setattr(
            "nixorb.envcheck.scan",
            lambda: (
                [],
                [
                    Conflict(
                        package="transformers",
                        installed="5.16.1",
                        holder="llamafactory",
                        holder_version="0.9.5.dev0",
                        specifier="<=4.57.1,>=4.51.0",
                    )
                ],
            ),
        )
        text = "\n".join(problems())
        assert "llamafactory" in text
        assert "do not overlap" in text
        assert "python -m venv" in text


class TestTorchReport:
    """torch is not a declared dependency, so absence is a note, not a fault.

    A torch stack whose halves came from different indexes is the fault:
    transformers>=5 imports torchaudio at module scope, so it takes down
    model loading with a linker error naming only a `.so`.
    """

    def test_missing_torch_is_a_note_not_a_failure(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name in ("torch", "torchaudio"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        lines = torch_report()
        assert len(lines) == 2
        assert all("not installed" in line for line in lines)
        assert not any("❌" in line for line in lines)

    def test_a_broken_torchaudio_is_reported_with_the_fix(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        class _Torch:
            __version__ = "2.13.0+cu126"

            class version:
                cuda = "12.6"

        def half_broken(name, *args, **kwargs):
            if name == "torch":
                return _Torch
            if name == "torchaudio":
                raise ImportError(
                    "libcudart.so.12: cannot open shared object file: "
                    "No such file or directory"
                )
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", half_broken)
        text = "\n".join(torch_report())
        assert "torch 2.13.0+cu126 (CUDA 12.6)" in text
        assert "torchaudio is installed but will not import" in text
        assert "libcudart.so.12" in text
        assert "download.pytorch.org" in text

    def test_a_healthy_cpu_stack_reports_cpu(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        class _Mod:
            __version__ = "2.13.0+cpu"

            class version:
                cuda = None

        monkeypatch.setattr(
            builtins,
            "__import__",
            lambda name, *a, **k: _Mod if name in ("torch", "torchaudio")
            else real_import(name, *a, **k),
        )
        text = "\n".join(torch_report())
        assert text.count("(CPU)") == 2
        assert "❌" not in text


class TestTorchReportLeftovers:
    """The second field report: an OSError, not an ImportError.

    Classifying on the exception type printed the raw loader message with
    no advice at all, which is what `nixorb check` did the first time
    someone hit it.
    """

    def _broken(self, monkeypatch, exc):
        import builtins

        real_import = builtins.__import__

        class _Torch:
            __version__ = "2.14.0+cpu"

            class version:
                cuda = None

        def half_broken(name, *args, **kwargs):
            if name == "torch":
                return _Torch
            if name == "torchaudio":
                raise exc
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", half_broken)

    def test_an_oserror_still_gets_advice(self, monkeypatch):
        self._broken(
            monkeypatch,
            OSError("Could not load this library: /x/torchaudio/lib/_torchaudio.so"),
        )
        monkeypatch.setattr("nixorb.hf.duplicate_extensions", lambda name="": [])
        text = "\n".join(torch_report())
        assert "torch 2.14.0+cpu (CPU)" in text
        assert "rm -rf" in text
        assert "optional" in text

    def test_leftover_files_are_listed_with_the_directory_to_delete(
        self, monkeypatch, tmp_path
    ):
        lib = tmp_path / "torchaudio" / "lib"
        lib.mkdir(parents=True)
        (lib / "_torchaudio.so").touch()
        (lib / "_torchaudio.abi3.so").touch()

        import importlib.util

        class _Spec:
            submodule_search_locations = [str(tmp_path / "torchaudio")]

        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name: _Spec() if name == "torchaudio" else None,
        )
        # The loader names the offending file in its own message, which is
        # where the directory to delete is derived from.
        self._broken(
            monkeypatch,
            OSError(
                "Could not load this library: "
                f"{tmp_path / 'torchaudio' / 'lib' / '_torchaudio.so'}"
            ),
        )
        text = "\n".join(torch_report())
        assert "2 extension files where there should be one" in text
        assert "_torchaudio.abi3.so" in text
        # The command must name the real directory, not a placeholder.
        assert f"rm -rf {tmp_path / 'torchaudio'}" in text
        assert "<site-packages>" not in text

    def test_a_plain_missing_module_stays_a_note(self, monkeypatch):
        self._broken(monkeypatch, ImportError("No module named 'torchaudio'"))
        monkeypatch.setattr("nixorb.hf.duplicate_extensions", lambda name="": [])
        text = "\n".join(torch_report())
        assert "torchaudio is not installed" in text
        assert "❌" not in text.split("torchaudio")[-1]
