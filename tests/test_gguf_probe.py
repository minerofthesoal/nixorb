"""Reading a GGUF's own header, so a load failure can say what is wrong.

From the field: NixOrb reported

    Could not load GGUF model 'empero-ai/Qwen3.8-2B-Distill-GGUF' /
    Qwen3.8-2B-Q4_K_M.gguf: Failed to load model from file: …

llama.cpp gives that one line whether the file is truncated, is not a GGUF
at all, or is a good GGUF whose architecture the bundled llama.cpp is too
old to build. That model's own card says which it was: "A recent llama.cpp
build with Qwen3.5 / Gated DeltaNet support is required — older builds will
fail to load the architecture." The header says so too, for free.

Where these tests build a GGUF they use llama.cpp's own `gguf` writer, not
hand-rolled bytes — a reader tested only against its own encoder proves
nothing about the real format.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nixorb.llm.gguf_probe import (
    MAGIC,
    GGUFInfo,
    NotGGUF,
    explain_load_failure,
    human_size,
    probe,
)


def write_gguf(path: Path, architecture: str = "qwen3next", extras: bool = True):
    """A real GGUF, written by the reference implementation."""
    gguf = pytest.importorskip("gguf")
    numpy = pytest.importorskip("numpy")

    writer = gguf.GGUFWriter(str(path), architecture)
    writer.add_block_count(4)
    writer.add_context_length(4096)
    if extras:
        # Every metadata shape the reader has to skip past on its way to
        # general.architecture: strings, string arrays, int arrays, bools.
        writer.add_string("general.name", "test model")
        writer.add_array("tokenizer.ggml.tokens", ["a", "b", "c"])
        writer.add_array("some.ints", [1, 2, 3, 4])
        writer.add_bool("general.some_flag", True)
    writer.add_tensor("blk.0.weight", numpy.zeros((8, 8), dtype=numpy.float32))
    writer.add_tensor("blk.1.weight", numpy.zeros((4, 4), dtype=numpy.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


class TestProbe:
    def test_reads_the_architecture_from_a_real_file(self, tmp_path):
        info = probe(write_gguf(tmp_path / "m.gguf", "qwen3next"))
        assert info.architecture == "qwen3next"
        assert info.tensor_count == 2
        assert info.version >= 3
        assert info.size > 0

    def test_walks_past_every_metadata_shape(self, tmp_path):
        # Arrays and bools sit between the header and, in other files, the
        # architecture; the reader must skip them without losing its place.
        from nixorb.llm.gguf_probe import _Reader

        path = write_gguf(tmp_path / "m.gguf", "llama")
        with path.open("rb") as handle:
            assert handle.read(4) == MAGIC
            reader = _Reader(handle)
            reader.u32()
            reader.u64()
            kv_count = reader.u64()
            for _ in range(kv_count):
                reader.string()
                reader.skip_value(reader.u32())

    def test_a_file_that_is_not_a_gguf_says_so(self, tmp_path):
        # What a failed download actually leaves behind.
        path = tmp_path / "m.gguf"
        path.write_bytes(b"<!DOCTYPE html><html><body>404</body></html>")
        with pytest.raises(NotGGUF, match="not a GGUF"):
            probe(path)

    def test_an_lfs_pointer_is_caught(self, tmp_path):
        path = tmp_path / "m.gguf"
        path.write_text(
            "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1312164224\n"
        )
        with pytest.raises(NotGGUF):
            probe(path)

    def test_a_header_cut_off_mid_value_is_caught(self, tmp_path):
        path = tmp_path / "m.gguf"
        path.write_bytes(MAGIC + b"\x03\x00\x00\x00" + b"\x02\x00")
        with pytest.raises(NotGGUF, match="ends mid-value"):
            probe(path)

    def test_a_truncated_file_still_reports_what_it_could_read(self, tmp_path):
        # Header intact, metadata cut off: version and tensor count are
        # real and worth reporting even with no architecture.
        full = write_gguf(tmp_path / "full.gguf").read_bytes()
        cut = tmp_path / "cut.gguf"
        cut.write_bytes(full[:28])
        info = probe(cut)
        assert info.version >= 3
        assert info.architecture == ""

    def test_an_implausible_string_length_does_not_allocate(self, tmp_path):
        # A corrupt length field must not become a 16-exabyte read.
        import struct

        path = tmp_path / "m.gguf"
        path.write_bytes(
            MAGIC
            + struct.pack("<I", 3)
            + struct.pack("<Q", 1)
            + struct.pack("<Q", 1)
            + struct.pack("<Q", 2**60)
        )
        info = probe(path)
        assert info.architecture == ""


class TestHumanSize:
    @pytest.mark.parametrize(
        "count,expected",
        [(800, "800 B"), (45_000, "45.0 KB"), (1_312_164_224, "1.31 GB")],
    )
    def test_reads_at_a_glance(self, count, expected):
        assert human_size(count) == expected


class TestExplainLoadFailure:
    def test_a_good_file_points_at_llama_cpp_not_the_download(self, tmp_path):
        path = write_gguf(tmp_path / "m.gguf", "qwen3next")
        message = explain_load_failure(
            path, ValueError(f"Failed to load model from file: {path}")
        )
        assert "qwen3next" in message
        assert "The file itself is fine" in message
        assert "llama-cpp-python" in message
        # It must not send them to re-download a file that is intact.
        assert "download again" not in message

    def test_a_bad_file_points_at_the_download(self, tmp_path):
        path = tmp_path / "m.gguf"
        path.write_bytes(b"<!DOCTYPE html>")
        message = explain_load_failure(path, ValueError("Failed to load"))
        assert "not a usable GGUF" in message
        assert "llama-cpp-python" not in message

    def test_a_missing_file_says_so(self, tmp_path):
        message = explain_load_failure(
            tmp_path / "nope.gguf", ValueError("Failed to load")
        )
        assert "Could not read" in message

    def test_the_original_error_is_always_kept(self, tmp_path):
        path = write_gguf(tmp_path / "m.gguf")
        message = explain_load_failure(path, ValueError("Failed to load model"))
        assert "Failed to load model" in message


class TestBackendWiring:
    def test_the_backend_enriches_a_llama_cpp_failure(self, tmp_path):
        from nixorb.llm.hf_llm_backend import HuggingFaceLLMBackend

        path = write_gguf(tmp_path / "m.gguf", "qwen3next")
        message = HuggingFaceLLMBackend._describe_gguf_failure(
            ValueError(f"Failed to load model from file: {path}")
        )
        assert "qwen3next" in message

    def test_an_unrelated_error_passes_through_untouched(self):
        from nixorb.llm.hf_llm_backend import HuggingFaceLLMBackend

        message = HuggingFaceLLMBackend._describe_gguf_failure(
            ValueError("some other llama.cpp problem")
        )
        assert message == "some other llama.cpp problem"

    def test_it_survives_a_path_it_cannot_stat(self):
        from nixorb.llm.hf_llm_backend import HuggingFaceLLMBackend

        message = HuggingFaceLLMBackend._describe_gguf_failure(
            ValueError("Failed to load model from file: /nope/missing.gguf")
        )
        assert "Could not read" in message


def test_gguf_info_is_immutable():
    import dataclasses

    info = GGUFInfo(path="x", size=1, version=3, tensor_count=2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.architecture = "changed"  # type: ignore[misc]
