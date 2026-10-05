"""Guards the GPU/CPU device selection in app/runtime.py.

The failure this exists to prevent is specific and quiet: `fastembed` requests
`CUDAExecutionProvider`, onnxruntime fails to dlopen it (missing
`libcublasLt.so.13`, because `pip install onnxruntime-gpu` does not install the
CUDA runtime), logs one ERROR line, and falls back to CPU. Everything still
returns correct vectors, so the only symptom is that no speedup ever appears.

`probe_cuda` therefore has to verify the *session*, not the request. These
tests cover the decision logic with the probe stubbed, so they run on a
CPU-only machine too.
"""
from __future__ import annotations

import os

import pytest

from app import runtime


@pytest.fixture(autouse=True)
def _clean_device_env(monkeypatch):
    monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)


# ── EMBEDDING_DEVICE parsing ─────────────────────────────────────────────

def test_default_is_auto(monkeypatch):
    assert runtime.requested_device() == "auto"


def test_explicit_values_are_case_and_space_insensitive(monkeypatch):
    for raw, want in (("cuda", "cuda"), ("cpu", "cpu"),
                      ("CUDA", "cuda"), (" cpu ", "cpu"), ("Auto", "auto")):
        monkeypatch.setenv("EMBEDDING_DEVICE", raw)
        assert runtime.requested_device() == want, raw


def test_unknown_value_fails_loudly(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DEVICE", "gpu-maybe")
    with pytest.raises(ValueError, match="Unknown EMBEDDING_DEVICE"):
        runtime.requested_device()


# ── resolution logic, probe stubbed ───────────────────────────────────────

@pytest.fixture
def probe(monkeypatch):
    """Replace the CUDA probe so these tests are hardware-independent."""
    def _set(ok: bool, detail: str = "stub"):
        monkeypatch.setattr(runtime, "probe_cuda", lambda model: (ok, detail))
    return _set


def test_cpu_request_never_probes(probe, monkeypatch):
    """Forcing CPU must not load a model just to find out CUDA exists."""
    monkeypatch.setenv("EMBEDDING_DEVICE", "cpu")
    calls = []
    monkeypatch.setattr(runtime, "probe_cuda", lambda m: calls.append(m) or (True, "x"))
    use_cuda, why = runtime.resolve_device("BAAI/bge-small-en-v1.5", "fastembed")
    assert use_cuda is False
    assert calls == [], "cpu must not pay for a CUDA probe"
    assert "cpu requested" in why


def test_auto_uses_cuda_when_probe_succeeds(probe):
    probe(True, "CUDA ok")
    use_cuda, why = runtime.resolve_device("BAAI/bge-small-en-v1.5", "fastembed")
    assert use_cuda is True and "CUDA ok" in why


def test_auto_falls_back_when_probe_fails(probe):
    probe(False, "libcublas.so.13 missing")
    use_cuda, why = runtime.resolve_device("BAAI/bge-small-en-v1.5", "fastembed")
    assert use_cuda is False
    assert "libcublas" in why, "the reason must name the actual cause"


def test_cuda_raises_instead_of_silently_downgrading(probe, monkeypatch):
    """`cuda` means cuda. A silent fallback would make the setting a lie."""
    monkeypatch.setenv("EMBEDDING_DEVICE", "cuda")
    probe(False, "no GPU")
    with pytest.raises(RuntimeError, match="unusable"):
        runtime.resolve_device("BAAI/bge-small-en-v1.5", "fastembed")


@pytest.mark.parametrize("backend", ["sentence-transformers", "openai"])
def test_non_fastembed_backends_are_told_the_truth(monkeypatch, backend):
    """Only fastembed routes through onnxruntime; asking about CUDA is wrong."""
    monkeypatch.setattr(runtime, "probe_cuda",
                        lambda m: pytest.fail("must not probe a non-onnx backend"))
    use_cuda, why = runtime.resolve_device("some/model", backend)
    assert use_cuda is False and backend in why


def test_cuda_on_unsupported_backend_raises(monkeypatch):
    monkeypatch.setenv("EMBEDDING_DEVICE", "cuda")
    with pytest.raises(ValueError, match="only supported for the fastembed"):
        runtime.resolve_device("text-embedding-3-small", "openai")


# ── reporting ─────────────────────────────────────────────────────────────

def test_resolve_device_returns_a_usable_pair():
    """Whatever the hardware, the answer is a (bool, non-empty reason) pair.

    The reason is not decoration: it is the only way a user learns *why* they
    are on CPU, and the screenshot/Makefile output prints it.
    """
    import os

    os.environ.pop("EMBEDDING_DEVICE", None)
    use_cuda, reason = runtime.resolve_device("BAAI/bge-small-en-v1.5", "fastembed")
    assert isinstance(use_cuda, bool)
    assert reason and reason.strip()


def test_describe_is_a_single_readable_line(monkeypatch):
    monkeypatch.setattr(runtime, "probe_cuda", lambda m: (True, "stubbed"))
    line = runtime.describe()
    assert "\n" not in line
    assert "device=" in line and "cuda" in line


# ── venv CUDA library discovery ───────────────────────────────────────────

def test_nvidia_lib_dirs_are_absolute_existing():
    for d in runtime.nvidia_lib_dirs():
        assert os.path.isabs(d) and os.path.isdir(d), d


def test_activate_cuda_libs_is_idempotent(monkeypatch):
    monkeypatch.setattr(runtime, "nvidia_lib_dirs", lambda: [])
    assert runtime.activate_cuda_libs() == []


def test_preload_finds_the_libraries_ort_needs():
    """Skip when the CUDA wheels are absent -- this asserts presence, not use."""
    dirs = runtime.nvidia_lib_dirs()
    if not dirs:
        pytest.skip("nvidia CUDA wheels not installed in this venv")
    found = set()
    for d in dirs:
        for so in os.listdir(d):
            found.add(so)
    # libcublasLt is the exact symbol onnxruntime's error message names, and
    # libcudart is its first unsatisfied dependency.
    assert any(f.startswith("libcublasLt.so") for f in found), sorted(found)[:20]
    assert any(f.startswith("libcudart.so") for f in found), sorted(found)[:20]
