"""Report and verify the compute device used for embeddings.

Run via `make device`, or directly:

    python scripts/verify_device.py              # report the resolved device
    python scripts/verify_device.py --benchmark  # + throughput on CPU and GPU

Exit code 0 whenever a device is usable (CPU counts), so this doubles as the
GPU-post-install check in setup-gpu.sh. It exits 1 only when
EMBEDDING_DEVICE=cuda was requested and is not available -- an explicit request
that could not be honoured is an error, whereas `auto` falling back is not.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from app.config import load_dotenv  # noqa: E402

load_dotenv()

from app.embeddings import BACKENDS, DEFAULT_BACKEND, Embedder  # noqa: E402
from app.runtime import (  # noqa: E402
    _ort_is_cpu_only,
    activate_cuda_libs,
    nvidia_lib_dirs,
    probe_cuda,
    quiet_onnxruntime,
    resolve_device,
)

# Onnxruntime logs to the C++ logger, not through Python's `logging`, so a
# `logging.disable()` here does nothing. These two warnings are informational
# (ORT deliberately puts shape-inference ops on the CPU) and they otherwise
# land in the middle of every notebook's evidence output.
quiet_onnxruntime()

N_BENCH = 512
BENCH_TEXT = "tài liệu cloud computing và tự động mở rộng hạ tầng tiếng Việt"
# Matches Embedder's default backend, read from the lab's own registry so the
# benchmark cannot drift from the configured model.
DEFAULT_MODEL = BACKENDS[DEFAULT_BACKEND].model


def report() -> tuple[bool, str]:
    """Print the resolved stack + device. Returns (cuda_in_use, reason)."""
    try:
        e = Embedder()
    except ValueError as exc:
        print(f"  ✗ config error: {exc}")
        return False, str(exc)

    # resolve_device is called directly rather than read off the Embedder:
    # app/embeddings.py belongs to the lab and is not modified here, so the
    # device decision lives entirely in app/runtime.py.
    use_cuda, reason = resolve_device(e.model_name, e.spec.provider)

    print(f"  backend    : {e.backend}")
    print(f"  model      : {e.model_name}")
    print(f"  dimension  : {e.dim}")
    print(f"  device     : {'cuda' if use_cuda else 'cpu'}")
    print(f"  reason     : {reason}")
    return use_cuda, reason


def benchmark() -> None:
    """Throughput for the current backend, measured per device.

    Drives fastembed directly rather than going through `app.embeddings.Embedder`.

    That indirection is deliberate: `Embedder._load` passes no `cuda=` argument,
    so it always takes fastembed's `cuda=auto` default and ignores
    EMBEDDING_DEVICE. Benchmarking through it would report the GPU twice — once
    mislabelled "cpu" — which is exactly the kind of measurement that makes a
    speedup look real when it is not. app/embeddings.py is the lab's own file and
    is left unmodified, so this script constructs the model itself.
    """
    import subprocess

    prev_rate: int | None = None
    for dev in ("cpu", "cuda"):
        if dev == "cuda" and _ort_is_cpu_only():
            print("  cuda       : skipped (onnxruntime is a CPU-only build — "
                  "run `bash setup-gpu.sh`)")
            continue
        # N_BENCH is interpolated because the child process has no access to
        # this module's globals -- a NameError here would otherwise surface as
        # a bare "unavailable".
        code = (
            "import sys, time, numpy as np;"
            f"sys.path.insert(0, {str(ROOT)!r});"
            "from fastembed import TextEmbedding;"
            f"want_cuda = {dev == 'cuda'};"
            f"m = TextEmbedding(model_name={DEFAULT_MODEL!r}, cuda=want_cuda);"
            f"texts = [{BENCH_TEXT!r}] * {N_BENCH};"
            "list(m.embed(texts[:8]));"
            "t0 = time.perf_counter();"
            "np.asarray(list(m.embed(texts)));"
            "dt = time.perf_counter() - t0;"
            f"print('BENCH', {dev!r}, round(dt, 3), round({N_BENCH}/dt))"
        )
        # The venv's activate script (patched by setup-gpu.sh) is what puts the
        # CUDA libs on the loader path, so re-read it here rather than assuming
        # the caller's shell is already activated.
        env = {**os.environ, "EMBEDDING_DEVICE": dev}
        ld = env.get("LD_LIBRARY_PATH", "")
        for d in nvidia_lib_dirs():
            if d not in ld.split(os.pathsep):
                ld = f"{d}{os.pathsep}{ld}" if ld else d
        env["LD_LIBRARY_PATH"] = ld
        res = subprocess.run([sys.executable, "-c", code], env=env,
                             capture_output=True, text=True)
        line = next(
            (l for l in res.stdout.splitlines() if l.startswith("BENCH ")), None
        )
        if line:
            _, actual, secs, rate = line.split()
            print(f"  {actual:11}: {N_BENCH} docs in {float(secs):.2f}s "
                  f"-> {int(rate):>6} docs/s")
        else:
            err = (res.stderr.strip().splitlines() or ["?"])[-1]
            print(f"  {dev:11}: unavailable ({err[:70]})")
            continue
        # Sanity check, not decoration. An earlier version of this script routed
        # the benchmark through app.embeddings.Embedder, which does not forward a
        # cuda= argument -- so both rows measured the GPU and the "cpu" number
        # came out implausibly fast. When the two rows are within 10% of each
        # other, something is being measured twice and the figures are void.
        if prev_rate:
            ratio = int(rate) / prev_rate if prev_rate else 0
            if 0.9 < ratio < 1.1:
                print("    ! cpu and cuda are within 10% -- both rows are almost")
                print("      certainly the same device; treat these numbers as void.")
        prev_rate = int(rate)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark", action="store_true",
                    help="also measure embedding throughput on each device")
    args = ap.parse_args()

    print("Day 19 — compute device")
    cuda, _ = report()

    if args.benchmark:
        print("\n  throughput:")
        benchmark()
        return 0

    if cuda:
        print("\n  ✓ GPU in use (session bound CUDAExecutionProvider)")
    else:
        libs = activate_cuda_libs()
        e = Embedder()
        ok, detail = probe_cuda(e.model_name)
        print("\n  CPU in use.")
        if not ok:
            print(f"    CUDA unavailable: {detail}")
            print(f"    CUDA runtime libs found in venv: {len(libs)}")
            print("    To enable: bash setup-gpu.sh")
        else:
            print("    CUDA is usable but not selected (EMBEDDING_DEVICE).")

    if (os.getenv("EMBEDDING_DEVICE") or "auto").strip().lower() == "cuda" and not cuda:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
