#!/usr/bin/env bash
# Optional: enable GPU acceleration for embeddings.
#
#   bash setup-gpu.sh
#
# Idempotent and non-fatal: if there is no NVIDIA GPU, or the install fails, it
# says so and leaves the lab working on CPU. Nothing else in the lab changes.
set -euo pipefail

cd "$(dirname "$0")"
VENV=".venv"
PY="$VENV/bin/python"
ACTIVATE="$VENV/bin/activate"

# ── 0. is there a GPU at all? ──────────────────────────────────────────────
if ! command -v nvidia-smi >/dev/null 2>&1; then
  cat <<'EOF'
  No nvidia-smi on PATH -> no NVIDIA GPU detected.

  Nothing to do: the lab runs entirely on CPU. The only difference is speed
  (embedding 1000 docs takes ~36 s on CPU vs ~1 s on a modern GPU).

  To set EMBEDDING_DEVICE=auto in .env anyway (harmless without a GPU, and
  correct if you later attach one), run:

      echo 'EMBEDDING_DEVICE=auto' >> .env
EOF
  exit 0
fi

echo "  · GPU detected:"
nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader | sed 's/^/      /'

# ── 1. python version / venv ───────────────────────────────────────────────
if [ ! -x "$PY" ]; then
  echo "  ! No .venv found. Run 'bash setup-lite.sh' first." >&2
  exit 1
fi
PYVER=$("$PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "  · venv python: $PYVER"

# ── 2. install onnxruntime-gpu + CUDA runtime ──────────────────────────────
# Reinstalling onnxruntime-gpu over the CPU onnxruntime is required: they ship
# the same `onnxruntime` module, so pip replaces rather than adds.
echo "  · installing onnxruntime-gpu + CUDA runtime libraries…"
"$PY" -m pip install --upgrade -r requirements-gpu.txt 2>&1 | tail -5

# ── 3. verify — and verify it is really on the GPU ─────────────────────────
# The check that matters is that the onnxruntime *session* bound
# CUDAExecutionProvider. Checking get_available_providers() is not enough: the
# provider can be listed and still fail to dlopen, after which onnxruntime
# silently continues on CPU with correct results.
echo "  · verifying device…"
if ! "$PY" scripts/verify_device.py; then
  cat <<'EOF'

  GPU acceleration is installed but not usable. The lab still works on CPU.

  Most likely cause: the CUDA runtime libraries are present but onnxruntime
  cannot dlopen them (the "libcublasLt.so.13: cannot open shared object file"
  message above). app/runtime.py preloads them from site-packages/nvidia/*/lib;
  if that failed, check that:
    - nvidia-cublas-cu13 and nvidia-cudnn-cu13 are installed
    - the driver supports CUDA 13 (nvidia-smi -> "CUDA Version")
    - for an older driver, use the cu12 wheels instead (see requirements-gpu.txt)

  Force CPU for now with:  EMBEDDING_DEVICE=cpu make benchmark
EOF
  exit 0
fi

# ── 4. put the CUDA libs on the loader path ─────────────────────────────────
# `pip install onnxruntime-gpu` leaves libcublasLt.so.13 etc. in
# `site-packages/nvidia/*/lib`, which is NOT searched by ld.so. This has to be
# fixed at *process start*, not at runtime: glibc snapshots LD_LIBRARY_PATH once,
# when the process begins, so a library loaded later cannot add to the search
# path. Mutating os.environ from inside Python is therefore too late, and a
# re-exec inside a Jupyter kernel would destroy the kernel.
#
# So the path is exported by the venv's activate script, which is the one hook
# that runs before any Python in this project does. The lab's own source files
# are left untouched.
SITE_NVIDIA="$PWD/$VENV/lib/python$PYVER/site-packages/nvidia"
CUDA_LIBS=""
for sub in cu13/lib cudnn/lib cublas/lib cudnn_cu13/lib cusparselt/lib; do
  if [ -d "$SITE_NVIDIA/$sub" ]; then
    CUDA_LIBS="$CUDA_LIBS${CUDA_LIBS:+:}$SITE_NVIDIA/$sub"
  fi
done

if [ -n "$CUDA_LIBS" ]; then
  if ! grep -q 'setup-gpu.sh' "$ACTIVATE" 2>/dev/null; then
    cat >> "$ACTIVATE" <<EOF

# ── added by setup-gpu.sh: CUDA runtime libs for onnxruntime-gpu ──
# Without this, onnxruntime's provider load fails with
# "libcublasLt.so.13: cannot open shared object file" and it SILENTLY falls
# back to CPU while still returning correct results.
export LD_LIBRARY_PATH="$CUDA_LIBS\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
EOF
    echo "  · added CUDA libs to $ACTIVATE"
  else
    echo "  · $ACTIVATE already patched"
  fi
  echo "    (re-run 'source $ACTIVATE', or use 'make' targets which read it)"
else
  echo "  ! no nvidia CUDA runtime wheels found — GPU load will fail."
  echo "    pip install -r requirements-gpu.txt"
fi

# ── 5. record the device preference in .env ────────────────────────────────
if [ -f .env ] && ! grep -q '^EMBEDDING_DEVICE=' .env; then
  printf '\n# GPU path (added by setup-gpu.sh)\nEMBEDDING_DEVICE=auto\n' >> .env
  echo "  · appended EMBEDDING_DEVICE=auto to .env"
elif [ ! -f .env ]; then
  printf 'EMBEDDING_DEVICE=auto\n' > .env
  echo "  · created .env with EMBEDDING_DEVICE=auto"
else
  echo "  · .env already sets EMBEDDING_DEVICE — leaving it as is"
fi

# ── 6. measure, so the speedup is on the record ────────────────────────────
echo
echo "  · embedding throughput (512 short docs):"
LD_LIBRARY_PATH="$CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$PY" scripts/verify_device.py --benchmark
