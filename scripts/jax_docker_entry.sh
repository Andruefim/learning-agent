#!/usr/bin/env bash
# CPU torch + mujoco in a named volume; JAX/ROCm stay in the image.
# Default Linux torch wheels pull CUDA — always use the CPU index.
# Extra numpy/jax/scipy must not shadow the image (ROCm JAX segfaults).
# pip mujoco has no mjx; overlay mujoco-mjx onto extra/mujoco (pip --target
# will not merge the two wheels).
set -euo pipefail
export PYTHONPATH="/opt/extra${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"

_wipe_extra_numpy() {
  rm -rf /opt/extra/numpy /opt/extra/numpy.libs \
    /opt/extra/numpy-*.dist-info 2>/dev/null || true
}

_wipe_extra_scipy() {
  rm -rf /opt/extra/scipy /opt/extra/scipy.libs \
    /opt/extra/scipy-*.dist-info 2>/dev/null || true
}

_wipe_extra_jax() {
  rm -rf /opt/extra/jax /opt/extra/jaxlib /opt/extra/jax_plugins \
    /opt/extra/jax-*.dist-info /opt/extra/jaxlib-*.dist-info 2>/dev/null || true
}

_wipe_extra_mujoco() {
  rm -rf /opt/extra/mujoco /opt/extra/mujoco.libs \
    /opt/extra/mujoco-*.dist-info /opt/extra/mujoco*.dist-info 2>/dev/null || true
}

_mjx_ok() {
  python -c "import jax; from mujoco import mjx" >/dev/null 2>&1
}

_mjx_why() {
  python -c "import jax; from mujoco import mjx" 2>&1 || true
}

if ! python -c "import torch" >/dev/null 2>&1; then
  echo "Installing CPU torch into /opt/extra (first run only)..." >&2
  python -m pip install --target /opt/extra \
    --index-url https://download.pytorch.org/whl/cpu \
    torch
fi

_wipe_extra_jax
_wipe_extra_numpy
_wipe_extra_scipy

_overlay_mjx() {
  # pip --target will not merge mujoco-mjx into an existing extra/mujoco.
  local tmp
  tmp="$(mktemp -d)"
  python -m pip install --target "$tmp" --no-deps "mujoco-mjx>=3.2.0" trimesh
  if [[ ! -d "$tmp/mujoco/mjx" ]]; then
    echo "FATAL: mujoco-mjx wheel has no mujoco/mjx" >&2
    ls -la "$tmp" >&2 || true
    rm -rf "$tmp"
    exit 1
  fi
  rm -rf /opt/extra/mujoco/mjx
  cp -a "$tmp/mujoco/mjx" /opt/extra/mujoco/mjx
  mkdir -p /opt/extra/trimesh
  if [[ -d "$tmp/trimesh" ]]; then
    rm -rf /opt/extra/trimesh
    cp -a "$tmp/trimesh" /opt/extra/trimesh
  fi
  rm -rf "$tmp"
}

_install_mjx() {
  echo "Installing mujoco + overlay mujoco-mjx into /opt/extra..." >&2
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -qq
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      libosmesa6 libgl1 libegl1 libgles2 >/dev/null
  fi
  _wipe_extra_mujoco
  python -m pip install --target /opt/extra --no-deps \
    "mujoco>=3.2.0" absl-py etils glfw pyopengl
  _overlay_mjx
  _wipe_extra_jax
  _wipe_extra_numpy
  _wipe_extra_scipy
}

# Skip a full wipe if the overlay is already on the volume (pip --target
# otherwise reinstalls mujoco every container start and drops mjx).
if [[ ! -d /opt/extra/mujoco/mjx ]] || ! python -c "from mujoco import mjx" >/dev/null 2>&1; then
  _install_mjx
fi

if ! _mjx_ok; then
  echo "mjx import failed after install:" >&2
  _mjx_why >&2
  echo "retrying overlay only..." >&2
  _overlay_mjx
fi

python - <<'PY'
import os, traceback
print("jax-docker probe PYTHONPATH=", os.environ.get("PYTHONPATH"), flush=True)
try:
    import numpy as np
    print("numpy", np.__version__, np.__file__, flush=True)
except Exception:
    traceback.print_exc()
try:
    import jax
    print("jax", jax.__version__, jax.__file__, flush=True)
    print("jax.devices", jax.devices(), flush=True)
except Exception:
    traceback.print_exc()
try:
    import mujoco
    print("mujoco", getattr(mujoco, "__version__", "?"), mujoco.__file__, flush=True)
except Exception:
    traceback.print_exc()
try:
    from mujoco import mjx  # noqa: F401
    print("mjx ok", mjx.__file__, flush=True)
except Exception:
    traceback.print_exc()
try:
    import torch
    print("torch", torch.__version__, flush=True)
except Exception:
    traceback.print_exc()
PY

if ! _mjx_ok; then
  echo "FATAL: jax + mujoco.mjx must import for GPU PPO" >&2
  _mjx_why >&2
  exit 1
fi

exec "$@"
