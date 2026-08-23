#!/usr/bin/env bash
# ROCm JAX via Docker Desktop on Windows/WSL (RX 9070 / gfx1201).
# Official AMD image expects /dev/kfd; WSL exposes /dev/dxg. Overlay the host
# HSA runtime (linked to libdxcore) so the plugin can see the GPU.
set -euo pipefail
IMAGE="${JAX_IMAGE:-rocm/jax:rocm7.14-jax0.10.0-py3.12}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CORE=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib
HOST_ROCM=/opt/rocm-7.2.0/lib
docker rm -f learning-agent-jax >/dev/null 2>&1 || true
exec docker run --rm --name learning-agent-jax \
  --device=/dev/dxg \
  --group-add video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /usr/lib/wsl:/usr/lib/wsl:ro \
  -v "${HOST_ROCM}/libhsa-runtime64.so.1:${CORE}/libhsa-runtime64.so.1:ro" \
  -v "${HOST_ROCM}/libhsa-amd-aqlprofile64.so.1:${CORE}/libhsa-amd-aqlprofile64.so.1:ro" \
  -v "${HOST_ROCM}/libhsa-amd-aqlprofile64.so:${CORE}/libhsa-amd-aqlprofile64.so:ro" \
  -v learning-agent-jax-extra:/opt/extra \
  -v "${ROOT}":/workspace/learning-agent \
  -w /workspace/learning-agent \
  -e HIP_VISIBLE_DEVICES=0 \
  -e MUJOCO_GL=osmesa \
  -e PYTHONPATH=/opt/extra \
  -e LD_LIBRARY_PATH="/usr/lib/wsl/lib:${CORE}" \
  "$IMAGE" \
  bash /workspace/learning-agent/scripts/jax_docker_entry.sh \
  "$@"
