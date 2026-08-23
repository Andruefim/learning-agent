# Official ROCm JAX image via Docker Desktop (WSL2 backend, RX 9070).
# Usage:
#   .\scripts\run_jax_docker.ps1
#   .\scripts\run_jax_docker.ps1 python train_l3_foundation.py --iters 200 --envs 512
$ErrorActionPreference = "Stop"
function ConvertTo-WslPath([string]$p) {
    $full = [System.IO.Path]::GetFullPath($p)
    $drive = $full.Substring(0, 1).ToLowerInvariant()
    $rest = $full.Substring(2).Replace('\', '/')
    return "/mnt/$drive$rest"
}
$sh = ConvertTo-WslPath (Join-Path $PSScriptRoot "run_jax_docker.sh")
if ($args.Count -eq 0) {
    wsl -e bash $sh python -c "import jax; print(jax.__version__); print(jax.devices())"
} else {
    wsl -e bash $sh @args
}
