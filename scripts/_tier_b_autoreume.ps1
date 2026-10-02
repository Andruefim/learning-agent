# Auto-resume Tier B until all modes reach N episodes (handles MuJoCo hard crashes).
$ErrorActionPreference = "Continue"
Set-Location "C:\Users\Andrey\Desktop\agi\learning-agent"
$env:MUJOCO_GL = "glfw"
$env:PYTHONPATH = "."
$n = 100
$run = "tier_b_gate100"
$maxRounds = 40
for ($r = 1; $r -le $maxRounds; $r++) {
  $done = $true
  foreach ($mode in @("k1_default", "k5_random", "k5_llm")) {
    $p = "reports\$run\$mode.jsonl"
    $lines = 0
    if (Test-Path $p) {
      $lines = @(Get-Content $p | Where-Object { $_.Trim() -ne "" }).Count
    }
    Write-Output ("round {0} check {1} = {2}/{3}" -f $r, $mode, $lines, $n)
    if ($lines -lt $n) { $done = $false }
  }
  if ($done) {
    Write-Output "ALL MODES COMPLETE"
    python -u scripts/run_tier_b.py --all --n $n --seed 2000 --run $run --seconds 12 --k 5 --resume
    break
  }
  Write-Output ("round {0} launching resume..." -f $r)
  python -u scripts/run_tier_b.py --all --n $n --seed 2000 --run $run --seconds 12 --k 5 --resume
  $code = $LASTEXITCODE
  Write-Output ("round {0} exit={1}" -f $r, $code)
  Start-Sleep -Seconds 2
}
