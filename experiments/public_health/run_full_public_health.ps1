# Full public-health FMG extension run.
# This run is intentionally separate from archived primary benchmark outputs.
# It is resumable: rerun the same command after interruption.

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$python = Join-Path $repo '.venv\Scripts\python.exe'
$script = Join-Path $PSScriptRoot 'run_public_health_fmg_audit.py'
$uci = Join-Path $repo '.local_data\diabetes+130-us-hospitals.zip'
$brfss = 'C:\Users\mvk80\Downloads\archive.zip'
$out = Join-Path $PSScriptRoot 'outputs_full'

if (!(Test-Path $python)) { throw "Python environment not found: $python" }
if (!(Test-Path $uci)) { throw "UCI archive not found: $uci" }
if (!(Test-Path $brfss)) { throw "CDC archive not found: $brfss" }

& $python $script `
  --diabetes-zip $uci `
  --brfss $brfss `
  --seeds 42 123 456 `
  --mask-repeats 5 `
  --bootstrap-reps 200 `
  --output-dir $out

if ($LASTEXITCODE -ne 0) { throw "Full audit failed with exit code $LASTEXITCODE" }
Write-Host "Completed. Outputs: $out"
