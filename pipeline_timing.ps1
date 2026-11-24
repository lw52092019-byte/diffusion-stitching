# Stitch-timing experiment, both models, strictly serial (one GPU job at a time).
# Usage (from the repository root):  powershell -ExecutionPolicy Bypass -File pipeline_timing.ps1
Set-Location $PSScriptRoot
. .\env.ps1
$py = ".\.venv\python.exe"
$log = "logs\pipeline_timing.log"
function Step($name, [scriptblock]$cmd) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $name" | Out-File -Append -Encoding utf8 $log
    & $cmd
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] END   $name (exit $LASTEXITCODE)" | Out-File -Append -Encoding utf8 $log
}
Step "selftest"      { & $py timing.py --selftest *> logs\timing_selftest.log }
Step "timing kappa1" { & $py -u timing.py --kappa 1.0 *> logs\timing_kappa1.log }
Step "timing kappa0.1" { & $py -u timing.py --kappa 0.1 *> logs\timing_kappa0.1.log }
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] PIPELINE DONE" | Out-File -Append -Encoding utf8 $log
