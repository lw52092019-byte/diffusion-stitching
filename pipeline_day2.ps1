# Serial run of everything left after the 2026-11-24 pause (one GPU job at a time).
# Usage (from the repository root):  powershell -ExecutionPolicy Bypass -File pipeline_day2.ps1
Set-Location $PSScriptRoot
. .\env.ps1
$py = ".\.venv\python.exe"
$log = "logs\pipeline_day2.log"
function Step($name, [scriptblock]$cmd) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $name" | Out-File -Append -Encoding utf8 $log
    & $cmd
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] END   $name (exit $LASTEXITCODE)" | Out-File -Append -Encoding utf8 $log
}

Step "kappa1 main grid + D + extra N" { & $py -u run_main.py --kappa 1.0 --extra-N 8 16 *> logs\main_kappa1_day2.log }
Step "kappa0.1 acceptance"            { & $py -u eval_model.py --kappa 0.1 --n 4000 *> logs\eval_kappa0.1.log }
Step "kappa0.1 lam0 check run 1"      { & $py -u check_lam0.py --kappa 0.1 --batches 1000 *> logs\lam0_kappa0.1_run1.log }
Step "kappa0.1 lam0 check run 2"      { & $py -u check_lam0.py --kappa 0.1 --batches 1000 --seed 100 --out logs\lam0_kappa0.1_seed100.json *> logs\lam0_kappa0.1_run2.log }
Step "kappa0.1 lam0 pooled"           { & $py -u pool_lam0.py --out logs\lam0_kappa0.1_pooled.json logs\lam0_kappa0.1_stepfinal.json logs\lam0_kappa0.1_seed100.json *> logs\lam0_kappa0.1_pooled.log }
Step "kappa0.1 main grid + D + extra N" { & $py -u run_main.py --kappa 0.1 --extra-N 8 16 *> logs\main_kappa0.1.log }
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] PIPELINE DONE" | Out-File -Append -Encoding utf8 $log
