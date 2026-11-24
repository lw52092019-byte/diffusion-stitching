# Level 3 pipeline, strictly serial (one GPU job at a time). Every step is resumable / skips
# finished work, so after an interruption just run this script again.
# Usage (from the repository root):  powershell -ExecutionPolicy Bypass -File level3_pipeline.ps1 [-Only train]
param([string]$Only = "")
Set-Location $PSScriptRoot
. .\env.ps1
$py = ".\.venv\python.exe"
New-Item -ItemType Directory -Force level3\logs | Out-Null
$log = "level3\logs\pipeline.log"
function Step($name, [scriptblock]$cmd) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $name" | Out-File -Append -Encoding utf8 $log
    & $cmd
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] END   $name (exit $LASTEXITCODE)" | Out-File -Append -Encoding utf8 $log
}
if ($Only -eq "" -or $Only -eq "classifiers") { Step "classifiers" { & $py -u level3_classifiers.py *>> level3\logs\classifiers_rerun.log } }
if ($Only -eq "" -or $Only -eq "train")       { Step "train diffusion" { & $py -u level3_train_diffusion.py *>> level3\logs\train.log } }
if ($Only -eq "train") { "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] TRAIN ONLY DONE" | Out-File -Append -Encoding utf8 $log; exit }
Step "model acceptance"  { & $py -u level3_eval_model.py *>> level3\logs\eval_model.log }
Step "exp1 survival"     { & $py -u level3_timing.py *>> level3\logs\exp1.log }
Step "exp2 methods"      { & $py -u level3_methods.py *>> level3\logs\exp2.log }
Step "report"            { & $py -u level3_report.py *>> level3\logs\report.log }
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] PIPELINE DONE" | Out-File -Append -Encoding utf8 $log
