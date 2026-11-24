# Reproduce everything from scratch (Windows PowerShell, from the repository root).
#   powershell -ExecutionPolicy Bypass -File reproduce.ps1
# Prerequisite: the .venv environment (Python 3.12 + torch/torchvision cu128 + diffusers matplotlib tqdm).
# Run strictly one GPU job at a time. Total time on this laptop: roughly 4-5 hours.
Set-Location $PSScriptRoot
. .\env.ps1
$py = ".\.venv\python.exe"
$ErrorActionPreference = "Continue"

& $py check_env.py                                   # milestone 1: GPU check + MNIST download
& $py train_classifiers.py                           # milestone 2: judge + evaluator
& $py -u train_diffusion.py --kappa 1.0 --steps 35000
& $py -u train_diffusion.py --kappa 0.1 --steps 35000 --ckpt-every 3000
& $py eval_model.py --kappa 1.0 --n 4000             # milestone 3 acceptance
& $py eval_model.py --kappa 0.1 --n 4000
& $py test_smc.py                                    # logic tests for A/B/C/D and metrics
foreach ($k in "1.0", "0.1") {
    $t = if ($k -eq "1.0") { "1" } else { "0.1" }
    & $py check_lam0.py --kappa $k --batches 1000                                                  # milestone 4
    & $py check_lam0.py --kappa $k --batches 1000 --seed 100 --out "logs\lam0_kappa${t}_seed100.json"
    & $py pool_lam0.py --out "logs\lam0_kappa${t}_pooled.json" "logs\lam0_kappa${t}_stepfinal.json" "logs\lam0_kappa${t}_seed100.json"
    & $py -u run_main.py --kappa $k --extra-N 8 16                                                 # milestones 5-7 + D
    & $py make_table.py --kappa $k --lam0 "logs\lam0_kappa${t}_stepfinal.json" "logs\lam0_kappa${t}_seed100.json" --lam0-pooled "logs\lam0_kappa${t}_pooled.json"
}
& $py build_results_json.py                          # results.json + samples_{A,B,C}.png
