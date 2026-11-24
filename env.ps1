# Dot-source before running anything for this project:  . .\env.ps1
# Keeps pip cache, temp files, and other caches inside the repository folder instead of the system drive.
$root = $PSScriptRoot
foreach ($d in ".cache\pip", ".cache\torch", ".cache\hf", ".cache\conda_pkgs", "tmp") { New-Item -ItemType Directory -Force (Join-Path $root $d) | Out-Null }
$env:PIP_CACHE_DIR   = "$root\.cache\pip"
$env:TEMP            = "$root\tmp"
$env:TMP             = "$root\tmp"
$env:TORCH_HOME      = "$root\.cache\torch"
$env:HF_HOME         = "$root\.cache\hf"
$env:CONDA_PKGS_DIRS = "$root\.cache\conda_pkgs"
$env:MPLCONFIGDIR    = "$root\.cache\matplotlib"
