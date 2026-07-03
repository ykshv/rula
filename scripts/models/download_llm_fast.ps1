param(
    [string]$RepoId = "Qwen/Qwen3-14B-FP8",
    [int]$MaxWorkers = 16,
    [switch]$UseXet
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root
. .\scripts\env\load_local_env.ps1

$env:HF_HUB_ENABLE_HF_TRANSFER = "1"
if ($UseXet) {
    $env:HF_XET_HIGH_PERFORMANCE = "1"
    Remove-Item Env:\HF_HUB_DISABLE_XET -ErrorAction SilentlyContinue
}
else {
    $env:HF_HUB_DISABLE_XET = "1"
    Remove-Item Env:\HF_XET_HIGH_PERFORMANCE -ErrorAction SilentlyContinue
}

if (-not $env:HF_TOKEN) {
    throw "HF_TOKEN is not set. Use download_llm_with_token.ps1 for hidden prompt auth."
}

$downloadArgs = @(".\scripts\models\download_models.py", "--max-workers", "$MaxWorkers", "--model", $RepoId)
if (-not $UseXet) {
    $downloadArgs += "--disable-xet"
}
python @downloadArgs
python .\scripts\models\verify_models.py
