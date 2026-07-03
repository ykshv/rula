param(
    [string]$RepoId = "Qwen/Qwen3-14B-FP8",
    [int]$MaxWorkers = 16,
    [switch]$UseXet
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root
. .\scripts\env\load_local_env.ps1

$hadToken = [bool]$env:HF_TOKEN
$plainToken = $null
$bstr = [IntPtr]::Zero

try {
    $env:HF_HUB_ENABLE_HF_TRANSFER = "1"
    if ($UseXet) {
        $env:HF_XET_HIGH_PERFORMANCE = "1"
        Remove-Item Env:\HF_HUB_DISABLE_XET -ErrorAction SilentlyContinue
    }
    else {
        $env:HF_HUB_DISABLE_XET = "1"
        Remove-Item Env:\HF_XET_HIGH_PERFORMANCE -ErrorAction SilentlyContinue
    }

    if (-not $hadToken) {
        $secureToken = Read-Host "Paste Hugging Face token (input is hidden)" -AsSecureString
        if ($secureToken.Length -eq 0) {
            throw "HF token is empty"
        }
        $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
        $plainToken = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
        $env:HF_TOKEN = $plainToken
    }

    $downloadArgs = @(".\scripts\models\download_models.py", "--max-workers", "$MaxWorkers", "--model", $RepoId)
    if (-not $UseXet) {
        $downloadArgs += "--disable-xet"
    }
    python @downloadArgs
    python .\scripts\models\verify_models.py
}
finally {
    if ($bstr -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
    if (-not $hadToken) {
        Remove-Item Env:\HF_TOKEN -ErrorAction SilentlyContinue
    }
}
