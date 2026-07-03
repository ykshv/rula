param(
    [string]$RepoId = "Qwen/Qwen3-14B-FP8",
    [string]$LocalDir = "models\hf\Qwen__Qwen3-14B-FP8",
    [int]$Parallel = 4,
    [switch]$PromptToken
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root
. .\scripts\env\load_local_env.ps1

New-Item -ItemType Directory -Force -Path $LocalDir | Out-Null

$token = $env:HF_TOKEN
$bstr = [IntPtr]::Zero
try {
    if ($PromptToken -and -not $token) {
        $secureToken = Read-Host "Paste Hugging Face token (input is hidden)" -AsSecureString
        if ($secureToken.Length -gt 0) {
            $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
            $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
        }
    }

    if ($token) {
        $env:HF_TOKEN = $token
    }

    python .\scripts\models\download_qwen_fp8_http.py --repo-id $RepoId --local-dir $LocalDir --parallel $Parallel
    python .\scripts\models\merge_model_manifest.py --repo-id $RepoId --local-dir $LocalDir
    python .\scripts\models\verify_models.py
}
finally {
    if ($bstr -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}
