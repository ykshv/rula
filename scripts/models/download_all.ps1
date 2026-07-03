$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root
. .\scripts\env\load_local_env.ps1

if (-not $env:HF_TOKEN) {
    Write-Warning "HF_TOKEN is not set. Run scripts\secrets\set_hf_token.ps1 first for better rate limits."
}

python .\scripts\models\download_models.py `
    --model ai-sage/GigaAM-v3 `
    --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice `
    --model nvidia/Audio2Face-3D-v3.0

powershell -ExecutionPolicy Bypass -File .\scripts\models\download_qwen_fp8_curl.ps1
python .\scripts\models\verify_models.py
