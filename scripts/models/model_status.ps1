$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root
. .\scripts\env\load_local_env.ps1

$modelsRoot = Join-Path $Root "models\hf"
$expected = @(
    "ai-sage__GigaAM-v3",
    "Qwen__Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen__Qwen3-TTS-12Hz-1.7B-Base",
    "nvidia__Audio2Face-3D-v3.0",
    "Qwen__Qwen3-14B-FP8"
)

Write-Output "HF_TOKEN configured: $([bool]$env:HF_TOKEN)"
Write-Output ""

$rows = foreach ($name in $expected) {
    $path = Join-Path $modelsRoot $name
    $exists = Test-Path $path
    $size = if ($exists) {
        (Get-ChildItem $path -Recurse -Force -File -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -notlike "*\.cache\*" } |
            Measure-Object Length -Sum).Sum
    }
    else {
        0
    }
    [pscustomobject]@{
        Model = $name
        Exists = $exists
        SizeGB = [math]::Round($size / 1GB, 3)
    }
}

$rows | Format-Table -AutoSize

Write-Output ""
Write-Output "Active model download processes:"
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like "*download_qwen_fp8*" -or $_.CommandLine -like "*download_models.py*" } |
    Select-Object ProcessId,Name,@{Name="Command";Expression={($_.CommandLine -replace "hf_[A-Za-z0-9_\\-]+", "hf_***")}} |
    Format-Table -AutoSize
