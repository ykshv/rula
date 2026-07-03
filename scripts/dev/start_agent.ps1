param(
    [int]$Port = 46181
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")

Set-Location $Root
. .\scripts\env\load_local_env.ps1
$env:PYTHONPATH = Join-Path $Root "apps\agent\src"
if (-not $env:RU_LOCAL_AVATAR_PORT) {
    $env:RU_LOCAL_AVATAR_PORT = "$Port"
}

python -m ru_local_avatar_agent
