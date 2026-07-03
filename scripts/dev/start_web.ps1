param(
    [int]$Port = 46174
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")

Set-Location $Root
. .\scripts\env\load_local_env.ps1
$effectivePort = if ($env:RU_LOCAL_AVATAR_WEB_PORT) { [int]$env:RU_LOCAL_AVATAR_WEB_PORT } else { $Port }
npm --workspace apps/web run dev -- --host 127.0.0.1 --port $effectivePort
