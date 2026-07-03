param(
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root
. .\scripts\env\load_local_env.ps1

$agentPort = if ($env:RU_LOCAL_AVATAR_PORT) { [int]$env:RU_LOCAL_AVATAR_PORT } else { 46181 }
$webPort = if ($env:RU_LOCAL_AVATAR_WEB_PORT) { [int]$env:RU_LOCAL_AVATAR_WEB_PORT } else { 46174 }
$runtimeDir = Join-Path $Root "runtime\logs"
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

function Stop-PortOwner {
    param([int]$Port)
    $owners = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($owner in $owners) {
        try {
            Stop-Process -Id $owner -Force -ErrorAction Stop
        }
        catch {}
    }
}

Stop-PortOwner -Port $agentPort
Stop-PortOwner -Port $webPort

$agentLog = Join-Path $runtimeDir "agent.log"
$webLog = Join-Path $runtimeDir "web.log"

$agentCommand = "`$env:PYTHONPATH='$Root\apps\agent\src'; `$env:RU_LOCAL_AVATAR_PORT='$agentPort'; python -m ru_local_avatar_agent *> '$agentLog'"
$webCommand = "npm --workspace apps/web run dev -- --host 127.0.0.1 --port $webPort *> '$webLog'"

Start-Process -FilePath "powershell" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $agentCommand -WorkingDirectory $Root -WindowStyle Hidden
Start-Sleep -Seconds 1
Start-Process -FilePath "powershell" -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $webCommand -WorkingDirectory $Root -WindowStyle Hidden

Write-Output "Agent: http://127.0.0.1:$agentPort/health"
Write-Output "Web:   http://127.0.0.1:$webPort/"
Write-Output "Logs:  $runtimeDir"

if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:$webPort/"
}
