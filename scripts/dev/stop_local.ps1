$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root
. .\scripts\env\load_local_env.ps1

$ports = @(
    $(if ($env:RU_LOCAL_AVATAR_PORT) { [int]$env:RU_LOCAL_AVATAR_PORT } else { 46181 }),
    $(if ($env:RU_LOCAL_AVATAR_WEB_PORT) { [int]$env:RU_LOCAL_AVATAR_WEB_PORT } else { 46174 })
)

foreach ($port in $ports) {
    $owners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($owner in $owners) {
        try {
            Stop-Process -Id $owner -Force -ErrorAction Stop
            Write-Output "Stopped PID $owner on port $port"
        }
        catch {}
    }
}
