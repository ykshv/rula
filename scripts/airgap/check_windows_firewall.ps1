param(
    [string]$ProjectName = "ru-local-avatar"
)

$ErrorActionPreference = "Stop"

$rules = Get-NetFirewallRule -PolicyStore ActiveStore -ErrorAction Stop |
    Where-Object { $_.DisplayName -like "*$ProjectName*" -and $_.Direction -eq "Outbound" }

if (-not $rules) {
    Write-Error "No active outbound firewall rule found for $ProjectName. Air-gap gate failed."
}

$blockingRules = $rules | Where-Object { $_.Action -eq "Block" -and $_.Enabled -eq "True" }
if (-not $blockingRules) {
    Write-Error "No enabled outbound block rule found for $ProjectName. Air-gap gate failed."
}

Write-Output "Windows firewall outbound block rule present for $ProjectName."
