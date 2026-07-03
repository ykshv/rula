$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$EnvFile = Join-Path $Root ".env.local"
$Example = Join-Path $Root ".env.local.example"

if (-not (Test-Path $EnvFile)) {
    Copy-Item -LiteralPath $Example -Destination $EnvFile
}

$secureToken = Read-Host "Paste Hugging Face token (input is hidden)" -AsSecureString
if ($secureToken.Length -eq 0) {
    throw "HF token is empty"
}

$bstr = [IntPtr]::Zero
try {
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureToken)
    $token = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)

    $lines = if (Test-Path $EnvFile) { @(Get-Content $EnvFile) } else { @() }
    $filtered = $lines | Where-Object { $_ -notmatch '^\s*HF_TOKEN\s*=' }
    $filtered += "HF_TOKEN=$token"
    Set-Content -LiteralPath $EnvFile -Value $filtered -Encoding UTF8

    try {
        icacls $EnvFile /inheritance:r /grant:r "$($env:USERNAME):(R,W)" | Out-Null
    }
    catch {
        Write-Warning "Could not tighten ACL for .env.local: $($_.Exception.Message)"
    }

    Write-Output "HF_TOKEN saved to .env.local for local use. Do not commit this file."
}
finally {
    if ($bstr -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
    }
}
