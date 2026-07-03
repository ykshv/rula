param(
    [string]$ManifestPath = "models\manifests\default_avatar.alicia_solid_vrm_0_51.json",
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$manifestFullPath = Join-Path $Root $ManifestPath
if (-not (Test-Path -LiteralPath $manifestFullPath)) {
    throw "Missing avatar manifest: $manifestFullPath"
}

$manifest = Get-Content -LiteralPath $manifestFullPath -Raw | ConvertFrom-Json
$target = Join-Path $Root $manifest.artifact.web_public_path
$targetDir = Split-Path -Parent $target
New-Item -ItemType Directory -Force -Path $targetDir | Out-Null

if ((Test-Path -LiteralPath $target) -and -not $Force) {
    $existingHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $target).Hash.ToLowerInvariant()
    if ($existingHash -eq $manifest.artifact.sha256) {
        Write-Output "Default avatar already present: $target"
        Write-Output "SHA256: $existingHash"
        exit 0
    }

    throw "Default avatar exists but checksum is wrong. Re-run with -Force to replace it."
}

$tmp = "$target.incomplete"
if (Test-Path -LiteralPath $tmp) {
    Remove-Item -LiteralPath $tmp -Force
}

Invoke-WebRequest -Uri $manifest.source.download_url -OutFile $tmp

$item = Get-Item -LiteralPath $tmp
if ($item.Length -ne [int64]$manifest.artifact.size_bytes) {
    Remove-Item -LiteralPath $tmp -Force
    throw "Downloaded avatar size mismatch. Expected $($manifest.artifact.size_bytes), got $($item.Length)."
}

$hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $tmp).Hash.ToLowerInvariant()
if ($hash -ne $manifest.artifact.sha256) {
    Remove-Item -LiteralPath $tmp -Force
    throw "Downloaded avatar checksum mismatch. Expected $($manifest.artifact.sha256), got $hash."
}

Move-Item -LiteralPath $tmp -Destination $target -Force
Write-Output "Downloaded default avatar: $target"
Write-Output "SHA256: $hash"
