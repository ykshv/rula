param(
    [switch]$ModelCache
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $Root

$targets = @()
$targets += Get-ChildItem $Root -Recurse -Force -Directory -Include __pycache__,.pytest_cache,.ruff_cache -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notlike "*\node_modules\*" }
$targets += @(
    Join-Path $Root "apps\web\dist"
    Join-Path $Root "packages\avatar_protocol\dist"
    Join-Path $Root "runtime"
    Join-Path $Root "output"
) | Where-Object { Test-Path $_ } | ForEach-Object { Get-Item $_ }

$files = @()
$files += Get-ChildItem $Root -Recurse -Force -File -Include *.tsbuildinfo -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notlike "*\node_modules\*" }

if ($ModelCache) {
    $modelsRoot = Join-Path $Root "models\hf"
    if (Test-Path $modelsRoot) {
        $targets += Get-ChildItem $modelsRoot -Recurse -Force -Directory -Filter .cache -ErrorAction SilentlyContinue
    }
}

$targets = $targets | Sort-Object FullName -Unique

foreach ($target in $targets) {
    $resolved = (Resolve-Path -LiteralPath $target.FullName).Path
    if (-not $resolved.StartsWith($Root, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to delete outside project: $resolved"
    }
}

foreach ($target in $targets) {
    Remove-Item -LiteralPath $target.FullName -Recurse -Force
    Write-Output "Removed $($target.FullName)"
}

foreach ($file in $files) {
    if (-not (Test-Path -LiteralPath $file.FullName)) {
        continue
    }
    $resolved = (Resolve-Path -LiteralPath $file.FullName).Path
    if (-not $resolved.StartsWith($Root, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to delete outside project: $resolved"
    }
    Remove-Item -LiteralPath $file.FullName -Force
    Write-Output "Removed $($file.FullName)"
}
