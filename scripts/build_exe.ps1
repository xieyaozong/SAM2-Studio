$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot
$buildDir = Join-Path $repoRoot "build"
$distDir = Join-Path $repoRoot "dist"

function Assert-InRepo {
    param([string]$PathToCheck)

    $parent = Split-Path -Parent $PathToCheck
    if (-not (Test-Path -LiteralPath $parent)) {
        throw "Parent path does not exist: $parent"
    }

    $resolvedParent = (Resolve-Path -LiteralPath $parent).Path
    if (-not $resolvedParent.StartsWith($repoRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to modify outside repo: $PathToCheck"
    }
}

$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }

$checkpointDir = Join-Path $repoRoot "checkpoints"
$checkpointData = "$checkpointDir;checkpoints"
$checkpointCount = 0
if (Test-Path -LiteralPath $checkpointDir) {
    $checkpointCount = @(Get-ChildItem -LiteralPath $checkpointDir -Filter "*.pt" -File -ErrorAction SilentlyContinue).Count
}
if ($checkpointCount -eq 0) {
    Write-Warning "No checkpoint .pt files found under checkpoints/. The exe will build, but model loading will fail until checkpoints are added."
}

& $python `
    -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --onedir `
    --name SAM2Studio `
    --specpath build `
    --collect-submodules sam2 `
    --collect-data sam2 `
    --collect-submodules pycocotools `
    --add-data $checkpointData `
    sam2_studio.py
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$sourceDir = Join-Path $repoRoot "dist\SAM2Studio"
$sourceExe = Join-Path $sourceDir "SAM2Studio.exe"
$sourceInternal = Join-Path $sourceDir "_internal"
$targetExe = Join-Path $repoRoot "SAM2Studio.exe"
$targetInternal = Join-Path $repoRoot "_internal"

Assert-InRepo $targetExe
Assert-InRepo $targetInternal

if (-not (Test-Path -LiteralPath $sourceExe)) {
    throw "Missing built exe: $sourceExe"
}
if (-not (Test-Path -LiteralPath $sourceInternal)) {
    throw "Missing packaged support directory: $sourceInternal"
}

if (Test-Path -LiteralPath $targetExe) {
    Remove-Item -LiteralPath $targetExe -Force
}
if (Test-Path -LiteralPath $targetInternal) {
    Remove-Item -LiteralPath $targetInternal -Recurse -Force
}

Move-Item -LiteralPath $sourceExe -Destination $targetExe

robocopy $sourceInternal $targetInternal /E /MOVE /NFL /NDL /NJH /NJS /NP | Out-Null
if ($LASTEXITCODE -gt 7) {
    throw "robocopy failed with exit code $LASTEXITCODE"
}

if (Test-Path -LiteralPath $sourceDir) {
    Remove-Item -LiteralPath $sourceDir -Force -ErrorAction SilentlyContinue
}
if (Test-Path -LiteralPath $distDir) {
    Remove-Item -LiteralPath $distDir -Recurse -Force
}
if (Test-Path -LiteralPath $buildDir) {
    Remove-Item -LiteralPath $buildDir -Recurse -Force
}

Write-Host "Built Windows app:"
Write-Host "  $targetExe"
Write-Host "  $targetInternal"
