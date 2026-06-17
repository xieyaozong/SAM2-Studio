$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$checkpointDir = Join-Path $repoRoot "checkpoints"
New-Item -ItemType Directory -Path $checkpointDir -Force | Out-Null

$files = @(
    @{
        Name = "sam2.1_hiera_tiny.pt"
        Url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
    },
    @{
        Name = "sam2.1_hiera_small.pt"
        Url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
    },
    @{
        Name = "sam2.1_hiera_base_plus.pt"
        Url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
    },
    @{
        Name = "sam2.1_hiera_large.pt"
        Url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    }
)

foreach ($file in $files) {
    $target = Join-Path $checkpointDir $file.Name
    if (Test-Path -LiteralPath $target) {
        Write-Host "Already exists: $target"
        continue
    }

    Write-Host "Downloading $($file.Name)..."
    Invoke-WebRequest -Uri $file.Url -OutFile $target
}

Write-Host "Done. Checkpoints are in $checkpointDir"
