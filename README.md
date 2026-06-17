# SAM2 Studio

SAM2 Studio is a small Windows-first desktop tool for interactive image segmentation with Meta's Segment Anything Model 2. It wraps SAM2 in a PySide6 GUI, adds a practical preprocessing workflow for centered product-like images, and exports training-friendly masks and YOLO segmentation labels.

This repository is intentionally kept as a personal tool repository. It does not vendor the upstream `sam2/` source tree; SAM2 is installed as an external dependency from Meta's official repository.

## Highlights

- Interactive SAM2 image segmentation with foreground and background clicks.
- Folder-based annotation workflow with previous / next navigation.
- Hough / foreground preprocessing for dark, wide, or centered-object images.
- `Preview Hough` and `Use Hough For SAM` are independent actions.
- Full masked image or adaptive center crop modes.
- YOLO polygon label export, class-id control, mask export, overlay previews, and object metadata.
- Windows PyInstaller build with no terminal window.

## Repository Layout

```text
sam2_studio.py          # Application entry point
run_sam_app.bat         # Convenience launcher for local development
SAM2Studio.spec         # PyInstaller configuration
requirements.txt        # Runtime and packaging dependencies
utils/                  # GUI, preprocessing, export, model, and IO code
checkpoints/            # Local checkpoint folder; *.pt files are ignored by git
scripts/                # Build and utility scripts
docs/                   # GitHub / release notes for this personal repo
```

Generated local artifacts are intentionally ignored:

```text
SAM2Studio.exe
_internal/
.venv/
outputs/
checkpoints/*.pt
```

## Requirements

- Windows 10 or later
- Python 3.11 recommended
- NVIDIA GPU with CUDA is recommended for practical speed, but CPU mode is available
- SAM2.1 checkpoint files downloaded locally

## Installation

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install PyTorch from the official selector first, then install the rest of the requirements.

## Download Checkpoints

Checkpoint files are large and should not be committed to git. Download them into `checkpoints/`.

```powershell
.\scripts\download_checkpoints.ps1
```

Expected files:

```text
checkpoints/sam2.1_hiera_tiny.pt
checkpoints/sam2.1_hiera_small.pt
checkpoints/sam2.1_hiera_base_plus.pt
checkpoints/sam2.1_hiera_large.pt
```

## Run the GUI

Development mode:

```powershell
.\.venv\Scripts\python.exe sam2_studio.py --gui
```

Convenience launcher:

```powershell
.\run_sam_app.bat
```

Packaged app:

```text
SAM2Studio.exe
```

When using the packaged build, keep `SAM2Studio.exe` and `_internal/` in the same folder.

## CLI Dry Run

```powershell
.\.venv\Scripts\python.exe sam2_studio.py `
  --input C:\path\to\images `
  --output C:\path\to\output `
  --recursive `
  --dry-run
```

## Build the Windows App

```powershell
.\scripts\build_exe.ps1
```

The build script uses `SAM2Studio.spec`, then moves the packaged app to the repository root:

```text
SAM2Studio.exe
_internal/
```

Those files are ignored by git. To share the app, zip both items together and publish the zip through GitHub Releases or another file host.

## GitHub Workflow

Recommended source commit:

```powershell
git add .
git status --short
git commit -m "Prepare SAM2 Studio personal tool"
git push origin main
```

The `.gitignore` file excludes local environments, checkpoints, generated datasets, and packaged binaries.

## Attribution

SAM2 Studio depends on Meta's Segment Anything Model 2. SAM2 is installed from the official upstream repository:

https://github.com/facebookresearch/sam2

Please review Meta's SAM2 license and model terms before redistributing checkpoints or packaged builds.
