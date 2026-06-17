# SAM2 Studio

SAM2 Studio is a compact Windows desktop app for interactive image segmentation with Meta's Segment Anything Model 2. It keeps the project focused on the annotation tool itself: SAM2 is installed from Meta's upstream repository instead of being copied into this repo.

## What It Does

- Open a single image or walk through a folder of images.
- Add foreground and background clicks to guide SAM2.
- Preview or apply Hough / foreground preprocessing before annotation.
- Use either the full masked image or an adaptive center crop.
- Export YOLO segmentation labels, mask images, overlays, and object metadata.
- Build a Windows executable that opens without a terminal window.

## Project Layout

```text
sam2_studio.py          # App entry point
run_sam_app.bat         # Local launcher for the virtual environment
requirements.txt        # Python dependencies
utils/                  # GUI, preprocessing, export, model, and IO code
scripts/                # Build and checkpoint helper scripts
checkpoints/            # Local model weights; *.pt files are ignored
```

## Install

Python 3.11 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you need a specific CUDA build of PyTorch, install PyTorch first from the official selector, then install the rest of the requirements.

## Download Checkpoints

```powershell
.\scripts\download_checkpoints.ps1
```

The app expects these files under `checkpoints/`:

```text
sam2.1_hiera_tiny.pt
sam2.1_hiera_small.pt
sam2.1_hiera_base_plus.pt
sam2.1_hiera_large.pt
```

## Run

```powershell
.\.venv\Scripts\python.exe sam2_studio.py --gui
```

or:

```powershell
.\run_sam_app.bat
```

Optional: set `SAM2_STUDIO_IMAGE_DIR` if you want the image picker to start in a specific folder.

## Build the Windows App

```powershell
.\scripts\build_exe.ps1
```

The build output is placed at the repo root:

```text
SAM2Studio.exe
_internal/
```

Keep both items together. The executable needs `_internal/` to find Python, PySide6, PyTorch, SAM2, and the packaged checkpoints.

## Notes

- `SAM2Studio.exe`, `_internal/`, `.venv/`, `outputs/`, and `checkpoints/*.pt` are ignored by git.
- To share a ready-to-run build, zip `SAM2Studio.exe` and `_internal/` together.
- To rebuild from source, install requirements, download checkpoints, then run `scripts/build_exe.ps1`.

## Attribution

SAM2 Studio depends on Meta's Segment Anything Model 2:

https://github.com/facebookresearch/sam2

Please review Meta's SAM2 license and model terms before redistributing checkpoints or packaged builds.
