# SAM2 Studio

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white&style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C?logo=pytorch&logoColor=white&style=flat-square)
![PySide6](https://img.shields.io/badge/GUI-PySide6-41CD52?logo=qt&logoColor=white&style=flat-square)
![SAM2](https://img.shields.io/badge/Model-SAM2-1877F2?style=flat-square)
![Platform](https://img.shields.io/badge/platform-Windows-0078D6?logo=windows&logoColor=white&style=flat-square)
![License](https://img.shields.io/badge/License-Apache_2.0-d97706?style=flat-square)

Windows desktop label-assist tool for building segmentation datasets with Meta's Segment Anything Model 2. It combines SAM2 prompts, optional Hough preprocessing, editable mask polygons, SAM-assisted hole cutting, whole-target moves, and export-ready YOLO / Mask R-CNN labels.

![Workflow](assets/workflow.png)

## What It Does

- Opens a single image or walks through a folder queue.
- Uses foreground/background prompts to create active SAM2 masks.
- Applies optional Hough masking or center-crop preprocessing before annotation.
- Converts masks into editable add/cut polygons, with undo support for manual edits.
- Keeps manual polygon edits intact when edit mode is toggled; Epsilon / Min area only rebuild polygons when those values are changed.
- Provides Epsilon / Min area controls from both the Labels and Edit tabs, including mouse-wheel value changes.
- Uses click-to-cut SAM Hole mode to create inner holes inside an existing mask target.
- Moves an active SAM mask or selected saved object as one target for quick alignment.
- Reuses crop and polygon templates across similar images.
- Restores existing annotations from the selected output folder so earlier work can be edited.
- Exports YOLO segmentation labels, binary/color masks, Mask R-CNN COCO/RLE annotations, overlays, and object metadata.
- Builds a ready-to-run Windows executable.

## Interface

![Interface layout](assets/interface.png)

The app centers on an image canvas with SAM prompts, mask overlays, editable polygons, middle-mouse panning, and wheel zoom. Side tabs separate image input, Hough preprocessing, label export, object review, reusable templates, polygon edits, whole-target moves, and SAM Hole cutting.

Canvas interaction is optimized for repeated annotation work: the base image/overlay pixmap is cached, polygon outlines are drawn as lightweight Qt scene items, wheel zoom and middle-button panning are coalesced, and CUDA runs with the available PyTorch acceleration settings.

## Typical Workflow

1. Open an image or folder and choose an output folder.
2. Add foreground/background prompts to create an active SAM2 mask.
3. Add the active mask to the object list.
4. Refine the target with Polygon mode, whole-target move, or SAM Hole mode.
5. Adjust Epsilon / Min area when you want a different polygon density.
6. Capture a template when several images share the same crop and mask layout.
7. Save labels, or use Save & Next to continue through the folder queue.

## Layout

```text
SAM2-Studio/
  sam2_studio.py
  run_sam_app.bat
  requirements.txt
  assets/
  utils/
  scripts/
  checkpoints/
```

## Installation

Python 3.11 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a specific CUDA build, install PyTorch first from the official selector, then the rest.

## Download Checkpoints

```powershell
.\scripts\download_checkpoints.ps1
```

Expected files under `checkpoints/`:

```text
sam2.1_hiera_tiny.pt
sam2.1_hiera_small.pt
sam2.1_hiera_base_plus.pt
sam2.1_hiera_large.pt
```

## Run

```powershell
.\.venv\Scripts\python.exe sam2_studio.py --gui
.\run_sam_app.bat
```

Optional: set `SAM2_STUDIO_IMAGE_DIR` to start the image picker in a specific folder.

Optional: set `SAM2_STUDIO_DISABLE_OPENGL=1` if a display driver has trouble with the accelerated Qt viewport.

## Build The Windows App

```powershell
.\scripts\build_exe.ps1
```

Output is placed at the repo root as `SAM2Studio.exe` and `_internal/`. Keep both together; the executable needs `_internal/` to find Python, PySide6, PyTorch, SAM2, and the packaged checkpoints.

## Notes

- `SAM2Studio.exe`, `_internal/`, `.venv/`, `outputs/`, and `checkpoints/*.pt` are git-ignored.
- To share a ready-to-run build, zip `SAM2Studio.exe` and `_internal/` together.
- To rebuild from source: install requirements, download checkpoints, and run `scripts/build_exe.ps1`.

## Attribution And License

SAM2 Studio depends on Meta's [Segment Anything Model 2](https://github.com/facebookresearch/sam2). Review Meta's SAM2 license and model terms before redistributing checkpoints or packaged builds.

This project's own code is released under the [Apache License 2.0](LICENSE).
