from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import os
import sys

import numpy as np


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", app_root())).resolve()
    return Path(__file__).resolve().parent.parent


APP_ROOT = app_root()
RESOURCE_ROOT = resource_root()
PROJECT_ROOT = APP_ROOT
PREFERRED_IMAGE_FOLDER = (
    Path(os.environ["SAM2_STUDIO_IMAGE_DIR"]).expanduser()
    if os.environ.get("SAM2_STUDIO_IMAGE_DIR")
    else None
)

DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
EXPORT_FORMATS = ("yolo", "mask", "mask_rcnn", "both", "all", "none")

MODEL_PRESETS = {
    "tiny": {
        "checkpoint": "sam2.1_hiera_tiny.pt",
        "cfg": "configs/sam2.1/sam2.1_hiera_t.yaml",
    },
    "small": {
        "checkpoint": "sam2.1_hiera_small.pt",
        "cfg": "configs/sam2.1/sam2.1_hiera_s.yaml",
    },
    "base_plus": {
        "checkpoint": "sam2.1_hiera_base_plus.pt",
        "cfg": "configs/sam2.1/sam2.1_hiera_b+.yaml",
    },
    "large": {
        "checkpoint": "sam2.1_hiera_large.pt",
        "cfg": "configs/sam2.1/sam2.1_hiera_l.yaml",
    },
}

LogFn = Callable[[str], None]


@dataclass
class SamBatchConfig:
    input_path: Path
    output_dir: Path
    model_size: str = "large"
    checkpoint: Path | None = None
    model_cfg: str | None = None
    device: str = "auto"
    recursive: bool = False
    extensions: tuple[str, ...] = DEFAULT_EXTENSIONS
    points_per_side: int = 32
    points_per_batch: int = 64
    pred_iou_thresh: float = 0.8
    stability_score_thresh: float = 0.95
    min_mask_region_area: int = 0
    alpha: float = 0.55
    max_masks: int = 0
    save_individual_masks: bool = False
    export_format: str = "yolo"
    yolo_class_id: int = 0
    yolo_epsilon: float = 2.0
    yolo_min_area: float = 8.0
    skip_existing: bool = False
    dry_run: bool = False
    stop_on_error: bool = False


@dataclass
class OutputPaths:
    train_image: Path
    overlay: Path
    mask_label: Path
    color_mask: Path
    objects_csv: Path
    masks_dir: Path
    yolo_label: Path
    mask_rcnn_annotation: Path
    mask_rcnn_masks_dir: Path


@dataclass
class ProcessResult:
    source: Path
    status: str
    mask_count: int = 0
    train_image: Path | None = None
    overlay: Path | None = None
    mask_label: Path | None = None
    color_mask: Path | None = None
    objects_csv: Path | None = None
    yolo_label: Path | None = None
    mask_rcnn_annotation: Path | None = None
    error: str = ""


@dataclass
class SavedObject:
    name: str
    mask: np.ndarray
    color: np.ndarray
    score: float
    class_id: int = 0
    yolo_polygons: list[dict[str, object]] = field(default_factory=list)
