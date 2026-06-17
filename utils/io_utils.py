from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
from PIL import Image, ImageOps
from utils.config import DEFAULT_EXTENSIONS, OutputPaths

import colorsys

import numpy as np


def parse_extensions(raw: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        items = raw.replace(",", " ").split()
    else:
        items = list(raw)

    extensions = []
    for item in items:
        item = item.strip().lower()
        if not item:
            continue
        if not item.startswith("."):
            item = f".{item}"
        extensions.append(item)
    return tuple(dict.fromkeys(extensions)) or DEFAULT_EXTENSIONS


def resolve_path(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def collect_images(
    input_path: Path,
    recursive: bool,
    extensions: Iterable[str],
    exclude_dir: Path | None = None,
) -> list[Path]:
    input_path = resolve_path(input_path)
    extensions = {ext.lower() for ext in extensions}
    exclude_dir = resolve_path(exclude_dir) if exclude_dir is not None else None

    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() in extensions else []

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    return sorted(
        path
        for path in iterator
        if path.is_file()
        and path.suffix.lower() in extensions
        and (exclude_dir is None or not is_within(path, exclude_dir))
    )


def output_paths_for(image_path: Path, input_path: Path, output_dir: Path) -> OutputPaths:
    input_path = resolve_path(input_path)
    output_dir = resolve_path(output_dir)
    input_root = input_path if input_path.is_dir() else input_path.parent
    rel_stem = image_path.resolve().relative_to(input_root).with_suffix("")
    rel_image = image_path.resolve().relative_to(input_root)
    preview_base = output_dir / "previews" / rel_stem
    metadata_base = output_dir / "metadata" / rel_stem
    label_base = output_dir / "labels" / rel_stem

    return OutputPaths(
        train_image=output_dir / "img" / rel_image,
        overlay=preview_base.parent / f"{preview_base.name}_overlay.png",
        mask_label=label_base.with_suffix(".png"),
        color_mask=preview_base.parent / f"{preview_base.name}_color_mask.png",
        objects_csv=metadata_base.parent / f"{metadata_base.name}_objects.csv",
        masks_dir=output_dir / "masks" / rel_stem,
        yolo_label=label_base.with_suffix(".txt"),
    )


def color_for_index(index: int) -> np.ndarray:
    hue = (index * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 1.0)
    return np.array([red * 255, green * 255, blue * 255], dtype=np.uint8)


def load_rgb_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.array(image, dtype=np.uint8)


def save_training_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    save_kwargs = {"quality": 95} if suffix in {".jpg", ".jpeg"} else {}
    Image.fromarray(image).save(path, **save_kwargs)


def save_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def wants_yolo_export(export_format: str) -> bool:
    return export_format in {"yolo", "both"}


def wants_mask_export(export_format: str) -> bool:
    return export_format in {"mask", "both"}
