from __future__ import annotations

from pathlib import Path
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from tqdm import tqdm
from utils.config import LogFn, ProcessResult, SamBatchConfig
from utils.export_utils import (
    build_class_label_mask,
    render_masks,
    sorted_annotations,
    write_objects_csv,
    write_mask_rcnn_annotation,
    write_yolo_segmentation_label,
)
from utils.io_utils import (
    collect_images,
    load_rgb_image,
    output_paths_for,
    resolve_path,
    save_png,
    save_training_image,
    wants_mask_export,
    wants_mask_rcnn_export,
    wants_yolo_export,
)
from utils.model_utils import build_mask_generator, inference_autocast

import csv
import logging

import numpy as np
import torch


def save_individual_masks(paths, annotations: list[dict]) -> None:
    paths.masks_dir.mkdir(parents=True, exist_ok=True)
    for mask_id, annotation in enumerate(annotations, start=1):
        mask = np.asarray(annotation["segmentation"], dtype=np.uint8) * 255
        save_png(mask, paths.masks_dir / f"mask_{mask_id:03d}.png")


def process_one_image(
    image_path: Path,
    input_path: Path,
    output_dir: Path,
    generator: SAM2AutomaticMaskGenerator,
    device: torch.device,
    config: SamBatchConfig,
) -> ProcessResult:
    paths = output_paths_for(image_path, input_path, output_dir)
    expected_label = None
    if wants_yolo_export(config.export_format):
        expected_label = paths.yolo_label
    elif wants_mask_export(config.export_format):
        expected_label = paths.mask_label
    elif wants_mask_rcnn_export(config.export_format):
        expected_label = paths.mask_rcnn_annotation

    if (
        config.skip_existing
        and paths.train_image.exists()
        and paths.overlay.exists()
        and (expected_label is None or expected_label.exists())
    ):
        return ProcessResult(
            source=image_path,
            status="skipped",
            train_image=paths.train_image,
            overlay=paths.overlay,
            yolo_label=paths.yolo_label if paths.yolo_label.exists() else None,
            mask_label=paths.mask_label if paths.mask_label.exists() else None,
            mask_rcnn_annotation=paths.mask_rcnn_annotation if paths.mask_rcnn_annotation.exists() else None,
        )

    image = load_rgb_image(image_path)

    with torch.inference_mode(), inference_autocast(device):
        annotations = generator.generate(image)

    annotations = sorted_annotations(annotations, config.max_masks)
    overlay, _label_mask, color_mask = render_masks(image, annotations, config.alpha)
    masks = [np.asarray(annotation["segmentation"], dtype=bool) for annotation in annotations]
    class_ids = [config.yolo_class_id] * len(annotations)

    save_training_image(image, paths.train_image)
    save_png(overlay, paths.overlay)
    save_png(color_mask, paths.color_mask)
    write_objects_csv(paths.objects_csv, annotations)

    saved_mask_label = None
    saved_yolo_label = None
    saved_mask_rcnn_annotation = None
    if wants_mask_export(config.export_format):
        save_png(build_class_label_mask(masks, class_ids, image.shape), paths.mask_label)
        saved_mask_label = paths.mask_label

    if wants_yolo_export(config.export_format):
        write_yolo_segmentation_label(
            paths.yolo_label,
            masks,
            class_ids,
            image.shape,
            epsilon=config.yolo_epsilon,
            min_area=config.yolo_min_area,
        )
        saved_yolo_label = paths.yolo_label

    if wants_mask_rcnn_export(config.export_format):
        write_mask_rcnn_annotation(
            paths.mask_rcnn_annotation,
            paths.mask_rcnn_masks_dir,
            paths.train_image.name,
            masks,
            class_ids,
            image.shape,
        )
        saved_mask_rcnn_annotation = paths.mask_rcnn_annotation

    if config.save_individual_masks:
        save_individual_masks(paths, annotations)

    return ProcessResult(
        source=image_path,
        status="ok",
        mask_count=len(annotations),
        train_image=paths.train_image,
        overlay=paths.overlay,
        mask_label=saved_mask_label,
        color_mask=paths.color_mask,
        objects_csv=paths.objects_csv,
        yolo_label=saved_yolo_label,
        mask_rcnn_annotation=saved_mask_rcnn_annotation,
    )


def write_summary_csv(output_dir: Path, results: list[ProcessResult]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    fieldnames = [
        "source",
        "status",
        "mask_count",
        "train_image",
        "overlay",
        "mask_label",
        "color_mask",
        "objects_csv",
        "yolo_label",
        "mask_rcnn_annotation",
        "error",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "source": str(result.source),
                    "status": result.status,
                    "mask_count": result.mask_count,
                    "train_image": str(result.train_image or ""),
                    "overlay": str(result.overlay or ""),
                    "mask_label": str(result.mask_label or ""),
                    "color_mask": str(result.color_mask or ""),
                    "objects_csv": str(result.objects_csv or ""),
                    "yolo_label": str(result.yolo_label or ""),
                    "mask_rcnn_annotation": str(result.mask_rcnn_annotation or ""),
                    "error": result.error,
                }
            )
    return summary_path


def process_batch(config: SamBatchConfig, log: LogFn = print, progress: bool = True) -> list[ProcessResult]:
    input_path = resolve_path(config.input_path)
    output_dir = resolve_path(config.output_dir)
    images = collect_images(input_path, config.recursive, config.extensions, exclude_dir=output_dir)

    if not images:
        raise RuntimeError(f"No supported images found in {input_path}")

    log(f"Found {len(images)} image(s).")
    if config.dry_run:
        for image in images:
            paths = output_paths_for(image, input_path, output_dir)
            label_targets = []
            if wants_yolo_export(config.export_format):
                label_targets.append(str(paths.yolo_label))
            if wants_mask_export(config.export_format):
                label_targets.append(str(paths.mask_label))
            if wants_mask_rcnn_export(config.export_format):
                label_targets.append(str(paths.mask_rcnn_annotation))
            label_text = ", ".join(label_targets) if label_targets else "no training label"
            log(f"DRY RUN: {image} -> image: {paths.train_image}; labels: {label_text}")
        return []

    generator, device = build_mask_generator(config, log)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[ProcessResult] = []
    iterator = enumerate(images, start=1)
    if progress:
        iterator = enumerate(tqdm(images, desc="Processing", unit="image"), start=1)

    total = len(images)
    for index, image_path in iterator:
        log(f"[{index}/{total}] Processing {image_path}")
        try:
            result = process_one_image(image_path, input_path, output_dir, generator, device, config)
            results.append(result)
            if result.status == "skipped":
                log(f"Skipped existing result: {image_path.name}")
            else:
                log(f"Saved {result.mask_count} mask(s): {result.overlay}")
        except Exception as exc:
            logging.exception("Failed to process %s", image_path)
            result = ProcessResult(source=image_path, status="error", error=str(exc))
            results.append(result)
            log(f"ERROR: {image_path} -> {exc}")
            if config.stop_on_error:
                raise

    summary_path = write_summary_csv(output_dir, results)
    ok_count = sum(1 for result in results if result.status == "ok")
    skipped_count = sum(1 for result in results if result.status == "skipped")
    error_count = sum(1 for result in results if result.status == "error")
    log(
        "Done. "
        f"Processed: {ok_count}, skipped: {skipped_count}, errors: {error_count}. "
        f"Summary: {summary_path}"
    )
    return results
