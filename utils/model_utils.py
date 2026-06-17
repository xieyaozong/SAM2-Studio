from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch

from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from utils.config import LogFn, MODEL_PRESETS, PROJECT_ROOT, RESOURCE_ROOT, SamBatchConfig


def select_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_name)


def resolve_model_paths(config: SamBatchConfig) -> tuple[str, Path]:
    preset = MODEL_PRESETS[config.model_size]
    model_cfg = config.model_cfg or preset["cfg"]
    if config.checkpoint is None:
        checkpoint = RESOURCE_ROOT / "checkpoints" / preset["checkpoint"]
    else:
        checkpoint = config.checkpoint
        if not checkpoint.is_absolute():
            resource_checkpoint = RESOURCE_ROOT / checkpoint
            app_checkpoint = PROJECT_ROOT / checkpoint
            checkpoint = resource_checkpoint if resource_checkpoint.exists() else app_checkpoint
    checkpoint = checkpoint.resolve()

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. Run checkpoints/download_ckpts.sh "
            "or choose another --checkpoint."
        )
    return model_cfg, checkpoint


def build_mask_generator(config: SamBatchConfig, log: LogFn) -> tuple[SAM2AutomaticMaskGenerator, torch.device]:
    device = select_device(config.device)
    model_cfg, checkpoint = resolve_model_paths(config)

    log(f"Loading SAM 2 model: {config.model_size} on {device}")
    log(f"Checkpoint: {checkpoint}")

    sam_model = build_sam2(model_cfg, str(checkpoint), device=str(device))
    generator = SAM2AutomaticMaskGenerator(
        sam_model,
        points_per_side=config.points_per_side,
        points_per_batch=config.points_per_batch,
        pred_iou_thresh=config.pred_iou_thresh,
        stability_score_thresh=config.stability_score_thresh,
        min_mask_region_area=config.min_mask_region_area,
        output_mode="binary_mask",
    )
    return generator, device


def build_image_predictor(
    model_size: str,
    checkpoint: Path | None,
    model_cfg: str | None,
    device_name: str,
    log: LogFn,
) -> tuple[SAM2ImagePredictor, torch.device]:
    config = SamBatchConfig(
        input_path=PROJECT_ROOT,
        output_dir=PROJECT_ROOT,
        model_size=model_size,
        checkpoint=checkpoint,
        model_cfg=model_cfg,
        device=device_name,
    )
    device = select_device(config.device)
    resolved_cfg, resolved_checkpoint = resolve_model_paths(config)

    log(f"Loading SAM 2 predictor: {model_size} on {device}")
    log(f"Checkpoint: {resolved_checkpoint}")

    sam_model = build_sam2(resolved_cfg, str(resolved_checkpoint), device=str(device))
    return SAM2ImagePredictor(sam_model), device


def inference_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()
