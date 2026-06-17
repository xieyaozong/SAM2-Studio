from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from utils.batch import process_batch
from utils.config import DEFAULT_EXTENSIONS, EXPORT_FORMATS, MODEL_PRESETS, SamBatchConfig
from utils.io_utils import parse_extensions
from utils.qt_gui import run_interactive_gui


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be greater than 0.")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be 0 or greater.")
    return parsed


def alpha_value(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("Alpha must be between 0 and 1.")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be 0 or greater.")
    return parsed


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAM 2 interactive and batch image segmentation app.",
    )
    parser.add_argument("--gui", action="store_true", help="Open the interactive point-click GUI.")
    parser.add_argument("--batch-gui", action="store_true", help="Open the batch folder GUI.")
    parser.add_argument("--input", "--input-dir", dest="input_path", type=Path)
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=Path)
    parser.add_argument("--recursive", action="store_true", help="Search subfolders too.")
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma or space separated image extensions.",
    )
    parser.add_argument(
        "--model-size",
        choices=tuple(MODEL_PRESETS.keys()),
        default="large",
        help="SAM 2.1 checkpoint preset.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Override checkpoint path.")
    parser.add_argument("--model-cfg", help="Override SAM 2 config name.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--points-per-side", type=positive_int, default=32)
    parser.add_argument("--points-per-batch", type=positive_int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.8)
    parser.add_argument("--stability-score-thresh", type=float, default=0.95)
    parser.add_argument("--min-mask-region-area", type=non_negative_int, default=0)
    parser.add_argument("--alpha", type=alpha_value, default=0.55)
    parser.add_argument("--max-masks", type=non_negative_int, default=0)
    parser.add_argument("--save-individual-masks", action="store_true")
    parser.add_argument(
        "--export-format",
        choices=EXPORT_FORMATS,
        default="yolo",
        help="Training label export format saved under labels/.",
    )
    parser.add_argument(
        "--no-yolo-labels",
        action="store_true",
        help="Deprecated: same as --export-format mask unless explicitly overridden.",
    )
    parser.add_argument("--yolo-class-id", type=non_negative_int, default=0)
    parser.add_argument(
        "--yolo-epsilon",
        type=non_negative_float,
        default=2.0,
        help="Polygon simplification tolerance in pixels.",
    )
    parser.add_argument("--yolo-min-area", type=non_negative_float, default=8.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List work without loading SAM 2.")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> SamBatchConfig:
    if args.input_path is None:
        parser.error("--input is required unless --gui or --batch-gui is used.")
    if args.output_dir is None:
        parser.error("--output is required unless --gui or --batch-gui is used.")

    export_format = args.export_format
    if args.no_yolo_labels and export_format in {"yolo", "both"}:
        export_format = "mask"

    return SamBatchConfig(
        input_path=args.input_path,
        output_dir=args.output_dir,
        model_size=args.model_size,
        checkpoint=args.checkpoint,
        model_cfg=args.model_cfg,
        device=args.device,
        recursive=args.recursive,
        extensions=parse_extensions(args.extensions),
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        min_mask_region_area=args.min_mask_region_area,
        alpha=args.alpha,
        max_masks=args.max_masks,
        save_individual_masks=args.save_individual_masks,
        export_format=export_format,
        yolo_class_id=args.yolo_class_id,
        yolo_epsilon=args.yolo_epsilon,
        yolo_min_area=args.yolo_min_area,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        stop_on_error=args.stop_on_error,
    )


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        run_interactive_gui()
        return 0

    parser = create_parser()
    args = parser.parse_args(argv)
    if args.gui:
        run_interactive_gui()
        return 0
    if args.batch_gui:
        from utils.tk_batch_gui import run_batch_gui

        run_batch_gui()
        return 0

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    config = config_from_args(args, parser)

    try:
        results = process_batch(config)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 1 if any(result.status == "error" for result in results) else 0
