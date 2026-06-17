from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - keeps the main app importable without OpenCV.
    cv2 = None

from utils.io_utils import save_png


MIN_HOUGH_CONFIDENCE = 2.05


@dataclass
class CircleDetection:
    x: float
    y: float
    radius: float
    method: str
    confidence: float


@dataclass
class CropBox:
    x1: int
    y1: int
    x2: int
    y2: int
    method: str


@dataclass
class HoughPreprocessSettings:
    work_size: int = 1280
    inner_radius_scale: float = 0.86
    crop_radius_scale: float = 0.55
    crop_size: int = 0
    no_circle_crop_padding: float = 0.35


@dataclass
class HoughPreprocessResult:
    full_rgb: np.ndarray
    crop_rgb: np.ndarray
    mask: np.ndarray
    debug_rgb: np.ndarray
    mode: str
    method: str
    metadata: dict[str, str | float | int]


def require_cv2():
    if cv2 is None:
        raise RuntimeError("Hough preprocessing needs opencv-python. Install it with: pip install opencv-python")
    return cv2


def resize_for_work(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    cv = require_cv2()
    h, w = image.shape[:2]
    if max_side <= 0:
        return image.copy(), 1.0
    scale = min(1.0, max_side / float(max(h, w)))
    if scale >= 1.0:
        return image.copy(), 1.0
    resized = cv.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv.INTER_AREA)
    return resized, scale


def odd_kernel_size(min_side: int, ratio: float, minimum: int, maximum: int) -> int:
    size = int(round(min_side * ratio))
    size = max(minimum, min(maximum, size))
    if size % 2 == 0:
        size += 1
    return size


def fitted_square_bounds(
    center_x: float,
    center_y: float,
    side: float,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    h, w = image_shape
    side_i = max(1, int(round(side)))
    side_i = min(side_i, max(1, w), max(1, h))
    x1 = int(round(center_x - side_i / 2.0))
    y1 = int(round(center_y - side_i / 2.0))
    x1 = max(0, min(x1, max(0, w - side_i)))
    y1 = max(0, min(y1, max(0, h - side_i)))
    return x1, y1, x1 + side_i, y1 + side_i


def circle_edge_support(edge_map: np.ndarray, x: float, y: float, radius: float) -> float:
    h, w = edge_map.shape[:2]
    angles = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    xs = np.rint(x + radius * np.cos(angles)).astype(np.int32)
    ys = np.rint(y + radius * np.sin(angles)).astype(np.int32)
    valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if valid.sum() < 32:
        return 0.0
    return float((edge_map[ys[valid], xs[valid]] > 0).mean())


def hough_candidates(gray: np.ndarray, edge_map: np.ndarray) -> list[CircleDetection]:
    cv = require_cv2()
    h, w = gray.shape[:2]
    min_side = min(h, w)
    aspect = max(h, w) / float(max(1, min_side))
    min_radius_ratio = 0.28 if aspect >= 1.6 else 0.32
    max_radius_ratio = 0.78
    min_dist = max(24, int(min_side * 0.30))
    candidates: list[CircleDetection] = []
    sources = [
        cv.GaussianBlur(gray, (0, 0), 2),
        cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray),
    ]

    hough_params = ((65, 14), (85, 18), (105, 26), (130, 36))
    for source in sources:
        for param1, param2 in hough_params:
            circles = cv.HoughCircles(
                source,
                cv.HOUGH_GRADIENT,
                dp=1.2,
                minDist=min_dist,
                param1=param1,
                param2=param2,
                minRadius=max(8, int(min_side * min_radius_ratio)),
                maxRadius=max(12, int(min_side * max_radius_ratio)),
            )
            if circles is None:
                continue
            for raw_x, raw_y, raw_r in circles[0]:
                center_offset = float(np.hypot((raw_x - w / 2.0) / w, (raw_y - h / 2.0) / h))
                center_limit = 0.42 if aspect >= 1.6 else 0.34
                if center_offset > center_limit:
                    continue
                support = circle_edge_support(edge_map, raw_x, raw_y, raw_r)
                if support < 0.14:
                    continue
                radius_score = raw_r / float(min_side)
                centrality = max(0.0, 1.0 - center_offset / center_limit)
                candidates.append(
                    CircleDetection(
                        x=float(raw_x),
                        y=float(raw_y),
                        radius=float(raw_r),
                        method="hough",
                        confidence=float(support * 4.0 + radius_score * 0.8 + centrality),
                    )
                )
    return candidates


def fit_dark_stage_circle(image: np.ndarray) -> CircleDetection | None:
    cv = require_cv2()
    h, w = image.shape[:2]
    min_side = min(h, w)
    lab = cv.cvtColor(image, cv.COLOR_RGB2LAB)
    luma = lab[:, :, 0]
    _threshold, dark = cv.threshold(luma, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    close_size = odd_kernel_size(min_side, 0.035, 9, 51)
    open_size = odd_kernel_size(min_side, 0.015, 5, 25)
    dark = cv.morphologyEx(dark, cv.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    dark = cv.morphologyEx(dark, cv.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    contours, _hierarchy = cv.findContours(dark, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    best: tuple[float, CircleDetection] | None = None
    for contour in contours:
        area = cv.contourArea(contour)
        if area < h * w * 0.12:
            continue
        (x, y), radius = cv.minEnclosingCircle(contour)
        if radius < min_side * 0.28 or radius > min_side * 0.78:
            continue
        center_offset = float(np.hypot((x - w / 2.0) / w, (y - h / 2.0) / h))
        if center_offset > 0.42:
            continue
        confidence = area / float(h * w) + max(0.0, 1.0 - center_offset / 0.42)
        detection = CircleDetection(float(x), float(y), float(radius), "fit_dark", float(confidence))
        if best is None or confidence > best[0]:
            best = (confidence, detection)
    return best[1] if best else None


def detect_stage_circle(image: np.ndarray, work_size: int) -> CircleDetection | None:
    cv = require_cv2()
    work, scale = resize_for_work(image, work_size)
    gray = cv.cvtColor(work, cv.COLOR_RGB2GRAY)
    gray = cv.medianBlur(gray, 5)
    edges = cv.Canny(gray, 50, 140)
    edges = cv.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    candidates = hough_candidates(gray, edges)
    hough_best = max(candidates, key=lambda item: item.confidence) if candidates else None
    best = hough_best if hough_best is not None and hough_best.confidence >= MIN_HOUGH_CONFIDENCE else fit_dark_stage_circle(work)
    if best is None:
        return None

    return CircleDetection(
        x=best.x / scale,
        y=best.y / scale,
        radius=best.radius / scale,
        method=best.method,
        confidence=best.confidence,
    )


def square_box_from_bounds(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_shape: tuple[int, int],
    padding: float,
) -> CropBox:
    h, w = image_shape
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    side = int(round(max(box_w, box_h) * (1.0 + padding * 2.0)))
    side = max(side, int(round(min(h, w) * 0.20)))
    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))
    fitted = fitted_square_bounds(cx, cy, side, image_shape)
    method = "foreground_bbox_fit" if fitted[2] - fitted[0] < side else "foreground_bbox"
    return CropBox(*fitted, method)


def detect_foreground_crop_box(image: np.ndarray, work_size: int, padding: float) -> CropBox:
    cv = require_cv2()
    work, scale = resize_for_work(image, work_size)
    h, w = work.shape[:2]
    min_side = min(h, w)
    hsv = cv.cvtColor(work, cv.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    chroma = work.max(axis=2).astype(np.int16) - work.min(axis=2).astype(np.int16)
    gray = cv.cvtColor(work, cv.COLOR_RGB2GRAY)
    edges = cv.Canny(gray, 35, 110)

    candidate = (((saturation > 35) & (value > 22)) | ((chroma > 20) & (value > 18)) | (edges > 0)).astype(np.uint8)
    open_size = odd_kernel_size(min_side, 0.006, 3, 9)
    dilate_size = odd_kernel_size(min_side, 0.022, 7, 31)
    close_size = odd_kernel_size(min_side, 0.035, 9, 45)
    candidate = cv.morphologyEx(candidate, cv.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    candidate = cv.dilate(candidate, np.ones((dilate_size, dilate_size), np.uint8), iterations=1)
    candidate = cv.morphologyEx(candidate, cv.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))

    count, labels, stats, centroids = cv.connectedComponentsWithStats(candidate, 8)
    keep = np.zeros((h, w), dtype=np.uint8)
    min_area = h * w * 0.0008
    for label in range(1, count):
        area = stats[label, cv.CC_STAT_AREA]
        if area < min_area:
            continue
        dx = (centroids[label][0] - w / 2.0) / max(w, 1)
        dy = (centroids[label][1] - h / 2.0) / max(h, 1)
        if np.hypot(dx, dy) > 0.46:
            continue
        keep[labels == label] = 1

    ys, xs = np.where(keep > 0)
    if len(xs) == 0:
        full_h, full_w = image.shape[:2]
        side = int(round(min(full_h, full_w) * 0.60))
        cx = full_w // 2
        cy = full_h // 2
        half = side // 2
        return CropBox(cx - half, cy - half, cx - half + side, cy - half + side, "center_fallback")

    x1 = int(np.floor(xs.min() / scale))
    y1 = int(np.floor(ys.min() / scale))
    x2 = int(np.ceil((xs.max() + 1) / scale))
    y2 = int(np.ceil((ys.max() + 1) / scale))
    return square_box_from_bounds(x1, y1, x2, y2, image.shape[:2], padding)


def make_circle_mask(shape: tuple[int, int], circle: CircleDetection, radius_scale: float) -> np.ndarray:
    cv = require_cv2()
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    center = (int(round(circle.x)), int(round(circle.y)))
    radius = max(1, int(round(circle.radius * radius_scale)))
    cv.circle(mask, center, radius, 255, -1)
    return mask


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = image.copy()
    output[mask == 0] = 0
    return output


def crop_square(
    image: np.ndarray,
    circle: CircleDetection,
    radius_scale: float,
    output_size: int | None,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    cv = require_cv2()
    h, w = image.shape[:2]
    half_size = max(1, int(round(circle.radius * radius_scale)))
    x1, y1, x2, y2 = fitted_square_bounds(circle.x, circle.y, half_size * 2, image.shape[:2])

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    safe_x1 = max(0, x1)
    safe_y1 = max(0, y1)
    safe_x2 = min(w, x2)
    safe_y2 = min(h, y2)
    crop = image[safe_y1:safe_y2, safe_x1:safe_x2]
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        crop = cv.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    if output_size and output_size > 0:
        crop = cv.resize(crop, (output_size, output_size), interpolation=cv.INTER_AREA)
    return crop, (x1, y1, x2, y2)


def crop_from_box(image: np.ndarray, box: CropBox, output_size: int | None) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    cv = require_cv2()
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - w)
    pad_bottom = max(0, y2 - h)

    safe_x1 = max(0, x1)
    safe_y1 = max(0, y1)
    safe_x2 = min(w, x2)
    safe_y2 = min(h, y2)
    crop = image[safe_y1:safe_y2, safe_x1:safe_x2]
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        crop = cv.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv.BORDER_CONSTANT, value=(0, 0, 0))
    if output_size and output_size > 0:
        crop = cv.resize(crop, (output_size, output_size), interpolation=cv.INTER_AREA)
    return crop, (x1, y1, x2, y2)


def make_debug_image(
    image: np.ndarray,
    circle: CircleDetection,
    inner_radius_scale: float,
    crop_radius_scale: float,
    max_side: int = 1400,
) -> np.ndarray:
    cv = require_cv2()
    debug, scale = resize_for_work(image, max_side)
    c = CircleDetection(
        x=circle.x * scale,
        y=circle.y * scale,
        radius=circle.radius * scale,
        method=circle.method,
        confidence=circle.confidence,
    )
    center = (int(round(c.x)), int(round(c.y)))
    cv.circle(debug, center, int(round(c.radius)), (0, 255, 0), 3)
    cv.circle(debug, center, int(round(c.radius * inner_radius_scale)), (255, 255, 0), 3)
    half_crop = int(round(c.radius * crop_radius_scale))
    cv.rectangle(
        debug,
        (center[0] - half_crop, center[1] - half_crop),
        (center[0] + half_crop, center[1] + half_crop),
        (0, 128, 255),
        3,
    )
    label = f"{c.method} r={circle.radius:.1f} conf={circle.confidence:.2f}"
    cv.putText(debug, label, (24, 48), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv.LINE_AA)
    cv.putText(debug, label, (24, 48), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1, cv.LINE_AA)
    return debug


def make_no_circle_debug_image(image: np.ndarray, crop_box: CropBox, max_side: int = 1400) -> np.ndarray:
    cv = require_cv2()
    debug, scale = resize_for_work(image, max_side)
    x1 = int(round(crop_box.x1 * scale))
    y1 = int(round(crop_box.y1 * scale))
    x2 = int(round(crop_box.x2 * scale))
    y2 = int(round(crop_box.y2 * scale))
    cv.rectangle(debug, (x1, y1), (x2, y2), (0, 128, 255), 3)
    label = f"no_circle {crop_box.method}"
    cv.putText(debug, label, (24, 48), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv.LINE_AA)
    cv.putText(debug, label, (24, 48), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 1, cv.LINE_AA)
    return debug


def preprocess_hough_circle(
    image_rgb: np.ndarray,
    settings: HoughPreprocessSettings | None = None,
) -> HoughPreprocessResult:
    require_cv2()
    settings = settings or HoughPreprocessSettings()
    image = np.ascontiguousarray(image_rgb)
    circle = detect_stage_circle(image, settings.work_size)

    if circle is None:
        mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        masked = image.copy()
        no_circle_crop_box = detect_foreground_crop_box(image, settings.work_size, settings.no_circle_crop_padding)
        crop, crop_box = crop_from_box(masked, no_circle_crop_box, settings.crop_size)
        debug = make_no_circle_debug_image(image, no_circle_crop_box)
        metadata: dict[str, str | float | int] = {
            "mode": "no_circle",
            "method": no_circle_crop_box.method,
            "circle_x": "",
            "circle_y": "",
            "circle_radius": "",
            "inner_radius": "",
            "confidence": "",
            "crop_x1": crop_box[0],
            "crop_y1": crop_box[1],
            "crop_x2": crop_box[2],
            "crop_y2": crop_box[3],
        }
        return HoughPreprocessResult(masked, crop, mask, debug, "no_circle", no_circle_crop_box.method, metadata)

    mask = make_circle_mask(image.shape[:2], circle, settings.inner_radius_scale)
    masked = apply_mask(image, mask)
    crop, crop_box = crop_square(masked, circle, settings.crop_radius_scale, settings.crop_size)
    debug = make_debug_image(image, circle, settings.inner_radius_scale, settings.crop_radius_scale)
    metadata = {
        "mode": "circle",
        "method": circle.method,
        "circle_x": round(circle.x, 3),
        "circle_y": round(circle.y, 3),
        "circle_radius": round(circle.radius, 3),
        "inner_radius": round(circle.radius * settings.inner_radius_scale, 3),
        "confidence": round(circle.confidence, 5),
        "crop_x1": crop_box[0],
        "crop_y1": crop_box[1],
        "crop_x2": crop_box[2],
        "crop_y2": crop_box[3],
    }
    return HoughPreprocessResult(masked, crop, mask, debug, "circle", circle.method, metadata)


def hough_output_name(image_path: Path, kind: str) -> str:
    return f"{image_path.stem}_hough_{kind}.png"


def hough_result_image(result: HoughPreprocessResult, kind: str) -> np.ndarray:
    if kind == "crop":
        return result.crop_rgb
    if kind == "debug":
        return result.debug_rgb
    return result.full_rgb


def save_hough_preprocess_result(
    image_path: Path,
    output_dir: Path,
    result: HoughPreprocessResult,
) -> dict[str, Path]:
    base = output_dir / "preprocess"
    stem = image_path.stem
    paths = {
        "full": base / "full" / hough_output_name(image_path, "full"),
        "crop": base / "crop" / hough_output_name(image_path, "crop"),
        "mask": base / "mask" / hough_output_name(image_path, "mask"),
        "debug": base / "debug" / hough_output_name(image_path, "debug"),
        "metadata": base / "metadata" / f"{stem}_hough.csv",
    }

    save_png(result.full_rgb, paths["full"])
    save_png(result.crop_rgb, paths["crop"])
    save_png(result.mask, paths["mask"])
    save_png(result.debug_rgb, paths["debug"])
    paths["metadata"].parent.mkdir(parents=True, exist_ok=True)
    with paths["metadata"].open("w", newline="", encoding="utf-8-sig") as file:
        fieldnames = ["input", "full", "crop", "mask", "debug", *result.metadata.keys()]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "input": str(image_path),
                "full": str(paths["full"]),
                "crop": str(paths["crop"]),
                "mask": str(paths["mask"]),
                "debug": str(paths["debug"]),
                **result.metadata,
            }
        )
    return paths
