from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence
from utils.config import SavedObject
from utils.io_utils import (
    color_for_index,
    save_png,
    save_training_image,
    wants_mask_rcnn_export,
    wants_mask_export,
    wants_yolo_export,
)

import csv
import json
import os

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None


def sorted_annotations(annotations: list[dict], max_masks: int) -> list[dict]:
    ordered = sorted(annotations, key=lambda ann: int(ann.get("area", 0)), reverse=True)
    if max_masks > 0:
        return ordered[:max_masks]
    return ordered


def render_masks(
    image: np.ndarray,
    annotations: list[dict],
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    overlay = image.astype(np.float32).copy()
    label_dtype = np.uint16 if len(annotations) <= np.iinfo(np.uint16).max else np.uint32
    label_mask = np.zeros(image.shape[:2], dtype=label_dtype)
    color_mask = np.zeros_like(image, dtype=np.uint8)

    for mask_id, annotation in enumerate(annotations, start=1):
        mask = np.asarray(annotation["segmentation"], dtype=bool)
        if mask.shape != label_mask.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match image shape {label_mask.shape}."
            )

        color = color_for_index(mask_id)
        overlay[mask] = overlay[mask] * (1.0 - alpha) + color.astype(np.float32) * alpha
        label_mask[mask] = mask_id
        color_mask[mask] = color

    return np.clip(overlay, 0, 255).astype(np.uint8), label_mask, color_mask


def safe_float(value: object) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def json_numbers(values: Sequence[object]) -> str:
    return json.dumps([safe_float(value) for value in values], ensure_ascii=False)


def write_objects_csv(path: Path, annotations: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "mask_id",
        "area",
        "bbox_xywh",
        "predicted_iou",
        "stability_score",
        "point_coords",
        "crop_box_xywh",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for mask_id, annotation in enumerate(annotations, start=1):
            point_coords = annotation.get("point_coords", [])
            writer.writerow(
                {
                    "mask_id": mask_id,
                    "area": int(annotation.get("area", 0)),
                    "bbox_xywh": json_numbers(annotation.get("bbox", [])),
                    "predicted_iou": safe_float(annotation.get("predicted_iou", 0.0)),
                    "stability_score": safe_float(annotation.get("stability_score", 0.0)),
                    "point_coords": json.dumps(point_coords, ensure_ascii=False),
                    "crop_box_xywh": json_numbers(annotation.get("crop_box", [])),
                }
            )


def blend_binary_mask(canvas: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> np.ndarray:
    output = canvas.astype(np.float32).copy()
    output[mask] = output[mask] * (1.0 - alpha) + color.astype(np.float32) * alpha
    return np.clip(output, 0, 255).astype(np.uint8)


def render_saved_overlay(
    image: np.ndarray,
    saved_objects: Sequence[SavedObject],
    alpha: float = 0.5,
) -> np.ndarray:
    overlay = image.copy()
    for saved in saved_objects:
        overlay = blend_binary_mask(overlay, saved.mask.astype(bool), saved.color, alpha)
    return overlay


def render_interactive_overlay(
    image: np.ndarray,
    saved_objects: Sequence[SavedObject],
    current_mask: np.ndarray | None,
    current_color: np.ndarray,
    saved_alpha: float = 0.45,
    current_alpha: float = 0.58,
) -> np.ndarray:
    overlay = render_saved_overlay(image, saved_objects, saved_alpha)
    if current_mask is not None:
        overlay = blend_binary_mask(overlay, current_mask.astype(bool), current_color, current_alpha)
    return overlay


def render_yolo_polygon_overlay(
    image: np.ndarray,
    saved_objects: Sequence[SavedObject],
    current_mask: np.ndarray | None = None,
    current_color: np.ndarray | None = None,
    current_class_id: int = 0,
    current_yolo_polygons: Sequence[dict[str, object]] | None = None,
    epsilon: float = 2.0,
    min_area: float = 8.0,
    fill_alpha: float = 0.18,
) -> tuple[np.ndarray, int]:
    overlay = image.copy()
    height, width = image.shape[:2]
    items: list[tuple[list[list[tuple[float, float]]], np.ndarray, int]] = [
        (yolo_export_polygons_for_saved_object(saved, epsilon=epsilon, min_area=min_area), saved.color, saved.class_id)
        for saved in saved_objects
    ]
    if current_mask is not None:
        color = current_color if current_color is not None else np.array([56, 217, 197], dtype=np.uint8)
        current_polygons = [
            [(float(x), float(y)) for x, y in item["points"]]  # type: ignore[index]
            for item in normalize_yolo_edit_polygons(current_yolo_polygons)
            if item["mode"] == "add"
        ]
        if not current_polygons:
            current_polygons = mask_to_polygons(current_mask.astype(bool), epsilon=epsilon, min_area=min_area)
        items.append((current_polygons, color, int(current_class_id)))

    polygon_count = 0
    for polygons, color, class_id in items:
        polygon_count += len(polygons)
        for polygon in polygons:
            points = np.rint(np.asarray(polygon, dtype=np.float32)).astype(np.int32)
            if len(points) < 3:
                continue

            color_tuple = tuple(int(value) for value in color.tolist())
            polygon_mask = np.zeros((height, width), dtype=np.uint8)
            if cv2 is not None:
                cv2.fillPoly(polygon_mask, [points.reshape((-1, 1, 2))], 255)
                overlay[polygon_mask > 0] = (
                    overlay[polygon_mask > 0].astype(np.float32) * (1.0 - fill_alpha)
                    + color.astype(np.float32) * fill_alpha
                ).astype(np.uint8)
                cv2.polylines(
                    overlay,
                    [points.reshape((-1, 1, 2))],
                    isClosed=True,
                    color=color_tuple,
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )
                for x, y in points:
                    cv2.circle(overlay, (int(x), int(y)), 4, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                    cv2.circle(overlay, (int(x), int(y)), 4, color_tuple, 1, lineType=cv2.LINE_AA)
                x0, y0 = points[0]
                cv2.putText(
                    overlay,
                    f"class {class_id}",
                    (int(x0) + 6, int(y0) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color_tuple,
                    2,
                    cv2.LINE_AA,
                )
            else:
                from PIL import Image, ImageDraw

                rgba = Image.fromarray(overlay).convert("RGBA")
                draw = ImageDraw.Draw(rgba, "RGBA")
                xy = [(float(x), float(y)) for x, y in points]
                draw.polygon(xy, fill=(*color_tuple, int(255 * fill_alpha)))
                draw.line(xy + [xy[0]], fill=(*color_tuple, 255), width=3)
                for x, y in xy:
                    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(255, 255, 255, 255), outline=(*color_tuple, 255))
                draw.text((xy[0][0] + 6, xy[0][1] - 16), f"class {class_id}", fill=(*color_tuple, 255))
                overlay = np.array(rgba.convert("RGB"), dtype=np.uint8)

    return np.clip(overlay, 0, 255).astype(np.uint8), polygon_count


def render_yolo_edit_polygon_overlay(
    image: np.ndarray,
    saved_objects: Sequence[SavedObject],
    selected_object_index: int = -2,
    selected_polygon_index: int = -1,
    selected_vertex_index: int = -1,
    draft_polygon: Sequence[tuple[float, float]] | None = None,
    draft_mode: str = "add",
    current_yolo_polygons: Sequence[dict[str, object]] | None = None,
    current_color: np.ndarray | None = None,
) -> np.ndarray:
    overlay = image.copy()
    if cv2 is None:
        return overlay

    items: list[tuple[int, np.ndarray, Sequence[dict[str, object]]]] = []
    if current_yolo_polygons is not None:
        color = current_color if current_color is not None else np.array([56, 217, 197], dtype=np.uint8)
        items.append((-1, color.astype(np.uint8), current_yolo_polygons))
    for object_index, saved in enumerate(saved_objects):
        items.append((object_index, saved.color.astype(np.uint8), saved.yolo_polygons))

    for object_index, color, polygons in items:
        selected_object = object_index == selected_object_index
        for polygon_index, item in enumerate(normalize_yolo_edit_polygons(polygons)):
            points = np.rint(np.asarray(item["points"], dtype=np.float32)).astype(np.int32)
            if len(points) < 2:
                continue
            mode = item["mode"]
            if mode == "subtract":
                color_tuple = (255, 210, 80)
            else:
                color_tuple = tuple(int(value) for value in color.tolist())
            thickness = 3 if selected_object and polygon_index == selected_polygon_index else 2
            cv2.polylines(
                overlay,
                [points.reshape((-1, 1, 2))],
                isClosed=True,
                color=color_tuple,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )
            if selected_object:
                for vertex_index, (x, y) in enumerate(points):
                    radius = 6 if polygon_index == selected_polygon_index and vertex_index == selected_vertex_index else 4
                    cv2.circle(overlay, (int(x), int(y)), radius, (255, 255, 255), -1, lineType=cv2.LINE_AA)
                    cv2.circle(overlay, (int(x), int(y)), radius, color_tuple, 2, lineType=cv2.LINE_AA)

    if draft_polygon:
        points = np.rint(np.asarray(draft_polygon, dtype=np.float32)).astype(np.int32)
        color_tuple = (255, 210, 80) if draft_mode == "subtract" else (56, 217, 197)
        if len(points) >= 2:
            cv2.polylines(
                overlay,
                [points.reshape((-1, 1, 2))],
                isClosed=False,
                color=color_tuple,
                thickness=2,
                lineType=cv2.LINE_AA,
            )
        for x, y in points:
            cv2.circle(overlay, (int(x), int(y)), 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(overlay, (int(x), int(y)), 5, color_tuple, 2, lineType=cv2.LINE_AA)
    return overlay


def build_color_mask(saved_objects: Sequence[SavedObject]) -> np.ndarray:
    if not saved_objects:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    height, width = saved_objects[0].mask.shape
    color_mask = np.zeros((height, width, 3), dtype=np.uint8)
    for saved in saved_objects:
        color_mask[saved.mask.astype(bool)] = saved.color
    return color_mask


def build_class_label_mask(
    masks: Sequence[np.ndarray],
    class_ids: Sequence[int],
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> np.ndarray:
    height = int(image_shape[0])
    width = int(image_shape[1])
    max_value = max([class_id + 1 for class_id in class_ids], default=1)
    dtype = np.uint8 if max_value <= np.iinfo(np.uint8).max else np.uint16
    label_mask = np.zeros((height, width), dtype=dtype)
    for mask, class_id in zip(masks, class_ids):
        label_mask[np.asarray(mask, dtype=bool)] = int(class_id) + 1
    return label_mask


def normalize_yolo_edit_polygons(raw_polygons: Sequence[dict[str, object]] | None) -> list[dict[str, object]]:
    polygons: list[dict[str, object]] = []
    for item in raw_polygons or []:
        mode = str(item.get("mode", "add"))
        if mode not in {"add", "subtract"}:
            mode = "add"
        points = []
        for raw_point in item.get("points", []):  # type: ignore[union-attr]
            try:
                x, y = raw_point  # type: ignore[misc]
                points.append((float(x), float(y)))
            except Exception:
                continue
        if len(points) >= 3:
            polygons.append({"mode": mode, "points": points})
    return polygons


def mask_to_yolo_edit_polygons(mask: np.ndarray, epsilon: float = 2.0, min_area: float = 8.0) -> list[dict[str, object]]:
    return [{"mode": "add", "points": polygon} for polygon in mask_to_polygons(mask, epsilon=epsilon, min_area=min_area)]


def mask_to_edit_polygons(mask: np.ndarray, epsilon: float = 2.0, min_area: float = 8.0) -> list[dict[str, object]]:
    mask_u8 = np.asarray(mask, dtype=bool).astype(np.uint8) * 255
    polygons: list[dict[str, object]] = []

    if cv2 is not None:
        contours, hierarchy = cv2.findContours(mask_u8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            return []
        for index, contour in enumerate(contours):
            area = float(cv2.contourArea(contour))
            if area <= min_area:
                continue
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 3:
                continue
            points = [(float(point[0][0]), float(point[0][1])) for point in approx]
            mode = "add" if int(hierarchy[0][index][3]) == -1 else "subtract"
            if polygon_area(points) < 0:
                points.reverse()
            polygons.append({"mode": mode, "points": points})
        polygons.sort(key=lambda item: abs(polygon_area(item["points"])), reverse=True)  # type: ignore[arg-type]
        return polygons

    for polygon in mask_to_polygons(mask, epsilon=epsilon, min_area=min_area):
        polygons.append({"mode": "add", "points": polygon})
    return polygons


def yolo_edit_polygons_to_mask(
    polygons: Sequence[dict[str, object]],
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> np.ndarray:
    height = int(image_shape[0])
    width = int(image_shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    normalized = normalize_yolo_edit_polygons(polygons)
    if cv2 is not None:
        for item in normalized:
            points = np.rint(np.asarray(item["points"], dtype=np.float32)).astype(np.int32)
            if len(points) < 3:
                continue
            value = 255 if item["mode"] == "add" else 0
            cv2.fillPoly(mask, [points.reshape((-1, 1, 2))], value)
        return mask.astype(bool)

    from PIL import Image, ImageDraw

    image = Image.fromarray(mask, mode="L")
    draw = ImageDraw.Draw(image)
    for item in normalized:
        value = 255 if item["mode"] == "add" else 0
        draw.polygon([(float(x), float(y)) for x, y in item["points"]], fill=value)
    return np.array(image, dtype=np.uint8).astype(bool)


def edit_polygons_to_mask(
    polygons: Sequence[dict[str, object]],
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> np.ndarray:
    return yolo_edit_polygons_to_mask(polygons, image_shape)


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    rows, cols = np.where(mask.astype(bool))
    if len(rows) == 0 or len(cols) == 0:
        return [0, 0, 0, 0]
    x_min = int(cols.min())
    y_min = int(rows.min())
    x_max = int(cols.max())
    y_max = int(rows.max())
    return [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]


def bbox_xyxy_from_mask(mask: np.ndarray) -> list[int]:
    x, y, width, height = bbox_from_mask(mask)
    if width <= 0 or height <= 0:
        return [0, 0, 0, 0]
    return [x, y, x + width - 1, y + height - 1]


def coco_rle_from_mask(mask: np.ndarray) -> tuple[dict[str, object], int, list[float]] | None:
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        return None

    mask_u8 = np.asfortranarray(np.asarray(mask, dtype=np.uint8))
    encoded = mask_utils.encode(mask_u8)
    area = int(mask_utils.area(encoded))
    bbox = [float(value) for value in mask_utils.toBbox(encoded).tolist()]
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(mask_u8.shape[0]), int(mask_u8.shape[1])], "counts": counts}, area, bbox


def polygon_area(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return area / 2.0


def point_line_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    dx = ex - sx
    dy = ey - sy
    if dx == 0 and dy == 0:
        return float(((px - sx) ** 2 + (py - sy) ** 2) ** 0.5)
    return abs(dy * px - dx * py + ex * sy - ey * sx) / float((dx * dx + dy * dy) ** 0.5)


def simplify_polyline(
    points: Sequence[tuple[float, float]],
    epsilon: float,
) -> list[tuple[float, float]]:
    if len(points) <= 2 or epsilon <= 0:
        return list(points)

    start = points[0]
    end = points[-1]
    max_distance = 0.0
    max_index = 0
    for index in range(1, len(points) - 1):
        distance = point_line_distance(points[index], start, end)
        if distance > max_distance:
            max_distance = distance
            max_index = index

    if max_distance > epsilon:
        left = simplify_polyline(points[: max_index + 1], epsilon)
        right = simplify_polyline(points[max_index:], epsilon)
        return left[:-1] + right
    return [start, end]


def simplify_polygon(
    points: Sequence[tuple[float, float]],
    epsilon: float,
) -> list[tuple[float, float]]:
    if len(points) < 4:
        return list(points)
    closed = list(points) + [points[0]]
    simplified = simplify_polyline(closed, epsilon)
    if simplified and simplified[-1] == simplified[0]:
        simplified = simplified[:-1]
    return simplified if len(simplified) >= 3 else list(points)


def mask_boundary_segments(mask: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    binary = mask.astype(bool)
    padded = np.pad(binary, 1, constant_values=False)
    segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for row, col in np.argwhere(binary):
        pr = int(row) + 1
        pc = int(col) + 1
        y = int(row)
        x = int(col)
        if not padded[pr - 1, pc]:
            segments.append(((x, y), (x + 1, y)))
        if not padded[pr, pc + 1]:
            segments.append(((x + 1, y), (x + 1, y + 1)))
        if not padded[pr + 1, pc]:
            segments.append(((x + 1, y + 1), (x, y + 1)))
        if not padded[pr, pc - 1]:
            segments.append(((x, y + 1), (x, y)))
    return segments


def connect_boundary_segments(
    segments: Sequence[tuple[tuple[int, int], tuple[int, int]]],
) -> list[list[tuple[float, float]]]:
    outgoing: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    unused = set(segments)
    for start, end in segments:
        outgoing[start].append(end)

    loops: list[list[tuple[float, float]]] = []
    while unused:
        start, end = unused.pop()
        loop = [start]
        current = end
        for _ in range(len(segments) + 1):
            if current == start:
                if len(loop) >= 3:
                    loops.append([(float(x), float(y)) for x, y in loop])
                break
            loop.append(current)
            next_point = None
            for candidate in outgoing.get(current, []):
                edge = (current, candidate)
                if edge in unused:
                    next_point = candidate
                    unused.remove(edge)
                    break
            if next_point is None:
                break
            current = next_point
    return loops


def mask_to_polygons(mask: np.ndarray, epsilon: float = 2.0, min_area: float = 8.0) -> list[list[tuple[float, float]]]:
    if cv2 is not None:
        mask_u8 = np.asarray(mask, dtype=np.uint8) * 255
        contours, _hierarchy = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= min_area:
                continue
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) < 3:
                continue
            polygon = [(float(point[0][0]), float(point[0][1])) for point in approx]
            if polygon_area(polygon) < 0:
                polygon.reverse()
            if abs(polygon_area(polygon)) > min_area:
                polygons.append(polygon)
        polygons.sort(key=lambda polygon: abs(polygon_area(polygon)), reverse=True)
        return polygons

    segments = mask_boundary_segments(mask)
    polygons: list[list[tuple[float, float]]] = []
    for loop in connect_boundary_segments(segments):
        area = polygon_area(loop)
        if area <= min_area:
            continue
        simplified = simplify_polygon(loop, epsilon)
        if len(simplified) < 3 or polygon_area(simplified) <= min_area:
            continue
        polygons.append(simplified)
    polygons.sort(key=lambda polygon: polygon_area(polygon), reverse=True)
    return polygons


def yolo_export_polygons_for_saved_object(
    saved: SavedObject,
    epsilon: float = 2.0,
    min_area: float = 8.0,
) -> list[list[tuple[float, float]]]:
    polygons = []
    for item in normalize_yolo_edit_polygons(saved.yolo_polygons):
        if item["mode"] != "add":
            continue
        points = [(float(x), float(y)) for x, y in item["points"]]  # type: ignore[index]
        if len(points) >= 3 and abs(polygon_area(points)) > min_area:
            polygons.append(points)
    if polygons:
        return polygons
    return mask_to_polygons(saved.mask, epsilon=epsilon, min_area=min_area)


def format_yolo_polygon_line(
    class_id: int,
    polygon: Sequence[tuple[float, float]],
    width: int,
    height: int,
) -> str:
    values = [str(int(class_id))]
    for x, y in polygon:
        norm_x = min(1.0, max(0.0, float(x) / float(width)))
        norm_y = min(1.0, max(0.0, float(y) / float(height)))
        values.append(f"{norm_x:.6f}")
        values.append(f"{norm_y:.6f}")
    return " ".join(values)


def write_yolo_segmentation_label(
    path: Path,
    masks: Sequence[np.ndarray],
    class_ids: Sequence[int],
    image_shape: tuple[int, int] | tuple[int, int, int],
    epsilon: float = 2.0,
    min_area: float = 8.0,
) -> int:
    height = int(image_shape[0])
    width = int(image_shape[1])
    lines: list[str] = []
    for mask, class_id in zip(masks, class_ids):
        for polygon in mask_to_polygons(mask, epsilon=epsilon, min_area=min_area):
            if len(polygon) >= 3:
                lines.append(format_yolo_polygon_line(class_id, polygon, width, height))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def write_saved_objects_yolo_segmentation_label(
    path: Path,
    saved_objects: Sequence[SavedObject],
    image_shape: tuple[int, int] | tuple[int, int, int],
    epsilon: float = 2.0,
    min_area: float = 8.0,
) -> int:
    height = int(image_shape[0])
    width = int(image_shape[1])
    lines: list[str] = []
    for saved in saved_objects:
        for polygon in yolo_export_polygons_for_saved_object(saved, epsilon=epsilon, min_area=min_area):
            if len(polygon) >= 3:
                lines.append(format_yolo_polygon_line(saved.class_id, polygon, width, height))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def write_mask_rcnn_annotation(
    path: Path,
    masks_dir: Path,
    image_file_name: str,
    masks: Sequence[np.ndarray],
    class_ids: Sequence[int],
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> int:
    height = int(image_shape[0])
    width = int(image_shape[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    annotations = []
    category_ids: set[int] = set()
    rle_count = 0
    for object_id, (mask, class_id) in enumerate(zip(masks, class_ids), start=1):
        mask_bool = np.asarray(mask, dtype=bool)
        area = int(mask_bool.sum())
        if area <= 0:
            continue
        class_id_int = int(class_id)
        category_ids.add(class_id_int)
        mask_path = masks_dir / f"object_{object_id:03d}.png"
        save_png(mask_bool.astype(np.uint8) * 255, mask_path)
        relative_mask = Path(os.path.relpath(mask_path, path.parent)).as_posix()
        bbox_xywh = bbox_from_mask(mask_bool)
        segmentation: dict[str, object] = {
            "format": "binary_mask_png",
            "path": relative_mask,
        }
        rle_values = coco_rle_from_mask(mask_bool)
        if rle_values is not None:
            segmentation, area, bbox_xywh = rle_values
            rle_count += 1
        annotations.append(
            {
                "id": object_id,
                "image_id": 1,
                "category_id": class_id_int,
                "class_id": class_id_int,
                "area": area,
                "bbox": bbox_xywh,
                "bbox_xywh": bbox_xywh,
                "bbox_xyxy": bbox_xyxy_from_mask(mask_bool),
                "iscrowd": 0,
                "segmentation": segmentation,
                "mask": relative_mask,
            }
        )

    categories = [{"id": category_id, "name": f"class_{category_id}"} for category_id in sorted(category_ids)]
    image_record = {
        "id": 1,
        "file_name": image_file_name,
        "height": height,
        "width": width,
    }
    payload = {
        "format": "coco_instance_segmentation_rle_v1" if rle_count == len(annotations) else "mask_rcnn_binary_masks_v1",
        "image": image_record,
        "images": [image_record],
        "annotations": annotations,
        "categories": categories,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(annotations)


def write_saved_objects_csv(path: Path, saved_objects: Sequence[SavedObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["object_id", "name", "class_id", "area", "bbox_xywh", "score", "color_rgb", "polygon_count"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for object_id, saved in enumerate(saved_objects, start=1):
            writer.writerow(
                {
                    "object_id": object_id,
                    "name": saved.name,
                    "class_id": saved.class_id,
                    "area": int(saved.mask.astype(bool).sum()),
                    "bbox_xywh": json.dumps(bbox_from_mask(saved.mask), ensure_ascii=False),
                    "score": saved.score,
                    "color_rgb": json.dumps([int(value) for value in saved.color], ensure_ascii=False),
                    "polygon_count": len(normalize_yolo_edit_polygons(saved.yolo_polygons)),
                }
            )


def saved_object_to_annotation_payload(saved: SavedObject, object_id: int) -> dict[str, object]:
    return {
        "id": object_id,
        "name": saved.name,
        "class_id": int(saved.class_id),
        "score": float(saved.score),
        "color_rgb": [int(value) for value in saved.color],
        "yolo_polygons": normalize_yolo_edit_polygons(saved.yolo_polygons),
    }


def write_saved_objects_annotation_json(
    path: Path,
    source_image_path: Path,
    output_image_name: str,
    image_shape: tuple[int, int] | tuple[int, int, int],
    saved_objects: Sequence[SavedObject],
) -> None:
    height = int(image_shape[0])
    width = int(image_shape[1])
    try:
        source_abs = str(source_image_path.resolve())
    except OSError:
        source_abs = str(source_image_path)
    payload = {
        "format": "sam2_studio_annotation_v1",
        "source": {
            "path": str(source_image_path),
            "absolute_path": source_abs,
            "file_name": source_image_path.name,
            "stem": source_image_path.stem,
        },
        "image": {
            "file_name": output_image_name,
            "stem": Path(output_image_name).stem,
            "height": height,
            "width": width,
        },
        "objects": [
            saved_object_to_annotation_payload(saved, object_id)
            for object_id, saved in enumerate(saved_objects, start=1)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_interactive_results(
    image_path: Path,
    output_dir: Path,
    image: np.ndarray,
    saved_objects: Sequence[SavedObject],
    export_format: str = "yolo",
    yolo_epsilon: float = 2.0,
    yolo_min_area: float = 8.0,
    save_object_masks: bool = False,
    image_name: str | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_image_name = image_name or image_path.name
    stem = Path(output_image_name).stem
    train_image_path = output_dir / "img" / output_image_name
    overlay_path = output_dir / "previews" / f"{stem}_overlay.png"
    mask_label_path = output_dir / "labels" / f"{stem}.png"
    color_path = output_dir / "previews" / f"{stem}_color_mask.png"
    csv_path = output_dir / "metadata" / f"{stem}_objects.csv"
    annotation_path = output_dir / "metadata" / f"{stem}_annotation.json"
    masks_dir = output_dir / "masks" / stem
    yolo_label_path = output_dir / "labels" / f"{stem}.txt"
    mask_rcnn_annotation_path = output_dir / "mask_rcnn" / f"{stem}.json"
    mask_rcnn_masks_dir = output_dir / "mask_rcnn" / "masks" / stem

    save_training_image(image, train_image_path)
    save_png(render_saved_overlay(image, saved_objects), overlay_path)
    save_png(build_color_mask(saved_objects), color_path)
    write_saved_objects_csv(csv_path, saved_objects)
    write_saved_objects_annotation_json(annotation_path, image_path, output_image_name, image.shape, saved_objects)

    outputs = {
        "train_image": train_image_path,
        "overlay": overlay_path,
        "color_mask": color_path,
        "objects_csv": csv_path,
        "annotation": annotation_path,
    }

    masks = [saved.mask for saved in saved_objects]
    class_ids = [saved.class_id for saved in saved_objects]
    if wants_mask_export(export_format):
        save_png(build_class_label_mask(masks, class_ids, image.shape), mask_label_path)
        outputs["mask_label"] = mask_label_path

    if wants_yolo_export(export_format):
        write_saved_objects_yolo_segmentation_label(
            yolo_label_path,
            saved_objects,
            image.shape,
            epsilon=yolo_epsilon,
            min_area=yolo_min_area,
        )
        outputs["yolo_label"] = yolo_label_path

    if wants_mask_rcnn_export(export_format):
        write_mask_rcnn_annotation(
            mask_rcnn_annotation_path,
            mask_rcnn_masks_dir,
            output_image_name,
            masks,
            class_ids,
            image.shape,
        )
        outputs["mask_rcnn_annotation"] = mask_rcnn_annotation_path
        outputs["mask_rcnn_masks_dir"] = mask_rcnn_masks_dir

    if save_object_masks:
        masks_dir.mkdir(parents=True, exist_ok=True)
        for object_id, saved in enumerate(saved_objects, start=1):
            save_png(saved.mask.astype(np.uint8) * 255, masks_dir / f"object_{object_id:03d}.png")
        outputs["masks_dir"] = masks_dir

    return outputs
