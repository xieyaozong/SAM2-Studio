from __future__ import annotations

from pathlib import Path
from typing import Sequence
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
except ImportError:
    QOpenGLWidget = None

from sam2.sam2_image_predictor import SAM2ImagePredictor
from utils.config import (
    DEFAULT_EXTENSIONS,
    EXPORT_FORMATS,
    MODEL_PRESETS,
    PREFERRED_IMAGE_FOLDER,
    PROJECT_ROOT,
    SavedObject,
)
from utils.export_utils import (
    mask_to_yolo_edit_polygons,
    normalize_yolo_edit_polygons,
    render_interactive_overlay,
    render_yolo_polygon_overlay,
    save_interactive_results,
    yolo_edit_polygons_to_mask,
)
from utils.hough_preprocess import (
    HoughPreprocessResult,
    HoughPreprocessSettings,
    hough_output_name,
    hough_result_image,
    preprocess_hough_circle,
    save_hough_preprocess_result,
)
from utils.io_utils import collect_images, color_for_index, load_rgb_image
from utils.model_utils import build_image_predictor, inference_autocast

import os
import json
import queue
import sys
import threading
import time

import numpy as np
import torch


def wheel_step_count(event) -> int:
    delta = event.angleDelta().y()
    if delta == 0:
        delta = event.pixelDelta().y()
    if delta == 0:
        return 0
    return int(delta / 120) if abs(delta) >= 120 else (1 if delta > 0 else -1)


class StepWheelSpinBox(QSpinBox):
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.WheelFocus)

    def wheelEvent(self, event) -> None:
        steps = wheel_step_count(event)
        if steps == 0:
            event.ignore()
            return
        self.stepBy(steps)
        event.accept()


class StepWheelDoubleSpinBox(QDoubleSpinBox):
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.WheelFocus)

    def wheelEvent(self, event) -> None:
        steps = wheel_step_count(event)
        if steps == 0:
            event.ignore()
            return
        self.stepBy(steps)
        event.accept()


class SamCanvasView(QGraphicsView):
    def __init__(self, on_click, on_edit_event):
        super().__init__()
        self.on_click = on_click
        self.on_edit_event = on_edit_event
        self.scene_obj = QGraphicsScene(self)
        self.scene_obj.setItemIndexMethod(QGraphicsScene.NoIndex)
        self.setScene(self.scene_obj)
        self.pixmap_item: QGraphicsPixmapItem | None = None
        self.pixmap_key: tuple[object, ...] | None = None
        self.image_size: tuple[int, int] | None = None
        self.point_items: list[QGraphicsItem] = []
        self.polygon_items: list[QGraphicsItem] = []
        self.has_image = False
        self.edit_mode = False
        self.dragging_edit = False
        self.middle_panning = False
        self.last_pan_pos = None
        self.pending_pan_dx = 0
        self.pending_pan_dy = 0
        self.pending_zoom_steps = 0.0
        self.pending_zoom_anchor = None
        self.fast_interaction = False
        self.opengl_viewport = False
        if QOpenGLWidget is not None and os.environ.get("SAM2_STUDIO_DISABLE_OPENGL") != "1":
            try:
                self.setViewport(QOpenGLWidget())
                self.opengl_viewport = True
            except Exception:
                pass
        self.setRenderHints(QPainter.Antialiasing)
        viewport_update_mode = (
            QGraphicsView.FullViewportUpdate
            if self.opengl_viewport
            else QGraphicsView.SmartViewportUpdate
        )
        self.setViewportUpdateMode(viewport_update_mode)
        self.setOptimizationFlag(QGraphicsView.DontSavePainterState, True)
        self.setOptimizationFlag(QGraphicsView.DontAdjustForAntialiasing, True)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#eef2f7"))
        self.setFrameShape(QFrame.NoFrame)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.pan_timer = QTimer(self)
        self.pan_timer.setSingleShot(True)
        self.pan_timer.timeout.connect(self.apply_pending_pan)
        self.zoom_timer = QTimer(self)
        self.zoom_timer.setSingleShot(True)
        self.zoom_timer.timeout.connect(self.apply_pending_zoom)
        self.fast_interaction_timer = QTimer(self)
        self.fast_interaction_timer.setSingleShot(True)
        self.fast_interaction_timer.timeout.connect(self.end_fast_interaction)

    def set_edit_mode(self, enabled: bool) -> None:
        self.edit_mode = enabled
        self.dragging_edit = False
        self.setDragMode(QGraphicsView.NoDrag if enabled else QGraphicsView.ScrollHandDrag)

    @staticmethod
    def array_image_key(image: np.ndarray) -> tuple[object, ...]:
        data_ptr = int(image.__array_interface__["data"][0])
        return (data_ptr, image.shape, image.strides, str(image.dtype))

    def remove_items(self, items: list[QGraphicsItem]) -> None:
        for item in items:
            self.scene_obj.removeItem(item)
        items.clear()

    def clear_prompt_points(self) -> None:
        self.remove_items(self.point_items)

    def clear_polygon_overlay(self) -> None:
        self.remove_items(self.polygon_items)

    def set_overlay(
        self,
        image: np.ndarray | None,
        points: Sequence[tuple[float, float, int]],
        image_key: tuple[object, ...] | None = None,
    ) -> None:
        self.clear_prompt_points()
        self.point_items.clear()
        self.has_image = image is not None

        if image is None:
            self.scene_obj.clear()
            self.pixmap_item = None
            self.pixmap_key = None
            self.image_size = None
            self.point_items.clear()
            self.polygon_items.clear()
            self.scene_obj.setSceneRect(0, 0, 100, 100)
            return

        image = np.ascontiguousarray(image)
        height, width = image.shape[:2]
        key = image_key or self.array_image_key(image)
        if self.pixmap_item is None or self.pixmap_key != key:
            qimage = QImage(image.data, width, height, int(image.strides[0]), QImage.Format_RGB888).copy()
            pixmap = QPixmap.fromImage(qimage)
            if self.pixmap_item is None:
                self.pixmap_item = self.scene_obj.addPixmap(pixmap)
                self.pixmap_item.setZValue(0)
                self.pixmap_item.setShapeMode(QGraphicsPixmapItem.BoundingRectShape)
                self.pixmap_item.setTransformationMode(Qt.FastTransformation)
            else:
                self.pixmap_item.setPixmap(pixmap)
            self.pixmap_key = key
            self.image_size = (height, width)
            self.scene_obj.setSceneRect(0, 0, width, height)
        self.draw_points(points)

    def draw_points(self, points: Sequence[tuple[float, float, int]]) -> None:
        for x, y, label in points:
            radius = 7
            item = QGraphicsEllipseItem(float(x) - radius, float(y) - radius, radius * 2, radius * 2)
            item.setBrush(QColor("#38d9c5") if label == 1 else QColor("#fb7185"))
            item.setPen(QPen(QColor("#111827"), 2))
            item.setZValue(10)
            self.scene_obj.addItem(item)
            self.point_items.append(item)

    def add_polygon_path(
        self,
        points: np.ndarray,
        color: tuple[int, int, int],
        thickness: int,
        *,
        closed: bool,
    ) -> None:
        if len(points) < 2:
            return
        path = QPainterPath()
        path.moveTo(float(points[0][0]), float(points[0][1]))
        for x, y in points[1:]:
            path.lineTo(float(x), float(y))
        if closed:
            path.closeSubpath()
        item = QGraphicsPathItem(path)
        pen = QPen(QColor(*color), thickness)
        pen.setJoinStyle(Qt.RoundJoin)
        pen.setCapStyle(Qt.RoundCap)
        item.setPen(pen)
        item.setBrush(QBrush(Qt.NoBrush))
        item.setZValue(20)
        self.scene_obj.addItem(item)
        self.polygon_items.append(item)

    def add_vertex_handle(self, x: float, y: float, radius: int, color: tuple[int, int, int]) -> None:
        item = QGraphicsEllipseItem(float(x) - radius, float(y) - radius, radius * 2, radius * 2)
        item.setBrush(QBrush(QColor(255, 255, 255)))
        item.setPen(QPen(QColor(*color), 2))
        item.setZValue(30)
        self.scene_obj.addItem(item)
        self.polygon_items.append(item)

    def set_polygon_overlay(
        self,
        saved_objects: Sequence[SavedObject],
        selected_object_index: int = -2,
        selected_polygon_index: int = -1,
        selected_vertex_index: int = -1,
        draft_polygon: Sequence[tuple[float, float]] | None = None,
        draft_mode: str = "add",
        current_yolo_polygons: Sequence[dict[str, object]] | None = None,
        current_color: np.ndarray | None = None,
    ) -> None:
        self.clear_polygon_overlay()
        if not self.has_image:
            return

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
                if item["mode"] == "subtract":
                    color_tuple = (255, 210, 80)
                else:
                    color_tuple = tuple(int(value) for value in color.tolist())
                thickness = 3 if selected_object and polygon_index == selected_polygon_index else 2
                self.add_polygon_path(points, color_tuple, thickness, closed=True)
                if selected_object:
                    for vertex_index, (x, y) in enumerate(points):
                        radius = 6 if polygon_index == selected_polygon_index and vertex_index == selected_vertex_index else 4
                        self.add_vertex_handle(float(x), float(y), radius, color_tuple)

        if draft_polygon:
            points = np.rint(np.asarray(draft_polygon, dtype=np.float32)).astype(np.int32)
            color_tuple = (255, 210, 80) if draft_mode == "subtract" else (56, 217, 197)
            self.add_polygon_path(points, color_tuple, 2, closed=False)
            for x, y in points:
                self.add_vertex_handle(float(x), float(y), 5, color_tuple)

    @staticmethod
    def event_pos(event):
        return event.position().toPoint() if hasattr(event, "position") else event.pos()

    def begin_fast_interaction(self) -> None:
        if not self.fast_interaction:
            self.fast_interaction = True
            self.setRenderHint(QPainter.Antialiasing, False)
        self.fast_interaction_timer.start(120)

    def end_fast_interaction(self) -> None:
        if self.pan_timer.isActive() or self.zoom_timer.isActive():
            self.fast_interaction_timer.start(120)
            return
        if self.fast_interaction:
            self.fast_interaction = False
            self.setRenderHint(QPainter.Antialiasing, True)
            self.viewport().update()

    def queue_pan_delta(self, dx: int, dy: int) -> None:
        if dx == 0 and dy == 0:
            return
        self.begin_fast_interaction()
        self.pending_pan_dx += int(dx)
        self.pending_pan_dy += int(dy)
        if not self.pan_timer.isActive():
            self.pan_timer.start(8)

    def apply_pending_pan(self) -> None:
        dx = self.pending_pan_dx
        dy = self.pending_pan_dy
        self.pending_pan_dx = 0
        self.pending_pan_dy = 0
        if dx == 0 and dy == 0:
            return
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - dx)
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() - dy)
        self.fast_interaction_timer.start(120)

    def flush_pending_pan(self) -> None:
        if self.pan_timer.isActive():
            self.pan_timer.stop()
        self.apply_pending_pan()

    def queue_zoom_steps(self, steps: float, anchor) -> None:
        if steps == 0:
            return
        self.begin_fast_interaction()
        self.pending_zoom_steps += float(steps)
        self.pending_zoom_anchor = anchor
        if not self.zoom_timer.isActive():
            self.zoom_timer.start(8)

    def apply_pending_zoom(self) -> None:
        steps = self.pending_zoom_steps
        anchor = self.pending_zoom_anchor or self.viewport().rect().center()
        self.pending_zoom_steps = 0.0
        self.pending_zoom_anchor = None
        if steps == 0:
            return
        factor = 1.15 ** steps
        self.zoom_at(anchor, factor)
        self.fast_interaction_timer.start(120)

    def flush_pending_zoom(self) -> None:
        if self.zoom_timer.isActive():
            self.zoom_timer.stop()
        self.apply_pending_zoom()

    def zoom_at(self, viewport_pos, factor: float) -> None:
        if factor <= 0:
            return
        scene_pos = self.mapToScene(viewport_pos)
        self.scale(factor, factor)
        new_view_pos = self.mapFromScene(scene_pos)
        delta = new_view_pos - viewport_pos
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + int(delta.x()))
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + int(delta.y()))

    def zoom_by(self, factor: float) -> None:
        self.begin_fast_interaction()
        self.zoom_at(self.viewport().rect().center(), factor)
        self.fast_interaction_timer.start(120)

    def fit_image(self) -> None:
        if self.has_image:
            self.flush_pending_pan()
            self.flush_pending_zoom()
            self.fitInView(self.scene_obj.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event) -> None:
        if not self.has_image:
            super().wheelEvent(event)
            return
        steps = float(event.angleDelta().y()) / 120.0
        if steps == 0:
            steps = float(event.pixelDelta().y()) / 240.0
        self.queue_zoom_steps(steps, self.event_pos(event))
        event.accept()

    def mousePressEvent(self, event) -> None:
        if self.has_image and event.button() == Qt.MiddleButton:
            self.middle_panning = True
            self.last_pan_pos = self.event_pos(event)
            self.begin_fast_interaction()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if self.has_image and self.edit_mode and event.button() in {Qt.LeftButton, Qt.RightButton}:
            point = self.mapToScene(event.pos())
            rect = self.scene_obj.sceneRect()
            if rect.contains(point):
                self.dragging_edit = event.button() == Qt.LeftButton
                self.on_edit_event("press", float(point.x()), float(point.y()), event.button(), event.modifiers())
                return
        if self.has_image and event.button() in {Qt.LeftButton, Qt.RightButton}:
            point = self.mapToScene(event.pos())
            rect = self.scene_obj.sceneRect()
            if rect.contains(point):
                label = 0 if event.button() == Qt.RightButton else None
                self.on_click(float(point.x()), float(point.y()), label)
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self.has_image and self.middle_panning and self.last_pan_pos is not None:
            pos = self.event_pos(event)
            delta = pos - self.last_pan_pos
            self.last_pan_pos = pos
            self.queue_pan_delta(int(delta.x()), int(delta.y()))
            event.accept()
            return
        if self.has_image and self.edit_mode and self.dragging_edit:
            point = self.mapToScene(event.pos())
            rect = self.scene_obj.sceneRect()
            if rect.contains(point):
                self.on_edit_event("move", float(point.x()), float(point.y()), Qt.LeftButton, event.modifiers())
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self.middle_panning and event.button() == Qt.MiddleButton:
            self.middle_panning = False
            self.last_pan_pos = None
            self.flush_pending_pan()
            self.fast_interaction_timer.start(80)
            self.unsetCursor()
            event.accept()
            return
        if self.has_image and self.edit_mode and self.dragging_edit:
            point = self.mapToScene(event.pos())
            self.dragging_edit = False
            self.on_edit_event("release", float(point.x()), float(point.y()), event.button(), event.modifiers())
            return
        super().mouseReleaseEvent(event)


class PySideSamWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM 2 Studio")
        self.resize(1440, 900)
        self.setMinimumSize(980, 620)

        default_model = "large" if torch.cuda.is_available() else "tiny"
        self.messages: queue.Queue[tuple[str, object]] = queue.Queue()
        self.predictor: SAM2ImagePredictor | None = None
        self.device: torch.device | None = None
        self.model_key: tuple[str, str] | None = None
        self.model_loading = False
        self.embedding = False
        self.image_ready = False
        self.predicting = False
        self.pending_prediction = False
        self.point_version = 0
        self.image_version = 0

        self.image_path: Path | None = None
        self.image_np: np.ndarray | None = None
        self.original_image_np: np.ndarray | None = None
        self.working_image_name: str | None = None
        self.current_mask: np.ndarray | None = None
        self.current_score = 0.0
        self.current_color = np.array([56, 217, 197], dtype=np.uint8)
        self.current_yolo_polygons: list[dict[str, object]] = []
        self.current_yolo_dirty = False
        self.saved_objects: list[SavedObject] = []
        self.points: list[tuple[float, float, int]] = []
        self.image_files: list[Path] = []
        self.image_index = -1
        self.image_folder: Path | None = None
        self.output_dir: Path | None = None
        self.output_dir_user_selected = False
        self.annotation_index_dir: Path | None = None
        self.annotation_records: list[dict[str, object]] = []
        self.hough_result: HoughPreprocessResult | None = None
        self.hough_preview_active = False
        self.hough_running = False
        self.hough_action_after_run: str | None = None
        self.hough_requested_output: str | None = None
        self.pending_template: dict[str, object] | None = None
        self.mask_template: dict[str, object] | None = None
        self.edit_drag_target: tuple[int, int, int] | None = None
        self.edit_move_target: tuple[int, float, float] | None = None
        self.edit_undo_stack: list[dict[str, object]] = []
        self.edit_undo_limit = 25
        self.sam_hole_predicting = False
        self.sam_hole_version = 0
        self.selected_polygon_index = -1
        self.selected_vertex_index = -1
        self.draft_polygon_points: list[tuple[float, float]] = []
        self.draft_polygon_active = False
        self.overlay_cache_key: tuple[object, ...] | None = None
        self.overlay_cache: np.ndarray | None = None
        self.render_pending = False
        self.render_pending_fit = False
        self.render_interval_ms = 16
        self.last_render_time = 0.0
        self.dirty = False

        self.build_ui(default_model)
        self.install_shortcuts()

        self.render_timer = QTimer(self)
        self.render_timer.setSingleShot(True)
        self.render_timer.timeout.connect(self.flush_render_canvas)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_messages)
        self.timer.start(80)

    def build_ui(self, default_model: str) -> None:
        toolbar = QToolBar("Main", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addWidget(QLabel("Model"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(MODEL_PRESETS.keys())
        self.model_combo.setCurrentText(default_model)
        toolbar.addWidget(self.model_combo)

        toolbar.addWidget(QLabel("Device"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "cuda", "cuda:0", "cpu"])
        toolbar.addWidget(self.device_combo)
        self.model_combo.currentTextChanged.connect(self.on_model_setting_changed)
        self.device_combo.currentTextChanged.connect(self.on_model_setting_changed)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        toolbar.addSeparator()
        toolbar.addWidget(self.status_label)

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 14)
        root_layout.setSpacing(12)
        self.setCentralWidget(root)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setObjectName("mainSplitter")
        root_layout.addWidget(splitter)

        left = self.make_panel()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)
        splitter.addWidget(left)

        self.left_tabs = QTabWidget()
        self.left_tabs.setObjectName("sideTabs")
        self.left_tabs.setDocumentMode(True)
        left_layout.addWidget(self.left_tabs, 1)
        self.populate_left_controls(self.left_tabs)

        self.save_button = QPushButton("Save Labels")
        self.save_button.setObjectName("primarySaveButton")
        self.save_button.clicked.connect(self.save_results)
        left_layout.addWidget(self.save_button)

        self.save_next_button = QPushButton("Save & Next")
        self.save_next_button.setObjectName("secondarySaveButton")
        self.save_next_button.clicked.connect(self.save_and_next)
        left_layout.addWidget(self.save_next_button)

        self.canvas = SamCanvasView(self.handle_canvas_click, self.handle_polygon_edit_event)
        self.canvas.setMinimumSize(260, 220)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(self.canvas)

        right = self.make_panel()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(10, 10, 10, 10)
        right_layout.setSpacing(8)
        splitter.addWidget(right)
        self.right_tabs = QTabWidget()
        self.right_tabs.setObjectName("sideTabs")
        self.right_tabs.setDocumentMode(True)
        right_layout.addWidget(self.right_tabs, 1)
        self.populate_right_controls(self.right_tabs)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([330, 760, 350])

    def make_tab_layout(self, tabs: QTabWidget, title: str, *, scroll: bool = True) -> QVBoxLayout:
        if scroll:
            scroll_area = QScrollArea()
            scroll_area.setWidgetResizable(True)
            scroll_area.setFrameShape(QFrame.NoFrame)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            scroll_area.setObjectName("controlScroll")
            body = QWidget()
            body.setObjectName("scrollBody")
            layout = QVBoxLayout(body)
            layout.setContentsMargins(4, 4, 4, 4)
            layout.setSpacing(8)
            scroll_area.setWidget(body)
            tabs.addTab(scroll_area, title)
            return layout

        page = QWidget()
        page.setObjectName("scrollBody")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(8)
        tabs.addTab(page, title)
        return layout

    @staticmethod
    def add_button_row(layout: QVBoxLayout, *buttons: QPushButton) -> None:
        row = QHBoxLayout()
        row.setSpacing(6)
        for button in buttons:
            row.addWidget(button)
        layout.addLayout(row)

    def populate_left_controls(self, tabs: QTabWidget) -> None:
        layout = self.make_tab_layout(tabs, "Input")
        layout.addWidget(self.section("Images"))
        open_image_button = QPushButton("Open Image")
        open_image_button.clicked.connect(self.open_image)

        open_folder_button = QPushButton("Open Folder")
        open_folder_button.clicked.connect(self.open_folder)
        self.add_button_row(layout, open_image_button, open_folder_button)

        output_button = QPushButton("Choose Output Folder")
        output_button.clicked.connect(self.choose_output_folder)
        layout.addWidget(output_button)

        self.folder_recursive_check = QCheckBox("Include subfolders")
        self.folder_recursive_check.setChecked(True)
        layout.addWidget(self.folder_recursive_check)

        self.image_counter_label = QLabel("No image loaded")
        self.image_counter_label.setObjectName("muted")
        layout.addWidget(self.image_counter_label)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(6)
        prev_button = QPushButton("Previous")
        prev_button.clicked.connect(self.previous_image)
        next_button = QPushButton("Next")
        next_button.clicked.connect(self.next_image)
        nav_row.addWidget(prev_button)
        nav_row.addWidget(next_button)
        layout.addLayout(nav_row)

        jump_row = QHBoxLayout()
        jump_row.setSpacing(6)
        self.image_jump_spin = StepWheelSpinBox()
        self.image_jump_spin.setRange(1, 1)
        self.image_jump_spin.setPrefix("Image ")
        self.image_jump_spin.setEnabled(False)
        jump_button = QPushButton("Go")
        jump_button.clicked.connect(self.jump_to_image)
        jump_row.addWidget(self.image_jump_spin)
        jump_row.addWidget(jump_button)
        layout.addLayout(jump_row)

        self.output_label = QLabel("Output: not selected")
        self.output_label.setObjectName("muted")
        self.output_label.setWordWrap(True)
        layout.addWidget(self.output_label)

        layout.addWidget(self.section("SAM Prompts"))
        mode_row = QHBoxLayout()
        mode_row.setSpacing(6)
        self.add_button = QPushButton("Foreground")
        self.add_button.setCheckable(True)
        self.add_button.setChecked(True)
        self.add_button.setProperty("mode", "add")
        self.remove_button = QPushButton("Background")
        self.remove_button.setCheckable(True)
        self.remove_button.setProperty("mode", "remove")
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.add_button)
        mode_group.addButton(self.remove_button)
        self.add_button.clicked.connect(lambda: self.set_status("Foreground point mode"))
        self.remove_button.clicked.connect(lambda: self.set_status("Background point mode"))
        mode_row.addWidget(self.add_button)
        mode_row.addWidget(self.remove_button)
        layout.addLayout(mode_row)

        layout.addWidget(self.section("Active Mask"))
        self.accept_button = QPushButton("Add Mask To Objects")
        self.accept_button.setProperty("accent", True)
        self.accept_button.clicked.connect(self.accept_current_mask)
        layout.addWidget(self.accept_button)

        undo_button = QPushButton("Undo Prompt")
        undo_button.clicked.connect(self.undo_point)
        clear_button = QPushButton("Clear Prompts")
        clear_button.clicked.connect(self.clear_points)
        self.add_button_row(layout, undo_button, clear_button)

        layout.addWidget(self.section("View"))
        zoom_out = QPushButton("-")
        zoom_out.clicked.connect(lambda: self.canvas.zoom_by(1 / 1.18))
        fit = QPushButton("Fit")
        fit.clicked.connect(self.canvas_fit)
        zoom_in = QPushButton("+")
        zoom_in.clicked.connect(lambda: self.canvas.zoom_by(1.18))
        self.add_button_row(layout, zoom_out, fit, zoom_in)
        layout.addStretch(1)

        layout = self.make_tab_layout(tabs, "Hough")
        self.hough_image_combo = QComboBox()
        self.hough_image_combo.addItem("Masked full image", "full")
        self.hough_image_combo.addItem("Center crop image", "crop")
        self.hough_image_combo.currentTextChanged.connect(self.refresh_hough_preview)
        layout.addWidget(self.hough_image_combo)

        self.hough_debug_check = QCheckBox("Show Hough debug")
        self.hough_debug_check.toggled.connect(self.refresh_hough_preview)
        layout.addWidget(self.hough_debug_check)

        self.hough_crop_size_spin = StepWheelSpinBox()
        self.hough_crop_size_spin.setRange(0, 8192)
        self.hough_crop_size_spin.setValue(0)
        self.hough_crop_size_spin.setSpecialValueText("native crop")
        self.hough_crop_size_spin.setSingleStep(128)
        self.hough_crop_size_spin.setPrefix("Crop ")
        self.hough_crop_size_spin.setSuffix(" px")
        self.hough_crop_size_spin.setToolTip("0 keeps the adaptive crop at its native size. Set a pixel value to force resize.")
        layout.addWidget(self.hough_crop_size_spin)

        self.hough_inner_scale_spin = StepWheelDoubleSpinBox()
        self.hough_inner_scale_spin.setRange(0.1, 1.5)
        self.hough_inner_scale_spin.setSingleStep(0.02)
        self.hough_inner_scale_spin.setDecimals(2)
        self.hough_inner_scale_spin.setValue(0.86)
        self.hough_inner_scale_spin.setPrefix("Inner x")
        layout.addWidget(self.hough_inner_scale_spin)

        self.hough_crop_scale_spin = StepWheelDoubleSpinBox()
        self.hough_crop_scale_spin.setRange(0.1, 1.5)
        self.hough_crop_scale_spin.setSingleStep(0.02)
        self.hough_crop_scale_spin.setDecimals(2)
        self.hough_crop_scale_spin.setValue(0.55)
        self.hough_crop_scale_spin.setPrefix("Crop x")
        layout.addWidget(self.hough_crop_scale_spin)

        preview_hough_button = QPushButton("Preview")
        preview_hough_button.clicked.connect(self.preview_hough_preprocess)

        use_hough_button = QPushButton("Use For SAM")
        use_hough_button.setProperty("accent", True)
        use_hough_button.clicked.connect(self.apply_hough_to_sam)
        self.add_button_row(layout, preview_hough_button, use_hough_button)

        save_hough_button = QPushButton("Save Hough Image")
        save_hough_button.clicked.connect(self.save_hough_result)

        restore_original_button = QPushButton("Restore Original")
        restore_original_button.clicked.connect(self.restore_original_image)
        self.add_button_row(layout, save_hough_button, restore_original_button)
        layout.addStretch(1)

        layout = self.make_tab_layout(tabs, "Labels")
        layout.addWidget(self.section("Dataset Export"))
        self.class_spin = StepWheelSpinBox()
        self.class_spin.setRange(0, 9999)
        self.class_spin.setValue(0)
        self.class_spin.setPrefix("Class ")
        self.class_spin.valueChanged.connect(lambda _value: self.render_canvas())
        layout.addWidget(self.class_spin)

        self.export_combo = QComboBox()
        self.export_combo.addItems(EXPORT_FORMATS)
        self.export_combo.setCurrentText("yolo")
        layout.addWidget(self.export_combo)

        self.object_masks_check = QCheckBox("Save per-object masks")
        layout.addWidget(self.object_masks_check)

        self.yolo_preview_check = QCheckBox("Preview YOLO polygons")
        self.yolo_preview_check.toggled.connect(lambda _checked: self.render_canvas())
        layout.addWidget(self.yolo_preview_check)

        self.yolo_epsilon_spin = StepWheelDoubleSpinBox()
        self.yolo_epsilon_spin.setRange(0.0, 30.0)
        self.yolo_epsilon_spin.setSingleStep(0.5)
        self.yolo_epsilon_spin.setDecimals(1)
        self.yolo_epsilon_spin.setValue(2.0)
        self.yolo_epsilon_spin.setPrefix("Epsilon ")
        self.yolo_epsilon_spin.valueChanged.connect(self.on_yolo_polygon_settings_changed)
        layout.addWidget(self.yolo_epsilon_spin)

        self.yolo_min_area_spin = StepWheelDoubleSpinBox()
        self.yolo_min_area_spin.setRange(0.0, 10000.0)
        self.yolo_min_area_spin.setSingleStep(1.0)
        self.yolo_min_area_spin.setDecimals(1)
        self.yolo_min_area_spin.setValue(8.0)
        self.yolo_min_area_spin.setPrefix("Min area ")
        self.yolo_min_area_spin.valueChanged.connect(self.on_yolo_polygon_settings_changed)
        layout.addWidget(self.yolo_min_area_spin)

        self.yolo_preview_label = QLabel("Polygon preview: off")
        self.yolo_preview_label.setObjectName("muted")
        layout.addWidget(self.yolo_preview_label)
        layout.addStretch(1)

    def populate_right_controls(self, tabs: QTabWidget) -> None:
        layout = self.make_tab_layout(tabs, "Objects", scroll=False)
        layout.addWidget(self.section("Objects"))
        self.object_count = QLabel("0 saved")
        self.object_count.setObjectName("muted")
        layout.addWidget(self.object_count)
        self.objects_list = QListWidget()
        self.objects_list.currentRowChanged.connect(self.on_selected_object_changed)
        layout.addWidget(self.objects_list, 1)

        remove_object = QPushButton("Remove Object")
        remove_object.setProperty("danger", True)
        remove_object.clicked.connect(self.remove_selected_object)

        clear_objects = QPushButton("Clear All")
        clear_objects.setProperty("danger", True)
        clear_objects.clicked.connect(self.clear_saved_objects)
        self.add_button_row(layout, remove_object, clear_objects)

        layout.addWidget(self.section("Reuse Template"))
        self.template_status_label = QLabel("Template: none")
        self.template_status_label.setObjectName("muted")
        self.template_status_label.setWordWrap(True)
        layout.addWidget(self.template_status_label)

        save_template = QPushButton("Capture Template")
        save_template.clicked.connect(self.save_mask_template)
        apply_template = QPushButton("Apply Template")
        apply_template.setProperty("accent", True)
        apply_template.clicked.connect(self.apply_mask_template)
        self.add_button_row(layout, save_template, apply_template)

        layout = self.make_tab_layout(tabs, "Edit")
        layout.addWidget(self.section("Mask Editing"))
        self.polygon_edit_check = QCheckBox("Enable mask editing")
        self.polygon_edit_check.setProperty("toolMode", True)
        self.polygon_edit_check.toggled.connect(self.set_polygon_edit_enabled)
        layout.addWidget(self.polygon_edit_check)

        tool_row = QHBoxLayout()
        tool_row.setSpacing(6)
        self.polygon_mode_button = QPushButton("Polygon")
        self.polygon_mode_button.setCheckable(True)
        self.polygon_mode_button.setChecked(True)
        self.polygon_mode_button.setProperty("activeAction", True)
        self.polygon_mode_button.clicked.connect(lambda: self.set_polygon_tool_mode("polygon"))
        self.sam_hole_mode_button = QPushButton("SAM Hole")
        self.sam_hole_mode_button.setCheckable(True)
        self.sam_hole_mode_button.setProperty("activeAction", True)
        self.sam_hole_mode_button.clicked.connect(lambda: self.set_polygon_tool_mode("sam_hole"))
        self.polygon_tool_group = QButtonGroup(self)
        self.polygon_tool_group.setExclusive(True)
        self.polygon_tool_group.addButton(self.polygon_mode_button)
        self.polygon_tool_group.addButton(self.sam_hole_mode_button)
        tool_row.addWidget(self.polygon_mode_button)
        tool_row.addWidget(self.sam_hole_mode_button)
        layout.addLayout(tool_row)

        self.whole_mask_drag_check = QCheckBox("Move whole target")
        self.whole_mask_drag_check.setProperty("toolMode", True)
        self.whole_mask_drag_check.setToolTip("Drag the active SAM mask or the selected saved object as one piece. Shift-drag also works.")
        layout.addWidget(self.whole_mask_drag_check)

        self.move_target_combo = QComboBox()
        self.move_target_combo.addItem("Move target: auto", "auto")
        self.move_target_combo.addItem("Move target: active mask", "current")
        self.move_target_combo.addItem("Move target: selected object", "object")
        layout.addWidget(self.move_target_combo)

        layout.addWidget(self.section("Auto Polygon Detail"))
        self.edit_yolo_epsilon_spin = StepWheelDoubleSpinBox()
        self.edit_yolo_epsilon_spin.setRange(0.0, 30.0)
        self.edit_yolo_epsilon_spin.setSingleStep(0.5)
        self.edit_yolo_epsilon_spin.setDecimals(1)
        self.edit_yolo_epsilon_spin.setValue(float(self.yolo_epsilon_spin.value()))
        self.edit_yolo_epsilon_spin.setPrefix("Epsilon ")
        self.edit_yolo_epsilon_spin.valueChanged.connect(self.set_yolo_epsilon_from_edit)
        layout.addWidget(self.edit_yolo_epsilon_spin)

        self.edit_yolo_min_area_spin = StepWheelDoubleSpinBox()
        self.edit_yolo_min_area_spin.setRange(0.0, 10000.0)
        self.edit_yolo_min_area_spin.setSingleStep(1.0)
        self.edit_yolo_min_area_spin.setDecimals(1)
        self.edit_yolo_min_area_spin.setValue(float(self.yolo_min_area_spin.value()))
        self.edit_yolo_min_area_spin.setPrefix("Min area ")
        self.edit_yolo_min_area_spin.valueChanged.connect(self.set_yolo_min_area_from_edit)
        layout.addWidget(self.edit_yolo_min_area_spin)

        self.draft_polygon_combo = QComboBox()
        self.draft_polygon_combo.addItem("Draw add region", "add")
        self.draft_polygon_combo.addItem("Draw cut hole", "subtract")
        layout.addWidget(self.draft_polygon_combo)

        self.new_polygon_button = QPushButton("Draw Polygon")
        self.new_polygon_button.setCheckable(True)
        self.new_polygon_button.setProperty("activeAction", True)
        self.new_polygon_button.clicked.connect(self.start_draft_polygon)

        finish_polygon = QPushButton("Finish")
        finish_polygon.clicked.connect(self.finish_draft_polygon)

        cancel_polygon = QPushButton("Cancel")
        cancel_polygon.clicked.connect(self.cancel_draft_polygon)
        self.add_button_row(layout, self.new_polygon_button, finish_polygon)
        undo_draft = QPushButton("Undo Draw Point")
        undo_draft.clicked.connect(self.undo_draft_polygon_point)
        self.add_button_row(layout, cancel_polygon, undo_draft)

        sam_hole = QPushButton("Active Mask -> Hole")
        sam_hole.clicked.connect(self.subtract_current_sam_mask_from_selected_object)
        layout.addWidget(sam_hole)

        delete_vertex = QPushButton("Delete Vertex")
        delete_vertex.setProperty("danger", True)
        delete_vertex.clicked.connect(self.delete_selected_vertex)
        delete_polygon = QPushButton("Delete Polygon")
        delete_polygon.setProperty("danger", True)
        delete_polygon.clicked.connect(self.delete_selected_polygon)
        self.add_button_row(layout, delete_vertex, delete_polygon)

        self.undo_edit_button = QPushButton("Undo Edit")
        self.undo_edit_button.clicked.connect(self.undo_edit)
        self.undo_edit_button.setEnabled(False)
        layout.addWidget(self.undo_edit_button)
        layout.addStretch(1)

    def make_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("panel")
        frame.setMinimumWidth(280)
        frame.setMaximumWidth(460)
        frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        return frame

    def section(self, text: str) -> QLabel:
        label = QLabel(text.upper())
        label.setObjectName("section")
        return label

    def install_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self.open_image)
        QShortcut(QKeySequence("Ctrl+Shift+O"), self, activated=self.open_folder)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self.save_results)
        QShortcut(QKeySequence("Ctrl+N"), self, activated=self.save_and_next)
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo_edit)
        QShortcut(QKeySequence("F"), self, activated=self.canvas_fit)
        QShortcut(QKeySequence("A"), self, activated=lambda: self.add_button.setChecked(True))
        QShortcut(QKeySequence("R"), self, activated=lambda: self.remove_button.setChecked(True))
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.previous_image)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.next_image)

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def on_model_setting_changed(self) -> None:
        self.predictor = None
        self.device = None
        self.model_key = None
        self.image_ready = False
        self.current_mask = None
        self.current_yolo_polygons.clear()
        self.current_yolo_dirty = False
        self.points.clear()
        self.point_version += 1
        self.render_canvas()
        if self.image_np is not None and not self.hough_preview_active:
            self.load_model_async(force=True)
        elif self.hough_preview_active:
            self.set_status("Hough preview active. Use it for SAM or restore original.")
        else:
            self.set_status("Model setting changed")

    def has_unsaved_work(self) -> bool:
        return self.dirty or bool(self.points) or self.current_mask is not None

    def confirm_discard_work(self) -> bool:
        if not self.has_unsaved_work():
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved annotation",
            "Discard current unsaved prompts, masks, or objects?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def update_image_counter(self) -> None:
        if self.image_path is None:
            self.image_counter_label.setText("No image loaded")
            if hasattr(self, "image_jump_spin"):
                self.image_jump_spin.setEnabled(False)
            return
        if self.image_files and self.image_index >= 0:
            self.image_counter_label.setText(f"{self.image_index + 1}/{len(self.image_files)}  {self.image_path.name}")
        else:
            self.image_counter_label.setText(f"Single  {self.image_path.name}")
        if hasattr(self, "image_jump_spin"):
            self.image_jump_spin.blockSignals(True)
            try:
                count = max(1, len(self.image_files))
                value = self.image_index + 1 if self.image_files and self.image_index >= 0 else 1
                self.image_jump_spin.setRange(1, count)
                self.image_jump_spin.setValue(max(1, min(value, count)))
                self.image_jump_spin.setEnabled(bool(self.image_files))
            finally:
                self.image_jump_spin.blockSignals(False)

    def update_output_label(self) -> None:
        text = f"Output: {self.output_dir}" if self.output_dir else "Output: not selected"
        self.output_label.setText(text)

    def invalidate_annotation_index(self) -> None:
        self.annotation_index_dir = None
        self.annotation_records.clear()

    @staticmethod
    def safe_resolve_text(path: Path) -> str:
        try:
            return str(path.resolve()).casefold()
        except OSError:
            return str(path).casefold()

    @staticmethod
    def meaningful_folder_tokens(parts: Sequence[str]) -> set[str]:
        ignored = {"", ".", "metadata", "labels", "img", "images", "previews", "masks", "mask_rcnn"}
        return {part.casefold() for part in parts if part.casefold() not in ignored}

    @staticmethod
    def base_image_stem(stem: str) -> str:
        lowered = stem.casefold()
        for suffix in ("_hough_crop", "_hough_full", "_hough_mask", "_hough_debug"):
            if lowered.endswith(suffix):
                return stem[: -len(suffix)].casefold()
        return lowered

    def annotation_path_tokens(self, annotation_path: Path, output_dir: Path) -> set[str]:
        try:
            parent_parts = annotation_path.relative_to(output_dir).parent.parts
        except ValueError:
            parent_parts = annotation_path.parent.parts
        return self.meaningful_folder_tokens(parent_parts)

    def candidate_output_images(self, image_name: str, label_path: Path | None = None) -> list[Path]:
        if self.output_dir is None:
            return []
        output_dir = self.output_dir
        candidates: list[Path] = []
        image_path = Path(image_name) if image_name else Path()
        if image_name:
            candidates.append(output_dir / "img" / image_path)
            candidates.append(output_dir / image_path)
            if label_path is not None:
                try:
                    label_relative = label_path.relative_to(output_dir)
                    if len(label_relative.parts) >= 2 and label_relative.parts[0].casefold() in {"labels", "mask_rcnn"}:
                        candidates.append(output_dir / "img" / Path(*label_relative.parts[1:]).with_name(image_path.name))
                except ValueError:
                    pass
        if image_path.suffix:
            return candidates

        stem = image_path.name if image_name else (label_path.stem if label_path is not None else "")
        if not stem:
            return candidates
        for extension in DEFAULT_EXTENSIONS:
            candidates.append(output_dir / "img" / f"{stem}{extension}")
            if label_path is not None:
                try:
                    label_relative = label_path.relative_to(output_dir)
                    if len(label_relative.parts) >= 2 and label_relative.parts[0].casefold() in {"labels", "mask_rcnn"}:
                        candidates.append(output_dir / "img" / Path(*label_relative.parts[1:]).with_suffix(extension))
                except ValueError:
                    pass
        return candidates

    def find_output_image_for_record(self, record: dict[str, object]) -> Path | None:
        image_path_text = str(record.get("image_path") or "")
        if image_path_text:
            image_path = Path(image_path_text)
            if image_path.exists():
                return image_path
            if self.output_dir is not None:
                candidate = self.output_dir / image_path
                if candidate.exists():
                    return candidate
        image_relative_text = str(record.get("image_relative_path") or "")
        if image_relative_text and record.get("path"):
            candidate = Path(record["path"]).parent / image_relative_text
            if candidate.exists():
                return candidate

        label_path = Path(record["path"]) if record.get("path") else None
        image_name = str(record.get("image_file_name") or record.get("image_name") or "")
        for candidate in self.candidate_output_images(image_name, label_path):
            if candidate.exists():
                return candidate
        return None

    def make_annotation_record(
        self,
        *,
        kind: str,
        path: Path,
        source_abs: str = "",
        source_name: str = "",
        source_stem: str = "",
        source_parent: str = "",
        image_name: str = "",
        image_stem: str = "",
        image_path: str = "",
        image_relative_path: str = "",
        width: int = 0,
        height: int = 0,
        path_tokens: set[str] | None = None,
    ) -> dict[str, object]:
        resolved_source_stem = source_stem or Path(source_name).stem
        resolved_image_stem = image_stem or Path(image_name).stem or path.stem
        return {
            "kind": kind,
            "path": path,
            "mtime": path.stat().st_mtime,
            "source_abs": source_abs.casefold(),
            "source_name": source_name.casefold(),
            "source_file_name": source_name,
            "source_stem": resolved_source_stem.casefold(),
            "source_base_stem": self.base_image_stem(resolved_source_stem),
            "source_parent": source_parent.casefold(),
            "image_name": image_name.casefold(),
            "image_file_name": image_name,
            "image_stem": resolved_image_stem.casefold(),
            "image_base_stem": self.base_image_stem(resolved_image_stem),
            "image_path": image_path,
            "image_relative_path": image_relative_path,
            "path_tokens": path_tokens or self.annotation_path_tokens(path, self.output_dir or path.parent),
            "width": int(width or 0),
            "height": int(height or 0),
        }

    def build_annotation_index(self) -> None:
        if self.output_dir is None:
            self.annotation_index_dir = None
            self.annotation_records.clear()
            return
        output_dir = self.output_dir
        if self.annotation_index_dir == output_dir and self.annotation_records:
            return

        records: list[dict[str, object]] = []
        if output_dir.exists():
            for annotation_path in output_dir.rglob("*_annotation.json"):
                try:
                    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                source = payload.get("source", {}) if isinstance(payload, dict) else {}
                image = payload.get("image", {}) if isinstance(payload, dict) else {}
                if not isinstance(source, dict) or not isinstance(image, dict):
                    continue
                source_path_text = str(source.get("absolute_path") or source.get("path") or "")
                source_parent = Path(source_path_text).parent.name if source_path_text else ""
                path_tokens = self.annotation_path_tokens(annotation_path, output_dir)
                if source_parent:
                    path_tokens.add(source_parent.casefold())
                records.append(
                    self.make_annotation_record(
                        kind="annotation",
                        path=annotation_path,
                        source_abs=source_path_text,
                        source_name=str(source.get("file_name") or ""),
                        source_stem=str(source.get("stem") or ""),
                        source_parent=source_parent,
                        image_name=str(image.get("file_name") or ""),
                        image_stem=str(image.get("stem") or annotation_path.stem.replace("_annotation", "")),
                        image_path=str(image.get("path") or ""),
                        image_relative_path=str(image.get("relative_path") or ""),
                        width=int(image.get("width") or 0),
                        height=int(image.get("height") or 0),
                        path_tokens=path_tokens,
                    )
                )

            for label_path in output_dir.rglob("*.txt"):
                if label_path.parent.name.lower() != "labels":
                    continue
                if label_path.name.startswith("."):
                    continue
                records.append(
                    self.make_annotation_record(
                        kind="yolo",
                        path=label_path,
                        source_stem=label_path.stem,
                        image_name=label_path.stem,
                        image_stem=label_path.stem,
                    )
                )

            for mask_rcnn_path in output_dir.rglob("*.json"):
                if "mask_rcnn" not in {part.casefold() for part in mask_rcnn_path.parts}:
                    continue
                try:
                    payload = json.loads(mask_rcnn_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                image = payload.get("image", {}) if isinstance(payload, dict) else {}
                annotations = payload.get("annotations", []) if isinstance(payload, dict) else []
                if not isinstance(image, dict) or not isinstance(annotations, list) or not annotations:
                    continue
                image_name = str(image.get("file_name") or mask_rcnn_path.stem)
                records.append(
                    self.make_annotation_record(
                        kind="mask_rcnn",
                        path=mask_rcnn_path,
                        source_stem=Path(image_name).stem,
                        image_name=image_name,
                        image_stem=Path(image_name).stem,
                        width=int(image.get("width") or 0),
                        height=int(image.get("height") or 0),
                    )
                )

        self.annotation_index_dir = output_dir
        self.annotation_records = records

    def annotation_match_score(self, record: dict[str, object], image_path: Path, image_shape=None) -> int:
        image_abs = self.safe_resolve_text(image_path)
        name = image_path.name.casefold()
        stem = image_path.stem.casefold()
        score = 0
        if record.get("source_abs") and str(record["source_abs"]) == image_abs:
            score += 1000
        if record.get("source_parent") and record.get("source_parent") == image_path.parent.name.casefold():
            score += 80
        if record.get("source_name") == name:
            score += 240
        if record.get("image_name") == name:
            score += 220
        if record.get("source_stem") == stem:
            score += 120
        if record.get("image_stem") == stem:
            score += 110
        if record.get("source_base_stem") == self.base_image_stem(stem):
            score += 115
        if record.get("image_base_stem") == self.base_image_stem(stem):
            score += 105
        record_tokens = record.get("path_tokens")
        if isinstance(record_tokens, set) and record_tokens:
            image_tokens = self.meaningful_folder_tokens(image_path.parent.parts)
            score += min(90, 30 * len(record_tokens.intersection(image_tokens)))
        if image_shape is not None:
            height = int(image_shape[0])
            width = int(image_shape[1])
            if int(record.get("width") or 0) == width and int(record.get("height") or 0) == height:
                score += 40
        return score

    def image_base_stem_is_unique(self, image_path: Path) -> bool:
        if not self.image_files:
            return True
        target = self.base_image_stem(image_path.stem)
        matches = 0
        for candidate in self.image_files:
            if self.base_image_stem(candidate.stem) == target:
                matches += 1
                if matches > 1:
                    return False
        return True

    def annotation_match_is_reliable(self, record: dict[str, object], image_path: Path) -> bool:
        image_abs = self.safe_resolve_text(image_path)
        name = image_path.name.casefold()
        stem = image_path.stem.casefold()
        base_stem = self.base_image_stem(stem)
        source_abs = str(record.get("source_abs") or "")
        if source_abs and source_abs == image_abs:
            return True

        image_tokens = self.meaningful_folder_tokens(image_path.parent.parts)
        record_tokens = record.get("path_tokens")
        token_match = isinstance(record_tokens, set) and bool(record_tokens.intersection(image_tokens))
        parent_match = bool(record.get("source_parent") and record.get("source_parent") == image_path.parent.name.casefold())
        unique_base = self.image_base_stem_is_unique(image_path)

        exact_name_match = (
            record.get("source_name") == name
            or record.get("image_name") == name
            or record.get("source_stem") == stem
            or record.get("image_stem") == stem
        )
        derived_name_match = record.get("source_base_stem") == base_stem or record.get("image_base_stem") == base_stem
        if exact_name_match:
            return parent_match or token_match or unique_base
        if derived_name_match:
            return parent_match or token_match or unique_base
        return False

    def find_annotation_record_for_image(self, image_path: Path, image_shape=None) -> dict[str, object] | None:
        self.build_annotation_index()
        best: tuple[int, float, dict[str, object]] | None = None
        for record in self.annotation_records:
            score = self.annotation_match_score(record, image_path, image_shape)
            if score < 100:
                continue
            if not self.annotation_match_is_reliable(record, image_path):
                continue
            mtime = float(record.get("mtime") or 0.0)
            if best is None or (score, mtime) > (best[0], best[1]):
                best = (score, mtime, record)
        return best[2] if best is not None else None

    def image_has_existing_annotation(self, image_path: Path) -> bool:
        return self.find_annotation_record_for_image(image_path) is not None

    def first_unprocessed_image_index(self) -> int:
        for index, image_path in enumerate(self.image_files):
            if not self.image_has_existing_annotation(image_path):
                return index
        return 0

    def saved_objects_from_annotation_json(self, path: Path) -> list[SavedObject]:
        if self.image_np is None:
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        objects = payload.get("objects", []) if isinstance(payload, dict) else []
        saved_objects: list[SavedObject] = []
        for object_index, item in enumerate(objects, start=1):
            if not isinstance(item, dict):
                continue
            polygons = item.get("yolo_polygons", [])
            if not isinstance(polygons, list):
                continue
            mask = yolo_edit_polygons_to_mask(polygons, self.image_np.shape)
            if not mask.any():
                continue
            color_values = item.get("color_rgb")
            try:
                color = np.array(color_values, dtype=np.uint8) if color_values else color_for_index(object_index)
                if color.shape != (3,):
                    color = color_for_index(object_index)
            except Exception:
                color = color_for_index(object_index)
            saved_objects.append(
                SavedObject(
                    name=str(item.get("name") or f"Object {object_index}"),
                    mask=mask,
                    color=color,
                    score=float(item.get("score") or 1.0),
                    class_id=int(item.get("class_id") or 0),
                    yolo_polygons=polygons,
                )
            )
        return saved_objects

    def saved_objects_from_yolo_label(self, path: Path) -> list[SavedObject]:
        if self.image_np is None:
            return []
        height, width = self.image_np.shape[:2]
        saved_objects: list[SavedObject] = []
        for object_index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.strip().split()
            if len(parts) < 7 or len(parts[1:]) % 2:
                continue
            try:
                class_id = int(float(parts[0]))
                coords = [float(value) for value in parts[1:]]
            except ValueError:
                continue
            points = []
            for index in range(0, len(coords), 2):
                points.append((coords[index] * width, coords[index + 1] * height))
            polygons = [{"mode": "add", "points": points}]
            mask = yolo_edit_polygons_to_mask(polygons, self.image_np.shape)
            if not mask.any():
                continue
            saved_objects.append(
                SavedObject(
                    name=f"Object {object_index}",
                    mask=mask,
                    color=color_for_index(object_index),
                    score=1.0,
                    class_id=class_id,
                    yolo_polygons=polygons,
                )
            )
        return saved_objects

    def decode_mask_rcnn_segmentation(self, item: dict[str, object], annotation_path: Path) -> np.ndarray | None:
        if self.image_np is None:
            return None
        segmentation = item.get("segmentation")
        if isinstance(segmentation, dict):
            mask_path_text = str(segmentation.get("path") or "")
            if mask_path_text:
                mask_path = annotation_path.parent / mask_path_text
                if mask_path.exists():
                    return load_rgb_image(mask_path)[:, :, 0] > 0
            counts = segmentation.get("counts")
            size = segmentation.get("size")
            if counts is not None and isinstance(size, list) and len(size) == 2:
                try:
                    from pycocotools import mask as mask_utils

                    rle = {"size": [int(size[0]), int(size[1])], "counts": counts.encode("utf-8") if isinstance(counts, str) else counts}
                    return np.asarray(mask_utils.decode(rle), dtype=bool)
                except Exception:
                    return None
        mask_path_text = str(item.get("mask") or "")
        if mask_path_text:
            mask_path = annotation_path.parent / mask_path_text
            if mask_path.exists():
                return load_rgb_image(mask_path)[:, :, 0] > 0
        return None

    def saved_objects_from_mask_rcnn_annotation(self, path: Path) -> list[SavedObject]:
        if self.image_np is None:
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        annotations = payload.get("annotations", []) if isinstance(payload, dict) else []
        if not isinstance(annotations, list):
            return []
        saved_objects: list[SavedObject] = []
        for object_index, item in enumerate(annotations, start=1):
            if not isinstance(item, dict):
                continue
            mask = self.decode_mask_rcnn_segmentation(item, path)
            if mask is None or not mask.any():
                continue
            if mask.shape[:2] != self.image_np.shape[:2]:
                continue
            class_id = int(item.get("class_id", item.get("category_id", 0)) or 0)
            saved_objects.append(
                SavedObject(
                    name=f"Object {object_index}",
                    mask=mask,
                    color=color_for_index(object_index),
                    score=1.0,
                    class_id=class_id,
                    yolo_polygons=mask_to_yolo_edit_polygons(mask),
                )
            )
        return saved_objects

    def prepare_working_image_for_record(self, record: dict[str, object]) -> bool:
        if self.image_np is None:
            return False
        record_width = int(record.get("width") or 0)
        record_height = int(record.get("height") or 0)
        current_height, current_width = self.image_np.shape[:2]
        record_image_name = str(record.get("image_file_name") or record.get("image_name") or "")
        record_stem = str(record.get("image_stem") or "")
        current_stem = self.image_path.stem.casefold() if self.image_path is not None else ""
        working_image_name = self.working_image_name or ""
        needs_output_image = bool(record_image_name and Path(record_image_name).name.casefold() != working_image_name.casefold())
        if record_width and record_height and (record_width != current_width or record_height != current_height):
            needs_output_image = True
        if record_stem and record_stem != current_stem and self.base_image_stem(record_stem) == self.base_image_stem(current_stem):
            needs_output_image = True

        output_image_path = self.find_output_image_for_record(record)
        if not needs_output_image:
            return True
        if output_image_path is None:
            if record_width and record_height and (record_width != current_width or record_height != current_height):
                self.set_status(
                    f"Found annotation for {record_image_name or record_stem}, but its {record_width}x{record_height} image is missing."
                )
                return False
            return True

        image = load_rgb_image(output_image_path)
        self.image_np = np.ascontiguousarray(image)
        self.working_image_name = output_image_path.name
        self.hough_preview_active = False
        self.image_version += 1
        return True

    def restore_existing_annotation_for_current_image(self) -> bool:
        if self.image_path is None or self.image_np is None or self.output_dir is None:
            return False
        record = self.find_annotation_record_for_image(self.image_path, self.image_np.shape)
        if record is None:
            return False
        path = Path(record["path"])
        try:
            if not self.prepare_working_image_for_record(record):
                return False
            if record.get("kind") == "annotation":
                restored = self.saved_objects_from_annotation_json(path)
            elif record.get("kind") == "mask_rcnn":
                restored = self.saved_objects_from_mask_rcnn_annotation(path)
            else:
                restored = self.saved_objects_from_yolo_label(path)
        except Exception as exc:
            self.set_status(f"Could not restore annotation: {exc}")
            return False
        if not restored:
            return False
        self.saved_objects = restored
        self.dirty = False
        self.update_object_list()
        self.set_status(f"Restored {len(restored)} object(s) from {path.name} on {self.working_image_name}")
        return True

    def reset_annotation_state(self) -> None:
        self.points.clear()
        self.saved_objects.clear()
        self.current_mask = None
        self.current_score = 0.0
        self.current_yolo_polygons.clear()
        self.current_yolo_dirty = False
        self.edit_drag_target = None
        self.edit_move_target = None
        self.sam_hole_predicting = False
        self.sam_hole_version += 1
        self.selected_polygon_index = -1
        self.selected_vertex_index = -1
        self.draft_polygon_points.clear()
        self.draft_polygon_active = False
        if hasattr(self, "new_polygon_button"):
            self.new_polygon_button.setChecked(False)
        self.image_ready = False
        self.dirty = False
        self.point_version += 1
        self.clear_edit_undo_stack()
        self.update_object_list()

    def hough_settings(self) -> HoughPreprocessSettings:
        return HoughPreprocessSettings(
            inner_radius_scale=float(self.hough_inner_scale_spin.value()),
            crop_radius_scale=float(self.hough_crop_scale_spin.value()),
            crop_size=int(self.hough_crop_size_spin.value()),
        )

    def initial_image_dialog_dir(self) -> str:
        candidates = [
            self.image_folder,
            self.image_path.parent if self.image_path is not None else None,
            PREFERRED_IMAGE_FOLDER,
            PROJECT_ROOT,
        ]
        for candidate in candidates:
            if candidate is not None and candidate.exists():
                return str(candidate)
        return ""

    def selected_hough_output(self) -> str:
        return str(self.hough_image_combo.currentData() or "full")

    def working_hough_variant(self) -> str | None:
        if self.image_path is None or not self.working_image_name:
            return None
        stem = Path(self.working_image_name).stem
        prefix = f"{self.image_path.stem}_hough_"
        if stem.startswith(prefix):
            variant = stem[len(prefix):]
            if variant in {"full", "crop", "mask", "debug"}:
                return variant
        return None

    def current_hough_settings_payload(self) -> dict[str, float | int]:
        return {
            "inner_radius_scale": float(self.hough_inner_scale_spin.value()),
            "crop_radius_scale": float(self.hough_crop_scale_spin.value()),
            "crop_size": int(self.hough_crop_size_spin.value()),
        }

    @staticmethod
    def hough_settings_from_payload(payload: dict[str, object] | None) -> HoughPreprocessSettings:
        payload = payload if isinstance(payload, dict) else {}
        return HoughPreprocessSettings(
            inner_radius_scale=float(payload.get("inner_radius_scale", 0.86)),
            crop_radius_scale=float(payload.get("crop_radius_scale", 0.55)),
            crop_size=int(payload.get("crop_size", 0)),
        )

    @staticmethod
    def clone_template_polygons(polygons: Sequence[dict[str, object]], scale_x: float, scale_y: float) -> list[dict[str, object]]:
        cloned: list[dict[str, object]] = []
        for item in polygons:
            mode = str(item.get("mode") or "add")
            points = item.get("points", [])
            cloned.append(
                {
                    "mode": mode,
                    "points": [
                        (float(point[0]) * scale_x, float(point[1]) * scale_y)
                        for point in points  # type: ignore[index]
                    ],
                }
            )
        return cloned

    def clone_template_saved_object(
        self,
        item: dict[str, object],
        index: int,
        scale_x: float,
        scale_y: float,
    ) -> SavedObject | None:
        if self.image_np is None:
            return None
        polygons = self.clone_template_polygons(item.get("yolo_polygons", []), scale_x, scale_y)  # type: ignore[arg-type]
        mask = yolo_edit_polygons_to_mask(polygons, self.image_np.shape)
        if not mask.any():
            return None
        color_values = item.get("color_rgb")
        try:
            color = np.array(color_values, dtype=np.uint8) if color_values else color_for_index(index)
            if color.shape != (3,):
                color = color_for_index(index)
        except Exception:
            color = color_for_index(index)
        return SavedObject(
            name=str(item.get("name") or f"Template {index}"),
            mask=mask,
            color=color,
            score=float(item.get("score") or 1.0),
            class_id=int(item.get("class_id") or 0),
            yolo_polygons=polygons,
        )

    def template_object_payload(self, saved: SavedObject, index: int) -> dict[str, object]:
        polygons = saved.yolo_polygons or mask_to_yolo_edit_polygons(
            saved.mask,
            epsilon=float(self.yolo_epsilon_spin.value()),
            min_area=float(self.yolo_min_area_spin.value()),
        )
        return {
            "name": saved.name or f"Template {index}",
            "class_id": int(saved.class_id),
            "score": float(saved.score),
            "color_rgb": [int(value) for value in saved.color],
            "yolo_polygons": [
                {
                    "mode": str(item.get("mode") or "add"),
                    "points": [(float(x), float(y)) for x, y in item.get("points", [])],  # type: ignore[misc]
                }
                for item in polygons
            ],
        }

    def update_template_status_label(self) -> None:
        if not hasattr(self, "template_status_label"):
            return
        if not self.mask_template:
            self.template_status_label.setText("Template: none")
            return
        count = len(self.mask_template.get("objects", []))  # type: ignore[arg-type]
        mode = str(self.mask_template.get("mode") or "original")
        shape = self.mask_template.get("shape", (0, 0))
        self.template_status_label.setText(f"Template: {count} object(s), {mode}, {shape[1]}x{shape[0]}")  # type: ignore[index]

    def save_mask_template(self) -> None:
        if self.image_np is None:
            self.set_status("Open an image first")
            return
        if self.current_mask is not None:
            answer = QMessageBox.question(
                self,
                "Active mask",
                "Accept the active mask into objects before saving the template?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )
            if answer == QMessageBox.Yes:
                self.accept_current_mask()
        if not self.saved_objects:
            self.set_status("No saved objects to use as template")
            return
        hough_variant = self.working_hough_variant()
        height, width = self.image_np.shape[:2]
        self.mask_template = {
            "mode": f"hough_{hough_variant}" if hough_variant else "original",
            "hough_variant": hough_variant,
            "hough_settings": self.current_hough_settings_payload(),
            "shape": (height, width),
            "objects": [
                self.template_object_payload(saved, index)
                for index, saved in enumerate(self.saved_objects, start=1)
            ],
        }
        self.update_template_status_label()
        self.set_status("Template captured for this session")

    def confirm_replace_with_template(self) -> bool:
        has_existing = bool(self.saved_objects or self.points or self.current_mask is not None)
        if not has_existing:
            return True
        answer = QMessageBox.question(
            self,
            "Apply template",
            "Replace current objects and prompts with the saved template?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def apply_template_objects_to_working_image(self, template: dict[str, object]) -> bool:
        if self.image_np is None:
            return False
        template_shape = template.get("shape", self.image_np.shape[:2])
        if not isinstance(template_shape, tuple) or len(template_shape) < 2:
            template_shape = self.image_np.shape[:2]
        template_height = max(1.0, float(template_shape[0]))
        template_width = max(1.0, float(template_shape[1]))
        height, width = self.image_np.shape[:2]
        scale_x = float(width) / template_width
        scale_y = float(height) / template_height
        objects: list[SavedObject] = []
        for index, item in enumerate(template.get("objects", []), start=1):  # type: ignore[arg-type]
            if not isinstance(item, dict):
                continue
            saved = self.clone_template_saved_object(item, index, scale_x, scale_y)
            if saved is not None:
                objects.append(saved)
        if not objects:
            self.set_status("Template had no usable polygons for this image")
            return False
        if self.saved_objects or self.points or self.current_mask is not None:
            self.push_undo_state("apply template")
        self.points.clear()
        self.current_mask = None
        self.current_score = 0.0
        self.current_yolo_polygons.clear()
        self.current_yolo_dirty = False
        self.saved_objects = objects
        self.edit_drag_target = None
        self.edit_move_target = None
        self.selected_polygon_index = -1
        self.selected_vertex_index = -1
        self.draft_polygon_points.clear()
        self.draft_polygon_active = False
        if hasattr(self, "new_polygon_button"):
            self.new_polygon_button.setChecked(False)
        self.dirty = True
        self.point_version += 1
        self.update_object_list()
        if hasattr(self, "polygon_edit_check") and self.saved_objects:
            self.polygon_edit_check.setChecked(True)
        self.render_canvas(fit=True)
        self.set_status(f"Template applied: {len(objects)} object(s)")
        return True

    def apply_mask_template(self) -> None:
        if not self.mask_template:
            self.set_status("No template saved")
            return
        if self.image_path is None or self.original_image_np is None:
            self.set_status("Open an image first")
            return
        if not self.confirm_replace_with_template():
            return
        hough_variant = self.mask_template.get("hough_variant")
        if hough_variant:
            if self.hough_running:
                self.set_status("Wait for current Hough preprocessing")
                return
            self.pending_template = self.mask_template
            self.run_hough_preprocess(
                action_after_run="template",
                output_variant=str(hough_variant),
                settings=self.hough_settings_from_payload(self.mask_template.get("hough_settings")),  # type: ignore[arg-type]
                confirm_discard=False,
            )
            return

        self.set_working_image(
            self.original_image_np.copy(),
            self.image_path.name,
            "Applying template to original image",
            prepare_sam=True,
            fit=True,
        )
        self.apply_template_objects_to_working_image(self.mask_template)

    def set_working_image(
        self,
        image: np.ndarray,
        image_name: str,
        status: str,
        *,
        prepare_sam: bool,
        fit: bool = True,
    ) -> None:
        self.image_np = np.ascontiguousarray(image)
        self.working_image_name = image_name
        self.image_version += 1
        self.reset_annotation_state()
        self.render_canvas(fit=fit)
        self.set_status(status)
        if prepare_sam:
            self.load_model_async(force=False)

    def refresh_hough_preview(self) -> None:
        if self.hough_result is None or not self.hough_preview_active or self.image_path is None:
            return
        output_variant = "debug" if self.hough_debug_check.isChecked() else self.selected_hough_output()
        image = hough_result_image(self.hough_result, output_variant)
        image_name = hough_output_name(self.image_path, output_variant)
        self.set_working_image(
            image,
            image_name,
            "Hough preview ready. Use it for SAM or restore original.",
            prepare_sam=False,
            fit=True,
        )
        self.image_ready = False

    def preview_hough_preprocess(self) -> None:
        self.run_hough_preprocess(action_after_run="preview")

    def run_hough_preprocess(
        self,
        action_after_run: str,
        output_variant: str | None = None,
        settings: HoughPreprocessSettings | None = None,
        confirm_discard: bool = True,
    ) -> None:
        if self.image_path is None or self.original_image_np is None:
            self.set_status("Open an image first")
            return
        if self.hough_running:
            if action_after_run == "use":
                self.hough_action_after_run = "use"
                self.hough_requested_output = output_variant or self.selected_hough_output()
                self.set_status("Hough preprocessing is running. Will use result for SAM.")
            else:
                self.set_status("Hough preprocessing is running")
            return
        if confirm_discard and not self.hough_preview_active and not self.confirm_discard_work():
            return

        image_path = self.image_path
        source_image = self.original_image_np.copy()
        settings = settings or self.hough_settings()
        self.hough_running = True
        self.hough_action_after_run = action_after_run
        self.hough_requested_output = output_variant or self.selected_hough_output()
        self.set_status("Running Hough preprocessing")

        def worker() -> None:
            try:
                hough_result = preprocess_hough_circle(source_image, settings)
                self.messages.put(("hough_ready", (image_path, hough_result)))
            except Exception as exc:
                self.messages.put(("hough_error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def apply_hough_to_sam(self) -> None:
        if self.image_path is None:
            self.set_status("Open an image first")
            return
        if self.hough_result is None:
            self.run_hough_preprocess(action_after_run="use", output_variant=self.selected_hough_output())
            return
        if not self.hough_preview_active and not self.confirm_discard_work():
            return

        output_variant = self.selected_hough_output()
        self.use_hough_image_for_sam(output_variant)

    def use_hough_image_for_sam(self, output_variant: str) -> None:
        if self.image_path is None or self.hough_result is None:
            return
        image = hough_result_image(self.hough_result, output_variant)
        self.hough_preview_active = False
        self.set_working_image(
            image,
            hough_output_name(self.image_path, output_variant),
            f"Using Hough {output_variant} image for SAM",
            prepare_sam=True,
            fit=True,
        )

    def restore_original_image(self) -> None:
        if self.image_path is None or self.original_image_np is None:
            self.set_status("Open an image first")
            return
        if not self.confirm_discard_work():
            return
        self.hough_preview_active = False
        self.set_working_image(
            self.original_image_np,
            self.image_path.name,
            "Restored original image",
            prepare_sam=True,
            fit=True,
        )

    def save_hough_result(self) -> None:
        if self.image_path is None:
            self.set_status("Open an image first")
            return
        if self.hough_result is None:
            self.set_status("Preview Hough first")
            return
        if self.output_dir is None:
            default_dir = self.image_path.parent / "sam2_interactive_results"
            selected = QFileDialog.getExistingDirectory(self, "Choose output folder", str(default_dir.parent))
            if not selected:
                return
            self.output_dir = Path(selected)
            self.output_dir_user_selected = True
            self.update_output_label()
        try:
            outputs = save_hough_preprocess_result(self.image_path, self.output_dir, self.hough_result)
        except Exception as exc:
            QMessageBox.critical(self, "Save Hough result", str(exc))
            return
        self.set_status(f"Saved Hough result: {outputs['full']}")

    def sync_polygon_edit_mode_after_load(self, restored: bool, was_polygon_editing: bool) -> None:
        if not hasattr(self, "polygon_edit_check"):
            return
        has_labels = bool(restored and self.saved_objects)
        if has_labels and was_polygon_editing:
            self.polygon_edit_check.setChecked(True)
        elif not has_labels:
            self.polygon_edit_check.setChecked(False)

    def load_image_path(self, path: Path) -> None:
        was_polygon_editing = bool(
            self.polygon_edit_check.isChecked() if hasattr(self, "polygon_edit_check") else False
        )
        try:
            self.image_path = path
            self.original_image_np = load_rgb_image(path)
            self.image_np = self.original_image_np.copy()
            self.working_image_name = path.name
            self.hough_result = None
            self.hough_preview_active = False
            self.image_version += 1
        except Exception as exc:
            QMessageBox.critical(self, "Open image", str(exc))
            return

        self.reset_annotation_state()
        restored = self.restore_existing_annotation_for_current_image()
        self.sync_polygon_edit_mode_after_load(restored, was_polygon_editing)
        self.render_canvas(fit=True)
        self.update_image_counter()
        self.setWindowTitle(f"SAM 2 Studio - {path.name}")
        if not restored:
            self.set_status(f"Opened {path.name}")
        self.load_model_async(force=False)

    def open_image(self) -> None:
        if not self.confirm_discard_work():
            return
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Open image",
            self.initial_image_dialog_dir(),
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return
        selected = Path(path)
        self.image_files = [selected]
        self.image_index = 0
        self.image_folder = None
        if self.output_dir is None:
            self.output_dir = selected.parent / "sam2_interactive_results"
            self.output_dir_user_selected = False
            self.update_output_label()
        self.load_image_path(selected)

    def open_folder(self) -> None:
        if not self.confirm_discard_work():
            return
        folder = QFileDialog.getExistingDirectory(self, "Open image folder", self.initial_image_dialog_dir())
        if not folder:
            return
        folder_path = Path(folder)
        images = collect_images(
            folder_path,
            recursive=self.folder_recursive_check.isChecked(),
            extensions=DEFAULT_EXTENSIONS,
        )
        if not images:
            QMessageBox.information(self, "Open folder", "No supported images found in this folder.")
            return
        self.image_folder = folder_path
        self.image_files = images
        if self.output_dir is None or not self.output_dir_user_selected:
            self.output_dir = folder_path / "sam2_dataset"
            self.output_dir_user_selected = False
        self.invalidate_annotation_index()
        self.image_index = self.first_unprocessed_image_index()
        self.update_output_label()
        self.load_image_path(self.image_files[self.image_index])

    def choose_output_folder(self) -> None:
        initial_dir = str(self.output_dir or (self.image_path.parent if self.image_path else PROJECT_ROOT))
        selected = QFileDialog.getExistingDirectory(self, "Choose output folder", initial_dir)
        if not selected:
            return
        self.output_dir = Path(selected)
        self.output_dir_user_selected = True
        self.invalidate_annotation_index()
        self.update_output_label()
        if self.image_files and self.image_folder is not None and not self.has_unsaved_work():
            self.image_index = self.first_unprocessed_image_index()
            self.load_image_path(self.image_files[self.image_index])
            return
        if self.image_path is not None and not self.has_unsaved_work():
            self.restore_existing_annotation_for_current_image()
            self.render_canvas()
        self.set_status(f"Output folder: {self.output_dir}")

    def jump_to_image(self) -> None:
        if not self.image_files:
            return
        index = int(self.image_jump_spin.value()) - 1
        if not (0 <= index < len(self.image_files)) or index == self.image_index:
            return
        if not self.confirm_discard_work():
            return
        self.image_index = index
        self.load_image_path(self.image_files[self.image_index])

    def previous_image(self) -> None:
        if not self.image_files or self.image_index <= 0:
            return
        if not self.confirm_discard_work():
            return
        self.image_index -= 1
        self.load_image_path(self.image_files[self.image_index])

    def next_image(self) -> None:
        if not self.image_files or self.image_index >= len(self.image_files) - 1:
            return
        if not self.confirm_discard_work():
            return
        self.image_index += 1
        self.load_image_path(self.image_files[self.image_index])

    def load_model_async(self, force: bool = False) -> None:
        requested = (self.model_combo.currentText(), self.device_combo.currentText())
        if self.model_loading:
            self.set_status("Model is loading")
            return
        if not force and self.predictor is not None and self.model_key == requested:
            if self.image_np is not None and not self.image_ready:
                self.embed_current_image_async()
            else:
                self.set_status("Model ready")
            return

        self.model_loading = True
        self.image_ready = False
        self.predictor = None
        self.device = None
        self.model_key = None
        self.set_status(f"Loading {requested[0]} model")

        def worker() -> None:
            try:
                predictor, device = build_image_predictor(
                    requested[0],
                    checkpoint=None,
                    model_cfg=None,
                    device_name=requested[1],
                    log=lambda message: self.messages.put(("status", message)),
                )
                self.messages.put(("model_ready", (predictor, device, requested)))
            except Exception as exc:
                self.messages.put(("error", f"Model load failed: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def embed_current_image_async(self) -> None:
        if self.predictor is None or self.device is None:
            self.load_model_async(force=False)
            return
        if self.image_np is None or self.embedding:
            return

        image = self.image_np
        predictor = self.predictor
        device = self.device
        version = self.image_version
        self.embedding = True
        self.image_ready = False
        self.current_mask = None
        self.points.clear()
        self.point_version += 1
        self.render_canvas()
        self.set_status("Preparing image")

        def worker() -> None:
            try:
                with torch.inference_mode(), inference_autocast(device):
                    predictor.set_image(image)
                self.messages.put(("image_ready", version))
            except Exception as exc:
                self.messages.put(("error", f"Image preparation failed: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def poll_messages(self) -> None:
        try:
            while True:
                message_type, message_data = self.messages.get_nowait()
                if message_type == "status":
                    self.set_status(str(message_data))
                elif message_type == "model_ready":
                    predictor, device, requested = message_data
                    self.predictor = predictor
                    self.device = device
                    self.model_key = requested
                    self.model_loading = False
                    self.set_status(f"Model ready on {device}")
                    if self.image_np is not None:
                        self.embed_current_image_async()
                elif message_type == "image_ready":
                    version = int(message_data)
                    if version != self.image_version:
                        self.embedding = False
                        if self.image_np is not None and not self.hough_preview_active:
                            self.embed_current_image_async()
                        continue
                    self.embedding = False
                    self.image_ready = True
                    self.set_status("Image ready")
                    self.render_canvas()
                elif message_type == "prediction":
                    version, mask, score = message_data
                    self.predicting = False
                    if version == self.point_version:
                        self.current_mask = mask
                        self.current_score = score
                        self.current_yolo_polygons = mask_to_yolo_edit_polygons(
                            mask,
                            epsilon=float(self.yolo_epsilon_spin.value()),
                            min_area=float(self.yolo_min_area_spin.value()),
                        )
                        self.current_yolo_dirty = False
                        self.edit_drag_target = None
                        self.selected_polygon_index = -1
                        self.selected_vertex_index = -1
                        self.draft_polygon_points.clear()
                        self.draft_polygon_active = False
                        self.set_status(f"Active mask score {score:.3f}")
                        self.render_canvas()
                    if self.pending_prediction or version != self.point_version:
                        self.pending_prediction = False
                        self.predict_current_mask_async()
                elif message_type == "hole_prediction":
                    version, image_version, target_index, hole_mask, score = message_data
                    self.sam_hole_predicting = False
                    if version != self.sam_hole_version or image_version != self.image_version:
                        continue
                    if self.apply_hole_mask_to_target(int(target_index), hole_mask, label="SAM hole"):
                        self.set_status(f"SAM hole score {float(score):.3f}")
                elif message_type == "hole_done":
                    if int(message_data) == self.sam_hole_version:
                        self.sam_hole_predicting = False
                elif message_type == "error":
                    self.model_loading = False
                    self.embedding = False
                    self.predicting = False
                    self.sam_hole_predicting = False
                    self.set_status(str(message_data))
                    QMessageBox.critical(self, "SAM 2 Studio", str(message_data))
                elif message_type == "hough_ready":
                    image_path, hough_result = message_data
                    self.hough_running = False
                    if image_path != self.image_path:
                        self.hough_action_after_run = None
                        self.hough_requested_output = None
                        self.pending_template = None
                        continue
                    self.hough_result = hough_result
                    action_after_run = self.hough_action_after_run or "preview"
                    output_variant = self.hough_requested_output or self.selected_hough_output()
                    self.hough_action_after_run = None
                    self.hough_requested_output = None
                    if action_after_run == "use":
                        self.use_hough_image_for_sam(output_variant)
                        self.set_status(f"Using Hough {output_variant}: {hough_result.method}")
                    elif action_after_run == "template":
                        template = self.pending_template
                        self.pending_template = None
                        self.hough_preview_active = False
                        image = hough_result_image(hough_result, output_variant)
                        self.set_working_image(
                            image,
                            hough_output_name(self.image_path, output_variant),
                            f"Using Hough {output_variant} for template",
                            prepare_sam=True,
                            fit=True,
                        )
                        if template is not None:
                            self.apply_template_objects_to_working_image(template)
                    else:
                        self.hough_preview_active = True
                        self.refresh_hough_preview()
                        self.set_status(f"Hough {hough_result.mode}: {hough_result.method}")
                elif message_type == "hough_error":
                    self.hough_running = False
                    self.hough_action_after_run = None
                    self.hough_requested_output = None
                    self.pending_template = None
                    self.set_status(str(message_data))
                    QMessageBox.critical(self, "Hough preprocessing", str(message_data))
        except queue.Empty:
            pass

    def handle_canvas_click(self, x: float, y: float, forced_label: int | None) -> None:
        if self.image_np is None:
            self.set_status("Open an image first")
            return
        if self.hough_preview_active:
            self.set_status("Use Hough For SAM before point selection")
            return
        if not self.image_ready:
            self.set_status("Wait for image preparation")
            return
        label = forced_label
        if label is None:
            label = 1 if self.add_button.isChecked() else 0
        self.points.append((x, y, label))
        self.point_version += 1
        self.render_canvas()
        self.predict_current_mask_async()

    def predict_current_mask_async(self) -> None:
        if self.predictor is None or self.device is None or not self.image_ready:
            return
        if not self.points:
            self.current_mask = None
            self.current_yolo_polygons.clear()
            self.current_yolo_dirty = False
            self.render_canvas()
            return
        if not any(label == 1 for _x, _y, label in self.points):
            self.current_mask = None
            self.current_yolo_polygons.clear()
            self.current_yolo_dirty = False
            self.set_status("Add at least one foreground point")
            self.render_canvas()
            return
        if self.predicting:
            self.pending_prediction = True
            return

        coords = np.array([(x, y) for x, y, _label in self.points], dtype=np.float32)
        labels = np.array([label for _x, _y, label in self.points], dtype=np.int32)
        version = self.point_version
        predictor = self.predictor
        device = self.device
        multimask = len(self.points) == 1

        self.predicting = True
        self.set_status("Predicting mask")

        def worker() -> None:
            try:
                with torch.inference_mode(), inference_autocast(device):
                    masks, scores, _low_res = predictor.predict(
                        point_coords=coords,
                        point_labels=labels,
                        multimask_output=multimask,
                    )
                best_index = int(np.argmax(scores))
                self.messages.put(("prediction", (version, masks[best_index].astype(bool), float(scores[best_index]))))
            except Exception as exc:
                self.messages.put(("error", f"Prediction failed: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def predict_sam_hole_async(self, x: float, y: float) -> None:
        if self.predictor is None or self.device is None or not self.image_ready:
            self.set_status("Wait for image preparation")
            return
        if self.hough_preview_active:
            self.set_status("Use Hough For SAM before hole selection")
            return
        if self.predicting or self.sam_hole_predicting:
            self.set_status("Wait for current SAM prediction")
            return
        target_index = self.sam_hole_target_index()
        if target_index is None:
            self.set_status("Create an active mask or select an object first")
            return
        target_mask = self.mask_for_target(target_index)
        if target_mask is None or not target_mask.any():
            self.set_status("Target mask is empty")
            return
        if not self.target_contains_point(target_index, x, y):
            self.set_status("Click inside the target mask to make a hole")
            return

        coords = np.array([(x, y)], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        target_mask = target_mask.astype(bool).copy()
        predictor = self.predictor
        device = self.device
        self.sam_hole_predicting = True
        self.sam_hole_version += 1
        version = self.sam_hole_version
        image_version = self.image_version
        self.set_status("Predicting SAM hole")

        def worker() -> None:
            try:
                with torch.inference_mode(), inference_autocast(device):
                    masks, scores, _low_res = predictor.predict(
                        point_coords=coords,
                        point_labels=labels,
                        multimask_output=True,
                    )
                best: tuple[float, np.ndarray] | None = None
                for index, score in sorted(
                    enumerate(scores),
                    key=lambda item: float(item[1]),
                    reverse=True,
                ):
                    clipped = masks[index].astype(bool) & target_mask
                    if not clipped.any():
                        continue
                    best = (float(score), clipped)
                    break
                if best is None:
                    self.messages.put(("status", "SAM hole did not overlap the target mask"))
                    self.messages.put(("hole_done", version))
                    return
                score, clipped_mask = best
                self.messages.put(("hole_prediction", (version, image_version, target_index, clipped_mask, score)))
            except Exception as exc:
                self.messages.put(("error", f"SAM hole prediction failed: {exc}"))

        threading.Thread(target=worker, daemon=True).start()

    def undo_point(self) -> None:
        if not self.points:
            return
        self.points.pop()
        self.point_version += 1
        if not self.points:
            self.current_mask = None
            self.current_yolo_polygons.clear()
            self.current_yolo_dirty = False
            self.render_canvas()
            self.set_status("Prompt removed")
            return
        self.render_canvas()
        self.predict_current_mask_async()

    def clear_points(self) -> None:
        self.points.clear()
        self.current_mask = None
        self.current_score = 0.0
        self.current_yolo_polygons.clear()
        self.current_yolo_dirty = False
        self.point_version += 1
        self.pending_prediction = False
        self.render_canvas()
        self.set_status("Prompts cleared")

    @staticmethod
    def clone_yolo_polygons(polygons: Sequence[dict[str, object]]) -> list[dict[str, object]]:
        cloned: list[dict[str, object]] = []
        for item in polygons:
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
                cloned.append({"mode": mode, "points": points})
        return cloned

    def clone_saved_objects(self, saved_objects: Sequence[SavedObject]) -> list[SavedObject]:
        return [
            SavedObject(
                name=saved.name,
                mask=saved.mask.copy(),
                color=saved.color.copy(),
                score=float(saved.score),
                class_id=int(saved.class_id),
                yolo_polygons=self.clone_yolo_polygons(saved.yolo_polygons),
            )
            for saved in saved_objects
        ]

    def annotation_snapshot(self, label: str) -> dict[str, object]:
        return {
            "label": label,
            "points": list(self.points),
            "current_mask": None if self.current_mask is None else self.current_mask.copy(),
            "current_score": float(self.current_score),
            "current_yolo_polygons": self.clone_yolo_polygons(self.current_yolo_polygons),
            "current_yolo_dirty": bool(self.current_yolo_dirty),
            "saved_objects": self.clone_saved_objects(self.saved_objects),
            "selected_object_row": self.objects_list.currentRow() if hasattr(self, "objects_list") else -1,
            "selected_polygon_index": int(self.selected_polygon_index),
            "selected_vertex_index": int(self.selected_vertex_index),
            "draft_polygon_points": list(self.draft_polygon_points),
            "draft_polygon_active": bool(self.draft_polygon_active),
            "new_polygon_checked": bool(
                self.new_polygon_button.isChecked() if hasattr(self, "new_polygon_button") else False
            ),
            "dirty": bool(self.dirty),
        }

    def update_undo_button_state(self) -> None:
        if hasattr(self, "undo_edit_button"):
            self.undo_edit_button.setEnabled(bool(self.edit_undo_stack))

    def clear_edit_undo_stack(self) -> None:
        self.edit_undo_stack.clear()
        self.update_undo_button_state()

    def push_undo_state(self, label: str) -> None:
        if self.image_np is None:
            return
        self.edit_undo_stack.append(self.annotation_snapshot(label))
        if len(self.edit_undo_stack) > self.edit_undo_limit:
            del self.edit_undo_stack[0 : len(self.edit_undo_stack) - self.edit_undo_limit]
        self.update_undo_button_state()

    def undo_edit(self) -> None:
        if not self.edit_undo_stack:
            if self.points:
                self.undo_point()
                return
            self.set_status("No edit to undo")
            return
        snapshot = self.edit_undo_stack.pop()
        self.points = list(snapshot.get("points", []))  # type: ignore[arg-type]
        mask = snapshot.get("current_mask")
        self.current_mask = None if mask is None else np.asarray(mask, dtype=bool).copy()
        self.current_score = float(snapshot.get("current_score") or 0.0)
        self.current_yolo_polygons = self.clone_yolo_polygons(
            snapshot.get("current_yolo_polygons", [])  # type: ignore[arg-type]
        )
        self.current_yolo_dirty = bool(snapshot.get("current_yolo_dirty", False))
        self.saved_objects = self.clone_saved_objects(
            snapshot.get("saved_objects", [])  # type: ignore[arg-type]
        )
        self.edit_drag_target = None
        self.edit_move_target = None
        self.selected_polygon_index = int(snapshot.get("selected_polygon_index", -1))
        self.selected_vertex_index = int(snapshot.get("selected_vertex_index", -1))
        self.draft_polygon_points = list(snapshot.get("draft_polygon_points", []))  # type: ignore[arg-type]
        self.draft_polygon_active = bool(snapshot.get("draft_polygon_active", False))
        if hasattr(self, "new_polygon_button"):
            self.new_polygon_button.setChecked(bool(snapshot.get("new_polygon_checked", False)))
        self.dirty = bool(snapshot.get("dirty", True))
        self.point_version += 1
        self.pending_prediction = False
        self.update_object_list()
        row = int(snapshot.get("selected_object_row", -1))
        if self.saved_objects and 0 <= row < len(self.saved_objects):
            self.objects_list.setCurrentRow(row)
        self.selected_polygon_index = int(snapshot.get("selected_polygon_index", -1))
        self.selected_vertex_index = int(snapshot.get("selected_vertex_index", -1))
        self.update_undo_button_state()
        self.render_canvas()
        self.set_status(f"Undid {snapshot.get('label', 'edit')}")

    def accept_current_mask(self) -> None:
        if self.current_mask is None:
            self.set_status("No active mask")
            return
        self.push_undo_state("accept mask")
        yolo_polygons = self.current_yolo_polygons or mask_to_yolo_edit_polygons(
            self.current_mask,
            epsilon=float(self.yolo_epsilon_spin.value()),
            min_area=float(self.yolo_min_area_spin.value()),
        )
        object_id = len(self.saved_objects) + 1
        saved = SavedObject(
            name=f"Object {object_id}",
            mask=self.current_mask.copy(),
            color=color_for_index(object_id),
            score=self.current_score,
            class_id=int(self.class_spin.value()),
            yolo_polygons=[{"mode": item["mode"], "points": list(item["points"])} for item in yolo_polygons],
        )
        self.saved_objects.append(saved)
        self.clear_points()
        self.dirty = True
        self.update_object_list()
        self.objects_list.setCurrentRow(len(self.saved_objects) - 1)
        self.set_status(f"Added {saved.name}")

    def remove_selected_object(self) -> None:
        row = self.objects_list.currentRow()
        if row < 0:
            return
        self.push_undo_state("remove object")
        del self.saved_objects[row]
        self.edit_drag_target = None
        self.edit_move_target = None
        self.selected_polygon_index = -1
        self.selected_vertex_index = -1
        self.dirty = True
        self.update_object_list()
        self.render_canvas()
        self.set_status("Object removed")

    def clear_saved_objects(self) -> None:
        if not self.saved_objects:
            return
        if QMessageBox.question(self, "Clear objects", "Clear all saved objects?") != QMessageBox.Yes:
            return
        self.push_undo_state("clear objects")
        self.saved_objects.clear()
        self.edit_drag_target = None
        self.edit_move_target = None
        self.selected_polygon_index = -1
        self.selected_vertex_index = -1
        self.dirty = True
        self.update_object_list()
        self.render_canvas()
        self.set_status("Objects cleared")

    def on_selected_object_changed(self, _row: int) -> None:
        self.edit_drag_target = None
        self.edit_move_target = None
        self.selected_polygon_index = -1
        self.selected_vertex_index = -1
        self.render_canvas()

    def selected_object_index(self) -> int:
        row = self.objects_list.currentRow()
        return row if 0 <= row < len(self.saved_objects) else -1

    def active_polygon_target_index(self) -> int | None:
        if self.current_mask is not None:
            return -1
        object_index = self.selected_object_index()
        return object_index if object_index >= 0 else None

    def sam_hole_target_index(self) -> int | None:
        if self.current_mask is not None:
            return -1
        object_index = self.selected_object_index()
        if object_index < 0 and len(self.saved_objects) == 1:
            self.objects_list.setCurrentRow(0)
            object_index = 0
        return object_index if object_index >= 0 else None

    def selected_move_target_index(self) -> int | None:
        mode = str(self.move_target_combo.currentData() if hasattr(self, "move_target_combo") else "auto")
        if mode == "current":
            return -1 if self.current_mask is not None else None
        if mode == "object":
            object_index = self.selected_object_index()
            return object_index if object_index >= 0 else None
        return self.active_polygon_target_index()

    def rendered_polygon_target_index(self) -> int:
        if self.sam_hole_mode_enabled():
            hole_target_index = self.sam_hole_target_index()
            if hole_target_index is not None:
                return hole_target_index
        if hasattr(self, "whole_mask_drag_check") and self.whole_mask_drag_check.isChecked():
            move_target_index = self.selected_move_target_index()
            if move_target_index is not None:
                return move_target_index
        target_index = self.active_polygon_target_index()
        return target_index if target_index is not None else -2

    def polygons_for_target(self, target_index: int) -> list[dict[str, object]]:
        if target_index == -1:
            return self.current_yolo_polygons
        return self.saved_objects[target_index].yolo_polygons

    def mask_for_target(self, target_index: int) -> np.ndarray | None:
        if target_index == -1:
            return self.current_mask
        if 0 <= target_index < len(self.saved_objects):
            return self.saved_objects[target_index].mask
        return None

    def ensure_polygons_for_target(self, target_index: int) -> bool:
        polygons = self.polygons_for_target(target_index)
        if polygons:
            return True
        mask = self.mask_for_target(target_index)
        if mask is None or not mask.any():
            return False
        polygons.extend(
            mask_to_yolo_edit_polygons(
                mask,
                epsilon=float(self.yolo_epsilon_spin.value()),
                min_area=float(self.yolo_min_area_spin.value()),
            )
        )
        return bool(polygons)

    def rebuild_target_mask_from_polygons(self, target_index: int) -> None:
        if self.image_np is None:
            return
        if target_index == -1:
            self.current_mask = yolo_edit_polygons_to_mask(self.current_yolo_polygons, self.image_np.shape)
            self.current_yolo_dirty = True
            return
        if 0 <= target_index < len(self.saved_objects):
            saved = self.saved_objects[target_index]
            saved.mask = yolo_edit_polygons_to_mask(saved.yolo_polygons, self.image_np.shape)
            self.dirty = True

    def sync_edit_polygon_detail_controls(self) -> None:
        if hasattr(self, "edit_yolo_epsilon_spin"):
            previous_block = self.edit_yolo_epsilon_spin.blockSignals(True)
            try:
                self.edit_yolo_epsilon_spin.setValue(float(self.yolo_epsilon_spin.value()))
            finally:
                self.edit_yolo_epsilon_spin.blockSignals(previous_block)
        if hasattr(self, "edit_yolo_min_area_spin"):
            previous_block = self.edit_yolo_min_area_spin.blockSignals(True)
            try:
                self.edit_yolo_min_area_spin.setValue(float(self.yolo_min_area_spin.value()))
            finally:
                self.edit_yolo_min_area_spin.blockSignals(previous_block)

    def set_yolo_epsilon_from_edit(self, value: float) -> None:
        previous_block = self.yolo_epsilon_spin.blockSignals(True)
        try:
            self.yolo_epsilon_spin.setValue(float(value))
        finally:
            self.yolo_epsilon_spin.blockSignals(previous_block)
        self.on_yolo_polygon_settings_changed(value)

    def set_yolo_min_area_from_edit(self, value: float) -> None:
        previous_block = self.yolo_min_area_spin.blockSignals(True)
        try:
            self.yolo_min_area_spin.setValue(float(value))
        finally:
            self.yolo_min_area_spin.blockSignals(previous_block)
        self.on_yolo_polygon_settings_changed(value)

    def polygon_detail_target_index(self) -> int | None:
        if self.current_mask is not None:
            return -1
        if hasattr(self, "whole_mask_drag_check") and self.whole_mask_drag_check.isChecked():
            move_target_index = self.selected_move_target_index()
            if move_target_index is not None:
                return move_target_index
        object_index = self.selected_object_index()
        if object_index >= 0:
            return object_index
        if len(self.saved_objects) == 1:
            return 0
        return None

    def regenerate_polygons_for_target(self, target_index: int) -> bool:
        mask = self.mask_for_target(target_index)
        if mask is None or not mask.any():
            return False
        existing_polygons = self.clone_yolo_polygons(self.polygons_for_target(target_index))
        polygons = mask_to_yolo_edit_polygons(
            mask,
            epsilon=float(self.yolo_epsilon_spin.value()),
            min_area=float(self.yolo_min_area_spin.value()),
        )
        if not polygons and existing_polygons:
            self.set_status("Polygon detail produced no usable polygon; kept current polygons")
            return False
        if polygons == existing_polygons:
            return False
        self.push_undo_state("polygon detail")
        if target_index == -1:
            self.current_yolo_polygons = polygons
            self.current_yolo_dirty = False
        elif 0 <= target_index < len(self.saved_objects):
            self.saved_objects[target_index].yolo_polygons = polygons
            self.dirty = True
        else:
            return False
        self.selected_polygon_index = -1
        self.selected_vertex_index = -1
        return True

    def on_yolo_polygon_settings_changed(self, _value) -> None:
        self.sync_edit_polygon_detail_controls()
        target_index = self.polygon_detail_target_index()
        edit_enabled = bool(
            hasattr(self, "polygon_edit_check") and self.polygon_edit_check.isChecked()
        )
        changed = False
        if target_index is not None:
            can_regenerate = not edit_enabled
            if target_index == -1 and not self.current_yolo_dirty:
                can_regenerate = True
            if can_regenerate:
                changed = self.regenerate_polygons_for_target(target_index)
        if changed and target_index >= 0:
            self.update_object_list()
        self.render_canvas()

    def set_polygon_edit_enabled(self, enabled: bool) -> None:
        if enabled and hasattr(self, "polygon_mode_button") and not (
            self.polygon_mode_button.isChecked() or self.sam_hole_mode_button.isChecked()
        ):
            self.polygon_mode_button.setChecked(True)
        self.canvas.set_edit_mode(enabled)
        if enabled and self.current_mask is None and self.selected_object_index() < 0 and self.saved_objects:
            self.objects_list.setCurrentRow(0)
        if not enabled:
            changed_target: int | None = None
            if self.edit_drag_target is not None:
                changed_target = self.edit_drag_target[0]
            elif self.edit_move_target is not None:
                changed_target = self.edit_move_target[0]
            if changed_target is not None:
                self.rebuild_target_mask_from_polygons(changed_target)
                self.update_object_list()
            self.edit_drag_target = None
            self.edit_move_target = None
            self.draft_polygon_points.clear()
            self.draft_polygon_active = False
            if hasattr(self, "new_polygon_button"):
                self.new_polygon_button.setChecked(False)
        self.render_canvas()
        self.set_status("Mask editing enabled" if enabled else "Prompt mode")

    def set_polygon_tool_mode(self, mode: str) -> None:
        if mode == "sam_hole":
            self.sam_hole_mode_button.setChecked(True)
            self.cancel_draft_polygon()
            self.whole_mask_drag_check.setChecked(False)
            self.polygon_edit_check.setChecked(True)
            if self.current_mask is None and self.selected_object_index() < 0 and self.saved_objects:
                self.objects_list.setCurrentRow(0)
            self.set_status("SAM hole tool: click inside the target mask")
            return
        self.polygon_mode_button.setChecked(True)
        self.polygon_edit_check.setChecked(True)
        self.set_status("Polygon tool")

    def sam_hole_mode_enabled(self) -> bool:
        return bool(
            hasattr(self, "sam_hole_mode_button")
            and self.polygon_edit_check.isChecked()
            and self.sam_hole_mode_button.isChecked()
        )

    def target_contains_point(self, target_index: int, x: float, y: float) -> bool:
        mask = self.mask_for_target(target_index)
        if mask is None:
            return False
        col = int(round(x))
        row = int(round(y))
        if row < 0 or col < 0 or row >= mask.shape[0] or col >= mask.shape[1]:
            return False
        return bool(mask[row, col])

    def translate_target_polygons(self, target_index: int, dx: float, dy: float) -> None:
        for item in self.polygons_for_target(target_index):
            translated = []
            for raw_point in item.get("points", []):  # type: ignore[union-attr]
                try:
                    px, py = raw_point  # type: ignore[misc]
                except Exception:
                    continue
                translated.append((float(px) + dx, float(py) + dy))
            item["points"] = translated
        self.selected_vertex_index = -1
        if target_index == -1:
            self.current_yolo_dirty = True
        else:
            self.dirty = True

    def polygon_hit_threshold(self) -> float:
        transform = self.canvas.transform()
        scale = max(abs(transform.m11()), 0.1)
        return max(6.0, 14.0 / scale)

    @staticmethod
    def distance_to_segment(
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
            return float(np.hypot(px - sx, py - sy))
        t = max(0.0, min(1.0, ((px - sx) * dx + (py - sy) * dy) / float(dx * dx + dy * dy)))
        proj_x = sx + t * dx
        proj_y = sy + t * dy
        return float(np.hypot(px - proj_x, py - proj_y))

    def nearest_polygon_vertex(self, x: float, y: float) -> tuple[int, int, int] | None:
        target_index = self.active_polygon_target_index()
        if target_index is None:
            return None
        threshold = self.polygon_hit_threshold()
        best: tuple[float, int, int] | None = None
        for polygon_index, item in enumerate(self.polygons_for_target(target_index)):
            points = item.get("points", [])
            for vertex_index, point in enumerate(points):  # type: ignore[assignment]
                px, py = point
                distance = float(np.hypot(float(px) - x, float(py) - y))
                if distance <= threshold and (best is None or distance < best[0]):
                    best = (distance, polygon_index, vertex_index)
        if best is None:
            return None
        return target_index, best[1], best[2]

    def nearest_polygon_edge(self, x: float, y: float) -> tuple[int, int, int] | None:
        target_index = self.active_polygon_target_index()
        if target_index is None:
            return None
        threshold = self.polygon_hit_threshold()
        best: tuple[float, int, int] | None = None
        for polygon_index, item in enumerate(self.polygons_for_target(target_index)):
            points = list(item.get("points", []))
            if len(points) < 3:
                continue
            for vertex_index, start in enumerate(points):
                end = points[(vertex_index + 1) % len(points)]
                distance = self.distance_to_segment((x, y), start, end)
                if distance <= threshold and (best is None or distance < best[0]):
                    best = (distance, polygon_index, vertex_index + 1)
        if best is None:
            return None
        return target_index, best[1], best[2]

    def insert_vertex_on_edge(self, edge: tuple[int, int, int], x: float, y: float) -> tuple[int, int, int] | None:
        target_index, polygon_index, insert_index = edge
        points = self.polygons_for_target(target_index)[polygon_index].get("points", [])
        if not hasattr(points, "insert"):
            return None
        if insert_index > len(points):
            insert_index = len(points)
        self.push_undo_state("insert vertex")
        points.insert(insert_index, (x, y))  # type: ignore[attr-defined]
        self.selected_polygon_index = polygon_index
        self.selected_vertex_index = insert_index
        self.edit_drag_target = (target_index, polygon_index, insert_index)
        self.rebuild_target_mask_from_polygons(target_index)
        self.update_object_list()
        self.render_canvas()
        return target_index, polygon_index, insert_index

    def delete_polygon(self, target_index: int, polygon_index: int, *, push_undo: bool = True) -> bool:
        polygons = self.polygons_for_target(target_index)
        if not (0 <= polygon_index < len(polygons)):
            return False
        if push_undo:
            self.push_undo_state("delete polygon")
        del polygons[polygon_index]
        self.selected_polygon_index = -1
        self.selected_vertex_index = -1
        self.edit_drag_target = None
        self.edit_move_target = None
        self.rebuild_target_mask_from_polygons(target_index)
        self.update_object_list()
        self.render_canvas()
        return True

    def delete_vertex(self, target_index: int, polygon_index: int, vertex_index: int) -> bool:
        polygons = self.polygons_for_target(target_index)
        if not (0 <= polygon_index < len(polygons)):
            return False
        points = polygons[polygon_index].get("points", [])
        if not (0 <= vertex_index < len(points)):
            return False
        self.push_undo_state("delete vertex")
        if len(points) <= 3:
            removed = self.delete_polygon(target_index, polygon_index, push_undo=False)
            if removed:
                self.set_status("Polygon removed")
            return removed
        del points[vertex_index]  # type: ignore[index]
        self.selected_polygon_index = polygon_index
        self.selected_vertex_index = -1
        self.edit_drag_target = None
        self.rebuild_target_mask_from_polygons(target_index)
        self.update_object_list()
        self.render_canvas()
        return True

    def closest_edge_vertex(
        self,
        edge: tuple[int, int, int],
        x: float,
        y: float,
    ) -> tuple[int, int, int] | None:
        target_index, polygon_index, insert_index = edge
        points = list(self.polygons_for_target(target_index)[polygon_index].get("points", []))
        if len(points) < 3:
            return None
        previous_index = (insert_index - 1) % len(points)
        next_index = insert_index % len(points)
        previous = points[previous_index]
        next_point = points[next_index]
        previous_distance = float(np.hypot(float(previous[0]) - x, float(previous[1]) - y))
        next_distance = float(np.hypot(float(next_point[0]) - x, float(next_point[1]) - y))
        vertex_index = previous_index if previous_distance <= next_distance else next_index
        return target_index, polygon_index, vertex_index

    def rebuild_object_mask_from_polygons(self, object_index: int) -> None:
        self.rebuild_target_mask_from_polygons(object_index)

    def start_draft_polygon(self) -> None:
        if hasattr(self, "polygon_mode_button"):
            self.polygon_mode_button.setChecked(True)
        if self.active_polygon_target_index() is None:
            self.set_status("Create a SAM2 mask first or select an object")
            self.new_polygon_button.setChecked(False)
            return
        self.polygon_edit_check.setChecked(True)
        self.draft_polygon_points.clear()
        self.draft_polygon_active = True
        self.new_polygon_button.setChecked(True)
        self.set_status("Drawing polygon")
        self.render_canvas()

    def undo_draft_polygon_point(self) -> None:
        if not self.draft_polygon_active or not self.draft_polygon_points:
            self.set_status("No draft point to undo")
            return
        self.draft_polygon_points.pop()
        self.render_canvas()
        self.set_status(f"Drawing polygon: {len(self.draft_polygon_points)} point(s)")

    def finish_draft_polygon(self) -> None:
        target_index = self.active_polygon_target_index()
        if target_index is None or not self.draft_polygon_active:
            return
        if len(self.draft_polygon_points) < 3:
            self.set_status("Polygon needs at least 3 points")
            return
        mode = str(self.draft_polygon_combo.currentData() or "add")
        self.push_undo_state("add polygon")
        polygons = self.polygons_for_target(target_index)
        polygons.append(
            {"mode": mode, "points": [(float(x), float(y)) for x, y in self.draft_polygon_points]}
        )
        self.selected_polygon_index = len(polygons) - 1
        self.selected_vertex_index = -1
        self.draft_polygon_points.clear()
        self.draft_polygon_active = False
        self.new_polygon_button.setChecked(False)
        self.rebuild_target_mask_from_polygons(target_index)
        self.update_object_list()
        self.render_canvas()
        self.set_status("Polygon applied")

    def cancel_draft_polygon(self) -> None:
        self.draft_polygon_points.clear()
        self.draft_polygon_active = False
        self.new_polygon_button.setChecked(False)
        self.render_canvas()
        self.set_status("Polygon drawing canceled")

    def delete_selected_vertex(self) -> None:
        target_index = self.active_polygon_target_index()
        if target_index is None or self.selected_polygon_index < 0 or self.selected_vertex_index < 0:
            self.set_status("No vertex selected")
            return
        if self.delete_vertex(target_index, self.selected_polygon_index, self.selected_vertex_index):
            self.set_status("Vertex deleted")

    def delete_selected_polygon(self) -> None:
        target_index = self.active_polygon_target_index()
        if target_index is None or self.selected_polygon_index < 0:
            self.set_status("No polygon selected")
            return
        if self.delete_polygon(target_index, self.selected_polygon_index):
            self.set_status("Polygon deleted")

    def apply_hole_mask_to_target(
        self,
        target_index: int,
        hole_mask: np.ndarray,
        *,
        label: str = "hole",
        clear_active_mask: bool = False,
    ) -> bool:
        if self.image_np is None:
            return False
        target_mask = self.mask_for_target(target_index)
        if target_mask is None or not target_mask.any():
            self.set_status("Target mask is empty")
            return False
        clipped_hole = np.asarray(hole_mask, dtype=bool) & target_mask.astype(bool)
        if not clipped_hole.any():
            self.set_status("Hole does not overlap the target mask")
            return False

        target_polygons = self.polygons_for_target(target_index)
        base_polygons: list[dict[str, object]] = []
        if not target_polygons:
            base_polygons = mask_to_yolo_edit_polygons(
                target_mask,
                epsilon=float(self.yolo_epsilon_spin.value()),
                min_area=float(self.yolo_min_area_spin.value()),
            )
            if not base_polygons:
                self.set_status("Target has no editable polygon")
                return False

        hole_polygons = mask_to_yolo_edit_polygons(
            clipped_hole,
            epsilon=float(self.yolo_epsilon_spin.value()),
            min_area=float(self.yolo_min_area_spin.value()),
        )
        if not hole_polygons:
            self.set_status("SAM hole had no usable polygon")
            return False

        self.push_undo_state(label)
        target_polygons.extend(self.clone_yolo_polygons(base_polygons))
        first_new_index = len(target_polygons)
        for item in hole_polygons:
            target_polygons.append(
                {
                    "mode": "subtract",
                    "points": [(float(x), float(y)) for x, y in item.get("points", [])],  # type: ignore[misc]
                }
            )

        if clear_active_mask:
            self.points.clear()
            self.current_mask = None
            self.current_score = 0.0
            self.current_yolo_polygons.clear()
            self.current_yolo_dirty = False
            self.point_version += 1

        self.selected_polygon_index = first_new_index
        self.selected_vertex_index = -1
        self.edit_drag_target = None
        self.edit_move_target = None
        self.rebuild_target_mask_from_polygons(target_index)
        self.update_object_list()
        if target_index >= 0:
            self.objects_list.setCurrentRow(target_index)
            self.selected_polygon_index = first_new_index
            self.selected_vertex_index = -1
        self.polygon_edit_check.setChecked(True)
        self.render_canvas()
        return True

    def subtract_current_sam_mask_from_selected_object(self) -> None:
        if self.current_mask is None:
            self.set_status("Create a SAM2 mask for the hole first")
            return
        object_index = self.selected_object_index()
        if object_index < 0 and len(self.saved_objects) == 1:
            self.objects_list.setCurrentRow(0)
            object_index = 0
        if object_index < 0:
            self.set_status("Select the saved object to cut from")
            return
        if self.apply_hole_mask_to_target(
            object_index,
            self.current_mask,
            label="SAM hole",
            clear_active_mask=True,
        ):
            self.set_status("Converted active SAM mask to hole")

    def handle_polygon_edit_event(self, event_type, x: float, y: float, button, modifiers) -> None:
        target_index = self.active_polygon_target_index()
        if self.image_np is None:
            self.set_status("Open an image first")
            return
        if self.sam_hole_mode_enabled():
            if event_type == "press" and button == Qt.LeftButton:
                self.predict_sam_hole_async(x, y)
            elif event_type == "press" and button == Qt.RightButton:
                self.set_status("SAM hole tool uses left-click inside the target mask")
            return
        if self.draft_polygon_active:
            if target_index is None or not self.ensure_polygons_for_target(target_index):
                self.set_status("Create a SAM2 mask first or select an object")
                return
            if event_type == "press" and button == Qt.LeftButton:
                self.draft_polygon_points.append((x, y))
                self.set_status(f"Drawing polygon: {len(self.draft_polygon_points)} point(s)")
                self.render_canvas()
                return
            if event_type == "press" and button == Qt.RightButton:
                self.finish_draft_polygon()
                return
            return

        if event_type == "press" and button == Qt.LeftButton:
            move_requested = bool(
                (hasattr(self, "whole_mask_drag_check") and self.whole_mask_drag_check.isChecked())
                or (modifiers & Qt.ShiftModifier)
            )
            if move_requested:
                move_target_index = self.selected_move_target_index()
                if move_target_index is None:
                    self.set_status("Select a mask/object to drag")
                    return
                if not self.ensure_polygons_for_target(move_target_index):
                    self.set_status("Selected target has no editable polygon")
                    return
                if not self.target_contains_point(move_target_index, x, y):
                    self.set_status("Click inside the selected mask/object to drag it")
                    return
                self.push_undo_state("move mask")
                self.edit_move_target = (move_target_index, x, y)
                self.edit_drag_target = None
                self.selected_polygon_index = -1
                self.selected_vertex_index = -1
                self.render_canvas()
                self.set_status("Dragging target")
                return

        if event_type == "move" and self.edit_move_target is not None:
            move_target_index, last_x, last_y = self.edit_move_target
            dx = float(x) - float(last_x)
            dy = float(y) - float(last_y)
            if dx or dy:
                self.translate_target_polygons(move_target_index, dx, dy)
                self.edit_move_target = (move_target_index, x, y)
                self.render_canvas()
            return

        if event_type == "release" and self.edit_move_target is not None:
            move_target_index, _last_x, _last_y = self.edit_move_target
            self.rebuild_target_mask_from_polygons(move_target_index)
            self.update_object_list()
            self.render_canvas()
            self.set_status("Target moved")
            self.edit_move_target = None
            return

        if target_index is None:
            self.set_status("Create a SAM2 mask first or select an object")
            return
        if not self.ensure_polygons_for_target(target_index):
            self.set_status("No editable polygon for this target")
            return

        if event_type == "press" and button == Qt.RightButton:
            hit = self.nearest_polygon_vertex(x, y)
            if hit is not None:
                target_index, polygon_index, vertex_index = hit
                if self.delete_vertex(target_index, polygon_index, vertex_index):
                    self.set_status("Vertex deleted")
                return
            edge = self.nearest_polygon_edge(x, y)
            if edge is not None:
                edge_vertex = self.closest_edge_vertex(edge, x, y)
                if edge_vertex is not None:
                    target_index, polygon_index, vertex_index = edge_vertex
                    if self.delete_vertex(target_index, polygon_index, vertex_index):
                        self.set_status("Nearest vertex deleted")
                    return
            self.set_status("Right-click near a vertex to delete it")
            return

        if event_type == "press" and button == Qt.LeftButton:
            hit = self.nearest_polygon_vertex(x, y)
            if hit is not None:
                self.push_undo_state("move vertex")
                self.edit_drag_target = hit
                _object_index, polygon_index, vertex_index = hit
                self.selected_polygon_index = polygon_index
                self.selected_vertex_index = vertex_index
                self.render_canvas()
                return

            edge = self.nearest_polygon_edge(x, y)
            if edge is not None:
                if self.insert_vertex_on_edge(edge, x, y) is not None:
                    self.set_status("Vertex inserted")
                return
            self.set_status("Drag a vertex, click an edge to add a vertex, or start a new polygon")
            return

        if event_type == "move" and self.edit_drag_target is not None:
            target_index, polygon_index, vertex_index = self.edit_drag_target
            points = self.polygons_for_target(target_index)[polygon_index].get("points", [])
            if 0 <= vertex_index < len(points):
                points[vertex_index] = (x, y)  # type: ignore[index]
                if target_index == -1:
                    self.current_yolo_dirty = True
                else:
                    self.dirty = True
                self.render_canvas()
            return

        if event_type == "release":
            if self.edit_drag_target is not None:
                target_index, _polygon_index, _vertex_index = self.edit_drag_target
                self.rebuild_target_mask_from_polygons(target_index)
                self.update_object_list()
                self.render_canvas()
                self.set_status("Polygon updated")
            self.edit_drag_target = None

    def update_object_list(self) -> None:
        current_row = self.objects_list.currentRow()
        previous_block = self.objects_list.blockSignals(True)
        try:
            self.objects_list.clear()
            for index, saved in enumerate(self.saved_objects, start=1):
                area = int(np.count_nonzero(saved.mask))
                polygon_count = len(saved.yolo_polygons)
                self.objects_list.addItem(
                    f"{index}. class={saved.class_id}  area={area}  polygons={polygon_count}  score={saved.score:.3f}"
                )
            if self.saved_objects:
                self.objects_list.setCurrentRow(max(0, min(current_row, len(self.saved_objects) - 1)))
        finally:
            self.objects_list.blockSignals(previous_block)
        self.object_count.setText(f"{len(self.saved_objects)} saved")

    def save_results(self) -> bool:
        if self.image_np is None or self.image_path is None:
            self.set_status("Open an image first")
            return False
        if self.current_mask is not None:
            if QMessageBox.question(self, "Accept active mask", "Accept the active mask before saving?") == QMessageBox.Yes:
                self.accept_current_mask()
        if not self.saved_objects:
            self.set_status("No saved objects")
            return False
        if self.output_dir is None:
            default_dir = self.image_path.parent / "sam2_interactive_results"
            selected = QFileDialog.getExistingDirectory(self, "Choose output folder", str(default_dir.parent))
            if not selected:
                return False
            self.output_dir = Path(selected)
            self.output_dir_user_selected = True
            self.update_output_label()
        try:
            outputs = save_interactive_results(
                self.image_path,
                self.output_dir,
                self.image_np,
                self.saved_objects,
                export_format=self.export_combo.currentText(),
                yolo_epsilon=float(self.yolo_epsilon_spin.value()),
                yolo_min_area=float(self.yolo_min_area_spin.value()),
                save_object_masks=self.object_masks_check.isChecked(),
                image_name=self.working_image_name,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save results", str(exc))
            return False
        self.dirty = False
        self.invalidate_annotation_index()
        self.set_status(f"Saved: {outputs['overlay']}")
        return True

    def save_and_next(self) -> None:
        if self.save_results():
            if self.image_files and self.image_index < len(self.image_files) - 1:
                self.image_index += 1
                self.load_image_path(self.image_files[self.image_index])
            else:
                self.set_status("Saved. No next image.")

    def canvas_fit(self) -> None:
        self.canvas.fit_image()

    def render_canvas(self, fit: bool = False) -> None:
        if self.image_np is None or not hasattr(self, "render_timer"):
            self.render_pending = False
            self.render_pending_fit = False
            self.render_canvas_now(fit=fit)
            return

        self.render_pending_fit = self.render_pending_fit or fit
        if self.render_pending:
            return

        elapsed_ms = (time.monotonic() - self.last_render_time) * 1000.0
        delay_ms = 0 if fit or self.last_render_time <= 0 else max(0, int(self.render_interval_ms - elapsed_ms))
        self.render_pending = True
        self.render_timer.start(delay_ms)

    def flush_render_canvas(self) -> None:
        fit = self.render_pending_fit
        self.render_pending = False
        self.render_pending_fit = False
        self.render_canvas_now(fit=fit)

    def interactive_overlay_cache_key(self) -> tuple[object, ...]:
        saved_key = tuple(
            (
                id(saved),
                id(saved.mask),
                tuple(int(value) for value in saved.color),
            )
            for saved in self.saved_objects
        )
        current_color = tuple(int(value) for value in self.current_color)
        return (self.image_version, saved_key, id(self.current_mask), current_color)

    def cached_interactive_overlay(self) -> np.ndarray:
        if self.image_np is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        key = self.interactive_overlay_cache_key()
        if self.overlay_cache_key != key or self.overlay_cache is None:
            self.overlay_cache = render_interactive_overlay(
                self.image_np,
                self.saved_objects,
                self.current_mask,
                self.current_color,
            )
            self.overlay_cache_key = key
        return self.overlay_cache

    def render_canvas_now(self, fit: bool = False) -> None:
        self.last_render_time = time.monotonic()
        if self.image_np is None:
            self.overlay_cache_key = None
            self.overlay_cache = None
            self.canvas.set_overlay(None, [])
            if hasattr(self, "yolo_preview_label"):
                self.yolo_preview_label.setText("Polygon preview: off")
            return
        overlay_key: tuple[object, ...] | None = None
        polygon_overlay = False
        if hasattr(self, "polygon_edit_check") and self.polygon_edit_check.isChecked():
            overlay = self.cached_interactive_overlay()
            overlay_key = ("interactive", self.overlay_cache_key)
            polygon_overlay = True
            total_polygons = len(self.current_yolo_polygons) + sum(len(saved.yolo_polygons) for saved in self.saved_objects)
            self.yolo_preview_label.setText(f"Editable polygons: {total_polygons}")
        elif self.yolo_preview_check.isChecked():
            overlay, polygon_count = render_yolo_polygon_overlay(
                self.image_np,
                self.saved_objects,
                self.current_mask,
                self.current_color,
                current_class_id=int(self.class_spin.value()),
                current_yolo_polygons=self.current_yolo_polygons,
                epsilon=float(self.yolo_epsilon_spin.value()),
                min_area=float(self.yolo_min_area_spin.value()),
            )
            overlay_key = ("yolo_preview", self.last_render_time, id(overlay))
            self.yolo_preview_label.setText(f"YOLO polygons: {polygon_count}")
        else:
            overlay = self.cached_interactive_overlay()
            overlay_key = ("interactive", self.overlay_cache_key)
            self.yolo_preview_label.setText("Polygon preview: off")
        show_prompt_points = not (hasattr(self, "polygon_edit_check") and self.polygon_edit_check.isChecked())
        self.canvas.set_overlay(overlay, self.points if show_prompt_points else [], image_key=overlay_key)
        if polygon_overlay:
            self.canvas.set_polygon_overlay(
                self.saved_objects,
                selected_object_index=self.rendered_polygon_target_index(),
                selected_polygon_index=self.selected_polygon_index,
                selected_vertex_index=self.selected_vertex_index,
                draft_polygon=self.draft_polygon_points,
                draft_mode=str(self.draft_polygon_combo.currentData() or "add"),
                current_yolo_polygons=self.current_yolo_polygons if self.current_mask is not None else None,
                current_color=self.current_color,
            )
        else:
            self.canvas.clear_polygon_overlay()
        if fit:
            QTimer.singleShot(0, self.canvas.fit_image)


def apply_style(app: QApplication) -> None:
    families = set(QFontDatabase.families())
    for family in ("Segoe UI", "Microsoft JhengHei UI", "Noto Sans TC", "Source Han Sans TC", "Noto Sans CJK TC"):
        if family in families:
            app.setFont(QFont(family, 10))
            break

    app.setStyleSheet(
        """
        QWidget {
            background: #f6f8fb;
            color: #1f2937;
            font-family: "Segoe UI", "Microsoft JhengHei UI", "Noto Sans TC", "Source Han Sans TC";
            font-size: 10pt;
        }
        QMainWindow, QToolBar { background: #f6f8fb; }
        QToolBar {
            border: 0;
            padding: 10px;
            spacing: 8px;
            background: #ffffff;
            border-bottom: 1px solid #d9e1ec;
        }
        QSplitter::handle { background: #d9e1ec; margin: 2px; }
        QTabWidget#sideTabs::pane {
            border: 0;
            top: -1px;
            background: #ffffff;
        }
        QTabWidget#sideTabs QTabBar::tab {
            background: #eef3f8;
            border: 1px solid #d9e1ec;
            border-bottom: 0;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            color: #475569;
            font-weight: 700;
            padding: 7px 9px;
            margin-right: 3px;
        }
        QTabWidget#sideTabs QTabBar::tab:selected {
            background: #ffffff;
            color: #0f766e;
            border-color: #cbd7e6;
        }
        QTabWidget#sideTabs QTabBar::tab:hover { background: #e2eaf3; }
        QScrollArea#controlScroll, QWidget#scrollBody { background: #ffffff; border: 0; }
        QScrollBar:vertical { background: #f1f5f9; width: 10px; margin: 0; }
        QScrollBar::handle:vertical {
            background: #cbd7e6;
            border-radius: 5px;
            min-height: 28px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QFrame#panel {
            background: #ffffff;
            border: 1px solid #d9e1ec;
            border-radius: 8px;
        }
        QLabel#section {
            color: #64748b;
            font-weight: 700;
            letter-spacing: 0;
            padding-top: 8px;
        }
        QLabel#muted, QLabel#statusLabel { color: #64748b; }
        QPushButton {
            background: #eef3f8;
            border: 1px solid #cbd7e6;
            border-radius: 6px;
            padding: 8px 10px;
            font-weight: 600;
        }
        QPushButton:hover { background: #e2eaf3; }
        QPushButton:checked {
            background: #d9f3ee;
            border-color: #0f9f8d;
            color: #0f766e;
        }
        QPushButton[mode="add"]:checked, QPushButton[activeAction="true"]:checked {
            background: #0f9f8d;
            border-color: #0f766e;
            color: #ffffff;
        }
        QPushButton[mode="remove"]:checked {
            background: #be123c;
            border-color: #9f1239;
            color: #ffffff;
        }
        QPushButton[accent="true"] {
            background: #0f9f8d;
            border-color: #0f766e;
            color: #ffffff;
        }
        QPushButton[danger="true"] {
            background: #fff1f2;
            border-color: #fda4af;
            color: #be123c;
        }
        QPushButton#primarySaveButton {
            background: #0f9f8d;
            border: 1px solid #0f766e;
            color: #ffffff;
            border-radius: 8px;
            padding: 14px 12px;
            min-height: 34px;
            font-size: 13pt;
            font-weight: 800;
        }
        QPushButton#primarySaveButton:hover { background: #0f8f80; }
        QPushButton#secondarySaveButton {
            background: #e0f2fe;
            border: 1px solid #93c5fd;
            color: #1d4ed8;
            border-radius: 8px;
            padding: 11px 12px;
            min-height: 28px;
            font-weight: 700;
        }
        QComboBox, QSpinBox, QDoubleSpinBox {
            background: #ffffff;
            border: 1px solid #cbd7e6;
            border-radius: 6px;
            padding: 7px 8px;
            min-height: 20px;
        }
        QCheckBox {
            color: #334155;
            spacing: 8px;
            padding: 3px 2px;
        }
        QCheckBox[toolMode="true"] {
            background: #f8fafc;
            border: 1px solid #d9e1ec;
            border-radius: 6px;
            padding: 7px 8px;
        }
        QCheckBox[toolMode="true"]:checked {
            background: #d9f3ee;
            border-color: #0f9f8d;
        }
        QCheckBox:checked {
            color: #0f766e;
            font-weight: 700;
        }
        QCheckBox::indicator {
            width: 15px;
            height: 15px;
            border-radius: 4px;
            border: 1px solid #94a3b8;
            background: #ffffff;
        }
        QCheckBox::indicator:checked {
            background: #0f9f8d;
            border-color: #0f766e;
        }
        QListWidget {
            background: #ffffff;
            border: 1px solid #cbd7e6;
            border-radius: 6px;
            padding: 6px;
        }
        QListWidget::item { padding: 8px; border-radius: 4px; }
        QListWidget::item:selected {
            background: #d9f3ee;
            color: #0f766e;
        }
        """
    )


def run_interactive_gui() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    apply_style(app)
    window = PySideSamWindow()
    window.show()
    if os.environ.get("SAM2_GUI_SMOKE") == "1":
        QTimer.singleShot(0, app.quit)
    raise SystemExit(app.exec())
