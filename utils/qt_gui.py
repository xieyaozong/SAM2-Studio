from __future__ import annotations

from pathlib import Path
from typing import Sequence
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QImage,
    QKeySequence,
    QPainter,
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
    QToolBar,
    QVBoxLayout,
    QWidget,
)
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
    render_interactive_overlay,
    render_yolo_edit_polygon_overlay,
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
import queue
import sys
import threading

import numpy as np
import torch


class SamCanvasView(QGraphicsView):
    def __init__(self, on_click, on_edit_event):
        super().__init__()
        self.on_click = on_click
        self.on_edit_event = on_edit_event
        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.pixmap_item: QGraphicsPixmapItem | None = None
        self.point_items: list[QGraphicsEllipseItem] = []
        self.has_image = False
        self.edit_mode = False
        self.dragging_edit = False
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#eef2f7"))
        self.setFrameShape(QFrame.NoFrame)
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def set_edit_mode(self, enabled: bool) -> None:
        self.edit_mode = enabled
        self.dragging_edit = False
        self.setDragMode(QGraphicsView.NoDrag if enabled else QGraphicsView.ScrollHandDrag)

    def set_overlay(self, image: np.ndarray | None, points: Sequence[tuple[float, float, int]]) -> None:
        self.scene_obj.clear()
        self.point_items.clear()
        self.pixmap_item = None
        self.has_image = image is not None

        if image is None:
            self.scene_obj.setSceneRect(0, 0, 100, 100)
            return

        image = np.ascontiguousarray(image)
        height, width = image.shape[:2]
        qimage = QImage(image.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        self.pixmap_item = self.scene_obj.addPixmap(QPixmap.fromImage(qimage))
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

    def fit_image(self) -> None:
        if self.has_image:
            self.fitInView(self.scene_obj.sceneRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event) -> None:
        if not self.has_image:
            super().wheelEvent(event)
            return
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:
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
        if self.has_image and self.edit_mode and self.dragging_edit:
            point = self.mapToScene(event.pos())
            rect = self.scene_obj.sceneRect()
            if rect.contains(point):
                self.on_edit_event("move", float(point.x()), float(point.y()), Qt.LeftButton, event.modifiers())
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
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
        self.resize(1280, 820)
        self.setMinimumSize(760, 520)

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
        self.hough_result: HoughPreprocessResult | None = None
        self.hough_preview_active = False
        self.hough_running = False
        self.hough_action_after_run: str | None = None
        self.hough_requested_output: str | None = None
        self.edit_drag_target: tuple[int, int, int] | None = None
        self.selected_polygon_index = -1
        self.selected_vertex_index = -1
        self.draft_polygon_points: list[tuple[float, float]] = []
        self.draft_polygon_active = False
        self.dirty = False

        self.build_ui(default_model)
        self.install_shortcuts()

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
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)
        splitter.addWidget(left)

        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setFrameShape(QFrame.NoFrame)
        control_scroll.setObjectName("controlScroll")
        control_body = QWidget()
        control_body.setObjectName("scrollBody")
        control_layout = QVBoxLayout(control_body)
        control_layout.setContentsMargins(4, 4, 4, 4)
        control_layout.setSpacing(10)
        control_scroll.setWidget(control_body)
        left_layout.addWidget(control_scroll, 1)

        self.populate_left_controls(control_layout)

        self.save_button = QPushButton("Save Current")
        self.save_button.setObjectName("primarySaveButton")
        self.save_button.clicked.connect(self.save_results)
        left_layout.addWidget(self.save_button)

        self.save_next_button = QPushButton("Save + Next")
        self.save_next_button.setObjectName("secondarySaveButton")
        self.save_next_button.clicked.connect(self.save_and_next)
        left_layout.addWidget(self.save_next_button)

        self.canvas = SamCanvasView(self.handle_canvas_click, self.handle_polygon_edit_event)
        self.canvas.setMinimumSize(260, 220)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.addWidget(self.canvas)

        right = self.make_panel()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(16, 16, 16, 16)
        right_layout.setSpacing(10)
        splitter.addWidget(right)
        self.populate_right_controls(right_layout)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([260, 760, 260])

    def populate_left_controls(self, layout: QVBoxLayout) -> None:
        layout.addWidget(self.section("File"))
        open_image_button = QPushButton("Open Image")
        open_image_button.clicked.connect(self.open_image)
        layout.addWidget(open_image_button)

        open_folder_button = QPushButton("Open Folder")
        open_folder_button.clicked.connect(self.open_folder)
        layout.addWidget(open_folder_button)

        output_button = QPushButton("Output Folder")
        output_button.clicked.connect(self.choose_output_folder)
        layout.addWidget(output_button)

        self.folder_recursive_check = QCheckBox("Recursive folder")
        self.folder_recursive_check.setChecked(True)
        layout.addWidget(self.folder_recursive_check)

        layout.addWidget(self.section("Preprocess"))
        self.hough_image_combo = QComboBox()
        self.hough_image_combo.addItem("Full masked image", "full")
        self.hough_image_combo.addItem("Center crop", "crop")
        self.hough_image_combo.currentTextChanged.connect(self.refresh_hough_preview)
        layout.addWidget(self.hough_image_combo)

        self.hough_debug_check = QCheckBox("Show debug overlay")
        self.hough_debug_check.toggled.connect(self.refresh_hough_preview)
        layout.addWidget(self.hough_debug_check)

        self.hough_crop_size_spin = QSpinBox()
        self.hough_crop_size_spin.setRange(0, 8192)
        self.hough_crop_size_spin.setValue(0)
        self.hough_crop_size_spin.setSpecialValueText("crop native")
        self.hough_crop_size_spin.setSingleStep(128)
        self.hough_crop_size_spin.setPrefix("crop ")
        self.hough_crop_size_spin.setSuffix(" px")
        self.hough_crop_size_spin.setToolTip("0 keeps the adaptive crop at its native size. Set a pixel value to force resize.")
        layout.addWidget(self.hough_crop_size_spin)

        self.hough_inner_scale_spin = QDoubleSpinBox()
        self.hough_inner_scale_spin.setRange(0.1, 1.5)
        self.hough_inner_scale_spin.setSingleStep(0.02)
        self.hough_inner_scale_spin.setDecimals(2)
        self.hough_inner_scale_spin.setValue(0.86)
        self.hough_inner_scale_spin.setPrefix("inner x")
        layout.addWidget(self.hough_inner_scale_spin)

        self.hough_crop_scale_spin = QDoubleSpinBox()
        self.hough_crop_scale_spin.setRange(0.1, 1.5)
        self.hough_crop_scale_spin.setSingleStep(0.02)
        self.hough_crop_scale_spin.setDecimals(2)
        self.hough_crop_scale_spin.setValue(0.55)
        self.hough_crop_scale_spin.setPrefix("crop x")
        layout.addWidget(self.hough_crop_scale_spin)

        preview_hough_button = QPushButton("Preview Hough")
        preview_hough_button.clicked.connect(self.preview_hough_preprocess)
        layout.addWidget(preview_hough_button)

        use_hough_button = QPushButton("Use Hough For SAM")
        use_hough_button.setProperty("accent", True)
        use_hough_button.clicked.connect(self.apply_hough_to_sam)
        layout.addWidget(use_hough_button)

        save_hough_button = QPushButton("Save Hough Result")
        save_hough_button.clicked.connect(self.save_hough_result)
        layout.addWidget(save_hough_button)

        restore_original_button = QPushButton("Restore Original")
        restore_original_button.clicked.connect(self.restore_original_image)
        layout.addWidget(restore_original_button)

        self.image_counter_label = QLabel("No image loaded")
        self.image_counter_label.setObjectName("muted")
        layout.addWidget(self.image_counter_label)

        nav_row = QHBoxLayout()
        prev_button = QPushButton("Prev")
        prev_button.clicked.connect(self.previous_image)
        next_button = QPushButton("Next")
        next_button.clicked.connect(self.next_image)
        nav_row.addWidget(prev_button)
        nav_row.addWidget(next_button)
        layout.addLayout(nav_row)

        self.output_label = QLabel("Output: not selected")
        self.output_label.setObjectName("muted")
        self.output_label.setWordWrap(True)
        layout.addWidget(self.output_label)

        layout.addWidget(self.section("Point Mode"))
        mode_row = QHBoxLayout()
        self.add_button = QPushButton("Add")
        self.add_button.setCheckable(True)
        self.add_button.setChecked(True)
        self.add_button.setProperty("accent", True)
        self.remove_button = QPushButton("Remove")
        self.remove_button.setCheckable(True)
        self.remove_button.setProperty("danger", True)
        mode_group = QButtonGroup(self)
        mode_group.addButton(self.add_button)
        mode_group.addButton(self.remove_button)
        self.add_button.clicked.connect(lambda: self.set_status("Foreground point mode"))
        self.remove_button.clicked.connect(lambda: self.set_status("Background point mode"))
        mode_row.addWidget(self.add_button)
        mode_row.addWidget(self.remove_button)
        layout.addLayout(mode_row)

        layout.addWidget(self.section("Class And Export"))
        self.class_spin = QSpinBox()
        self.class_spin.setRange(0, 9999)
        self.class_spin.setValue(0)
        self.class_spin.setPrefix("class ")
        self.class_spin.valueChanged.connect(lambda _value: self.render_canvas())
        layout.addWidget(self.class_spin)

        self.export_combo = QComboBox()
        self.export_combo.addItems(EXPORT_FORMATS)
        self.export_combo.setCurrentText("yolo")
        layout.addWidget(self.export_combo)

        self.object_masks_check = QCheckBox("Save object masks")
        layout.addWidget(self.object_masks_check)

        self.yolo_preview_check = QCheckBox("Show YOLO polygons")
        self.yolo_preview_check.toggled.connect(lambda _checked: self.render_canvas())
        layout.addWidget(self.yolo_preview_check)

        self.yolo_epsilon_spin = QDoubleSpinBox()
        self.yolo_epsilon_spin.setRange(0.0, 30.0)
        self.yolo_epsilon_spin.setSingleStep(0.5)
        self.yolo_epsilon_spin.setDecimals(1)
        self.yolo_epsilon_spin.setValue(2.0)
        self.yolo_epsilon_spin.setPrefix("epsilon ")
        self.yolo_epsilon_spin.valueChanged.connect(self.on_yolo_polygon_settings_changed)
        layout.addWidget(self.yolo_epsilon_spin)

        self.yolo_min_area_spin = QDoubleSpinBox()
        self.yolo_min_area_spin.setRange(0.0, 10000.0)
        self.yolo_min_area_spin.setSingleStep(1.0)
        self.yolo_min_area_spin.setDecimals(1)
        self.yolo_min_area_spin.setValue(8.0)
        self.yolo_min_area_spin.setPrefix("min area ")
        self.yolo_min_area_spin.valueChanged.connect(self.on_yolo_polygon_settings_changed)
        layout.addWidget(self.yolo_min_area_spin)

        self.yolo_preview_label = QLabel("YOLO polygons: off")
        self.yolo_preview_label.setObjectName("muted")
        layout.addWidget(self.yolo_preview_label)

        layout.addWidget(self.section("Mask Actions"))
        self.accept_button = QPushButton("Accept Mask")
        self.accept_button.setProperty("accent", True)
        self.accept_button.clicked.connect(self.accept_current_mask)
        layout.addWidget(self.accept_button)

        undo_button = QPushButton("Undo Point")
        undo_button.clicked.connect(self.undo_point)
        layout.addWidget(undo_button)

        clear_button = QPushButton("Clear Points")
        clear_button.clicked.connect(self.clear_points)
        layout.addWidget(clear_button)

        layout.addWidget(self.section("View"))
        view_row = QHBoxLayout()
        zoom_out = QPushButton("-")
        zoom_out.clicked.connect(lambda: self.canvas.scale(1 / 1.18, 1 / 1.18))
        fit = QPushButton("Fit")
        fit.clicked.connect(self.canvas_fit)
        zoom_in = QPushButton("+")
        zoom_in.clicked.connect(lambda: self.canvas.scale(1.18, 1.18))
        view_row.addWidget(zoom_out)
        view_row.addWidget(fit)
        view_row.addWidget(zoom_in)
        layout.addLayout(view_row)
        layout.addStretch(1)

    def populate_right_controls(self, layout: QVBoxLayout) -> None:
        layout.addWidget(self.section("Objects"))
        self.object_count = QLabel("0 saved")
        self.object_count.setObjectName("muted")
        layout.addWidget(self.object_count)
        self.objects_list = QListWidget()
        self.objects_list.currentRowChanged.connect(self.on_selected_object_changed)
        layout.addWidget(self.objects_list, 1)

        remove_object = QPushButton("Remove Selected")
        remove_object.setProperty("danger", True)
        remove_object.clicked.connect(self.remove_selected_object)
        layout.addWidget(remove_object)

        clear_objects = QPushButton("Clear Objects")
        clear_objects.setProperty("danger", True)
        clear_objects.clicked.connect(self.clear_saved_objects)
        layout.addWidget(clear_objects)

        layout.addWidget(self.section("YOLO Polygon Edit"))
        self.polygon_edit_check = QCheckBox("Edit YOLO polygons")
        self.polygon_edit_check.toggled.connect(self.set_polygon_edit_enabled)
        layout.addWidget(self.polygon_edit_check)

        self.draft_polygon_combo = QComboBox()
        self.draft_polygon_combo.addItem("Add YOLO polygon", "add")
        self.draft_polygon_combo.addItem("Cut mask hole", "subtract")
        layout.addWidget(self.draft_polygon_combo)

        new_polygon = QPushButton("New YOLO Polygon")
        new_polygon.clicked.connect(self.start_draft_polygon)
        layout.addWidget(new_polygon)

        finish_polygon = QPushButton("Finish Polygon")
        finish_polygon.clicked.connect(self.finish_draft_polygon)
        layout.addWidget(finish_polygon)

        cancel_polygon = QPushButton("Cancel Polygon")
        cancel_polygon.clicked.connect(self.cancel_draft_polygon)
        layout.addWidget(cancel_polygon)

        delete_polygon = QPushButton("Delete Selected Polygon")
        delete_polygon.setProperty("danger", True)
        delete_polygon.clicked.connect(self.delete_selected_polygon)
        layout.addWidget(delete_polygon)

    def make_panel(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("panel")
        frame.setMinimumWidth(190)
        frame.setMaximumWidth(340)
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
        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.undo_point)
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
            "Discard current unsaved points or accepted masks?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def update_image_counter(self) -> None:
        if self.image_path is None:
            self.image_counter_label.setText("No image loaded")
            return
        if self.image_files and self.image_index >= 0:
            self.image_counter_label.setText(f"{self.image_index + 1}/{len(self.image_files)}  {self.image_path.name}")
        else:
            self.image_counter_label.setText(f"Single  {self.image_path.name}")

    def update_output_label(self) -> None:
        text = f"Output: {self.output_dir}" if self.output_dir else "Output: not selected"
        self.output_label.setText(text)

    def reset_annotation_state(self) -> None:
        self.points.clear()
        self.saved_objects.clear()
        self.current_mask = None
        self.current_score = 0.0
        self.current_yolo_polygons.clear()
        self.current_yolo_dirty = False
        self.edit_drag_target = None
        self.selected_polygon_index = -1
        self.selected_vertex_index = -1
        self.draft_polygon_points.clear()
        self.draft_polygon_active = False
        self.image_ready = False
        self.dirty = False
        self.point_version += 1
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

    def run_hough_preprocess(self, action_after_run: str, output_variant: str | None = None) -> None:
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
        if not self.hough_preview_active and not self.confirm_discard_work():
            return

        image_path = self.image_path
        source_image = self.original_image_np.copy()
        settings = self.hough_settings()
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
            self.update_output_label()
        try:
            outputs = save_hough_preprocess_result(self.image_path, self.output_dir, self.hough_result)
        except Exception as exc:
            QMessageBox.critical(self, "Save Hough result", str(exc))
            return
        self.set_status(f"Saved Hough result: {outputs['full']}")

    def load_image_path(self, path: Path) -> None:
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
        self.render_canvas(fit=True)
        self.update_image_counter()
        self.setWindowTitle(f"SAM 2 Studio - {path.name}")
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
        self.image_index = 0
        self.output_dir = folder_path / "sam2_dataset"
        self.update_output_label()
        self.load_image_path(self.image_files[self.image_index])

    def choose_output_folder(self) -> None:
        initial_dir = str(self.output_dir or (self.image_path.parent if self.image_path else PROJECT_ROOT))
        selected = QFileDialog.getExistingDirectory(self, "Choose output folder", initial_dir)
        if not selected:
            return
        self.output_dir = Path(selected)
        self.update_output_label()
        self.set_status(f"Output folder: {self.output_dir}")

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
                        self.set_status(f"Mask score {score:.3f}")
                        self.render_canvas()
                    if self.pending_prediction or version != self.point_version:
                        self.pending_prediction = False
                        self.predict_current_mask_async()
                elif message_type == "error":
                    self.model_loading = False
                    self.embedding = False
                    self.predicting = False
                    self.set_status(str(message_data))
                    QMessageBox.critical(self, "SAM 2 Studio", str(message_data))
                elif message_type == "hough_ready":
                    image_path, hough_result = message_data
                    self.hough_running = False
                    if image_path != self.image_path:
                        self.hough_action_after_run = None
                        self.hough_requested_output = None
                        continue
                    self.hough_result = hough_result
                    action_after_run = self.hough_action_after_run or "preview"
                    output_variant = self.hough_requested_output or self.selected_hough_output()
                    self.hough_action_after_run = None
                    self.hough_requested_output = None
                    if action_after_run == "use":
                        self.use_hough_image_for_sam(output_variant)
                        self.set_status(f"Using Hough {output_variant}: {hough_result.method}")
                    else:
                        self.hough_preview_active = True
                        self.refresh_hough_preview()
                        self.set_status(f"Hough {hough_result.mode}: {hough_result.method}")
                elif message_type == "hough_error":
                    self.hough_running = False
                    self.hough_action_after_run = None
                    self.hough_requested_output = None
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
            self.set_status("Point removed")
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
        self.set_status("Points cleared")

    def accept_current_mask(self) -> None:
        if self.current_mask is None:
            self.set_status("No active mask")
            return
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
        self.set_status(f"Accepted {saved.name}")

    def remove_selected_object(self) -> None:
        row = self.objects_list.currentRow()
        if row < 0:
            return
        del self.saved_objects[row]
        self.edit_drag_target = None
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
        self.saved_objects.clear()
        self.edit_drag_target = None
        self.selected_polygon_index = -1
        self.selected_vertex_index = -1
        self.dirty = True
        self.update_object_list()
        self.render_canvas()
        self.set_status("Objects cleared")

    def on_selected_object_changed(self, _row: int) -> None:
        self.edit_drag_target = None
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

    def polygons_for_target(self, target_index: int) -> list[dict[str, object]]:
        if target_index == -1:
            return self.current_yolo_polygons
        return self.saved_objects[target_index].yolo_polygons

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

    def on_yolo_polygon_settings_changed(self, _value) -> None:
        if self.current_mask is not None and not self.current_yolo_dirty:
            self.current_yolo_polygons = mask_to_yolo_edit_polygons(
                self.current_mask,
                epsilon=float(self.yolo_epsilon_spin.value()),
                min_area=float(self.yolo_min_area_spin.value()),
            )
            self.selected_polygon_index = -1
            self.selected_vertex_index = -1
        self.render_canvas()

    def set_polygon_edit_enabled(self, enabled: bool) -> None:
        self.canvas.set_edit_mode(enabled)
        if enabled and self.current_mask is None and self.selected_object_index() < 0 and self.saved_objects:
            self.objects_list.setCurrentRow(0)
        if not enabled:
            self.edit_drag_target = None
            self.draft_polygon_points.clear()
            self.draft_polygon_active = False
        self.render_canvas()
        self.set_status("YOLO polygon edit mode" if enabled else "Point mode")

    def polygon_hit_threshold(self) -> float:
        transform = self.canvas.transform()
        scale = max(abs(transform.m11()), 0.1)
        return max(4.0, 10.0 / scale)

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

    def rebuild_object_mask_from_polygons(self, object_index: int) -> None:
        self.rebuild_target_mask_from_polygons(object_index)

    def start_draft_polygon(self) -> None:
        if self.active_polygon_target_index() is None:
            self.set_status("Create a SAM2 mask first or select an object")
            return
        self.polygon_edit_check.setChecked(True)
        self.draft_polygon_points.clear()
        self.draft_polygon_active = True
        self.set_status("Draft polygon started")
        self.render_canvas()

    def finish_draft_polygon(self) -> None:
        target_index = self.active_polygon_target_index()
        if target_index is None or not self.draft_polygon_active:
            return
        if len(self.draft_polygon_points) < 3:
            self.set_status("Draft polygon needs at least 3 points")
            return
        mode = str(self.draft_polygon_combo.currentData() or "add")
        polygons = self.polygons_for_target(target_index)
        polygons.append(
            {"mode": mode, "points": [(float(x), float(y)) for x, y in self.draft_polygon_points]}
        )
        self.selected_polygon_index = len(polygons) - 1
        self.selected_vertex_index = -1
        self.draft_polygon_points.clear()
        self.draft_polygon_active = False
        self.rebuild_target_mask_from_polygons(target_index)
        self.update_object_list()
        self.render_canvas()
        self.set_status("Polygon added")

    def cancel_draft_polygon(self) -> None:
        self.draft_polygon_points.clear()
        self.draft_polygon_active = False
        self.render_canvas()
        self.set_status("Draft polygon canceled")

    def delete_selected_polygon(self) -> None:
        target_index = self.active_polygon_target_index()
        if target_index is None or self.selected_polygon_index < 0:
            self.set_status("No polygon selected")
            return
        polygons = self.polygons_for_target(target_index)
        if not (0 <= self.selected_polygon_index < len(polygons)):
            return
        del polygons[self.selected_polygon_index]
        self.selected_polygon_index = -1
        self.selected_vertex_index = -1
        self.rebuild_target_mask_from_polygons(target_index)
        self.update_object_list()
        self.render_canvas()
        self.set_status("Polygon deleted")

    def handle_polygon_edit_event(self, event_type, x: float, y: float, button, modifiers) -> None:
        target_index = self.active_polygon_target_index()
        if self.image_np is None or target_index is None:
            self.set_status("Create a SAM2 mask first or select an object")
            return
        if self.draft_polygon_active:
            if event_type == "press" and button == Qt.LeftButton:
                self.draft_polygon_points.append((x, y))
                self.render_canvas()
                return
            if event_type == "press" and button == Qt.RightButton:
                self.finish_draft_polygon()
                return
            return

        if event_type == "press" and button == Qt.RightButton:
            hit = self.nearest_polygon_vertex(x, y)
            if hit is not None:
                target_index, polygon_index, vertex_index = hit
                points = self.polygons_for_target(target_index)[polygon_index].get("points", [])
                if len(points) > 3:
                    del points[vertex_index]  # type: ignore[index]
                    self.selected_polygon_index = polygon_index
                    self.selected_vertex_index = -1
                    self.rebuild_target_mask_from_polygons(target_index)
                    self.update_object_list()
                    self.render_canvas()
                    self.set_status("Vertex deleted")
                return
            self.set_status("Right-click a vertex to delete it")
            return

        if event_type == "press" and button == Qt.LeftButton:
            if modifiers & Qt.ShiftModifier:
                edge = self.nearest_polygon_edge(x, y)
                if edge is not None:
                    target_index, polygon_index, insert_index = edge
                    points = self.polygons_for_target(target_index)[polygon_index].get("points", [])
                    points.insert(insert_index, (x, y))  # type: ignore[attr-defined]
                    self.selected_polygon_index = polygon_index
                    self.selected_vertex_index = insert_index
                    self.edit_drag_target = (target_index, polygon_index, insert_index)
                    self.rebuild_target_mask_from_polygons(target_index)
                    self.update_object_list()
                    self.render_canvas()
                    self.set_status("Vertex inserted")
                    return

            hit = self.nearest_polygon_vertex(x, y)
            if hit is not None:
                self.edit_drag_target = hit
                _object_index, polygon_index, vertex_index = hit
                self.selected_polygon_index = polygon_index
                self.selected_vertex_index = vertex_index
                self.render_canvas()
                return

            edge = self.nearest_polygon_edge(x, y)
            if edge is not None:
                _object_index, polygon_index, _insert_index = edge
                self.selected_polygon_index = polygon_index
                self.selected_vertex_index = -1
                self.render_canvas()
                self.set_status("Polygon selected")
                return
            self.set_status("Drag a vertex, Shift-click an edge, or start a new polygon")
            return

        if event_type == "move" and self.edit_drag_target is not None:
            target_index, polygon_index, vertex_index = self.edit_drag_target
            points = self.polygons_for_target(target_index)[polygon_index].get("points", [])
            if 0 <= vertex_index < len(points):
                points[vertex_index] = (x, y)  # type: ignore[index]
                self.rebuild_target_mask_from_polygons(target_index)
                self.render_canvas()
            return

        if event_type == "release":
            if self.edit_drag_target is not None:
                target_index, _polygon_index, _vertex_index = self.edit_drag_target
                self.rebuild_target_mask_from_polygons(target_index)
                self.update_object_list()
                self.set_status("Polygon updated")
            self.edit_drag_target = None

    def update_object_list(self) -> None:
        current_row = self.objects_list.currentRow()
        previous_block = self.objects_list.blockSignals(True)
        try:
            self.objects_list.clear()
            for index, saved in enumerate(self.saved_objects, start=1):
                area = int(saved.mask.astype(bool).sum())
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
        if self.image_np is None:
            self.canvas.set_overlay(None, [])
            if hasattr(self, "yolo_preview_label"):
                self.yolo_preview_label.setText("YOLO polygons: off")
            return
        if hasattr(self, "polygon_edit_check") and self.polygon_edit_check.isChecked():
            overlay = render_interactive_overlay(
                self.image_np,
                self.saved_objects,
                self.current_mask,
                self.current_color,
            )
            overlay = render_yolo_edit_polygon_overlay(
                overlay,
                self.saved_objects,
                selected_object_index=self.active_polygon_target_index() if self.active_polygon_target_index() is not None else -2,
                selected_polygon_index=self.selected_polygon_index,
                selected_vertex_index=self.selected_vertex_index,
                draft_polygon=self.draft_polygon_points,
                draft_mode=str(self.draft_polygon_combo.currentData() or "add"),
                current_yolo_polygons=self.current_yolo_polygons if self.current_mask is not None else None,
                current_color=self.current_color,
            )
            total_polygons = len(self.current_yolo_polygons) + sum(len(saved.yolo_polygons) for saved in self.saved_objects)
            self.yolo_preview_label.setText(f"Editable YOLO polygons: {total_polygons}")
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
            self.yolo_preview_label.setText(f"YOLO polygons: {polygon_count}")
        else:
            overlay = render_interactive_overlay(
                self.image_np,
                self.saved_objects,
                self.current_mask,
                self.current_color,
            )
            self.yolo_preview_label.setText("YOLO polygons: off")
        show_prompt_points = not (hasattr(self, "polygon_edit_check") and self.polygon_edit_check.isChecked())
        self.canvas.set_overlay(overlay, self.points if show_prompt_points else [])
        if fit:
            QTimer.singleShot(0, self.canvas.fit_image)


def apply_style(app: QApplication) -> None:
    families = set(QFontDatabase.families())
    for family in ("Noto Sans TC", "Source Han Sans TC", "Noto Sans CJK TC", "Microsoft JhengHei UI", "Segoe UI"):
        if family in families:
            app.setFont(QFont(family, 10))
            break

    app.setStyleSheet(
        """
        QWidget {
            background: #f6f8fb;
            color: #1f2937;
            font-family: "Noto Sans TC", "Source Han Sans TC", "Microsoft JhengHei UI", "Segoe UI";
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
        QCheckBox { color: #334155; spacing: 8px; }
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
