"""Windows-friendly Qt desktop workflow. All expensive image work runs in a worker."""
from __future__ import annotations

from pathlib import Path
import os
import sys
import traceback

import cv2
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QAction, QKeySequence, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QListWidget, QListWidgetItem, QComboBox, QSpinBox, QCheckBox,
    QDoubleSpinBox, QSplitter, QFileDialog, QMessageBox, QProgressBar, QFrame,
    QAbstractItemView, QGroupBox, QScrollArea, QLayout,
)

from .core import (RepairError, MODES, read_image, auto_align, manual_align,
                   composite, overlay, export_image, validate_matrix)
from .model import Document, Source, Region, save_project, load_project
from .widgets import ImageCanvas, ElidedLabel
from .colors import ColorSettings, estimate_color_transform, adjust_image
from .selections import source_masks, region_contours

IMAGE_FILTER = "图片 (*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff);;所有文件 (*)"
PROJECT_FILTER = "图片互补工程 (*.mrepair)"
APP_ICON_PATH = Path(__file__).resolve().parent / "assets" / "app-icon.png"
MODE_NAMES = [("相似：缩放 / 旋转 / 平移", "similarity"), ("仿射：非等比缩放", "affine"),
              ("透视：四角变换", "perspective")]
STYLE = """
QWidget { font-family: 'Microsoft YaHei', 'Noto Sans CJK SC', sans-serif; font-size: 12px; color: #263443; }
QMainWindow, QWidget#root { background: #f3f5f7; }
QFrame#header { background: #ffffff; border-bottom: 1px solid #dce3e8; }
QLabel#brand { font-size: 23px; font-weight: 700; color: #142e3b; }
QLabel#muted { color: #6b7986; }
QLabel#step { color: #208267; font-size: 11px; font-weight: 700; }
QLabel#panelTitle { font-size: 15px; font-weight: 700; }
QLabel#hint { padding: 8px 10px; background: #e8f2ee; color: #386455; border-radius: 5px; }
QPushButton { background: white; border: 1px solid #cfd9df; border-radius: 5px; padding: 7px 11px; }
QPushButton:hover { background: #eaf3ef; border-color: #80b8a4; }
QPushButton:pressed, QPushButton:checked { background: #d7ece3; border-color: #208267; color: #11694e; }
QPushButton:disabled { color: #a3adb5; background: #edf0f2; border-color: #dce2e7; }
QPushButton#primary { background: #167b5a; color: white; border-color: #167b5a; font-weight: 600; }
QPushButton#primary:hover { background: #126648; }
QPushButton#primary:disabled { background: #a2bdb2; border-color: #a2bdb2; }
QGroupBox { border: 1px solid #d9e1e6; border-radius: 7px; margin-top: 10px; padding: 12px 9px 8px; font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QListWidget { border: 1px solid #d6dfe5; border-radius: 5px; background: white; padding: 3px; outline: none; }
QListWidget::item { padding: 7px 3px; border-radius: 3px; }
QListWidget::item:selected { background: #dfeee7; color: #18583f; }
QComboBox, QSpinBox, QDoubleSpinBox { border: 1px solid #cfd9df; border-radius: 4px; background: white; padding: 5px; }
QFrame#panel { background: white; border: 1px solid #d9e1e6; border-radius: 7px; }
QSplitter::handle { background: #e3e9ed; width: 5px; }
QStatusBar { background: white; border-top: 1px solid #dce3e8; }
QProgressBar { border: none; background: #e4ece7; max-height: 6px; }
"""


class Job(QThread):
    def __init__(self, function, parent):
        super().__init__(parent)
        self.function = function
        self.result = None
        self.error = None

    def run(self):
        try:
            self.result = self.function()
        except Exception as exc:
            self.error = "可用内存不足，请减少图片尺寸或补充图数量。" if isinstance(exc, MemoryError) else (str(exc) or type(exc).__name__)
            traceback.print_exc()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.document = None
        self.project_path = None
        self.dirty = False
        self.undo_stack = []
        self.redo_stack = []
        self.active_index = -1
        self.active_region = -1
        self.manual = False
        self.pending_point = None
        self.pending_crop = None
        self.result_image = None
        self.overlay_image = None
        self.job = None
        self.callback = None
        self.preview_pending = False
        self.selection_mode = "brush"
        self.source_preview = None
        self.source_preview_key = None
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(120)
        self.preview_timer.timeout.connect(self.render_preview)
        self.setWindowTitle("图片互补修补")
        self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self.resize(1480, 900)
        self.setMinimumSize(1080, 700)
        self.setStyleSheet(STYLE)
        self.build_ui()
        self.build_shortcuts()
        self.refresh()

    @property
    def source(self):
        if self.document and 0 <= self.active_index < len(self.document.sources):
            return self.document.sources[self.active_index]
        return None

    @staticmethod
    def label(text, name=None, wrap=False):
        result = QLabel(text)
        if name:
            result.setObjectName(name)
        result.setWordWrap(wrap)
        return result

    @staticmethod
    def button(text, callback, primary=False):
        button = QPushButton(text)
        button.clicked.connect(callback)
        if primary:
            button.setObjectName("primary")
        return button

    def build_ui(self):
        self.root = QWidget()
        self.root.setObjectName("root")
        root = QVBoxLayout(self.root)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.setCentralWidget(self.root)
        header = QFrame()
        header.setObjectName("header")
        h = QHBoxLayout(header)
        h.setContentsMargins(20, 13, 20, 13)
        brand_icon = self.label("")
        brand_icon.setPixmap(self.windowIcon().pixmap(44, 44))
        brand_icon.setFixedSize(44, 44)
        h.addWidget(brand_icon)
        brand = QVBoxLayout()
        brand.addWidget(self.label("图片互补", "brand"))
        brand.addWidget(self.label("从补充图取清晰内容，修补模板对应位置", "muted"))
        h.addLayout(brand)
        h.addStretch()
        self.open_button = self.button("打开模板", self.open_base)
        self.add_button = self.button("＋ 添加补充图", self.add_sources)
        self.load_button = self.button("打开工程", self.open_project)
        self.save_button = self.button("保存工程", self.save_current)
        self.undo_button = self.button("撤销", self.undo)
        self.redo_button = self.button("重做", self.redo)
        self.export_button = self.button("导出 JPEG", self.export_current, True)
        for button in (self.open_button, self.add_button, self.load_button, self.save_button,
                       self.undo_button, self.redo_button, self.export_button):
            h.addWidget(button)
        root.addWidget(header)
        workflow = self.label("  拖入模板和补充图  →  裁切 / 对齐  →  自动匹配颜色  →  在左侧模板涂抹或圈选  →  导出 JPEG", "muted")
        workflow.setContentsMargins(17, 12, 10, 12)
        root.addWidget(workflow)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        sidebar = QWidget()
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(15, 0, 10, 12)
        side.addWidget(self.label("补充图", "panelTitle"))
        side.addWidget(self.label("靠后的图片优先覆盖，可勾选启停", "muted"))
        self.source_list = QListWidget()
        self.source_list.setMinimumHeight(110)
        self.source_list.currentRowChanged.connect(self.select_source)
        self.source_list.itemChanged.connect(self.source_checked)
        side.addWidget(self.source_list, 2)
        row = QHBoxLayout()
        self.up_button = self.button("上移", lambda: self.move_source(-1))
        self.down_button = self.button("下移", lambda: self.move_source(1))
        self.remove_button = self.button("移除", self.remove_source)
        for button in (self.up_button, self.down_button, self.remove_button):
            row.addWidget(button)
        side.addLayout(row)
        align_group = QGroupBox("对齐到模板")
        align = QVBoxLayout(align_group)
        self.mode_combo = QComboBox()
        for name, value in MODE_NAMES:
            self.mode_combo.addItem(name, value)
        self.mode_combo.currentIndexChanged.connect(self.change_alignment_mode)
        align.addWidget(self.mode_combo)
        row = QHBoxLayout()
        self.auto_button = self.button("自动对齐", self.align_automatically, True)
        self.manual_button = self.button("手动校准", self.toggle_manual)
        row.addWidget(self.auto_button)
        row.addWidget(self.manual_button)
        align.addLayout(row)
        self.alignment_info = self.label("尚未导入补充图", "muted", True)
        self.alignment_info.setTextFormat(Qt.TextFormat.PlainText)
        align.addWidget(self.alignment_info)
        self.manual_tools = QWidget()
        manual_layout = QVBoxLayout(self.manual_tools)
        manual_layout.setContentsMargins(0, 2, 0, 0)
        self.point_info = self.label("", "hint", True)
        manual_layout.addWidget(self.point_info)
        row = QHBoxLayout()
        row.addWidget(self.button("撤回一对点", self.undo_point))
        row.addWidget(self.button("清空点", self.clear_points))
        manual_layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.button("应用校准", self.apply_manual, True))
        row.addWidget(self.button("结束选点", self.toggle_manual))
        manual_layout.addLayout(row)
        align.addWidget(self.manual_tools)
        self.nudge_tools = QWidget()
        nudge = QHBoxLayout(self.nudge_tools)
        nudge.setContentsMargins(0, 0, 0, 0)
        self.nudge_step = QDoubleSpinBox()
        self.nudge_step.setRange(0.1, 20)
        self.nudge_step.setSingleStep(0.1)
        self.nudge_step.setValue(1)
        self.nudge_step.setSuffix(" px")
        self.nudge_step.setToolTip("按模板像素微调对齐位置")
        nudge.addWidget(self.nudge_step)
        for text, dx, dy in (("←", -1, 0), ("→", 1, 0), ("↑", 0, -1), ("↓", 0, 1)):
            b = self.button(text, lambda checked=False, x=dx, y=dy: self.nudge(x, y))
            b.setFixedWidth(30)
            nudge.addWidget(b)
        align.addWidget(self.nudge_tools)
        side.addWidget(align_group)
        side.addWidget(self.label("当前图片的修补选区", "panelTitle"))
        self.region_list = QListWidget()
        self.region_list.setMinimumHeight(90)
        self.region_list.currentRowChanged.connect(self.select_region)
        self.region_list.itemChanged.connect(self.region_checked)
        side.addWidget(self.region_list, 2)
        self.delete_region_button = self.button("删除选区  ·  Delete", self.delete_region)
        side.addWidget(self.delete_region_button)
        row = QHBoxLayout()
        row.addWidget(self.label("边缘羽化"))
        self.feather_spin = QSpinBox()
        self.feather_spin.setRange(0, 50)
        self.feather_spin.setSuffix(" px")
        self.feather_spin.setKeyboardTracking(False)
        self.feather_spin.setToolTip("模板像素；0 表示直接覆盖。小选区建议减小羽化。")
        self.feather_spin.valueChanged.connect(self.change_feather)
        row.addWidget(self.feather_spin)
        side.addLayout(row)
        self.color_group = QGroupBox("补充图颜色")
        color_layout = QVBoxLayout(self.color_group)
        self.auto_color_check = QCheckBox("对齐后自动匹配模板色调")
        self.auto_color_check.toggled.connect(self.change_auto_color)
        color_layout.addWidget(self.auto_color_check)
        self.match_color_button = self.button("重新自动匹配颜色",self.match_colors,True)
        color_layout.addWidget(self.match_color_button)
        self.color_info = self.label("先对齐，再自动匹配亮度、对比度和饱和度。","muted",True)
        color_layout.addWidget(self.color_info)
        self.color_spins = {}
        for key,label in (("strength","自动匹配强度"),("brightness","亮度"),("contrast","对比度"),
                          ("saturation","饱和度"),("temperature","冷暖")):
            row = QHBoxLayout()
            row.addWidget(self.label(label))
            spin = QSpinBox()
            spin.setRange(0 if key=="strength" else -100,100)
            spin.setKeyboardTracking(False)
            spin.valueChanged.connect(lambda value,k=key:self.change_color(k,value))
            self.color_spins[key] = spin
            row.addWidget(spin)
            color_layout.addLayout(row)
        color_layout.addWidget(self.button("恢复原始颜色",self.reset_colors))
        side.addWidget(self.color_group)
        side.addWidget(self.label("左侧模板涂抹 / 圈选 · 中间补充图裁切\n滚轮缩放 · 中键拖动画布", "muted", True))
        side.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.side_scroll = QScrollArea()
        self.side_scroll.setWidgetResizable(True)
        self.side_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.side_scroll.setWidget(sidebar)
        self.side_scroll.setMinimumWidth(275)
        self.side_scroll.setMaximumWidth(345)
        self.splitter.addWidget(self.side_scroll)

        self.source_panel = QFrame()
        self.source_panel.setObjectName("panel")
        source_layout = QVBoxLayout(self.source_panel)
        source_layout.setContentsMargins(10, 12, 10, 10)
        source_layout.addWidget(self.label("02   补充图：裁切 / 校准 / 调色预览", "step"))
        self.source_title = ElidedLabel("请添加补充图")
        self.source_title.setObjectName("panelTitle")
        source_layout.addWidget(self.source_title)
        self.source_title.setTextFormat(Qt.TextFormat.PlainText)
        row = QHBoxLayout()
        self.crop_tool = self.button("裁切图片", lambda: self.set_tool("crop"))
        self.crop_tool.setCheckable(True)
        self.reset_crop_button = self.button("恢复完整图片", self.reset_crop)
        for button in (self.crop_tool, self.reset_crop_button):
            row.addWidget(button)
        self.source_original = QCheckBox("对比原补充图")
        self.source_original.toggled.connect(lambda:self.show_source_preview())
        row.addWidget(self.source_original)
        row.addStretch()
        source_layout.addLayout(row)
        row = QHBoxLayout()
        self.quick_match_button = self.button("自动匹配颜色",self.match_colors,True)
        row.addWidget(self.quick_match_button)
        row.addWidget(self.button("颜色微调",lambda:self.side_scroll.ensureWidgetVisible(self.color_group)))
        row.addWidget(self.label("亮度 / 对比度 / 饱和度","muted"),1)
        source_layout.addLayout(row)
        self.crop_tools = QWidget()
        crop_row = QHBoxLayout(self.crop_tools)
        crop_row.setContentsMargins(0, 0, 0, 0)
        self.apply_crop_button = self.button("应用裁切", self.apply_crop, True)
        crop_row.addWidget(self.apply_crop_button)
        crop_row.addWidget(self.button("取消", lambda: self.set_tool(self.selection_mode)))
        crop_row.addWidget(self.label("拖出需要保留的范围", "muted"))
        source_layout.addWidget(self.crop_tools)
        self.source_canvas = ImageCanvas()
        self.source_canvas.rectangle_drawn.connect(self.add_rectangle)
        self.source_canvas.files_dropped.connect(self.add_source_paths)
        self.source_canvas.empty_text = "将一张或多张补充图拖到这里"
        self.source_canvas.point_picked.connect(self.pick_source_point)
        self.source_canvas.region_picked.connect(lambda i: self.region_list.setCurrentRow(i))
        source_layout.addWidget(self.source_canvas, 1)
        source_layout.addLayout(self.canvas_footer(self.source_canvas, "与模板同步显示选区 · 勾选原图可对比颜色"))
        self.splitter.addWidget(self.source_panel)

        result_panel = QFrame()
        result_panel.setObjectName("panel")
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(10, 12, 10, 10)
        result_layout.addWidget(self.label("01 / 03   模板：绘制修补选区与结果预览", "step"))
        self.base_title = ElidedLabel("请先打开模板")
        self.base_title.setObjectName("panelTitle")
        self.base_title.setTextFormat(Qt.TextFormat.PlainText)
        result_layout.addWidget(self.base_title)
        row = QHBoxLayout()
        self.preview_combo = QComboBox()
        self.preview_combo.addItems(["修补结果", "原始模板", "对齐叠加 · 当前补充图 50%"])
        self.preview_combo.currentIndexChanged.connect(self.change_preview)
        row.addWidget(self.preview_combo, 1)
        self.show_guides = QCheckBox("显示选框")
        self.show_guides.setChecked(True)
        self.show_guides.toggled.connect(self.update_guides)
        row.addWidget(self.show_guides)
        result_layout.addLayout(row)
        row = QHBoxLayout()
        self.selection_buttons = {}
        for mode,name in (("brush","涂抹"),("polygon","多边形"),("select","矩形"),("eraser","橡皮")):
            button = self.button(name,lambda checked=False,m=mode:self.set_tool(m))
            button.setCheckable(True)
            row.addWidget(button)
            self.selection_buttons[mode] = button
        self.select_tool = self.selection_buttons["select"]
        result_layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(self.label("画笔直径"))
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1,500)
        self.brush_spin.setValue(24)
        self.brush_spin.setSuffix(" px")
        self.brush_spin.valueChanged.connect(lambda value:setattr(self.result_canvas,"brush_size",value))
        row.addWidget(self.brush_spin)
        row.addWidget(self.label("多边形：单击加点，双击或 Enter 完成","muted"),1)
        result_layout.addLayout(row)
        self.result_hint = self.label("先对齐补充图，再框选清晰区域。", "hint", True)
        result_layout.addWidget(self.result_hint)
        self.result_canvas = ImageCanvas()
        self.result_canvas.point_picked.connect(self.pick_target_point)
        self.result_canvas.rectangle_drawn.connect(self.add_template_rectangle)
        self.result_canvas.shape_drawn.connect(self.add_template_shape)
        self.result_canvas.files_dropped.connect(self.drop_template)
        self.result_canvas.empty_text = "将模板图片拖到这里"
        result_layout.addWidget(self.result_canvas, 1)
        result_layout.addLayout(self.canvas_footer(self.result_canvas, "导出保持模板尺寸 · 不含选框线"))
        self.splitter.addWidget(result_panel)
        self.splitter.insertWidget(0,result_panel)
        self.splitter.addWidget(self.side_scroll)
        self.splitter.setSizes([580,580,300])
        root.addWidget(self.splitter, 1)
        self.status_label = QLabel("就绪 · 图片仅在本地处理")
        self.statusBar().addWidget(self.status_label, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setFixedWidth(170)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.label("Windows 10 x64  ·  JPEG 95", "muted"))

    def canvas_footer(self, canvas, hint):
        row = QHBoxLayout()
        row.addWidget(self.label(hint, "muted"), 1)
        row.addWidget(self.button("适应", canvas.fit_image))
        row.addWidget(self.button("1:1", canvas.actual_size))
        return row

    def build_shortcuts(self):
        for name, sequence, handler in (("打开模板", "Ctrl+O", self.open_base),
                ("添加补充图", "Ctrl+I", self.add_sources), ("保存工程", "Ctrl+S", self.save_current),
                ("导出", "Ctrl+E", self.export_current), ("撤销", "Ctrl+Z", self.undo),
                ("重做", "Ctrl+Y", self.redo), ("删除选框", "Delete", self.delete_region),
                ("取消", "Esc", self.escape)):
            action = QAction(name, self)
            action.setShortcut(QKeySequence(sequence))
            action.triggered.connect(handler)
            self.addAction(action)

    def notice(self, message):
        QMessageBox.warning(self, "图片互补", str(message))

    def start_task(self, message, function, callback):
        if self.job is not None:
            return False
        self.preview_timer.stop()
        self.root.setEnabled(False)
        self.status_label.setText(message)
        self.progress.show()
        self.callback = callback
        self.job = Job(function, self)
        self.job.finished.connect(self.finish_task)
        self.job.start()
        return True

    def finish_task(self):
        job, callback = self.job, self.callback
        self.job, self.callback = None, None
        self.root.setEnabled(True)
        self.progress.hide()
        self.status_label.setText("就绪 · 图片仅在本地处理")
        if job.error:
            self.notice(job.error)
        else:
            try:
                callback(job.result)
            except Exception as exc:
                traceback.print_exc()
                self.notice(exc)
        job.deleteLater()
        if self.preview_pending:
            self.preview_pending = False
            self.schedule_preview()

    def remember(self):
        if self.document:
            self.undo_stack.append((self.document.clone(), self.active_index, self.active_region))
            self.undo_stack = self.undo_stack[-30:]
            self.redo_stack.clear()
            self.dirty = True

    def undo(self):
        if self.job or not self.undo_stack:
            return
        self.escape()
        self.redo_stack.append((self.document.clone(), self.active_index, self.active_region))
        self.document, self.active_index, self.active_region = self.undo_stack.pop()
        self.dirty = True
        self.refresh()

    def redo(self):
        if self.job or not self.redo_stack:
            return
        self.escape()
        self.undo_stack.append((self.document.clone(), self.active_index, self.active_region))
        self.document, self.active_index, self.active_region = self.redo_stack.pop()
        self.dirty = True
        self.refresh()

    def confirm_discard(self, continuation=None):
        if not self.dirty:
            return True
        answer = QMessageBox.question(self, "未保存的工程", "当前工程有未保存的更改，是否保存？",
                     QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                     QMessageBox.StandardButton.Save)
        if answer == QMessageBox.StandardButton.Discard:
            return True
        if answer == QMessageBox.StandardButton.Save:
            self.save_current(after=continuation)
        return False

    def open_base(self):
        if self.job:
            return
        path,_ = QFileDialog.getOpenFileName(self,"选择模板图片","",IMAGE_FILTER)
        if path:
            self.load_base_path(path)

    def drop_template(self,paths):
        if len(paths)!=1:
            self.notice("模板区域一次请拖入一张图片；多张补充图请拖入中间区域。")
            return
        self.load_base_path(paths[0])

    def load_base_path(self,path):
        if self.job or not self.confirm_discard(lambda:self.load_base_path(path)):
            return
        def done(image):
            resolved = str(Path(path).resolve())
            self.install_document(Document(Path(path).name,image,original_paths=(resolved,),base_path=resolved))
        self.start_task("正在读取模板…",lambda:read_image(path),done)

    def install_document(self, document, project_path=None):
        self.document = document
        self.project_path = project_path
        self.dirty = False
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.active_index = 0 if document.sources else -1
        self.active_region = -1
        self.manual = False
        self.pending_point = None
        self.pending_crop = None
        self.result_image = document.base.copy()
        self.overlay_image = None
        self.source_preview = None
        self.source_preview_key = None
        self.source_canvas.set_mode("pan")
        self.result_canvas.set_image(document.base, fit=True)
        self.refresh(fit_source=True)

    def add_sources(self):
        if self.job or not self.document:
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "选择一张或多张补充图", "", IMAGE_FILTER)
        if not paths:
            return
        self.add_source_paths(paths)

    def add_source_paths(self,paths):
        if self.job:
            return
        if not self.document:
            self.notice("请先将模板图片拖入左侧，再添加补充图。")
            return
        if not paths:
            return
        snapshot = self.document.clone()
        def read_all():
            for path in paths:
                image = read_image(path)
                snapshot.check_capacity(image.shape[0] * image.shape[1])
                snapshot.sources.append(Source(Path(path).name, image))
            return snapshot.sources[len(self.document.sources):]
        def done(sources):
            self.remember()
            self.document.sources.extend(sources)
            self.document.original_paths = tuple(dict.fromkeys(self.document.original_paths + tuple(str(Path(p).resolve()) for p in paths)))
            self.active_index = len(self.document.sources) - len(sources)
            self.refresh(fit_source=True)
        self.start_task("正在读取补充图…", read_all, done)

    def select_source(self, row):
        if row == self.active_index:
            return
        self.manual = False
        self.pending_point = None
        self.active_index = row
        self.active_region = -1
        self.pending_crop = None
        self.overlay_image = None
        self.source_canvas.set_mode("pan")
        self.refresh(fit_source=True)

    def source_checked(self, item):
        if not self.document or self.job:
            return
        index = self.source_list.row(item)
        enabled = item.checkState() == Qt.CheckState.Checked
        if self.document.sources[index].enabled != enabled:
            self.remember()
            self.document.sources[index].enabled = enabled
            self.refresh()

    def move_source(self, direction):
        if self.job or not self.source:
            return
        new_index = self.active_index + direction
        if not 0 <= new_index < len(self.document.sources):
            return
        self.remember()
        self.document.sources[self.active_index], self.document.sources[new_index] = self.document.sources[new_index], self.source
        self.active_index = new_index
        self.refresh()

    def remove_source(self):
        if self.job or not self.source:
            return
        self.remember()
        self.document.sources.pop(self.active_index)
        self.active_index = min(self.active_index, len(self.document.sources)-1)
        self.manual = False
        self.pending_point = None
        self.source_canvas.set_mode("pan")
        self.refresh(fit_source=True)

    def change_alignment_mode(self, index):
        if not self.source or self.job:
            return
        mode = self.mode_combo.currentData()
        if self.source.mode != mode:
            self.remember()
            self.source.mode = mode
            self.refresh()

    def align_automatically(self):
        if not self.source or self.job:
            return
        source = self.source
        def done(alignment):
            self.remember()
            source.matrix, source.message = alignment.matrix, alignment.message
            source.color.transform = None
            self.preview_combo.setCurrentIndex(2)
            self.refresh()
            if source.color.auto:
                QTimer.singleShot(0,self.match_colors)
        self.start_task("正在匹配特征并对齐，请稍候…",
                        lambda: auto_align(source.image, self.document.base, source.mode, source.crop), done)

    def toggle_manual(self):
        if not self.source or self.job:
            return
        self.manual = not self.manual
        self.pending_point = None
        self.pending_crop = None
        self.source_canvas.set_mode("point" if self.manual else "pan")
        self.refresh()
        if self.manual:
            QTimer.singleShot(0, lambda: self.side_scroll.ensureWidgetVisible(self.manual_tools))

    def pick_source_point(self, point):
        if self.manual and self.source and not self.job:
            self.pending_point = point
            self.update_guides()
            self.update_manual_info()

    def pick_target_point(self, point):
        if not self.manual or self.pending_point is None or self.job:
            return
        if len(self.source.source_points) >= 100:
            self.notice("每张图最多保存 100 对校准点。")
            return
        self.remember()
        self.source.source_points.append(self.pending_point)
        self.source.target_points.append(point)
        self.pending_point = None
        self.refresh()

    def undo_point(self):
        if self.pending_point is not None:
            self.pending_point = None
            self.update_manual_info()
            self.update_guides()
        elif self.source and self.source.source_points:
            self.remember()
            self.source.source_points.pop()
            self.source.target_points.pop()
            self.refresh()

    def clear_points(self):
        if self.source and self.source.source_points:
            self.remember()
            self.source.source_points.clear()
            self.source.target_points.clear()
        self.pending_point = None
        self.refresh()

    def apply_manual(self):
        if not self.source or self.job:
            return
        source = self.source
        src, dst = source.visible_pairs()
        try:
            alignment = manual_align(src, dst, source.image.shape, self.document.base.shape, source.mode)
        except (RepairError, cv2.error) as exc:
            self.notice(exc)
            return
        self.remember()
        source.matrix, source.message = alignment.matrix, alignment.message
        source.color.transform = None
        self.manual = False
        self.pending_point = None
        self.source_canvas.set_mode("pan")
        self.preview_combo.setCurrentIndex(2)
        self.refresh()

        if source.color.auto:
            QTimer.singleShot(0,self.match_colors)

    def update_manual_info(self):
        if not self.source:
            return
        src, _ = self.source.visible_pairs()
        required = MODES[self.source.mode]
        next_step = "在左侧原模板点击对应位置" if self.pending_point is not None else "先在中间补充图点击一个清晰特征"
        self.point_info.setText(f"有效点 {len(src)} 对 / 至少 {required} 对\n{next_step}。点应尽量分散。")

    def nudge(self, dx, dy):
        if self.job or not self.source or self.source.matrix is None:
            return
        shift = np.eye(3)
        shift[:2, 2] = (dx*self.nudge_step.value(), dy*self.nudge_step.value())
        try:
            matrix = validate_matrix(shift @ self.source.matrix, self.source.image.shape, self.document.base.shape)
        except RepairError as exc:
            self.notice(exc)
            return
        self.remember()
        self.source.matrix = matrix
        if " · 已微调" not in self.source.message:
            self.source.message += " · 已微调"
        self.refresh()

    def set_tool(self,tool):
        if self.job or not self.source:
            return
        self.manual = False
        self.pending_point = None
        self.pending_crop = None
        if tool=="crop":
            self.source_canvas.set_mode("crop")
        else:
            self.selection_mode = tool
            self.source_canvas.set_mode("pan")
        self.result_canvas.cancel_drawing()
        self.refresh(schedule=False)

    def add_rectangle(self, rect):
        if self.job or not self.source:
            return
        if self.source_canvas.mode == "crop":
            self.pending_crop = rect
            self.apply_crop_button.setEnabled(True)
            return
    def add_template_rectangle(self,rect):
        self.commit_template_region(Region(rect,space="template"))

    def add_template_shape(self,payload):
        if not self.document:
            return
        try:
            region = Region.from_shape(payload["kind"],payload["points"],self.document.base.shape,payload["brush_size"])
            self.commit_template_region(region)
        except RepairError as exc:
            self.notice(exc)

    def commit_template_region(self,region):
        if self.job or not self.source or self.source.matrix is None:
            self.notice("请先将当前补充图对齐到模板，再在模板上选择修补区域。")
            return
        if len(self.source.regions)>=1000:
            self.notice("每张补充图最多保存 1000 个选区或笔画。")
            return
        self.remember()
        self.source.regions.append(region)
        self.active_region = len(self.source.regions)-1
        self.preview_combo.setCurrentIndex(0)
        self.refresh()

    def apply_crop(self):
        if not self.source or self.pending_crop is None or self.job:
            return
        self.remember()
        self.source.crop = self.pending_crop
        self.pending_crop = None
        self.source_canvas.set_mode("pan")
        self.refresh(fit_source=True)

    def reset_crop(self):
        if not self.source or self.source.crop is None or self.job:
            return
        self.remember()
        self.source.crop = None
        self.pending_crop = None
        self.source_canvas.set_mode("pan")
        self.refresh(fit_source=True)

    def select_region(self, row):
        self.active_region = row
        self.delete_region_button.setEnabled(self.source is not None and row >= 0)
        self.update_guides()

    def region_checked(self, item):
        if not self.source or self.job:
            return
        index = self.region_list.row(item)
        enabled = item.checkState() == Qt.CheckState.Checked
        if self.source.regions[index].enabled != enabled:
            self.remember()
            self.source.regions[index].enabled = enabled
            self.refresh()

    def delete_region(self):
        if self.job or not self.source or not 0 <= self.active_region < len(self.source.regions):
            return
        if self.manual or self.source_canvas.mode == "crop":
            return
        self.remember()
        self.source.regions.pop(self.active_region)
        self.active_region = min(self.active_region, len(self.source.regions)-1)
        self.refresh()

    def change_feather(self, value):
        if self.document and not self.job and self.document.feather != value:
            self.remember()
            self.document.feather = value
            self.refresh()

    def match_colors(self):
        if self.job or not self.source or self.source.matrix is None:
            return
        source = self.source
        base = self.document.base
        def estimate():
            exclusion,_ = source_masks(source,base.shape)
            return estimate_color_transform(source.image,base,source.matrix,source.crop,exclusion)
        def done(transform):
            self.remember()
            source.color.auto = True
            source.color.transform = transform
            self.refresh()
        self.start_task("正在分析重叠清晰区域并匹配亮度、对比度和饱和度…",estimate,done)

    def change_auto_color(self,enabled):
        if not self.source or self.job or self.source.color.auto==enabled:
            return
        self.remember()
        self.source.color.auto = enabled
        if enabled and self.source.color.transform is None and self.source.matrix is not None:
            self.match_colors()
        else:
            self.refresh()

    def change_color(self,key,value):
        if not self.source or self.job or getattr(self.source.color,key)==value:
            return
        self.remember()
        setattr(self.source.color,key,value)
        self.refresh()

    def reset_colors(self):
        if not self.source or self.job:
            return
        self.remember()
        self.source.color = ColorSettings(auto=False)
        self.refresh()

    def show_source_preview(self):
        if not self.source:
            return
        key = (id(self.source.image),self.source.color.key())
        image = (self.source_preview if self.source_preview_key==key and not self.source_original.isChecked()
                 else self.source.image)
        self.source_canvas.set_image(image,self.source.bounds)

    def change_preview(self):
        if self.document:
            self.show_result()
            self.schedule_preview()

    def escape(self):
        if self.job:
            return
        self.manual = False
        self.pending_point = None
        self.pending_crop = None
        self.source_canvas.set_mode("pan")
        self.refresh(schedule=False)

    def refresh(self, fit_source=False, schedule=True):
        document, source = self.document, self.source
        if document:
            self.active_index = min(self.active_index, len(document.sources)-1)
            source = self.source
        self.source_list.blockSignals(True)
        self.source_list.clear()
        for index, s in enumerate(document.sources if document else []):
            badge = "已对齐" if s.matrix is not None else "待对齐"
            item = QListWidgetItem(f"{index+1}. {s.name}\n    {badge} · {len(s.regions)} 个框")
            item.setToolTip(s.name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if s.enabled else Qt.CheckState.Unchecked)
            self.source_list.addItem(item)
        self.source_list.setCurrentRow(self.active_index)
        self.source_list.blockSignals(False)
        self.region_list.blockSignals(True)
        self.region_list.clear()
        for index, region in enumerate(source.regions if source else []):
            x, y, w, h = region.rect
            intersects = region.space=="template" or not (x+w <= source.bounds[0] or y+h <= source.bounds[1] or
                         x >= source.bounds[0]+source.bounds[2] or y >= source.bounds[1]+source.bounds[3])
            name = {"rectangle":"矩形","polygon":"多边形","brush":"涂抹","eraser":"橡皮"}[region.kind]
            item = QListWidgetItem(f"{index+1}. {name} · {w} × {h}" + (" · 裁切外" if not intersects else ""))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if region.enabled else Qt.CheckState.Unchecked)
            self.region_list.addItem(item)
        if source:
            self.active_region = min(self.active_region, len(source.regions)-1)
        else:
            self.active_region = -1
        self.region_list.setCurrentRow(self.active_region)
        self.region_list.blockSignals(False)
        self.mode_combo.blockSignals(True)
        self.mode_combo.setCurrentIndex(self.mode_combo.findData(source.mode) if source else 0)
        self.mode_combo.blockSignals(False)
        self.feather_spin.blockSignals(True)
        self.feather_spin.setValue(document.feather if document else 3)
        self.feather_spin.blockSignals(False)
        color = source.color if source else ColorSettings()
        self.auto_color_check.blockSignals(True)
        self.auto_color_check.setChecked(color.auto)
        self.auto_color_check.blockSignals(False)
        for name,spin in self.color_spins.items():
            spin.blockSignals(True)
            spin.setValue(getattr(color,name))
            spin.blockSignals(False)
        self.color_group.setEnabled(source is not None)
        self.match_color_button.setEnabled(source is not None and source.matrix is not None)
        self.quick_match_button.setEnabled(source is not None and source.matrix is not None)
        if color.auto and color.transform is not None:
            self.color_info.setText(f"已自动匹配 · {color.transform.samples} 个参考像素\n中间显示调色后补充图，修补与导出使用相同参数。")
        elif color.auto:
            self.color_info.setText("尚未匹配：先对齐，或点击自动匹配颜色。")
        else:
            self.color_info.setText("自动匹配已关闭；手动微调仍会应用于补充图。")
        self.add_button.setEnabled(document is not None)
        self.save_button.setEnabled(document is not None)
        self.export_button.setEnabled(document is not None)
        self.undo_button.setEnabled(bool(self.undo_stack))
        self.redo_button.setEnabled(bool(self.redo_stack))
        for widget in (self.up_button, self.down_button, self.remove_button, self.mode_combo,
                       self.auto_button, self.manual_button, self.select_tool, self.crop_tool):
            widget.setEnabled(source is not None)
        self.feather_spin.setEnabled(document is not None)
        self.reset_crop_button.setEnabled(source is not None and source.crop is not None)
        self.delete_region_button.setEnabled(source is not None and self.active_region >= 0)
        self.manual_tools.setVisible(self.manual)
        self.manual_button.setText("退出校准" if self.manual else "手动校准")
        self.nudge_tools.setVisible(source is not None and source.matrix is not None and not self.manual)
        is_crop = self.source_canvas.mode == "crop"
        self.crop_tools.setVisible(is_crop)
        self.apply_crop_button.setEnabled(self.pending_crop is not None)
        can_select = source is not None and source.matrix is not None and not self.manual and not is_crop
        for mode,button in self.selection_buttons.items():
            button.setChecked(mode==self.selection_mode and can_select)
            button.setEnabled(can_select)
        self.brush_spin.setEnabled(can_select)
        self.crop_tool.setChecked(is_crop)
        self.result_canvas.set_mode("point" if self.manual else (self.selection_mode if can_select else "pan"))
        self.preview_combo.setEnabled(not self.manual)
        if source:
            x, y, w, h = source.bounds
            crop_text = " · 已裁切" if source.crop else ""
            tone_text = " · 已自动调色" if source.color.auto and source.color.transform is not None else ""
            self.source_title.setText(f"{source.name}  ·  {w} × {h}{crop_text}{tone_text}")
            self.alignment_info.setText(source.message)
            self.source_canvas.set_image(source.image, source.bounds, fit=fit_source)
            self.show_source_preview()
        else:
            self.source_title.setText("请添加补充图")
            self.alignment_info.setText("尚未导入补充图")
            self.source_canvas.set_image(None)
        if document:
            h, w = document.base.shape[:2]
            self.base_title.setText(f"{document.base_name}  ·  {w} × {h}")
            self.setWindowTitle(f"{'* ' if self.dirty else ''}{Path(self.project_path).name if self.project_path else document.base_name} — 图片互补")
        else:
            self.base_title.setText("请先打开模板")
        if self.manual:
            self.result_hint.setText("手动校准：在中间补充图点一下 → 左侧原模板对应位置点一下，重复后应用校准。")
        elif source and source.matrix is None:
            self.result_hint.setText("当前补充图尚未对齐；可先裁切，再自动对齐或手动校准。")
        elif self.preview_combo.currentIndex() == 2:
            self.result_hint.setText("对齐检查：当前补充图以 50% 透明度叠加；重影明显时请校准或微调。")
        else:
            self.result_hint.setText("在此模板上涂抹或圈选要修补的位置；中间补充图同步显示对应选区。")
        self.update_manual_info()
        self.show_result()
        if schedule and document:
            self.schedule_preview()

    def update_guides(self,*args):
        source = self.source
        src_points,dst_points,src_shapes,dst_shapes = [],[],[],[]
        if source:
            x,y,w,h = source.bounds
            for i,(s,t) in enumerate(zip(source.source_points,source.target_points)):
                if self.manual and x<=s[0]<x+w and y<=s[1]<y+h:
                    src_points.append((i,s))
                    dst_points.append((i,t))
            if self.pending_point is not None:
                src_points.append((len(source.source_points),self.pending_point))
            if self.show_guides.isChecked() and not self.manual:
                for i,region in enumerate(source.regions):
                    for contour in region_contours(region):
                        points = np.asarray(contour,dtype=np.float32)[None]
                        if region.space=="source":
                            src_shapes.append((i,contour,region.enabled))
                            if source.matrix is not None:
                                dst_shapes.append((i,cv2.perspectiveTransform(points,source.matrix)[0],region.enabled and source.enabled))
                        else:
                            dst_shapes.append((i,contour,region.enabled and source.enabled))
                            if source.matrix is not None:
                                src_shapes.append((i,cv2.perspectiveTransform(points,np.linalg.inv(source.matrix))[0],region.enabled))
        self.source_canvas.set_guides(points=src_points,polygons=src_shapes,selected=self.active_region)
        self.result_canvas.set_guides(points=dst_points,polygons=dst_shapes,selected=self.active_region)

    def show_result(self):
        if not self.document:
            self.result_canvas.set_image(None)
            self.update_guides()
            return
        index = self.preview_combo.currentIndex()
        image = self.document.base if self.manual or index == 1 else (
                self.overlay_image if index == 2 and self.overlay_image is not None else self.result_image)
        self.result_canvas.set_image(image if image is not None else self.document.base)
        self.update_guides()

    def schedule_preview(self):
        if self.job:
            self.preview_pending = True
        else:
            self.preview_timer.start()

    def render_preview(self):
        if not self.document:
            return
        if self.job:
            self.preview_pending = True
            return
        if any(c.start is not None or c.pan_start is not None or c.polygon_points for c in (self.source_canvas,self.result_canvas)):
            self.preview_timer.start()
            return
        snapshot = self.document.clone()
        source = snapshot.sources[self.active_index] if self.source else None
        need_overlay = self.preview_combo.currentIndex() == 2 and not self.manual
        def render():
            result = composite(snapshot.base, snapshot.sources, snapshot.feather)
            # Crop restricts the alignment preview as well as the real patches.
            aligned = overlay(snapshot.base, source) if need_overlay else None
            adjusted = adjust_image(source.image,source.color) if source else None
            return result, aligned, adjusted
        def done(images):
            self.result_image, self.overlay_image, self.source_preview = images
            self.source_preview_key = (id(self.source.image),self.source.color.key()) if self.source else None
            self.show_result()
            self.show_source_preview()
            count = sum(len(s.regions) for s in self.document.sources)
            self.status_label.setText(f"{len(self.document.sources)} 张补充图 · {count} 个选框 · 本地处理")
        self.start_task("正在生成修补预览…", render, done)

    def save_current(self, checked=False, after=None):
        if self.job or not self.document:
            return
        path = self.project_path
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "保存工程（含全部图片）", "修补工程.mrepair", PROJECT_FILTER)
        if not path:
            return
        if not path.lower().endswith(".mrepair"):
            path += ".mrepair"
        snapshot = self.document.clone()
        def done(_):
            self.project_path = path
            self.dirty = False
            self.refresh(schedule=False)
            self.status_label.setText(f"工程已保存：{path}")
            if after:
                after()
        self.start_task("正在保存工程和图片…", lambda: save_project(path, snapshot), done)

    def open_project(self):
        if self.job or not self.confirm_discard(self.open_project):
            return
        path, _ = QFileDialog.getOpenFileName(self, "打开工程", "", PROJECT_FILTER)
        if path:
            self.start_task("正在读取工程…", lambda: load_project(path), lambda d: self.install_document(d, path))

    def default_export_path(self):
        name = Path(self.document.base_name).stem + "_修补.jpg"
        if self.document.base_path:
            template = Path(self.document.base_path)
            if template.is_absolute() and template.parent.is_dir():
                return str(template.parent / name)
        if self.project_path:
            parent = Path(self.project_path).resolve().parent
            if parent.is_dir():
                return str(parent / name)
        return name

    def export_current(self):
        if self.job or not self.document:
            return
        unaligned = [s.name for s in self.document.sources if s.enabled and s.matrix is None and any(r.enabled for r in s.regions)]
        if unaligned:
            self.notice("以下补充图已框选但尚未对齐，请先对齐或取消勾选：\n" + "\n".join(unaligned))
            return
        path, selected = QFileDialog.getSaveFileName(self, "导出修补图片", self.default_export_path(),
                       "JPEG 图片 (*.jpg *.jpeg);;PNG 图片 (*.png)")
        if not path:
            return
        extension = ".png" if selected.startswith("PNG") else ".jpg"
        if Path(path).suffix.lower() not in (".jpg", ".jpeg", ".png"):
            path += extension
        canonical = os.path.normcase(str(Path(path).resolve()))
        protected = {os.path.normcase(str(Path(p).resolve())) for p in self.document.original_paths}
        if self.document.base_path:
            protected.add(os.path.normcase(str(Path(self.document.base_path).resolve())))
        if self.project_path:
            protected.add(os.path.normcase(str(Path(self.project_path).resolve())))
        if canonical in protected:
            self.notice("请另选导出文件名，保留原始图片和工程。")
            return
        snapshot = self.document.clone()
        def write():
            export_image(path, composite(snapshot.base, snapshot.sources, snapshot.feather))
        self.start_task("正在导出原分辨率图片…", write,
                        lambda _: self.status_label.setText(f"图片已导出：{path}"))

    def closeEvent(self, event):
        if self.job:
            self.notice("当前任务正在处理，请完成后再关闭。")
            event.ignore()
        elif self.confirm_discard(self.close):
            self.preview_timer.stop()
            event.accept()
        else:
            event.ignore()


def run():
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MosaicRepair.Desktop")
    app = QApplication(sys.argv)
    app.setApplicationName("图片互补修补")
    app.setOrganizationName("MosaicRepair")
    app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    app.setFont(QFont("Microsoft YaHei", 10))
    window = MainWindow()
    window.show()
    if "--smoke-test" in sys.argv:
        QTimer.singleShot(150, app.quit)
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith(".mrepair"):
        path = sys.argv[1]
        QTimer.singleShot(0, lambda: window.start_task("正在读取工程…", lambda: load_project(path),
                                                    lambda d: window.install_document(d, path)))
    return app.exec()
