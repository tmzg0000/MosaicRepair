"""Zoomable original-coordinate image canvas for selecting regions and control points."""
from __future__ import annotations

import math
from pathlib import Path
import numpy as np
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QEvent
from PySide6.QtGui import QColor, QImage, QPixmap, QPen, QBrush, QPainter, QPolygonF, QPainterPath
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem, QLabel, QSizePolicy


class ElidedLabel(QLabel):
    """Keep long file names from expanding the window beyond the screen."""
    def __init__(self, text="", parent=None):
        super().__init__(parent)
        self.full_text = text
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setText(text)

    def setText(self, text):
        self.full_text = text
        self.setToolTip(text)
        super().setText(self.fontMetrics().elidedText(text, Qt.TextElideMode.ElideMiddle, max(40, self.width())))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.setText(self.full_text)


class ImageCanvas(QGraphicsView):
    rectangle_drawn = Signal(object)
    point_picked = Signal(object)
    region_picked = Signal(int)
    shape_drawn = Signal(object)
    wand_picked = Signal(object)
    files_dropped = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#232c36"))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.mode = "pan"
        self.image_bounds = QRectF()
        self.pixmap_item = None
        self.overlays = []
        self.temp_items = []
        self.rectangles = []
        self.start = None
        self.pan_start = None
        self.auto_fit = True
        self.brush_size = 24
        self.hover_position = None
        self.stroke = []
        self.polygon_points = []
        self.empty_text = "拖入图片，或使用顶部按钮打开"
        self.horizontalScrollBar().valueChanged.connect(self.update_brush_cursor)
        self.verticalScrollBar().valueChanged.connect(self.update_brush_cursor)

    def set_brush_size(self, size):
        self.brush_size = size
        self.update_brush_cursor()

    def brush_cursor_rect(self):
        if (self.hover_position is None or self.mode not in ("brush","eraser")
                or self.pan_start is not None or not self.isEnabled()):
            return None
        center = self.mapToScene(self.hover_position)
        if not self.image_bounds.contains(center):
            return None
        radius = self.brush_size / 2.0
        return QRectF(center.x()-radius,center.y()-radius,self.brush_size,self.brush_size)

    def update_brush_cursor(self, *args):
        if self.pan_start is not None:
            shape = Qt.CursorShape.ClosedHandCursor
        elif self.brush_cursor_rect() is not None:
            shape = Qt.CursorShape.BlankCursor
        elif self.mode in ("select","crop","point","brush","eraser","polygon","wand"):
            shape = Qt.CursorShape.CrossCursor
        else:
            shape = Qt.CursorShape.OpenHandCursor
        self.setCursor(shape)
        self.viewport().update()

    def viewportEvent(self, event):
        if event.type()==QEvent.Type.Leave:
            self.hover_position = None
            self.update_brush_cursor()
        elif event.type()==QEvent.Type.Enter:
            self.hover_position = event.position().toPoint()
            self.update_brush_cursor()
        return super().viewportEvent(event)

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type()==QEvent.Type.EnabledChange and hasattr(self,"hover_position"):
            self.update_brush_cursor()

    def drawForeground(self, painter, rect):
        super().drawForeground(painter,rect)
        footprint = self.brush_cursor_rect()
        if footprint is None:
            return
        painter.save()
        painter.setClipRect(self.image_bounds,Qt.ClipOperation.IntersectClip)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        # Two contrasting outlines remain visible on both light and dark pixels.
        painter.setPen(self._pen("#18222c",3))
        painter.drawEllipse(footprint)
        painter.setPen(self._pen("#ffffff",1))
        painter.drawEllipse(footprint)
        painter.restore()

    @staticmethod
    def dropped_paths(mime):
        if not mime.hasUrls():
            return []
        allowed = {".jpg",".jpeg",".png",".webp",".bmp",".tif",".tiff"}
        return [u.toLocalFile() for u in mime.urls() if u.isLocalFile()
                and Path(u.toLocalFile()).suffix.lower() in allowed and Path(u.toLocalFile()).is_file()]

    def dragEnterEvent(self,event):
        if self.dropped_paths(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self,event):
        self.dragEnterEvent(event)

    def dropEvent(self,event):
        paths = self.dropped_paths(event.mimeData())
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    def set_image(self, image, bounds=None, fit=False):
        if image is None:
            self.scene().clear()
            self.pixmap_item = None
            self.overlays.clear()
            self.temp_items.clear()
            self.image_bounds = QRectF()
            self.update_brush_cursor()
            return
        h, w = image.shape[:2]
        x, y, rw, rh = bounds or (0, 0, w, h)
        pixels = np.ascontiguousarray(image[y:y+rh, x:x+rw])
        qimage = QImage(pixels.data, rw, rh, pixels.strides[0], QImage.Format.Format_RGBA8888).copy()
        old_bounds = self.image_bounds
        self.image_bounds = QRectF(x, y, rw, rh)
        if self.pixmap_item is None:
            self.pixmap_item = self.scene().addPixmap(QPixmap.fromImage(qimage))
            self.pixmap_item.setZValue(0)
            fit = True
        else:
            self.pixmap_item.setPixmap(QPixmap.fromImage(qimage))
        self.pixmap_item.setOffset(x, y)
        self.setSceneRect(self.image_bounds)
        if fit or old_bounds != self.image_bounds:
            self.fit_image()
        self.update_brush_cursor()

    def set_mode(self, mode):
        self.cancel_drawing()
        self.mode = mode
        self.update_brush_cursor()

    def fit_image(self):
        if not self.image_bounds.isEmpty():
            self.fitInView(self.image_bounds.adjusted(-4, -4, 4, 4), Qt.AspectRatioMode.KeepAspectRatio)
        self.auto_fit = True
        self.update_brush_cursor()

    def actual_size(self):
        self.resetTransform()
        self.auto_fit = False
        self.update_brush_cursor()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.auto_fit:
            self.fit_image()

    def wheelEvent(self, event):
        if self.image_bounds.isEmpty():
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        current = self.transform().m11()
        if 0.01 <= current * factor <= 80:
            self.scale(factor, factor)
            self.auto_fit = False
        self.update_brush_cursor()
        event.accept()

    def _point(self, event):
        p = self.mapToScene(event.position().toPoint())
        b = self.image_bounds
        return QPointF(min(max(p.x(), b.left()), b.right()), min(max(p.y(), b.top()), b.bottom()))

    def mousePressEvent(self, event):
        self.hover_position = event.position().toPoint()
        self.update_brush_cursor()
        if event.button()==Qt.MouseButton.RightButton and self.mode=="polygon":
            if self.polygon_points:
                self.polygon_points.pop()
                self._show_path(self.polygon_points)
            event.accept()
            return
        if event.button() == Qt.MouseButton.MiddleButton or (event.button() == Qt.MouseButton.LeftButton and self.mode == "pan"):
            self.pan_start = event.position().toPoint()
            self.update_brush_cursor()
            event.accept()
            return
        if event.button() != Qt.MouseButton.LeftButton or self.image_bounds.isEmpty():
            return super().mousePressEvent(event)
        p = self.mapToScene(event.position().toPoint())
        if not self.image_bounds.contains(p):
            return
        if self.mode == "point":
            self.point_picked.emit((p.x(), p.y()))
        elif self.mode == "wand":
            self.wand_picked.emit((p.x(), p.y()))
        elif self.mode in ("brush","eraser"):
            self.start = self._point(event)
            self.stroke = [(self.start.x(),self.start.y())]
            self._show_path(self.stroke)
        elif self.mode=="polygon":
            if len(self.polygon_points)<4096:
                self.polygon_points.append((p.x(),p.y()))
                self._show_path(self.polygon_points)
        elif self.mode in ("select", "crop"):
            self.start = self._point(event)
        event.accept()

    def mouseMoveEvent(self, event):
        self.hover_position = event.position().toPoint()
        self.update_brush_cursor()
        if self.pan_start is not None:
            point = event.position().toPoint()
            delta = point - self.pan_start
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.pan_start = point
            self.auto_fit = False
            return
        if self.start is not None:
            if self.mode in ("brush","eraser"):
                p = self._point(event)
                if np.linalg.norm(np.array(self.stroke[-1])-[p.x(),p.y()])>=.5:
                    if len(self.stroke)>=4096:
                        self.stroke = self.stroke[::2]
                    self.stroke.append((p.x(),p.y()))
                self._show_path(self.stroke)
                return
            self._show_temporary(QRectF(self.start, self._point(event)).normalized())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.pan_start is not None:
            self.pan_start = None
            self.set_mode(self.mode)
            return
        if self.start is not None and event.button() == Qt.MouseButton.LeftButton:
            if self.mode in ("brush","eraser"):
                p = self._point(event)
                self.stroke.append((p.x(),p.y()))
                points = self.stroke[:4096]
                kind = self.mode
                self.cancel_drawing()
                self.shape_drawn.emit({"kind":kind,"points":points,"brush_size":self.brush_size})
                return
            rect = QRectF(self.start, self._point(event)).normalized().intersected(self.image_bounds)
            self.start = None
            if rect.width() >= 2 and rect.height() >= 2:
                x, y = math.floor(rect.x()), math.floor(rect.y())
                w, h = math.ceil(rect.right())-x, math.ceil(rect.bottom())-y
                self.rectangle_drawn.emit((x, y, w, h))
            else:
                p = rect.topLeft()
                for index, coordinates in reversed(self.rectangles):
                    if QRectF(*coordinates).contains(p):
                        self.region_picked.emit(index)
                        break
            if self.mode != "crop":
                self.cancel_drawing()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self,event):
        if self.mode=="polygon" and event.button()==Qt.MouseButton.LeftButton:
            self.finish_polygon()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def keyPressEvent(self,event):
        if self.mode=="polygon" and event.key() in (Qt.Key.Key_Return,Qt.Key.Key_Enter):
            self.finish_polygon()
            event.accept()
        else:
            super().keyPressEvent(event)

    def finish_polygon(self):
        if len(self.polygon_points)>=3:
            points = list(self.polygon_points)
            self.cancel_drawing()
            self.shape_drawn.emit({"kind":"polygon","points":points,"brush_size":self.brush_size})

    def _show_path(self,points):
        self._clear_items(self.temp_items)
        if not points:
            return
        path = QPainterPath(QPointF(*points[0]))
        for point in points[1:]:
            path.lineTo(QPointF(*point))
        if self.mode=="polygon":
            pen = self._pen("#ffc96b",2,True)
        else:
            pen = QPen(QColor("#ed7a7a" if self.mode=="eraser" else "#6aebbf"),self.brush_size)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            if len(points)==1:
                path.lineTo(QPointF(points[0][0]+.01,points[0][1]))
        item = self.scene().addPath(path,pen)
        item.setOpacity(.6)
        item.setZValue(50)
        self.temp_items.append(item)

    def _clear_items(self, items):
        for item in items:
            self.scene().removeItem(item)
        items.clear()

    def cancel_drawing(self):
        self.start = None
        self.stroke = []
        self.polygon_points = []
        self._clear_items(self.temp_items)

    @staticmethod
    def _pen(color, width=1.5, dashed=False):
        pen = QPen(QColor(color), width)
        pen.setCosmetic(True)
        if dashed:
            pen.setStyle(Qt.PenStyle.DashLine)
        return pen

    def _show_temporary(self, rect):
        self._clear_items(self.temp_items)
        if self.mode == "crop":
            path = QPainterPath()
            path.addRect(self.image_bounds)
            path.addRect(rect)
            path.setFillRule(Qt.FillRule.OddEvenFill)
            shade = self.scene().addPath(path, QPen(Qt.PenStyle.NoPen), QBrush(QColor(0, 0, 0, 145)))
            shade.setZValue(50)
            self.temp_items.append(shade)
        item = self.scene().addRect(rect, self._pen("#f9c96b", 2, True))
        item.setZValue(51)
        self.temp_items.append(item)

    def set_guides(self, rectangles=(), points=(), polygons=(), selected=-1):
        self._clear_items(self.overlays)
        self.rectangles = []
        for index, coordinates, enabled in rectangles:
            rect = QRectF(*coordinates).intersected(self.image_bounds)
            if rect.isEmpty():
                continue
            self.rectangles.append((index, coordinates))
            color = "#ffc96b" if index == selected else ("#6aebbf" if enabled else "#89929c")
            item = self.scene().addRect(rect, self._pen(color, 2, not enabled), QBrush(Qt.BrushStyle.NoBrush))
            item.setZValue(10)
            self.overlays.append(item)
            self._label(str(index+1), rect.topLeft(), color)
        for index, point in points:
            p = QPointF(*point)
            if not self.image_bounds.contains(p):
                continue
            radius = 4 / max(self.transform().m11(), 0.01)
            dot = self.scene().addEllipse(p.x()-radius, p.y()-radius, radius*2, radius*2,
                                         self._pen("#ffc96b", 2), QBrush(QColor("#232c36")))
            dot.setZValue(20)
            self.overlays.append(dot)
            self._label(str(index+1), p, "#ffc96b")
        for index, coordinates, enabled in polygons:
            if not enabled:
                continue
            polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in coordinates])
            path = QPainterPath()
            path.addPolygon(polygon)
            path.closeSubpath()
            clip = QPainterPath()
            clip.addRect(self.image_bounds)
            path = path.intersected(clip)
            if path.isEmpty():
                continue
            color = "#ffc96b" if index==selected else "#6aebbf"
            item = self.scene().addPath(path,self._pen(color,1.8,True))
            item.setZValue(10)
            self.overlays.append(item)
            self._label(str(index+1), path.boundingRect().topLeft(), color)

    def _label(self, value, point, color):
        text = self.scene().addSimpleText(value)
        text.setBrush(QBrush(QColor(color)))
        text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        text.setPos(point)
        text.setZValue(30)
        self.overlays.append(text)

    def drawBackground(self, painter, rect):
        super().drawBackground(painter, rect)
        if self.pixmap_item is None:
            painter.save()
            painter.resetTransform()
            painter.setPen(QColor("#9aa8b6"))
            painter.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter, self.empty_text)
            painter.restore()
