"""Image alignment and strictly masked compositing. Coordinates are original pixels."""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import math
import os
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_PIXELS = 36_000_000
MAX_EDGE = 16000
MODES = {"similarity": 2, "affine": 3, "perspective": 4}


class RepairError(ValueError):
    """An actionable error suitable for display in the application."""


@dataclass
class Alignment:
    matrix: np.ndarray
    message: str
    inliers: int = 0
    error: float = 0.0


def read_image(value: str | Path | bytes) -> np.ndarray:
    try:
        with Image.open(BytesIO(value) if isinstance(value, bytes) else value) as image:
            w, h = image.size
            if w * h > MAX_PIXELS or max(w, h) > MAX_EDGE:
                raise RepairError("图片过大：每张最多 3600 万像素，单边最多 16000 像素。")
            array = np.array(ImageOps.exif_transpose(image).convert("RGBA"))
            array.setflags(write=False)
            return array
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise RepairError(f"无法读取图片：{exc}") from exc


def png_bytes(array: np.ndarray) -> bytes:
    output = BytesIO()
    Image.fromarray(array).save(output, format="PNG")
    return output.getvalue()


def atomic_write(path: str | Path, data: bytes) -> None:
    path = Path(path)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".mrepair-", delete=False) as f:
            tmp = f.name
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def export_image(path: str | Path, image: np.ndarray) -> None:
    output = BytesIO()
    pil = Image.fromarray(image)
    suffix = Path(path).suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        background = Image.new("RGB", pil.size, "white")
        background.paste(pil, mask=pil.getchannel("A"))
        background.save(output, "JPEG", quality=95, subsampling=0)
    elif suffix == ".png":
        pil.save(output, "PNG")
    else:
        raise RepairError("导出格式请选择 PNG 或 JPEG。")
    atomic_write(path, output.getvalue())


def validate_matrix(matrix: np.ndarray, source_shape, target_shape) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-10:
        raise RepairError("对齐矩阵无效，请重新校准。")
    matrix = matrix / matrix[2, 2]
    if abs(np.linalg.det(matrix)) < 1e-10 or np.linalg.cond(matrix) > 1e12:
        raise RepairError("校准点无法确定稳定的变换，请选择分散且不共线的点。")
    h, w = source_shape[:2]
    th, tw = target_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    denominators = np.c_[corners, np.ones(4)] @ matrix[2]
    if np.any(denominators <= 1e-6):
        raise RepairError("透视变换经过无穷远点，请重新选择校准点。")
    polygon = cv2.perspectiveTransform(corners[None], matrix)[0]
    area = abs(cv2.contourArea(polygon))
    if not cv2.isContourConvex(polygon) or area < 4 or area > tw * th * 100:
        raise RepairError("对齐结果变形过大，请更换对齐方式或手动校准。")
    target_polygon = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    overlap, _ = cv2.intersectConvexConvex(polygon, target_polygon)
    if overlap < 1:
        raise RepairError("对齐后的补充图与模板没有重叠。")
    return matrix


def manual_align(source_points, target_points, source_shape, target_shape, mode="similarity") -> Alignment:
    if mode not in MODES:
        raise RepairError("未知的对齐方式。")
    src = np.asarray(source_points, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(target_points, dtype=np.float64).reshape(-1, 2)
    count = MODES[mode]
    if len(src) != len(dst) or len(src) < count or not np.isfinite(src).all() or not np.isfinite(dst).all():
        raise RepairError(f"这种对齐方式至少需要 {count} 对有效对应点。")
    if len(np.unique(src, axis=0)) != len(src) or len(np.unique(dst, axis=0)) != len(dst):
        raise RepairError("校准点重复，请在不同位置选择对应点。")
    if mode == "similarity":
        # x' = a*x - b*y + tx, y' = b*x + a*y + ty
        equations = np.zeros((2 * len(src), 4))
        equations[0::2] = np.c_[src[:, 0], -src[:, 1], np.ones(len(src)), np.zeros(len(src))]
        equations[1::2] = np.c_[src[:, 1], src[:, 0], np.zeros(len(src)), np.ones(len(src))]
        solution, _, rank, _ = np.linalg.lstsq(equations, dst.ravel(), rcond=None)
        if rank < 4:
            raise RepairError("对应点太接近，无法校准。")
        a, b, tx, ty = solution
        matrix = np.array([[a, -b, tx], [b, a, ty], [0, 0, 1.]])
    elif mode == "affine":
        equations = np.c_[src, np.ones(len(src))]
        solution, _, rank, _ = np.linalg.lstsq(equations, dst, rcond=None)
        if rank < 3 or np.linalg.matrix_rank(dst - dst.mean(axis=0)) < 2:
            raise RepairError("仿射校准点不能在同一直线上。")
        matrix = np.vstack([solution.T, [0, 0, 1]])
    else:
        # Four points must span an area in both images.
        for points in (src, dst):
            hull = cv2.convexHull(points.astype(np.float32))
            if cv2.contourArea(hull) < 4:
                raise RepairError("透视校准点不能共线，请选在画面四周。")
        matrix, _ = cv2.findHomography(src, dst, 0)
        if matrix is None:
            raise RepairError("无法计算透视变换，请重新选点。")
    matrix = validate_matrix(matrix, source_shape, target_shape)
    projected = cv2.perspectiveTransform(src[None], matrix)[0]
    error = float(np.sqrt(np.mean(np.sum((projected - dst) ** 2, axis=1))))
    return Alignment(matrix, f"手动校准 · {len(src)} 对点 · 均方根误差 {error:.2f} px", len(src), error)


def _working_gray(image):
    h, w = image.shape[:2]
    scale = min(1.0, 1600.0 / max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    small = cv2.resize(image, size, interpolation=cv2.INTER_AREA) if scale < 1 else image
    gray = cv2.cvtColor(small, cv2.COLOR_RGBA2GRAY)
    gray = cv2.createCLAHE(2.0, (8, 8)).apply(gray)
    mask = np.uint8(small[:, :, 3] > 240) * 255
    return gray, mask, np.diag([w / size[0], h / size[1], 1.0])


def auto_align(source: np.ndarray, target: np.ndarray, mode="similarity", crop=None) -> Alignment:
    if mode not in MODES:
        raise RepairError("未知的对齐方式。")
    working_source = source
    offset = np.eye(3)
    if crop is not None:
        x, y, w, h = crop
        working_source = source[y:y+h, x:x+w]
        offset[:2, 2] = (x, y)
    gray_s, mask_s, scale_s = _working_gray(working_source)
    scale_s = offset @ scale_s
    gray_t, mask_t, scale_t = _working_gray(target)
    sift = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.015)
    ks, ds = sift.detectAndCompute(gray_s, mask_s)
    kt, dt = sift.detectAndCompute(gray_t, mask_t)
    if ds is None or dt is None or min(len(ds), len(dt)) < 8:
        raise RepairError("图片可用特征太少。请用手动校准，在两张图上点选对应位置。")
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def good_matches(first, second):
        return {pair[0].queryIdx: pair[0].trainIdx for pair in matcher.knnMatch(first, second, k=2)
                if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance}
    forward, reverse = good_matches(ds, dt), good_matches(dt, ds)
    matches = [(i, j) for i, j in forward.items() if reverse.get(j) == i]
    required = 12 if mode == "perspective" else 8
    if len(matches) < required:
        raise RepairError(f"可靠匹配点不足（{len(matches)}/{required}）。请改用手动校准。")
    src = np.float32([ks[i].pt for i, _ in matches])
    dst = np.float32([kt[j].pt for _, j in matches])
    if mode == "perspective":
        matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0, maxIters=4000, confidence=0.995)
    else:
        estimate = cv2.estimateAffinePartial2D if mode == "similarity" else cv2.estimateAffine2D
        affine, mask = estimate(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0,
                                maxIters=4000, confidence=0.995, refineIters=15)
        matrix = None if affine is None else np.vstack([affine, [0, 0, 1]])
    if matrix is None or mask is None:
        raise RepairError("自动对齐失败，请改用手动校准。")
    inlier_mask = mask.ravel().astype(bool)
    n = int(inlier_mask.sum())
    if n < (8 if mode == "perspective" else 6) or n / len(matches) < 0.4:
        raise RepairError(f"匹配一致性不足（{n}/{len(matches)}），请手动校准。")
    # A cluster of features in a tiny patch does not constrain the whole image.
    for points, gray in ((src[inlier_mask], gray_s), (dst[inlier_mask], gray_t)):
        if cv2.contourArea(cv2.convexHull(points)) < gray.size * 0.005:
            raise RepairError("匹配点过于集中，整图对齐不可靠。请手动选择分散的对应点。")
    original_matrix = validate_matrix(scale_t @ matrix @ np.linalg.inv(scale_s), source.shape, target.shape)
    projected = cv2.perspectiveTransform(src[inlier_mask][None], matrix)[0]
    error = float(np.median(np.linalg.norm(projected - dst[inlier_mask], axis=1)) * scale_t[0, 0])
    return Alignment(original_matrix, f"自动对齐 · {n}/{len(matches)} 个匹配点 · 中位误差 {error:.2f} px", n, error)


def region_mask(shape, rectangles, feather=0, crop=None) -> np.ndarray:
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for x, y, rw, rh in rectangles:
        x0, y0 = max(0, math.floor(x)), max(0, math.floor(y))
        x1, y1 = min(w, math.ceil(x + rw)), min(h, math.ceil(y + rh))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = 1
    if crop is not None:
        x, y, rw, rh = crop
        mask[:y] = 0
        mask[y+rh:] = 0
        mask[:, :x] = 0
        mask[:, x+rw:] = 0
    if feather <= 0 or not mask.any():
        return mask.astype(np.float32)
    # Padding makes an image-edge selection feather inward as well.
    distance = cv2.distanceTransform(np.pad(mask, 1), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)[1:-1, 1:-1]
    return np.minimum(distance / float(feather), 1.0).astype(np.float32) * mask


def composite(base: np.ndarray, sources, feather=3) -> np.ndarray:
    """Merge each donor's union of enabled rectangles; later donors take priority."""
    from .selections import source_masks
    from .colors import adjust_rgb
    result = base.copy()
    h,w = base.shape[:2]
    for source in sources:
        if not source.enabled or source.matrix is None or not any(r.enabled for r in source.regions):
            continue
        target_mask,support = source_masks(source,base.shape)
        if not cv2.countNonZero(support) or not target_mask.any():
            continue
        sx,sy,rw,rh = cv2.boundingRect(support)
        ex,ey = sx+rw,sy+rh
        points = np.float32([[[sx,sy],[ex,sy],[ex,ey],[sx,ey]]])
        projected = cv2.perspectiveTransform(points,source.matrix)[0]
        dx0 = max(0,math.floor(float(projected[:,0].min()))-2)
        dy0 = max(0,math.floor(float(projected[:,1].min()))-2)
        dx1 = min(w,math.ceil(float(projected[:,0].max()))+2)
        dy1 = min(h,math.ceil(float(projected[:,1].max()))+2)
        if dx1<=dx0 or dy1<=dy0:
            continue
        rgba = source.image[sy:ey,sx:ex].astype(np.float32)/255.
        alpha = rgba[:,:,3]*(support[sy:ey,sx:ex].astype(np.float32)/255.)
        color = adjust_rgb(rgba[:,:,:3],getattr(source,"color",None))
        premult = color*alpha[:,:,None]
        to_original = np.array([[1,0,sx],[0,1,sy],[0,0,1.]])
        to_local = np.array([[1,0,-dx0],[0,1,-dy0],[0,0,1.]])
        transform = to_local @ source.matrix @ to_original
        size = (dx1-dx0,dy1-dy0)
        warped_alpha = cv2.warpPerspective(alpha,transform,size,flags=cv2.INTER_LINEAR)
        warped_rgb = cv2.warpPerspective(premult,transform,size,flags=cv2.INTER_LINEAR)
        mask = np.uint8(target_mask[dy0:dy1,dx0:dx1]>0)
        if feather>0:
            distance = cv2.distanceTransform(np.pad(mask,1),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)[1:-1,1:-1]
            selection_alpha = np.minimum(distance/float(feather),1.)*mask
        else:
            selection_alpha = mask.astype(np.float32)
        warped_alpha *= selection_alpha
        warped_rgb *= selection_alpha[:,:,None]
        destination = result[dy0:dy1,dx0:dx1]
        old = destination.astype(np.float32)/255.
        a = warped_alpha[:,:,None]
        old_a = old[:,:,3:4]
        out_a = a+old_a*(1-a)
        out_rgb = warped_rgb+old[:,:,:3]*old_a*(1-a)
        rgb = np.divide(out_rgb,out_a,out=np.zeros_like(out_rgb),where=out_a>1e-8)
        active = warped_alpha>0
        destination[active,:3] = np.uint8(np.clip(np.rint(rgb[active]*255),0,255))
        destination[active,3] = np.uint8(np.clip(np.rint(out_a[active,0]*255),0,255))
    return result


def overlay(base: np.ndarray, source, opacity=0.5) -> np.ndarray:
    """Alignment preview, independent of repair rectangles."""
    if source is None or source.matrix is None:
        return base.copy()
    from types import SimpleNamespace
    h, w = source.image.shape[:2]
    from .colors import adjust_image
    rgba = adjust_image(source.image,getattr(source,"color",None)).copy()
    rgba[:, :, 3] = np.uint8(rgba[:, :, 3] * opacity)
    donor = SimpleNamespace(image=rgba, matrix=source.matrix, enabled=True, crop=getattr(source, "crop", None),
                            regions=[SimpleNamespace(rect=(0, 0, w, h), enabled=True)])
    return composite(base, [donor], 0)
