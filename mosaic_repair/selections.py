"""Selections rasterized in original template/source pixel coordinates."""
import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

from .core import RepairError


def read_region_mask(region):
    """Decode a saved wand mask only after checking its dimensions and format."""
    try:
        raw = base64.b64decode(region.mask_png, validate=True)
        with Image.open(BytesIO(raw)) as image:
            if image.format != "PNG" or image.mode != "L" or image.size != region.rect[2:]:
                raise ValueError("mask dimensions or format")
            mask = np.array(image)
        if not np.any(mask) or np.any((mask != 0) & (mask != 255)):
            raise ValueError("empty or non-binary mask")
        return mask
    except (ValueError, TypeError, OSError, Image.DecompressionBombError) as exc:
        raise RepairError("工程魔法棒选区数据无效。") from exc


def magic_wand_mask(image, point, tolerance=32, expansion=0):
    """Select a connected color region, then optionally expand in template pixels."""
    h, w = image.shape[:2]
    point = np.asarray(point, dtype=float)
    if (point.shape != (2,) or not np.isfinite(point).all()
            or not 0 <= point[0] < w or not 0 <= point[1] < h):
        raise RepairError("请在模板图片内点击魔法棒选区。")
    if type(tolerance) is not int or not 0 <= tolerance <= 255:
        raise RepairError("魔法棒容差需在 0–255 之间。")
    if type(expansion) is not int or not 0 <= expansion <= 500:
        raise RepairError("魔术棒扩边需在 0–500 像素之间。")
    seed = (int(point[0]), int(point[1]))
    if image[seed[1], seed[0], 3] == 0:
        raise RepairError("点击处完全透明，请选择可见的模板区域。")
    # Mark transparent pixels as barriers; use a different value for the fill.
    flood = np.zeros((h+2, w+2), np.uint8)
    flood[1:-1, 1:-1][image[:, :, 3] == 0] = 1
    flags = 4 | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | (255 << 8)
    delta = (tolerance,) * 3
    cv2.floodFill(np.ascontiguousarray(image[:, :, :3]), flood, seed, 0,
                 loDiff=delta, upDiff=delta, flags=flags)
    selection = np.uint8(flood[1:-1, 1:-1] == 255) * 255
    if expansion:
        # Restrict the distance calculation to the selected bounds plus the margin.
        x,y,rw,rh = cv2.boundingRect(selection)
        x0,y0 = max(0,x-expansion),max(0,y-expansion)
        x1,y1 = min(w,x+rw+expansion),min(h,y+rh+expansion)
        local = selection[y0:y1,x0:x1]
        distance = cv2.distanceTransform(cv2.bitwise_not(local),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
        local[:] = np.uint8(distance<=expansion) * 255
        selection[image[:,:,3]==0] = 0
    return selection


def paint_region(mask,region,offset=(0,0)):
    kind = getattr(region,"kind","rectangle")
    color = 0 if kind=="eraser" else 255
    ox,oy = offset
    if kind=="wand":
        x,y,w,h = region.rect
        selection = read_region_mask(region)
        # Painting adds only selected pixels, preserving earlier selections in holes.
        x0,y0 = max(x,ox),max(y,oy)
        x1,y1 = min(x+w,ox+mask.shape[1]),min(y+h,oy+mask.shape[0])
        if x1>x0 and y1>y0:
            view = mask[y0-oy:y1-oy,x0-ox:x1-ox]
            view[selection[y0-y:y1-y,x0-x:x1-x]>0] = 255
    elif kind=="rectangle":
        x,y,w,h = region.rect
        cv2.rectangle(mask,(x-ox,y-oy),(x+w-1-ox,y+h-1-oy),color,-1)
    else:
        points = np.rint(np.asarray(region.points)-[ox,oy]).astype(np.int32)
        if kind=="polygon":
            cv2.fillPoly(mask,[points],color)
        elif len(points):
            width = region.brush_size
            radius = max(1,width//2)
            for point in points:
                cv2.circle(mask,tuple(point),radius,color,-1)
            if len(points)>1:
                cv2.polylines(mask,[points],False,color,width)


def region_contours(region):
    x,y,w,h = region.rect
    mask = np.zeros((h,w),np.uint8)
    from dataclasses import replace
    draw = replace(region,kind="brush") if region.kind=="eraser" else region
    paint_region(mask,draw,(x,y))
    contours,_ = cv2.findContours(mask,cv2.RETR_LIST if region.kind=="wand" else cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    outlines = []
    for contour in contours:
        if len(contour)>=3:
            outlines.append(contour[:,0,:].astype(np.float32)+[x,y])
        elif region.kind=="wand":
            cx,cy,cw,ch = cv2.boundingRect(contour)
            outlines.append(np.float32([[cx,cy],[cx+cw,cy],[cx+cw,cy+ch],[cx,cy+ch]])+[x,y])
    return outlines


def source_masks(source,target_shape):
    h,w = target_shape[:2]
    sh,sw = source.image.shape[:2]
    legacy = np.zeros((sh,sw),np.uint8)
    for region in source.regions:
        if region.enabled and getattr(region,"space","source")=="source":
            paint_region(legacy,region)
    cx,cy,cw,ch = getattr(source,"crop",None) or (0,0,sw,sh)
    legacy[:cy]=0; legacy[cy+ch:]=0
    legacy[:,:cx]=0; legacy[:,cx+cw:]=0
    target = cv2.warpPerspective(legacy,source.matrix,(w,h),flags=cv2.INTER_LINEAR)
    target[target>0]=255
    template_only = np.zeros((h,w),np.uint8)
    for region in source.regions:
        if region.enabled and getattr(region,"space","source")=="template":
            paint_region(target,region)
            paint_region(template_only,region)
    if template_only.any():
        support = cv2.warpPerspective(template_only,np.linalg.inv(source.matrix),(sw,sh),flags=cv2.INTER_NEAREST)
        support = np.maximum(support,legacy)
    else:
        support = legacy
    support[:cy]=0; support[cy+ch:]=0
    support[:,:cx]=0; support[:,cx+cw:]=0
    return target,support
