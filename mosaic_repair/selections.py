"""Vector selections rasterized in original template/source pixel coordinates."""
import cv2
import numpy as np


def paint_region(mask,region,offset=(0,0)):
    kind = getattr(region,"kind","rectangle")
    color = 0 if kind=="eraser" else 255
    ox,oy = offset
    if kind=="rectangle":
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
    contours,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    return [c[:,0,:].astype(np.float32)+[x,y] for c in contours if len(c)>=3]


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
