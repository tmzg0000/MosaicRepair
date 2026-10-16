"""Robust global tone matching for registered versions of the same photograph."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class ColorTransform:
    gains: tuple = (1., 1., 1.)
    offsets: tuple = (0., 0., 0.)
    samples: int = 0


@dataclass
class ColorSettings:
    auto: bool = True
    strength: int = 100
    brightness: int = 0
    contrast: int = 0
    saturation: int = 0
    temperature: int = 0
    transform: ColorTransform | None = None

    def key(self):
        return (self.auto, self.strength, self.brightness, self.contrast,
                self.saturation, self.temperature, self.transform)


def estimate_color_transform(source, target, matrix, crop=None, exclusion=None):
    h,w = target.shape[:2]
    scale = min(1.,900/max(h,w))
    tw,th = max(1,round(w*scale)),max(1,round(h*scale))
    small_target = cv2.resize(target,(tw,th),interpolation=cv2.INTER_AREA)
    transform = np.diag([tw/w,th/h,1.]) @ matrix
    sh,sw = source.shape[:2]
    sx,sy,cw,ch = crop or (0,0,sw,sh)
    offset = np.array([[1,0,sx],[0,1,sy],[0,0,1.]])
    warped = cv2.warpPerspective(source[sy:sy+ch,sx:sx+cw],transform @ offset,(tw,th))
    valid = (warped[:,:,3]>250) & (small_target[:,:,3]>250)
    if exclusion is not None:
        valid &= cv2.resize(exclusion,(tw,th),interpolation=cv2.INTER_NEAREST)==0
    valid &= (warped[:,:,:3].min(axis=2)>3) & (warped[:,:,:3].max(axis=2)<252)
    valid &= (small_target[:,:,:3].min(axis=2)>3) & (small_target[:,:,:3].max(axis=2)<252)
    a = cv2.cvtColor(warped[:,:,:3].astype(np.float32)/255,cv2.COLOR_RGB2LAB)[valid]
    b = cv2.cvtColor(small_target[:,:,:3].astype(np.float32)/255,cv2.COLOR_RGB2LAB)[valid]
    if len(a)<64:
        raise ValueError("自动调色可用参考像素不足，请扩大裁切范围，或使用手动颜色调整。")
    if len(a)>100000:
        index = np.linspace(0,len(a)-1,100000).astype(int)
        a,b = a[index],b[index]
    keep = np.ones(len(a),bool)
    for _ in range(3):
        aa,bb = a[keep],b[keep]
        ma,mb = np.median(aa,axis=0),np.median(bb,axis=0)
        sa = np.percentile(aa,85,axis=0)-np.percentile(aa,15,axis=0)
        sb = np.percentile(bb,85,axis=0)-np.percentile(bb,15,axis=0)
        gains = np.clip(np.divide(sb,sa,out=np.ones(3),where=sa>1.5),.5,2.)
        ca = np.median(np.linalg.norm(aa[:,1:]-ma[1:],axis=1))
        cb = np.median(np.linalg.norm(bb[:,1:]-mb[1:],axis=1))
        gains[1:] = np.clip(cb/ca,.4,2.5) if ca>1 else 1.
        offsets = mb-gains*ma
        residual = np.linalg.norm((a*gains+offsets-b)*[1.,.7,.7],axis=1)
        keep = residual<=max(float(np.percentile(residual,80)),1.)
    return ColorTransform(tuple(float(v) for v in gains),tuple(float(v) for v in offsets),int(keep.sum()))


def adjust_rgb(rgb,settings):
    if settings is None:
        return rgb
    automatic = settings.auto and settings.transform is not None and settings.strength>0
    if not automatic and not any((settings.brightness,settings.contrast,settings.saturation,settings.temperature)):
        return rgb
    lab = cv2.cvtColor(np.ascontiguousarray(rgb,dtype=np.float32),cv2.COLOR_RGB2LAB)
    if automatic:
        amount = settings.strength/100.
        tr = settings.transform
        lab = lab*(1+(np.array(tr.gains,dtype=np.float32)-1)*amount)+np.array(tr.offsets,dtype=np.float32)*amount
    lab[:,:,0] = (lab[:,:,0]-50)*(2.**(settings.contrast/100.))+50+settings.brightness*.5
    lab[:,:,1:] *= 2.**(settings.saturation/100.)
    lab[:,:,2] += settings.temperature*.2
    lab[:,:,0] = np.clip(lab[:,:,0],0,100)
    return np.clip(cv2.cvtColor(lab.astype(np.float32),cv2.COLOR_LAB2RGB),0,1)


def adjust_image(image,settings):
    automatic = settings is not None and settings.auto and settings.transform is not None and settings.strength>0
    if settings is None or (not automatic and not any((settings.brightness,settings.contrast,settings.saturation,settings.temperature))):
        return image
    rgb = image[:,:,:3].astype(np.float32)/255
    adjusted = adjust_rgb(rgb,settings)
    if adjusted is rgb:
        return image
    result = image.copy()
    result[:,:,:3] = np.uint8(np.clip(np.rint(adjusted*255),0,255))
    return result
