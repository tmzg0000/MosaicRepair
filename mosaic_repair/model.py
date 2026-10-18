"""Editable state and portable ZIP-based project persistence (no pickle)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace, asdict
from io import BytesIO
from pathlib import Path, PureWindowsPath
import base64
import json
import zipfile

import numpy as np

from .core import MODES, RepairError, atomic_write, png_bytes, read_image, validate_matrix
from .colors import ColorSettings, ColorTransform
from .selections import read_region_mask

MAX_TOTAL_PIXELS = 100_000_000
MAX_MANIFEST_BYTES = 16_000_000


@dataclass
class Region:
    rect: tuple[int, int, int, int]
    enabled: bool = True
    space: str = "source"
    kind: str = "rectangle"
    points: tuple = ()
    brush_size: int = 20
    mask_png: str = ""

    @classmethod
    def from_mask(cls, mask):
        import cv2
        mask = np.uint8(mask > 0) * 255
        x,y,w,h = cv2.boundingRect(mask)
        if w == 0 or h == 0:
            raise RepairError("魔法棒未选中任何像素。")
        encoded = base64.b64encode(png_bytes(mask[y:y+h,x:x+w])).decode("ascii")
        return cls((x,y,w,h), space="template", kind="wand", mask_png=encoded)

    @classmethod
    def from_shape(cls,kind,points,shape,brush_size=20):
        pts = np.asarray(points,dtype=float).reshape(-1,2)
        if len(pts)<(3 if kind=="polygon" else 1) or len(pts)>4096:
            raise RepairError("选区点数无效。多边形至少需要 3 个点。")
        h,w = shape[:2]
        margin = brush_size//2+1 if kind in ("brush","eraser") else 0
        low = np.floor(pts.min(axis=0)).astype(int)-margin
        high = np.ceil(pts.max(axis=0)).astype(int)+margin+1
        x,y = max(0,int(low[0])),max(0,int(low[1]))
        ex,ey = min(w,int(high[0])),min(h,int(high[1]))
        if ex<=x or ey<=y:
            raise RepairError("选区超出模板范围。")
        return cls((x,y,ex-x,ey-y),True,"template",kind,
                   tuple(tuple(float(v) for v in p) for p in pts),brush_size)


@dataclass
class Source:
    name: str
    image: np.ndarray
    matrix: np.ndarray | None = None
    mode: str = "similarity"
    message: str = "尚未对齐"
    regions: list[Region] = field(default_factory=list)
    source_points: list[tuple[float, float]] = field(default_factory=list)
    target_points: list[tuple[float, float]] = field(default_factory=list)
    enabled: bool = True
    crop: tuple[int, int, int, int] | None = None
    color: ColorSettings = field(default_factory=ColorSettings)

    @property
    def bounds(self):
        h, w = self.image.shape[:2]
        return self.crop or (0, 0, w, h)

    def visible_pairs(self):
        x, y, w, h = self.bounds
        pairs = [(s, t) for s, t in zip(self.source_points, self.target_points)
                 if x <= s[0] < x+w and y <= s[1] < y+h]
        return ([s for s, _ in pairs], [t for _, t in pairs])


@dataclass
class Document:
    base_name: str
    base: np.ndarray
    sources: list[Source] = field(default_factory=list)
    feather: int = 3
    original_paths: tuple[str, ...] = ()
    base_path: str | None = None

    def clone(self):
        # Images are immutable and intentionally shared between undo snapshots.
        return replace(self, sources=[replace(s, matrix=None if s.matrix is None else s.matrix.copy(),
                       regions=[replace(r) for r in s.regions], source_points=list(s.source_points),
                       target_points=list(s.target_points),color=replace(s.color)) for s in self.sources])

    def check_capacity(self, additional_pixels=0):
        pixels = self.base.shape[0] * self.base.shape[1] + additional_pixels
        pixels += sum(s.image.shape[0] * s.image.shape[1] for s in self.sources)
        if pixels > MAX_TOTAL_PIXELS or len(self.sources) >= 99:
            raise RepairError("当前工程图片总量过大（总计最多 1 亿像素、99 张补充图）。")


def save_project(path, document: Document):
    version = 3 if any(r.kind=="wand" for s in document.sources for r in s.regions) else 2
    manifest = {"version": version, "base_name": document.base_name, "feather": document.feather,
                "base_path": document.base_path, "original_paths": document.original_paths, "sources": []}
    output = BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("images/base.png", png_bytes(document.base))
        for i, source in enumerate(document.sources):
            image_path = f"images/source-{i}.png"
            archive.writestr(image_path, png_bytes(source.image))
            manifest["sources"].append({"name": source.name, "image": image_path,
                "matrix": None if source.matrix is None else source.matrix.tolist(),
                "mode": source.mode, "message": source.message, "enabled": source.enabled, "crop": source.crop,
                "regions": [asdict(r) for r in source.regions], "color": asdict(source.color),
                "source_points": source.source_points, "target_points": source.target_points})
        parameters = json.dumps(manifest, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(parameters)>MAX_MANIFEST_BYTES:
            raise RepairError("工程选区数据过多，请删除部分选区后再保存。")
        archive.writestr("manifest.json", parameters)
    atomic_write(path, output.getvalue())


def load_project(path) -> Document:
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if len(entries) > 102 or sum(i.file_size for i in entries) > 600_000_000:
                raise RepairError("工程文件过大，无法安全打开。")
            if len({i.filename for i in entries}) != len(entries):
                raise RepairError("工程包含重复文件。")
            if archive.getinfo("manifest.json").file_size > MAX_MANIFEST_BYTES:
                raise RepairError("工程参数过大。")
            data = json.loads(archive.read("manifest.json"))
            if data["version"] not in (1,2,3):
                raise RepairError("不支持这个工程版本。")
            sources = data["sources"]
            if not isinstance(sources, list) or len(sources) > 99:
                raise RepairError("工程补充图数量无效。")
            base = read_image(archive.read("images/base.png"))
            feather = data["feather"]
            if type(feather) is not int or not 0 <= feather <= 50:
                raise RepairError("工程羽化参数无效。")
            document = Document(str(data["base_name"])[:512], base, feather=feather)
            original_paths = data.get("original_paths", [])
            if (not isinstance(original_paths, list) or len(original_paths) > 100
                    or any(not isinstance(p, str) or len(p) > 32768 for p in original_paths)):
                raise RepairError("工程原图路径记录无效。")
            document.original_paths = tuple(original_paths)
            base_path = data.get("base_path")
            if base_path is not None and (not isinstance(base_path,str) or len(base_path)>32768 or "\x00" in base_path):
                raise RepairError("工程模板路径记录无效。")
            if "base_path" not in data:
                # Older projects only recorded a list of input files. Identify
                # the template by its filename; never pick an arbitrary donor.
                base_path = next((p for p in original_paths
                    if document.base_name.casefold() in (Path(p).name.casefold(),PureWindowsPath(p).name.casefold())),None)
            document.base_path = base_path or None
            for i, item in enumerate(sources):
                if item["image"] != f"images/source-{i}.png" or item["mode"] not in MODES:
                    raise RepairError("工程图片路径或对齐方式无效。")
                image = read_image(archive.read(item["image"]))
                document.check_capacity(image.shape[0] * image.shape[1])
                matrix = item["matrix"]
                if matrix is not None:
                    matrix = validate_matrix(matrix, image.shape, base.shape)
                source = Source(str(item["name"])[:512], image, matrix, item["mode"],
                                str(item["message"])[:512], enabled=bool(item["enabled"]))
                source.color = read_color_settings(item.get("color"))
                regions = item["regions"]
                if not isinstance(regions, list) or len(regions) > 1000:
                    raise RepairError("工程选框数量无效。")
                h, w = image.shape[:2]
                crop = item.get("crop")
                if crop is not None:
                    if (not isinstance(crop, list) or len(crop) != 4
                            or any(type(v) is not int for v in crop)):
                        raise RepairError("工程裁切范围无效。")
                    cx, cy, cw, ch = crop
                    if cx < 0 or cy < 0 or cw < 2 or ch < 2 or cx+cw > w or cy+ch > h:
                        raise RepairError("工程裁切范围超出图片。")
                    source.crop = tuple(crop)
                for region in regions:
                    rect = region["rect"]
                    if (not isinstance(rect, list) or len(rect) != 4
                            or any(type(v) is not int for v in rect)):
                        raise RepairError("工程选框坐标无效。")
                    space = region.get("space","source")
                    kind = region.get("kind","rectangle")
                    if space not in ("source","template") or kind not in ("rectangle","polygon","brush","eraser","wand"):
                        raise RepairError("工程选区类型无效。")
                    if kind=="wand" and (data["version"]<3 or space!="template"):
                        raise RepairError("工程魔法棒选区版本或坐标空间无效。")
                    gh,gw = base.shape[:2] if space=="template" else image.shape[:2]
                    x, y, rw, rh = rect
                    if x < 0 or y < 0 or rw < 1 or rh < 1 or x + rw > gw or y + rh > gh:
                        raise RepairError("工程选框超出图片范围。")
                    width = region.get("brush_size",20)
                    if type(width) is not int or not 1<=width<=500:
                        raise RepairError("工程画笔大小无效。")
                    points = region.get("points",[])
                    values = ()
                    if kind not in ("rectangle","wand"):
                        pts = np.asarray(points,dtype=float)
                        if (pts.ndim!=2 or pts.shape[1]!=2 or not (3 if kind=="polygon" else 1)<=len(pts)<=4096
                                or not np.isfinite(pts).all() or (pts<0).any()
                                or (pts[:,0]>gw).any() or (pts[:,1]>gh).any()):
                            raise RepairError("工程自由选区坐标无效。")
                        values = tuple(tuple(p) for p in pts.tolist())
                        calculated = Region.from_shape(kind,values,(gh,gw),width)
                        if tuple(rect)!=calculated.rect:
                            raise RepairError("工程自由选区边界与轮廓不一致。")
                    selection = Region(tuple(rect),bool(region["enabled"]),space,kind,values,width)
                    if kind=="wand":
                        selection.mask_png = region.get("mask_png")
                        if not isinstance(selection.mask_png,str):
                            raise RepairError("工程魔法棒选区数据无效。")
                        read_region_mask(selection)
                    source.regions.append(selection)
                for key, shape in (("source_points", image.shape), ("target_points", base.shape)):
                    points = np.asarray(item.get(key, []), dtype=float)
                    if points.size == 0:
                        values = []
                    elif (points.ndim != 2 or points.shape[1] != 2 or len(points) > 100
                          or not np.isfinite(points).all() or (points < 0).any()
                          or (points[:, 0] > shape[1]).any() or (points[:, 1] > shape[0]).any()):
                        raise RepairError("工程校准点无效。")
                    else:
                        values = [tuple(p) for p in points.tolist()]
                    setattr(source, key, values)
                if len(source.source_points) != len(source.target_points):
                    raise RepairError("工程校准点不成对。")
                document.sources.append(source)
            return document
    except RepairError:
        raise
    except (OSError, zipfile.BadZipFile, KeyError, TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise RepairError(f"工程文件损坏或格式不正确：{exc}") from exc


def read_color_settings(data):
    # Version 1 projects open without changing their old appearance.
    if data is None:
        return ColorSettings(auto=False)
    if not isinstance(data,dict) or type(data.get("auto")) is not bool:
        raise RepairError("工程调色参数无效。")
    settings = ColorSettings(auto=data["auto"])
    for name in ("strength","brightness","contrast","saturation","temperature"):
        value = data.get(name,100 if name=="strength" else 0)
        if type(value) is not int or not (0 if name=="strength" else -100)<=value<=100:
            raise RepairError("工程调色数值无效。")
        setattr(settings,name,value)
    transform = data.get("transform")
    if transform is not None:
        gains = np.asarray(transform["gains"],dtype=float)
        offsets = np.asarray(transform["offsets"],dtype=float)
        samples = transform["samples"]
        if (gains.shape!=(3,) or offsets.shape!=(3,) or not np.isfinite(gains).all()
                or not np.isfinite(offsets).all() or (gains<.1).any() or (gains>10).any()
                or (np.abs(offsets)>300).any() or type(samples) is not int or not 0<=samples<=1000000):
            raise RepairError("工程自动色调变换无效。")
        settings.transform = ColorTransform(tuple(gains.tolist()),tuple(offsets.tolist()),samples)
    return settings
