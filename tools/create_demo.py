"""Create an original geometric demo, with complementary masks and unequal resolutions."""
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mosaic_repair.core import auto_align, export_image, read_image
from mosaic_repair.model import Document, Source, Region, save_project
from mosaic_repair.colors import estimate_color_transform
from mosaic_repair.selections import source_masks


def create_demo(root):
    output = root / "samples"
    output.mkdir(exist_ok=True)
    rng = np.random.default_rng(940)
    h, w = 540, 800
    rgb = np.empty((h,w,3), np.uint8)
    rgb[:] = (211,230,224)
    cv2.circle(rgb, (634,118), 47, (241,185,85), -1, cv2.LINE_AA)
    cv2.fillPoly(rgb, [np.int32([[0,345],[211,95],[443,345]])], (91,138,135))
    cv2.fillPoly(rgb, [np.int32([[238,345],[475,152],[800,345]])], (65,109,113))
    cv2.fillPoly(rgb, [np.int32([[134,186],[211,95],[295,190],[237,170],[205,190],[180,164]])], (246,244,226))
    cv2.rectangle(rgb, (0,344), (800,540), (51,90,86), -1)
    cv2.fillPoly(rgb, [np.int32([[0,390],[180,352],[400,375],[620,345],[800,400],[800,540],[0,540]])], (66,118,112))
    cv2.fillPoly(rgb, [np.int32([[335,540],[454,352],[483,352],[435,540]])], (187,199,158))
    for x,y,size in ((76,303,35),(124,317,42),(668,324,45),(710,343,53),(613,350,28)):
        cv2.rectangle(rgb, (x-3,y), (x+3,y+size), (56,69,63), -1)
        cv2.fillPoly(rgb, [np.int32([[x-size//2,y+size//3],[x,y-size],[x+size//2,y+size//3]])], (35,79,76))
    cv2.rectangle(rgb, (173,253), (315,339), (238,212,157), -1)
    cv2.fillPoly(rgb, [np.int32([[155,259],[241,199],[333,259]])], (160,90,63))
    for x in (192,266):
        cv2.rectangle(rgb, (x,275), (x+24,300), (57,101,107), -1)
        cv2.line(rgb, (x+12,275), (x+12,300), (238,212,157), 2)
    cv2.rectangle(rgb, (232,294), (252,339), (94,87,66), -1)
    for _ in range(85):
        x,y = rng.integers([8,370],[790,530])
        cv2.circle(rgb, (int(x),int(y)), int(rng.integers(1,4)), (174,183,119), -1)
    cv2.putText(rgb, "MOUNTAIN", (34,63), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (34,66,72), 3, cv2.LINE_AA)
    cv2.putText(rgb, "FIELD NOTES  /  2026", (37,89), cv2.FONT_HERSHEY_SIMPLEX, .45, (67,103,106), 1, cv2.LINE_AA)
    cv2.putText(rgb, "07", (704,506), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (223,232,201), 2, cv2.LINE_AA)
    noise = rng.normal(0, 1.8, rgb.shape)
    rgb = np.uint8(np.clip(rgb.astype(float)+noise,0,255))
    clear = np.dstack([rgb, np.full((h,w),255,np.uint8)])
    region_a = (168,241,154,71)
    region_b = (599,340,130,85)
    template = clear.copy()
    def mosaic(image, rect):
        x,y,rw,rh = rect
        patch = image[y:y+rh,x:x+rw]
        reduced = cv2.resize(patch,(max(2,rw//15),max(2,rh//15)), interpolation=cv2.INTER_AREA)
        image[y:y+rh,x:x+rw] = cv2.resize(reduced,(rw,rh), interpolation=cv2.INTER_NEAREST)
    mosaic(template, region_a)
    export_image(output / "template.jpg", template)
    first = clear.copy()
    mosaic(first, region_b)
    first = cv2.resize(first, (640,432), interpolation=cv2.INTER_AREA)
    framed = np.full((522,720,4), (239,235,225,255), np.uint8)
    framed[45:477,40:680] = first
    cv2.putText(framed, "UNNEEDED BORDER", (45,27), cv2.FONT_HERSHEY_SIMPLEX, .5, (90,100,100,255), 1)
    lab = cv2.cvtColor(framed[:,:,:3].astype(np.float32)/255,cv2.COLOR_RGB2LAB)
    lab[:,:,0] = lab[:,:,0]*.9+3
    lab[:,:,1:] = lab[:,:,1:]*.75+[2,-3]
    framed[:,:,:3] = np.uint8(np.clip(np.rint(cv2.cvtColor(lab,cv2.COLOR_LAB2RGB)*255),0,255))
    export_image(output / "source_1.jpg", framed)
    second = clear.copy()
    mosaic(second, region_a)
    second = cv2.resize(second,(1000,675),interpolation=cv2.INTER_CUBIC)
    export_image(output / "source_2.jpg", second)
    base, first, second = [read_image(output/name) for name in ("template.jpg", "source_1.jpg", "source_2.jpg")]
    source_a = Source("source_1.jpg", first, crop=(40,45,640,432),
                      regions=[Region.from_shape("polygon",[(155,240),(165,228),(338,228),(338,326),(155,326)],base.shape)])
    source_b = Source("source_2.jpg",second)
    for source in (source_a,source_b):
        alignment = auto_align(source.image,base,crop=source.crop)
        source.matrix, source.message = alignment.matrix, alignment.message
        exclusion,_ = source_masks(source,base.shape)
        source.color.transform = estimate_color_transform(source.image,base,source.matrix,source.crop,exclusion)
    document = Document("template.jpg",base,[source_a,source_b],feather=3)
    save_project(output / "demo.mrepair", document)
    print("Demo project:",output / "demo.mrepair")
    return document


if __name__ == "__main__":
    create_demo(Path(__file__).resolve().parents[1])
