# 图标设计

图标用两张叠放的照片与四块互补像素，表达从补充图取回清晰内容的功能。绿色呼应程序界面的主色，暖色太阳提供识别点。图标无文字，保留透明背景。

- `app-icon.png`：ImageGen 透明原稿，1254 × 1254。
- `app-icon.ico`：Windows 多尺寸图标，包含 16、20、24、32、40、48、64、128、256 像素。
- 程序窗口、应用图标、页头与 PyInstaller 构建共用本设计。

生成方式：内置 ImageGen（非 CLI/API）。下列提示词保存了初次生成和定稿调整。

## 初始提示词

```text
Use case: logo-brand.
Asset type: production-ready Windows desktop application icon for "图片互补修补" (MosaicRepair), a utility that takes clear pixels from a second version of a photograph to repair a selected damaged or mosaicked area.
Primary request: Design ONE polished, distinctive app icon, simple enough to be recognizable at 32 pixels. Two overlapping, offset rounded-square photograph cards merge through a small set of large square pixel tiles, visually conveying complementary images and pixel repair.
Subject: A bold front rounded-square photo card with an elegantly simplified mountain landscape (one angular forest-green mountain ridge, light mint sky, one small warm apricot sun), and a second complementary emerald photo card offset behind toward the upper right. Near the front photo's upper-right edge, a compact 2 by 2 arrangement of large square mosaic tiles joins the two cards: the tiles form a clean stepped transition between them. Cards should read as two complementary image layers with interlocking pixels, not a collage of many windows.
Style/medium: Premium desktop icon design, crisp vector-like geometry with subtle depth, gentle carefully controlled shading, almost flat. Confident silhouette, large shapes, optically balanced. Restrained and professional rather than toy-like.
Color palette: emerald green #167B5A, deep forest #113F38, pale mint #DCF4E8, ivory, one tiny apricot sun accent. The actual application's interface is white and emerald green.
Composition/framing: square 1:1 canvas, centered standalone icon, mark occupies approximately 84 percent of the canvas with clear padding on all sides. Transparent outside the silhouette, rounded photo cards, no enclosing background tile. Mostly upright cards with at most very subtle rotation. Excellent contrast on light and dark desktops.
Text: none.
Constraints: exactly one finished icon; actual transparent background; no typography, no letters, no watermark, no mockup, no sample sheet, no caption, no desktop background. No magic wand, sparkle symbol, shield, magnifying glass, brush, puzzle-piece knobs, arrows, camera lens, gradients flooding the canvas, or excessive tiny tiles. This is a real pixel-complement repair tool. Make the image layer and complementary mosaic concept immediately legible.
```

## 定稿调整提示词

```text
Use case: precise-object-edit / logo-brand.
Edit the app icon just generated. Preserve its concept, composition, two overlapping photo cards, emerald and ivory palette, apricot sun, and the four complementary square tiles at the cards' meeting point.
Make ONE focused refinement: simplify the landscape and finish into a cleaner small-size application icon. Remove ALL trees and fine faceted rock detail. Each photo's landscape should have only two or three large smooth geometric mountain silhouettes with a single simple angular peak. Use flat clean colors with very subtle material shading only on the outside frame and pixel tiles, no painted texture, no glows. Refine all exterior silhouettes to be perfectly crisp and smoothly anti-aliased, without ANY stray pixels, white halos or speckling outside the icon. Keep attractive broad ivory front photo border and green rear photo frame. Keep foreground broad shapes and strong contrast so it reads at 32x32. The four tiles should remain a 2x2 cluster of large near-square tiles, clean geometry, narrow even gaps, light mint top left, emerald top right, mint bottom left, dark forest bottom right. Keep exactly two cards, one sun on the front card. Do not add text, extra symbols or decorative elements. Deliver one centered finished square icon with actual transparent background and 8 percent transparent padding.
```
