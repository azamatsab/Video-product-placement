"""
v2 step 3: composite the product RGBA onto each frame with the real label
preserved pixel-perfect.

Placement:
  - Find the top edge of the detected table mask on frame 0 (topmost mask
    pixel per column). Pick a point roughly in the centre-back of the visible
    table top surface. This is the anchor where the bottle's *base* sits.
  - Scale the product so its width is ~30 % of the table bbox width.
  - For subsequent frames, analytically project the anchor point through the
    known camera pan+zoom (same formulas as v1 source synthesis). In a real
    video we'd use SAM2 video-mode tracking on the table instead.

Compositing:
  - Alpha blend the warped product PNG onto each frame
  - Optional light color-match to the local ambient
  - Soft drop shadow for grounding

Output: /root/vpp/output/v2_composite/frames/*.png + v2_composite.mp4
"""
import os
import json
import numpy as np
from PIL import Image, ImageFilter
import torch
import torchvision.io as tvio

SOURCE_FRAMES = "/root/vpp/source/frames"
TABLE_MASK = "/root/vpp/assets/table_mask_frame0.png"
PRODUCT_RGBA = "/root/vpp/assets/product_rgba.png"

OUT = "/root/vpp/output/v2_composite"
OUT_FRAMES = os.path.join(OUT, "frames")

N_FRAMES = 72
FRAME_SIZE = (896, 672)   # (w, h)
CANVAS = (1024, 768)      # source canvas for pan+zoom


def pan_zoom_params(i):
    """Same pan+zoom as in v1 source synthesis."""
    t = i / max(1, N_FRAMES - 1)
    scale = 1.0 + 0.10 * t
    fw, fh = FRAME_SIZE
    cw = int(fw / scale)
    ch = int(fh / scale)
    pan_x = int(30 * t)
    W, H = CANVAS
    cx = (W - cw) // 2 + pan_x - 15
    cy = (H - ch) // 2
    return cx, cy, cw, ch


def frame_to_canvas(px, py, i):
    """Map a point in frame i coordinates to canvas coordinates."""
    cx, cy, cw, ch = pan_zoom_params(i)
    fw, fh = FRAME_SIZE
    return cx + px * cw / fw, cy + py * ch / fh


def canvas_to_frame(canvas_x, canvas_y, i):
    cx, cy, cw, ch = pan_zoom_params(i)
    fw, fh = FRAME_SIZE
    return (canvas_x - cx) * fw / cw, (canvas_y - cy) * fh / ch


def find_table_top_anchor(table_mask):
    """
    Returns (anchor_x, anchor_y, table_top_width).

    Strategy: use morphological opening with a horizontal kernel to remove
    the thin vertical legs. What remains is the flat top surface. Compute
    its centroid for the anchor, and its bbox width as the "usable" width.
    """
    import cv2
    # Horizontal rectangle kernel: wide enough to span the gap between legs
    # but narrow enough to not collapse the top surface edges.
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 5))
    top_only = cv2.morphologyEx(table_mask, cv2.MORPH_OPEN, k, iterations=2)
    ys, xs = np.where(top_only > 128)
    if len(ys) == 0:
        # fallback to bbox of full mask
        ys, xs = np.where(table_mask > 128)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    # Anchor at centroid of top_only (tends to be middle of top surface)
    anchor_x = int(xs.mean())
    anchor_y = int(ys.mean()) + int((y2 - y1) * 0.04)
    table_top_width = x2 - x1
    print(f"  table top bbox: ({x1},{y1})-({x2},{y2}), width: {table_top_width}")
    # save the cleaned mask for inspection
    Image.fromarray(top_only).save("/root/vpp/assets/table_top_only.png")
    return anchor_x, anchor_y, table_top_width


def main():
    os.makedirs(OUT_FRAMES, exist_ok=True)

    table_mask = np.array(Image.open(TABLE_MASK).convert("L"))
    anchor_x, anchor_y, table_w = find_table_top_anchor(table_mask)
    print(f"Anchor (frame0): ({anchor_x}, {anchor_y}), table_top_width: {table_w}")

    # Convert anchor to canvas coords (fixed for the whole clip)
    anchor_canvas = frame_to_canvas(anchor_x, anchor_y, 0)
    print(f"Anchor (canvas): {anchor_canvas}")

    # Load product
    product = Image.open(PRODUCT_RGBA).convert("RGBA")
    prod_w0, prod_h0 = product.size

    # Target product width = 42 % of table top width in frame 0
    target_prod_w = int(table_w * 0.42)
    aspect = prod_h0 / prod_w0
    target_prod_h = int(target_prod_w * aspect)
    product_base = product.resize((target_prod_w, target_prod_h), Image.LANCZOS)
    print(f"Product base size: {product_base.size}")

    # Per-frame compositing
    placements = []
    for i in range(N_FRAMES):
        frame = Image.open(os.path.join(SOURCE_FRAMES, f"{i:04d}.png")).convert("RGB")

        # Project anchor from canvas → this frame's coords
        fx, fy = canvas_to_frame(anchor_canvas[0], anchor_canvas[1], i)
        # Scale the product by the current zoom factor vs. frame 0 (slight grow)
        _, _, _, ch_i = pan_zoom_params(i)
        _, _, _, ch_0 = pan_zoom_params(0)
        zoom_ratio = ch_0 / ch_i  # ch shrinks as zoom in → ratio > 1
        cur_w = int(target_prod_w * zoom_ratio)
        cur_h = int(target_prod_h * zoom_ratio)
        product_i = product.resize((cur_w, cur_h), Image.LANCZOS)

        # paste: base centre at (fx, fy)
        paste_x = int(fx - cur_w / 2)
        paste_y = int(fy - cur_h)  # bottom of product at anchor

        # Drop shadow: darkened ellipse beneath the product
        shadow = Image.new("RGBA", product_i.size, (0, 0, 0, 0))
        from PIL import ImageDraw
        sh_draw = ImageDraw.Draw(shadow)
        sh_y1 = int(cur_h * 0.92)
        sh_y2 = cur_h - 1
        sh_draw.ellipse((int(cur_w * 0.08), sh_y1, int(cur_w * 0.92), sh_y2),
                        fill=(0, 0, 0, 120))
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=4))

        composite = frame.convert("RGBA")
        composite.alpha_composite(shadow, dest=(paste_x, paste_y))
        composite.alpha_composite(product_i, dest=(paste_x, paste_y))
        composite.convert("RGB").save(os.path.join(OUT_FRAMES, f"{i:04d}.png"))

        placements.append({"frame": i, "paste_xy": [paste_x, paste_y], "size": [cur_w, cur_h]})
        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{N_FRAMES} @ paste=({paste_x},{paste_y}) size=({cur_w}x{cur_h})")

    with open(os.path.join(OUT, "placements.json"), "w") as f:
        json.dump(placements, f, indent=2)

    # Encode mp4
    frames_np = np.stack([
        np.array(Image.open(os.path.join(OUT_FRAMES, f"{i:04d}.png")).convert("RGB"))
        for i in range(N_FRAMES)
    ])
    tvio.write_video(
        os.path.join(OUT, "v2_composite.mp4"),
        torch.from_numpy(frames_np),
        fps=24,
        video_codec="libx264",
        options={"crf": "18", "preset": "fast"},
    )
    print(f"Wrote {OUT}/v2_composite.mp4")


if __name__ == "__main__":
    main()
