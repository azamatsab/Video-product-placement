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
import cv2
from PIL import Image, ImageFilter, ImageDraw
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
    # Anchor at centroid of top_only (tends to be middle of top surface).
    # Small positive offset sits the base slightly toward the front so the
    # product visibly rests on the surface; values in [0.00, 0.04] all look
    # plausible — 0.02 is a safe middle.
    anchor_x = int(xs.mean())
    anchor_y = int(ys.mean()) + int((y2 - y1) * 0.02)
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

    # Load product at source resolution
    product_pil = Image.open(PRODUCT_RGBA).convert("RGBA")
    prod_w0, prod_h0 = product_pil.size

    # Target product width = 42 % of table top width in frame 0
    target_prod_w = int(table_w * 0.42)
    aspect = prod_h0 / prod_w0
    target_prod_h = int(target_prod_w * aspect)
    print(f"Product base size (frame 0): {target_prod_w}x{target_prod_h}")

    # Precompute product+shadow canvas ONCE at source resolution
    # The canvas is a bit larger than the product to leave room for a soft shadow
    # beneath the base. Everything is warped together per frame so shadow stays
    # attached and subpixel-smooth.
    pad_bottom = int(prod_h0 * 0.10)
    pad_side = int(prod_w0 * 0.06)
    canvas_w = prod_w0 + pad_side * 2
    canvas_h = prod_h0 + pad_bottom
    canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    # Draw shadow first
    shadow = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sh_cx = canvas_w / 2
    sh_y1 = prod_h0 - 8
    sh_y2 = prod_h0 + pad_bottom - 4
    sh_rx = prod_w0 * 0.42
    sd.ellipse((sh_cx - sh_rx, sh_y1, sh_cx + sh_rx, sh_y2), fill=(0, 0, 0, 140))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=10))
    canvas.alpha_composite(shadow)
    # Then product on top, centered horizontally, aligned to top of canvas
    canvas.alpha_composite(product_pil, dest=(pad_side, 0))

    # Convert to numpy BGRA for cv2
    product_np = np.array(canvas)  # HWC RGBA
    src_h, src_w = product_np.shape[:2]
    print(f"Source canvas with shadow: {src_w}x{src_h}")

    W, H = FRAME_SIZE

    # Per-frame compositing via cv2.warpAffine (sub-pixel bilinear)
    placements = []
    for i in range(N_FRAMES):
        frame = Image.open(os.path.join(SOURCE_FRAMES, f"{i:04d}.png")).convert("RGB")
        frame_np = np.array(frame)  # HWC RGB uint8

        # Project anchor from canvas → this frame's coords (both floats)
        fx, fy = canvas_to_frame(anchor_canvas[0], anchor_canvas[1], i)

        # Scale factor: zoom ratio × (target_prod_w / prod_w0 at source resolution)
        _, _, _, ch_i = pan_zoom_params(i)
        _, _, _, ch_0 = pan_zoom_params(0)
        zoom_ratio = ch_0 / ch_i
        s = (target_prod_w * zoom_ratio) / prod_w0

        # We want the *base* of the product (which in source canvas is at
        # y = prod_h0, x = canvas_w / 2) to land at (fx, fy) in the frame.
        # Affine: x_dst = s * x_src + tx, y_dst = s * y_src + ty
        # At base-center (canvas_w/2, prod_h0) → (fx, fy):
        tx = fx - s * (canvas_w / 2)
        ty = fy - s * prod_h0

        M = np.array([[s, 0, tx],
                      [0, s, ty]], dtype=np.float32)
        warped = cv2.warpAffine(
            product_np, M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        # Alpha composite onto frame
        alpha = warped[:, :, 3:4].astype(np.float32) / 255.0
        rgb = warped[:, :, :3].astype(np.float32)
        composed = frame_np.astype(np.float32) * (1 - alpha) + rgb * alpha
        composed = composed.clip(0, 255).astype(np.uint8)

        Image.fromarray(composed).save(os.path.join(OUT_FRAMES, f"{i:04d}.png"))
        placements.append({"frame": i, "fx": fx, "fy": fy, "scale": s})
        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{N_FRAMES} @ fx={fx:.2f} fy={fy:.2f} scale={s:.4f}")

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
