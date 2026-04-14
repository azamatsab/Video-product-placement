"""
Step 2: define a static insertion mask on the first frame.

For MVP we use a hand-placed rectangle on the table surface. In a production
pipeline this would come from SAM2 prompted with a click on frame 0.

The mask is propagated to all frames via the known pan+zoom transform from
step 01, yielding per-frame masks that track the table area under camera
motion. This mimics what real optical-flow / SAM2 tracking would produce.

Output:
  /root/vpp/source/mask_static.png   — mask on frame 0
  /root/vpp/source/masks/0000.png..  — per-frame tracked masks
"""
import os
import numpy as np
from PIL import Image, ImageDraw

SOURCE = "/root/vpp/source"
FRAMES = os.path.join(SOURCE, "frames")
MASKS = os.path.join(SOURCE, "masks")

# Box on frame 0 where we will insert the product.
# The frame size is 896x672. The table top is roughly in the lower-right
# quadrant. We pick a rectangle on the flat area of the table surface.
MASK_BOX = (560, 370, 740, 490)  # (x1, y1, x2, y2) on frame 0

# Recreate the same pan+zoom transform from 01_make_source_video.py
N_FRAMES = 72
FRAME_SIZE = (896, 672)
CANVAS = (1024, 768)

def pan_zoom_params(i):
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


def main():
    os.makedirs(MASKS, exist_ok=True)

    # Build static mask on frame-0 with feathered edges for smooth blending
    from PIL import ImageFilter
    mask0 = Image.new("L", FRAME_SIZE, 0)
    draw = ImageDraw.Draw(mask0)
    draw.ellipse(MASK_BOX, fill=255)
    mask0 = mask0.filter(ImageFilter.GaussianBlur(radius=8))
    mask0.save(os.path.join(SOURCE, "mask_static.png"))
    print(f"Static mask defined: {MASK_BOX} on {FRAME_SIZE} canvas")

    # Back-project MASK_BOX through the pan+zoom transform to the canvas frame
    # Frame-0 params → source canvas coords of mask box
    cx0, cy0, cw0, ch0 = pan_zoom_params(0)
    fw, fh = FRAME_SIZE
    # frame-0 mask point (fx, fy) corresponds to canvas point:
    #   canvas_x = cx0 + fx * cw0 / fw
    #   canvas_y = cy0 + fy * ch0 / fh
    fx1, fy1, fx2, fy2 = MASK_BOX
    canvas_x1 = cx0 + fx1 * cw0 / fw
    canvas_y1 = cy0 + fy1 * ch0 / fh
    canvas_x2 = cx0 + fx2 * cw0 / fw
    canvas_y2 = cy0 + fy2 * ch0 / fh
    print(f"Mask in canvas coords: {canvas_x1:.0f},{canvas_y1:.0f}  {canvas_x2:.0f},{canvas_y2:.0f}")

    # For each frame: forward-project canvas box → this frame's coords
    for i in range(N_FRAMES):
        cx, cy, cw, ch = pan_zoom_params(i)
        # canvas → frame: fx = (cx_canvas - cx) * fw / cw
        mx1 = int((canvas_x1 - cx) * fw / cw)
        my1 = int((canvas_y1 - cy) * fh / ch)
        mx2 = int((canvas_x2 - cx) * fw / cw)
        my2 = int((canvas_y2 - cy) * fh / ch)
        m = Image.new("L", FRAME_SIZE, 0)
        d = ImageDraw.Draw(m)
        d.ellipse((mx1, my1, mx2, my2), fill=255)
        from PIL import ImageFilter
        m = m.filter(ImageFilter.GaussianBlur(radius=8))
        m.save(os.path.join(MASKS, f"{i:04d}.png"))

    print(f"Wrote {N_FRAMES} per-frame masks to {MASKS}")

if __name__ == "__main__":
    main()
