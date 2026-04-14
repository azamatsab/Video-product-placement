"""
Step 5: quantify temporal stability and build a side-by-side comparison video.

Temporal stability metric:
  For each pair of consecutive frames (i, i+1), compute pixel RMSE inside the
  mask region. A stable video has small frame-to-frame deltas; a flickering
  video has large ones. We report mean and max across all pairs, for both
  `naive` and `warped` outputs.
"""
import os, json
import numpy as np
from PIL import Image
import torch
import torchvision.io as tvio

N_FRAMES = 72
SOURCE_FRAMES = "/root/vpp/source/frames"
MASKS_DIR = "/root/vpp/source/masks"

CONFIGS = [
    ("naive",  "/root/vpp/output/naive/frames"),
    ("warped", "/root/vpp/output/warped/frames"),
]

def load(path, as_gray=False):
    img = Image.open(path).convert("L" if as_gray else "RGB")
    return np.array(img)

def stability_metric(frame_dir):
    frames = [load(os.path.join(frame_dir, f"{i:04d}.png")) for i in range(N_FRAMES)]
    masks  = [load(os.path.join(MASKS_DIR, f"{i:04d}.png"), as_gray=True) for i in range(N_FRAMES)]
    deltas = []
    for i in range(N_FRAMES - 1):
        m = (masks[i] + masks[i+1]) / 2 > 64  # union-ish, soft
        if m.sum() == 0:
            continue
        d = np.abs(frames[i].astype(float) - frames[i+1].astype(float))
        # only pixels in mask
        dm = d[m].mean()
        deltas.append(dm)
    arr = np.array(deltas)
    return {
        "mean_frame_delta":  round(float(arr.mean()), 3),
        "max_frame_delta":   round(float(arr.max()), 3),
        "median_frame_delta": round(float(np.median(arr)), 3),
        "std_frame_delta":   round(float(arr.std()), 3),
    }

def build_comparison_video():
    """Horizontal stack: naive | warped, with labels baked in via top title bar."""
    from PIL import ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    W_single = 896
    bar_h = 36
    out_w = W_single * 2
    out_h = 672 + bar_h
    out_frames = []
    for i in range(N_FRAMES):
        naive = Image.open(f"/root/vpp/output/naive/frames/{i:04d}.png").convert("RGB")
        warp  = Image.open(f"/root/vpp/output/warped/frames/{i:04d}.png").convert("RGB")
        canvas = Image.new("RGB", (out_w, out_h), "white")
        d = ImageDraw.Draw(canvas)
        d.rectangle([0, 0, out_w, bar_h], fill="black")
        d.text((20, 6), "NAIVE per-frame inpainting (flicker)", fill="red", font=font)
        d.text((W_single + 20, 6), "WARPED via RAFT optical flow (stable)", fill="lime", font=font)
        canvas.paste(naive, (0, bar_h))
        canvas.paste(warp,  (W_single, bar_h))
        out_frames.append(np.array(canvas))
    arr = np.stack(out_frames)
    tvio.write_video(
        "/root/vpp/output/comparison.mp4",
        torch.from_numpy(arr),
        fps=24,
        video_codec="libx264",
        options={"crf": "18", "preset": "fast"},
    )
    # also save a representative gif-like still (triptych)
    strip = Image.new("RGB", (out_w, out_h), "white")
    draw = ImageDraw.Draw(strip)
    draw.rectangle([0, 0, out_w, bar_h], fill="black")
    draw.text((20, 6), "NAIVE per-frame inpainting (flicker)", fill="red", font=font)
    draw.text((W_single + 20, 6), "WARPED via RAFT optical flow (stable)", fill="lime", font=font)
    mid = Image.open(f"/root/vpp/output/naive/frames/0036.png").convert("RGB")
    mid_w = Image.open(f"/root/vpp/output/warped/frames/0036.png").convert("RGB")
    strip.paste(mid, (0, bar_h))
    strip.paste(mid_w, (W_single, bar_h))
    strip.save("/root/vpp/output/comparison_still.jpg", quality=92)
    print(f"Wrote comparison.mp4 and comparison_still.jpg")

def main():
    results = {}
    for name, fdir in CONFIGS:
        results[name] = stability_metric(fdir)
        print(f"{name:>8}: {results[name]}")
    with open("/root/vpp/output/stability_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print("metrics saved")
    build_comparison_video()

if __name__ == "__main__":
    main()
