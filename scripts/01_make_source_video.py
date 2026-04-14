"""
Step 1: synthesize a source video from a Flux-generated still image with a
programmatic camera pan/zoom-in.

Why synthetic source:
  - No dependency on external CC0 videos
  - Full control over scene (static camera-ready scene with empty surface)
  - Optical flow has a known ground-truth (useful for debugging RAFT later)
  - Reproducible

Output: /root/vpp/source/scene.mp4 (72 frames @ 24 fps = 3 seconds)
"""
import os, sys
import torch
import numpy as np
from PIL import Image
from diffusers import FluxPipeline
import torchvision.io as tvio

OUT_DIR = "/root/vpp/source"
N_FRAMES = 72            # 3 seconds at 24 fps
FPS = 24
CANVAS = (1024, 768)     # generation canvas (w, h)
FRAME_SIZE = (896, 672)  # final frame (slightly smaller for pan margin)

PROMPT = (
    "a modern minimalist living room interior, a polished wooden side table "
    "in the center of frame, soft natural window light from the left, "
    "empty clean table surface, a plain light-grey wall in the background, "
    "professional photography, cinematic composition, shallow depth of field"
)

def gen_still():
    print(f"Loading Flux from /root/avatar/models/flux1-dev...")
    pipe = FluxPipeline.from_pretrained("/root/avatar/models/flux1-dev", torch_dtype=torch.bfloat16)
    pipe.to("cuda")
    print(f"Generating base still: {PROMPT[:70]}...")
    gen = torch.Generator("cuda").manual_seed(101)
    img = pipe(
        prompt=PROMPT,
        num_inference_steps=25,
        guidance_scale=3.5,
        height=CANVAS[1], width=CANVAS[0],
        generator=gen,
    ).images[0]
    del pipe
    torch.cuda.empty_cache()
    return img

def make_frames(still):
    """Apply a slow zoom-in + slight left-to-right pan across N_FRAMES."""
    W, H = still.size
    fw, fh = FRAME_SIZE
    frames = []
    for i in range(N_FRAMES):
        t = i / max(1, N_FRAMES - 1)  # 0..1
        # zoom: scale 1.0 → 1.10 (subtle push-in)
        scale = 1.0 + 0.10 * t
        cw = int(fw / scale)
        ch = int(fh / scale)
        # pan: 0 → +30 px to the right
        pan_x = int(30 * t)
        cx = (W - cw) // 2 + pan_x - 15  # start slightly to the left
        cy = (H - ch) // 2
        crop = still.crop((cx, cy, cx + cw, cy + ch))
        crop = crop.resize((fw, fh), Image.LANCZOS)
        frames.append(np.array(crop))
    return frames

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    still = gen_still()
    still.save(os.path.join(OUT_DIR, "still.jpg"), quality=92)
    print(f"Still saved: {still.size}")

    frames = make_frames(still)
    arr = np.stack(frames)  # (N, H, W, 3) uint8
    print(f"Frames: {arr.shape}")

    tvio.write_video(
        os.path.join(OUT_DIR, "scene.mp4"),
        torch.from_numpy(arr),
        fps=FPS,
        video_codec="libx264",
        options={"crf": "18", "preset": "fast"},
    )
    # also save as PNG sequence for easy reuse
    seq_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(seq_dir, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(os.path.join(seq_dir, f"{i:04d}.png"))
    print(f"Wrote {N_FRAMES} frames to {seq_dir} and {OUT_DIR}/scene.mp4")

if __name__ == "__main__":
    main()
