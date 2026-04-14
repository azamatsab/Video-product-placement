"""
Step 4: TEMPORALLY-STABLE inpainting via RAFT optical flow warping.

Strategy:
  1. Inpaint ONE reference frame (frame 0) with SDXL Inpainting.
  2. For every subsequent frame, compute optical flow from frame 0 → frame i
     via RAFT (pretrained, torchvision).
  3. Warp the inpainted masked region of frame 0 forward to frame i using the
     flow field. This gives a geometrically-correct placement of the SAME
     product at frame i.
  4. Composite the warped product onto the original frame i using the warped
     mask as blend alpha.

Why this is fundamentally more stable than naive:
  - Product geometry is generated ONCE and then rigidly transported by the
    observed camera motion — no per-frame stochastic variation.
  - The only "drift" is accumulated optical-flow error, which for short clips
    with a static background is small.

Output: /root/vpp/output/warped/frames/*.png + warped.mp4
"""
import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
import torchvision.io as tvio
from diffusers import StableDiffusionXLInpaintPipeline

SOURCE = "/root/vpp/source"
FRAMES = os.path.join(SOURCE, "frames")
MASKS = os.path.join(SOURCE, "masks")
OUT = "/root/vpp/output/warped"
OUT_FRAMES = os.path.join(OUT, "frames")

PROMPT = "a luxury glass perfume bottle standing on a polished wooden table, minimalist living room, warm natural light, cinematic, sharp focus"
NEG_PROMPT = "blurry, low quality, text, watermark, multiple objects, grey background, studio backdrop, floating"
N_FRAMES = 72
SEED = 77


def inpaint_frame0():
    """Inpaint the reference frame once."""
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "/root/vpp/models/sdxl-inpaint", torch_dtype=torch.float16
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    img = Image.open(os.path.join(FRAMES, "0000.png")).convert("RGB")
    mask = Image.open(os.path.join(MASKS, "0000.png")).convert("L")
    print("Inpainting reference frame 0...")
    out = pipe(
        prompt=PROMPT,
        negative_prompt=NEG_PROMPT,
        image=img,
        mask_image=mask,
        guidance_scale=9.0,
        num_inference_steps=30,
        strength=1.0,
        generator=torch.Generator("cuda").manual_seed(SEED),
        padding_mask_crop=192,
    ).images[0]
    del pipe
    torch.cuda.empty_cache()
    return np.array(img), np.array(mask), np.array(out)


def load_raft():
    weights = Raft_Large_Weights.C_T_SKHT_V2
    model = raft_large(weights=weights, progress=False).eval().to("cuda")
    return model, weights.transforms()


def compute_flow(model, transforms, img1, img2):
    """Compute optical flow from img1 → img2. img1, img2 are uint8 HWC numpy arrays."""
    t1 = torch.from_numpy(img1).permute(2, 0, 1).float() / 255.0  # CHW
    t2 = torch.from_numpy(img2).permute(2, 0, 1).float() / 255.0
    # RAFT needs dims divisible by 8 — our 896x672 is fine
    t1b, t2b = transforms(t1[None].to("cuda"), t2[None].to("cuda"))
    with torch.no_grad():
        flows = model(t1b, t2b)  # list of refined flows
    return flows[-1][0].cpu().numpy()  # HWC? no, CHW with C=2


def warp_with_flow(src_img, src_mask, flow):
    """
    Warp src_img and src_mask forward using backward-warping with the flow.
    flow: (2, H, W) giving (dx, dy) per pixel, flow points from src to dst.
    We use grid_sample backward-mode: for each dst pixel, sample src at
    (x - flow[0,y,x], y - flow[1,y,x]).
    """
    H, W = src_img.shape[:2]
    # Build grid
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    src_x = xx - flow[0]
    src_y = yy - flow[1]

    # Normalize to [-1, 1] for grid_sample
    gx = 2.0 * src_x / (W - 1) - 1.0
    gy = 2.0 * src_y / (H - 1) - 1.0
    grid = torch.from_numpy(np.stack([gx, gy], axis=-1)[None]).float().to("cuda")

    src_t = torch.from_numpy(src_img).permute(2, 0, 1)[None].float().to("cuda") / 255.0
    warped = F.grid_sample(src_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    warped_np = (warped[0].cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)

    mask_t = torch.from_numpy(src_mask)[None, None].float().to("cuda") / 255.0
    warped_mask = F.grid_sample(mask_t, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    warped_mask_np = (warped_mask[0, 0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

    return warped_np, warped_mask_np


def main():
    os.makedirs(OUT_FRAMES, exist_ok=True)

    ref_img, ref_mask, ref_inpainted = inpaint_frame0()
    Image.fromarray(ref_inpainted).save(os.path.join(OUT, "reference_frame0.png"))
    print(f"reference saved, shape: {ref_inpainted.shape}")

    # Extract just the inpainted product: diff between ref_inpainted and ref_img,
    # masked by ref_mask. Actually, we'll use full ref_inpainted and compose via
    # mask during warping.
    model, transforms = load_raft()
    print("RAFT loaded")

    # Save reference frame unchanged
    Image.fromarray(ref_inpainted).save(os.path.join(OUT_FRAMES, "0000.png"))

    for i in range(1, N_FRAMES):
        # Load frame i
        frame_i = np.array(Image.open(os.path.join(FRAMES, f"{i:04d}.png")).convert("RGB"))
        # Compute flow from ref (frame 0) to frame i
        flow = compute_flow(model, transforms, ref_img, frame_i)
        # Warp ref_inpainted and ref_mask forward
        warped_img, warped_mask = warp_with_flow(ref_inpainted, ref_mask, flow)
        # Composite: final = frame_i * (1-m) + warped_img * m
        alpha = (warped_mask.astype(np.float32) / 255.0)[:, :, None]
        composed = frame_i.astype(np.float32) * (1 - alpha) + warped_img.astype(np.float32) * alpha
        composed = composed.clip(0, 255).astype(np.uint8)
        Image.fromarray(composed).save(os.path.join(OUT_FRAMES, f"{i:04d}.png"))
        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{N_FRAMES}")

    # Encode
    frames_np = np.stack([
        np.array(Image.open(os.path.join(OUT_FRAMES, f"{i:04d}.png")).convert("RGB"))
        for i in range(N_FRAMES)
    ])
    tvio.write_video(
        os.path.join(OUT, "warped.mp4"),
        torch.from_numpy(frames_np),
        fps=24,
        video_codec="libx264",
        options={"crf": "18", "preset": "fast"},
    )
    print(f"Wrote {OUT}/warped.mp4")


if __name__ == "__main__":
    main()
