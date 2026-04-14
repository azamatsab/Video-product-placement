"""
Step 3a: NAIVE per-frame inpainting.

Insert a product into every frame independently using SDXL Inpainting, with
the same prompt and the same seed. This produces the baseline "flicker
disaster" that motivates temporal consistency methods.

Key: even with identical seed and identical mask, each frame's input image is
slightly different (due to camera pan/zoom), so the inpainted product varies
between frames.

Output: /root/vpp/output/naive/frames/*.png and naive.mp4
"""
import os
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline
import torchvision.io as tvio

SOURCE = "/root/vpp/source"
FRAMES = os.path.join(SOURCE, "frames")
MASKS = os.path.join(SOURCE, "masks")
OUT = "/root/vpp/output/naive"
OUT_FRAMES = os.path.join(OUT, "frames")

PROMPT = "a luxury glass perfume bottle, isolated on a clean wooden table surface, sharp focus, detailed glass"
NEG_PROMPT = "blurry background, bokeh, out of focus elements, soft circle, background blob, multiple objects, text, watermark, floating, duplicate"
SEED = 77
N_FRAMES = 72
INFERENCE_STEPS = 40


def main():
    os.makedirs(OUT_FRAMES, exist_ok=True)

    print("Loading SDXL Inpainting...")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        "/root/vpp/models/sdxl-inpaint",
        torch_dtype=torch.float16,
        variant="fp16" if os.path.exists("/root/vpp/models/sdxl-inpaint/unet/diffusion_pytorch_model.fp16.safetensors") else None,
    )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print("loaded")

    for i in range(N_FRAMES):
        img = Image.open(os.path.join(FRAMES, f"{i:04d}.png")).convert("RGB")
        mask = Image.open(os.path.join(MASKS, f"{i:04d}.png")).convert("L")
        gen = torch.Generator("cuda").manual_seed(SEED)
        out = pipe(
            prompt=PROMPT,
            negative_prompt=NEG_PROMPT,
            image=img,
            mask_image=mask,
            guidance_scale=12.0,
            num_inference_steps=INFERENCE_STEPS,
            strength=1.0,
            generator=gen,
            padding_mask_crop=64,  # tight context — matches warped config for fair A/B
        ).images[0]
        out.save(os.path.join(OUT_FRAMES, f"{i:04d}.png"))
        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{N_FRAMES}")

    # Encode to MP4
    frames_np = np.stack([
        np.array(Image.open(os.path.join(OUT_FRAMES, f"{i:04d}.png")).convert("RGB"))
        for i in range(N_FRAMES)
    ])
    tvio.write_video(
        os.path.join(OUT, "naive.mp4"),
        torch.from_numpy(frames_np),
        fps=24,
        video_codec="libx264",
        options={"crf": "18", "preset": "fast"},
    )
    print(f"Wrote {OUT}/naive.mp4")


if __name__ == "__main__":
    main()
