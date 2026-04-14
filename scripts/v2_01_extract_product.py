"""
v2 step 1: extract product silhouette from the raw photo using SAM2.

Input:  assets/product_raw.jpg (Dior Hypnotic Poison on white background)
Output: assets/product_rgba.png — RGBA with transparent background, label intact
"""
import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForMaskGeneration

RAW = "/root/vpp/assets/product_raw.jpg"
OUT = "/root/vpp/assets/product_rgba.png"
MASK_OUT = "/root/vpp/assets/product_mask.png"
MODEL = "/root/vpp/models/sam2"


def main():
    print(f"Loading SAM2 from {MODEL}")
    processor = AutoProcessor.from_pretrained(MODEL)
    model = AutoModelForMaskGeneration.from_pretrained(MODEL, torch_dtype=torch.float32).to("cuda")
    model.eval()

    img = Image.open(RAW).convert("RGB")
    W, H = img.size
    print(f"Product image: {W}x{H}")

    # Point prompt at the center — the bottle is centered in the raw photo
    # SAM2 expects [image, object, point, coords] nesting
    input_points = [[[[W // 2, H // 2]]]]
    input_labels = [[[1]]]
    inputs = processor(
        images=img,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=True)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(),
        original_sizes=inputs["original_sizes"].cpu(),
    )[0][0]  # first image in batch, all mask candidates

    scores = outputs.iou_scores[0, 0].cpu().numpy()
    print(f"SAM2 generated {len(masks)} candidates, IoU scores: {scores}")
    best_idx = int(np.argmax(scores))
    print(f"Picked mask {best_idx} (score {scores[best_idx]:.3f})")

    mask_np = masks[best_idx].numpy().astype(np.uint8) * 255
    # mask may be (H, W) or (1, H, W)
    if mask_np.ndim == 3:
        mask_np = mask_np[0]

    # Save mask
    Image.fromarray(mask_np).save(MASK_OUT)
    print(f"Mask saved: {mask_np.shape}, non-zero: {(mask_np > 0).sum()} px")

    # Build RGBA
    img_np = np.array(img)
    rgba = np.dstack([img_np, mask_np])
    Image.fromarray(rgba).save(OUT)
    print(f"RGBA saved: {OUT}")


if __name__ == "__main__":
    main()
