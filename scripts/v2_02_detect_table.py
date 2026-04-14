"""
v2 step 2: auto-detect the insertion surface (table) in frame 0 using
GroundingDINO + SAM2 (NO hand-coded mask).

This replaces the analytical back-projection in v1's 02_define_mask.py.

Process:
  1. GroundingDINO("wooden table top.") on frame 0 → bbox(es)
  2. SAM2 with that bbox as prompt → precise table mask on frame 0
  3. Return the bbox center as the "anchor point" for product placement
  4. Track across all frames via analytical pan+zoom (the scene transform is
     known; in a real video we'd use SAM2 video-mode tracking)
"""
import os
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForMaskGeneration,
    AutoModelForZeroShotObjectDetection,
)

SCENE = "/root/vpp/source/frames/0000.png"
SAM2 = "/root/vpp/models/sam2"
GDINO = "/root/vpp/models/grounding-dino"
OUT_MASK = "/root/vpp/assets/table_mask_frame0.png"
OUT_PREVIEW = "/root/vpp/assets/table_detection_preview.jpg"


def main():
    img = Image.open(SCENE).convert("RGB")
    W, H = img.size
    print(f"Scene: {W}x{H}")

    # ─── Step 1: Grounding DINO → bbox for "wooden table" ───────────────
    print(f"Loading Grounding DINO from {GDINO}")
    gdino_proc = AutoProcessor.from_pretrained(GDINO)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO).to("cuda")
    gdino_model.eval()

    text_prompt = "wooden table."  # grounding dino expects "." at end, lower case
    inputs = gdino_proc(images=img, text=text_prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = gdino_model(**inputs)
    results = gdino_proc.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.3,
        text_threshold=0.25,
        target_sizes=[img.size[::-1]],  # (H, W)
    )[0]
    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]
    print(f"Grounding DINO: {len(boxes)} detections")
    for i, (b, s, l) in enumerate(zip(boxes, scores, labels)):
        print(f"  [{i}] label={l!r} score={s:.3f} box={b.astype(int).tolist()}")
    if len(boxes) == 0:
        raise RuntimeError("No table detected")

    # pick highest-score detection
    best_idx = int(np.argmax(scores))
    box = boxes[best_idx]  # [x1, y1, x2, y2]
    print(f"Picked box {best_idx}: {box.astype(int).tolist()}")

    del gdino_model
    torch.cuda.empty_cache()

    # ─── Step 2: SAM2 with box prompt → precise table mask ──────────────
    print(f"Loading SAM2 from {SAM2}")
    sam_proc = AutoProcessor.from_pretrained(SAM2)
    sam_model = AutoModelForMaskGeneration.from_pretrained(SAM2, dtype=torch.float32).to("cuda")
    sam_model.eval()

    input_boxes = [[box.tolist()]]  # [image, object, box]
    sam_inputs = sam_proc(
        images=img,
        input_boxes=input_boxes,
        return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        sam_out = sam_model(**sam_inputs, multimask_output=True)
    masks = sam_proc.post_process_masks(
        sam_out.pred_masks.cpu(),
        original_sizes=sam_inputs["original_sizes"].cpu(),
    )[0][0]
    iou = sam_out.iou_scores[0, 0].cpu().numpy()
    print(f"SAM2 iou scores: {iou}")
    pick = int(np.argmax(iou))
    mask = masks[pick].numpy().astype(np.uint8) * 255
    if mask.ndim == 3:
        mask = mask[0]
    print(f"Table mask: {mask.shape}, area: {(mask > 0).sum()} px")

    Image.fromarray(mask).save(OUT_MASK)
    print(f"Wrote {OUT_MASK}")

    # ─── Preview: overlay box + mask on scene ───────────────────────────
    from PIL import ImageDraw
    preview = img.copy().convert("RGBA")
    overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    ov = np.array(overlay)
    ov[mask > 128] = [0, 255, 0, 100]
    overlay = Image.fromarray(ov)
    preview = Image.alpha_composite(preview, overlay).convert("RGB")
    d = ImageDraw.Draw(preview)
    d.rectangle(box.tolist(), outline="red", width=3)
    preview.save(OUT_PREVIEW, quality=92)
    print(f"Wrote {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
