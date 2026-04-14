"""
v2 step 4: compare v1 (diffusion inpaint + RAFT warp) vs v2 (composite real
product) across two axes: temporal stability + label readability.

Label readability metric: run OCR (easyocr) on the product region of a
reference frame (frame 35) and measure how many characters of the ground
truth "HYPNOTIC POISON EAU SECRÈTE Dior" are detected.

Stability metric: same as before — mean frame-to-frame RGB delta in a tight
box around the product.
"""
import os, json
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import torchvision.io as tvio

N_FRAMES = 72
GT_LABEL = "HYPNOTIC POISON EAU SECRETE Dior"  # ASCII version for matching

CONFIGS = {
    "v1_warped": "/root/vpp/output/warped/frames",
    "v2_composite": "/root/vpp/output/v2_composite/frames",
}

PRODUCT_BOX = (520, 315, 762, 550)  # union of product bboxes across all frames


def stability(frame_dir):
    frames = [np.array(Image.open(os.path.join(frame_dir, f"{i:04d}.png")).convert("RGB"))
              for i in range(N_FRAMES)]
    x1, y1, x2, y2 = PRODUCT_BOX
    deltas = []
    for i in range(N_FRAMES - 1):
        a = frames[i][y1:y2, x1:x2].astype(float)
        b = frames[i+1][y1:y2, x1:x2].astype(float)
        deltas.append(np.abs(a - b).mean())
    arr = np.array(deltas)
    return {
        "mean": round(float(arr.mean()), 3),
        "median": round(float(np.median(arr)), 3),
        "max": round(float(arr.max()), 3),
        "std": round(float(arr.std()), 3),
    }


def label_readability(frame_dir):
    """
    Use easyocr to read text in the product area on frame 35. Return:
      - detected_text: joined string
      - char_overlap: longest common substring length with GT
      - substring_match: fraction of GT chars found in detected text
    """
    try:
        import easyocr
    except ImportError:
        print("easyocr not available, using simple heuristic")
        return {"detected": "", "match_ratio": 0.0, "note": "easyocr not installed"}

    reader = easyocr.Reader(["en"], gpu=True)
    img_path = os.path.join(frame_dir, "0035.png")
    img = np.array(Image.open(img_path).convert("RGB"))
    x1, y1, x2, y2 = PRODUCT_BOX
    crop = img[y1:y2, x1:x2]
    results = reader.readtext(crop)
    detected = " ".join(r[1] for r in results).upper()
    print(f"  OCR on {img_path}: {detected!r}")
    # Character-level match: how many unique 3-grams of GT appear in detected
    gt = GT_LABEL.upper().replace(" ", "")
    det = detected.replace(" ", "")
    ngrams = set()
    for n in range(3, 6):
        for i in range(len(gt) - n + 1):
            ngrams.add(gt[i:i+n])
    matched = sum(1 for ng in ngrams if ng in det)
    match_ratio = matched / max(1, len(ngrams))
    return {"detected": detected, "match_ratio": round(match_ratio, 3),
            "chars_detected": len(det)}


def build_side_by_side():
    """Horizontal stack: v1 | v2 with labels."""
    out_w = 896 * 2
    bar_h = 36
    out_h = 672 + bar_h
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception:
        font = ImageFont.load_default()
    out_frames = []
    for i in range(N_FRAMES):
        v1 = Image.open(f"/root/vpp/output/warped/frames/{i:04d}.png").convert("RGB")
        v2 = Image.open(f"/root/vpp/output/v2_composite/frames/{i:04d}.png").convert("RGB")
        canvas = Image.new("RGB", (out_w, out_h), "white")
        d = ImageDraw.Draw(canvas)
        d.rectangle([0, 0, out_w, bar_h], fill="black")
        d.text((20, 6), "v1  SDXL Inpaint + RAFT warp (label = gibberish)", fill="red", font=font)
        d.text((896 + 20, 6), "v2  SAM2 + GroundingDINO composite (label = pixel-perfect)", fill="lime", font=font)
        canvas.paste(v1, (0, bar_h))
        canvas.paste(v2, (896, bar_h))
        out_frames.append(np.array(canvas))
    arr = np.stack(out_frames)
    tvio.write_video(
        "/root/vpp/output/v1_vs_v2.mp4",
        torch.from_numpy(arr),
        fps=24,
        video_codec="libx264",
        options={"crf": "18", "preset": "fast"},
    )
    # still at frame 35
    v1 = Image.open("/root/vpp/output/warped/frames/0035.png")
    v2 = Image.open("/root/vpp/output/v2_composite/frames/0035.png")
    still = Image.new("RGB", (out_w, out_h), "white")
    d = ImageDraw.Draw(still)
    d.rectangle([0, 0, out_w, bar_h], fill="black")
    d.text((20, 6), "v1  SDXL Inpaint + RAFT warp", fill="red", font=font)
    d.text((896 + 20, 6), "v2  SAM2 + GroundingDINO composite", fill="lime", font=font)
    still.paste(v1, (0, bar_h))
    still.paste(v2, (896, bar_h))
    still.save("/root/vpp/output/v1_vs_v2_still.jpg", quality=92)
    print("wrote v1_vs_v2.mp4 and still")


def main():
    results = {}
    print("=== STABILITY (lower = better) ===")
    for name, d in CONFIGS.items():
        s = stability(d)
        results[name] = {"stability": s}
        print(f"  {name}: {s}")

    print("\n=== LABEL READABILITY (higher = better) ===")
    for name, d in CONFIGS.items():
        r = label_readability(d)
        results[name]["label"] = r
        print(f"  {name}: {r}")

    with open("/root/vpp/output/v2_metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote v2_metrics.json")

    build_side_by_side()


if __name__ == "__main__":
    main()
