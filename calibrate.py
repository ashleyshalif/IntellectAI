import os
import sys
import glob
import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from padim_core import FeatureExtractor, TRANSFORM, mahalanobis_score_map, load_stats

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data", "mvtec")
MODELS_ROOT = os.path.join(os.path.dirname(__file__), "models")
CATEGORIES = ["bottle", "cable", "tile"]

def score_image(path, extractor, mean, inv_cov, hw):
    pil_img = Image.open(path).convert("RGB")
    x = TRANSFORM(pil_img).unsqueeze(0)
    with torch.no_grad():
        emb = extractor(x).cpu().numpy()[0]
    score_map = mahalanobis_score_map(emb, mean, inv_cov, hw)
    # Top-1% mean instead of a single max patch -- must match app.py's
    # scoring exactly, or these thresholds won't be on the same scale.
    flat = score_map.flatten()
    k = max(1, int(flat.size * 0.01))
    return float(np.partition(flat, -k)[-k:].mean())

def main():
    extractor = FeatureExtractor()
    results = {}

    for category in CATEGORIES:
        pkl_path = os.path.join(MODELS_ROOT, f"{category}_padim.pkl")
        if not os.path.exists(pkl_path):
            print(f"[SKIP] {category}: no trained model at {pkl_path}")
            continue

        mean, inv_cov, hw = load_stats(pkl_path)

        good_dir = os.path.join(DATA_ROOT, category, "test", "good")
        image_paths = sorted(glob.glob(os.path.join(good_dir, "*.png")))
        if not image_paths:
            print(f"[SKIP] {category}: no images found in {good_dir}")
            continue

        scores = []
        for p in image_paths:
            scores.append(score_image(p, extractor, mean, inv_cov, hw))

        scores = np.array(scores)
        mean_s, std_s = scores.mean(), scores.std()
        p50, p90, p95, p99, p100 = np.percentile(scores, [50, 90, 95, 99, 100])

        suggested_minor = mean_s + 1.5 * std_s
        suggested_defect = mean_s + 3.0 * std_s

        results[category] = {
            "n": len(scores), "mean": mean_s, "std": std_s,
            "p50": p50, "p90": p90, "p95": p95, "p99": p99, "p100": p100,
            "suggested_minor": suggested_minor, "suggested_defect": suggested_defect,
        }

        print(f"\n=== {category} (n={len(scores)} normal test images) ===")
        print(f"  mean={mean_s:.2f}  std={std_s:.2f}")
        print(f"  p50={p50:.2f}  p90={p90:.2f}  p95={p95:.2f}  p99={p99:.2f}  max={p100:.2f}")
        print(f"  SUGGESTED minor_threshold={suggested_minor:.2f}  defect_threshold={suggested_defect:.2f}")

    print("\n\n=== Summary: paste these into a per-category threshold dict in app.py ===")
    for category, r in results.items():
        print(f'  "{category}": {{"minor": {r["suggested_minor"]:.2f}, "defect": {r["suggested_defect"]:.2f}}},')

if __name__ == "__main__":
    main()