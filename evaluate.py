"""
evaluate.py
Compares scoring aggregation strategies for InspectIQ's PaDiM models,
using cached raw score maps (each image is scored by the CNN only once).

Strategies compared:
  - max        : current app.py behavior (single most anomalous patch)
  - top1pct    : mean of the top 1% highest-scoring patches
  - top5pct    : mean of the top 5% highest-scoring patches
  - mean       : mean of the entire score map (for reference/baseline)

Run from the project root:
    python evaluate.py
"""

from pathlib import Path
import numpy as np
from PIL import Image
import torch

from src.padim_core import FeatureExtractor, TRANSFORM, mahalanobis_score_map, load_stats

DATA_ROOT = Path("data/mvtec")
MODELS_ROOT = Path("models")
CATEGORIES = ["bottle", "cable", "tile"]
IMG_EXTS = (".png", ".jpg", ".jpeg")


def raw_score_map(path, extractor, mean, inv_cov, hw):
    img = Image.open(path).convert("RGB")
    x = TRANSFORM(img).unsqueeze(0)
    with torch.no_grad():
        emb = extractor(x).cpu().numpy()[0]
    return mahalanobis_score_map(emb, mean, inv_cov, hw)


def aggregate(score_map, method):
    flat = score_map.flatten()
    if method == "max":
        return float(flat.max())
    if method == "mean":
        return float(flat.mean())
    if method in ("top1pct", "top5pct"):
        pct = 0.01 if method == "top1pct" else 0.05
        k = max(1, int(len(flat) * pct))
        top_k = np.partition(flat, -k)[-k:]
        return float(top_k.mean())
    raise ValueError(method)


METHODS = ["max", "top1pct", "top5pct", "mean"]


def evaluate_category(category, extractor):
    model_path = MODELS_ROOT / f"{category}_padim.pkl"
    if not model_path.exists():
        print(f"  [skip] no model found at {model_path}")
        return

    mean, inv_cov, hw = load_stats(str(model_path))
    test_root = DATA_ROOT / category / "test"
    if not test_root.exists():
        print(f"  [skip] no test folder at {test_root}")
        return
    subfolders = sorted(p for p in test_root.iterdir() if p.is_dir())

    cache = {}
    for sub in subfolders:
        imgs = [p for p in sorted(sub.glob("*")) if p.suffix.lower() in IMG_EXTS]
        cache[sub.name] = [raw_score_map(p, extractor, mean, inv_cov, hw) for p in imgs]

    print(f"\n=== {category.upper()} ===")
    for method in METHODS:
        print(f"\n  -- method={method} --")
        good_maps = cache.get("good", [])
        good_scores = np.array([aggregate(sm, method) for sm in good_maps])
        good_mean = good_scores.mean() if len(good_scores) else 0.0
        if len(good_scores):
            print(f"  good/                  n={len(good_scores):3d}  "
                  f"mean={good_mean:6.2f}  min={good_scores.min():6.2f}  max={good_scores.max():6.2f}")
        for name, maps in cache.items():
            if name == "good" or not maps:
                continue
            scores = np.array([aggregate(sm, method) for sm in maps])
            gap = scores.mean() - good_mean
            # rough separation quality: how many defect images score above the
            # highest good/ score seen (a "would this threshold work" proxy)
            above_worst_good = int((scores > good_scores.max()).sum()) if len(good_scores) else 0
            print(f"  {name:<22} n={len(scores):3d}  "
                  f"mean={scores.mean():6.2f}  min={scores.min():6.2f}  max={scores.max():6.2f}  "
                  f"gap={gap:+6.2f}  above_worst_good={above_worst_good}/{len(scores)}")


def main():
    extractor = FeatureExtractor()
    for category in CATEGORIES:
        evaluate_category(category, extractor)


if __name__ == "__main__":
    main()