"""
calibrate.py
Computes minor/defect anomaly-score thresholds per category using each
category's own in-category test set (good + all defect types), scored
with the same top-1%-mean formula as app.py.

defect threshold = F1-optimal point on the precision-recall curve
minor threshold  = 90th percentile of GOOD-only scores (catches mild
                    deviations before the defect line), clamped to never
                    exceed the defect threshold.

Usage:
    python src/calibrate.py --categories bottle cable tile
"""

import argparse
import json
import os
import numpy as np

from evaluate import evaluate_pair
from padim_core import FeatureExtractor

MODELS_ROOT = os.path.join(os.path.dirname(__file__), "..", "models")
OUT_PATH = os.path.join(MODELS_ROOT, "calibrated_thresholds.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=["bottle", "cable", "tile"])
    args = parser.parse_args()

    extractor = FeatureExtractor()
    thresholds = {}

    for cat in args.categories:
        print(f"\n[{cat}] scoring in-category test set...")
        metrics, _ = evaluate_pair(cat, cat, extractor)
        if metrics is None:
            print(f"[{cat}] no test images found, skipping")
            continue

        defect_thr = metrics["threshold"]
        good_scores = np.array(metrics["good_scores"])
        if len(good_scores) > 0:
            minor_thr = float(np.percentile(good_scores, 90))
            minor_thr = min(minor_thr, defect_thr - 0.01)  # never let minor cross defect
        else:
            minor_thr = defect_thr * 0.85

        thresholds[cat] = {"minor": round(minor_thr, 2), "defect": round(defect_thr, 2)}
        print(f"[{cat}] minor={thresholds[cat]['minor']}  defect={thresholds[cat]['defect']}  "
              f"AUROC={metrics['auroc']:.3f}  F1={metrics['f1']:.3f}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")
    print(json.dumps(thresholds, indent=2))