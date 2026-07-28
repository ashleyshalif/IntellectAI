"""
evaluate.py
1. Per-category evaluation: AUROC, Precision, Recall, F1 using the MVTec
   test/ split (good + each defect subfolder = anomalous). Uses the same
   top-1%-mean scorer as app.py so results reflect the live app.
2. Cross-Category Structural Robustness Profiling.

Usage:
    python src/evaluate.py --categories bottle cable tile
"""

import argparse
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, precision_recall_curve
from tqdm import tqdm

from padim_core import FeatureExtractor, load_image, mahalanobis_score_map, load_stats, apply_pca, DEVICE

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "mvtec")
MODELS_ROOT = os.path.join(os.path.dirname(__file__), "..", "models")


def collect_test_images(category: str):
    test_dir = os.path.join(DATA_ROOT, category, "test")
    items = []
    for subfolder in sorted(os.listdir(test_dir)):
        sub_path = os.path.join(test_dir, subfolder)
        if not os.path.isdir(sub_path):
            continue
        label = 0 if subfolder == "good" else 1
        for fname in sorted(os.listdir(sub_path)):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                items.append((os.path.join(sub_path, fname), label))
    return items


def top1pct_score(score_map: np.ndarray) -> float:
    """Same formula as app.py: top-1% of the score map, averaged."""
    flat = score_map.flatten()
    k = max(1, int(flat.size * 0.01))
    return float(np.partition(flat, -k)[-k:].mean())


def score_image(path, extractor, mean, inv_cov, hw, pca_mean, pca_components):
    x = load_image(path).unsqueeze(0)
    with torch.no_grad():
        raw_emb = extractor(x).cpu().numpy()[0]
    emb = apply_pca(raw_emb, pca_mean, pca_components)
    score_map = mahalanobis_score_map(emb, mean, inv_cov, hw)
    return top1pct_score(score_map)


def evaluate_pair(model_category: str, test_category: str, extractor):
    stats_path = os.path.join(MODELS_ROOT, f"{model_category}_padim.pkl")
    mean, inv_cov, hw, pca_mean, pca_components, _raw_centroid = load_stats(stats_path)

    items = collect_test_images(test_category)
    if not items:
        return None, None

    scores, labels = [], []
    for path, label in tqdm(items, desc=f"{model_category}->{test_category}", leave=False):
        scores.append(score_image(path, extractor, mean, inv_cov, hw, pca_mean, pca_components))
        labels.append(label)

    scores = np.array(scores)
    labels = np.array(labels)

    if len(set(labels)) < 2:
        auroc = float("nan")
        threshold = float(np.percentile(scores, 90))
    else:
        auroc = roc_auc_score(labels, scores)
        prec_arr, rec_arr, thresh_arr = precision_recall_curve(labels, scores)
        f1_arr = np.where((prec_arr + rec_arr) > 0,
                           2 * prec_arr * rec_arr / (prec_arr + rec_arr + 1e-12), 0)
        best_idx = np.argmax(f1_arr[:-1])
        threshold = float(thresh_arr[best_idx])

    preds = (scores >= threshold).astype(int)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)

    metrics = {"auroc": auroc, "precision": precision, "recall": recall, "f1": f1,
               "n": len(items), "threshold": threshold,
               "good_scores": scores[labels == 0].tolist() if (labels == 0).any() else []}
    return metrics, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=["bottle", "cable", "tile"])
    args = parser.parse_args()

    extractor = FeatureExtractor()
    cats = args.categories

    print("\n=== Cross-Category Structural Robustness Matrix (AUROC) ===")
    header = "model\\test".ljust(12) + "".join(c.ljust(12) for c in cats)
    print(header)

    results = {}
    for model_cat in cats:
        row = f"{model_cat}".ljust(12)
        for test_cat in cats:
            r, _ = evaluate_pair(model_cat, test_cat, extractor)
            results[(model_cat, test_cat)] = r
            row += f"{r['auroc']:.3f}".ljust(12) if r else "N/A".ljust(12)
        print(row)

    print("\n=== Full metrics (diagonal = in-category performance) ===")
    for (model_cat, test_cat), r in results.items():
        if r:
            print(f"{model_cat} -> {test_cat}: AUROC={r['auroc']:.3f} "
                  f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
                  f"threshold={r['threshold']:.2f} (n={r['n']})")