"""
train.py
Fits PaDiM statistics (mean + inverse covariance) for each category using
ONLY the 'good' training images (data/mvtec/<category>/train/good/*.png).

Usage:
    python src/train.py --categories bottle cable tile
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from padim_core import FeatureExtractor, load_image, fit_gaussian, save_stats, DEVICE

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "mvtec")
MODELS_ROOT = os.path.join(os.path.dirname(__file__), "..", "models")


def train_category(category: str, extractor: FeatureExtractor, batch_size: int = 8):
    good_dir = os.path.join(DATA_ROOT, category, "train", "good")
    if not os.path.isdir(good_dir):
        raise FileNotFoundError(
            f"Expected training images at: {good_dir}\n"
            f"Check that you copied the MVTec '{category}' folder into data/mvtec/"
        )

    image_files = sorted(
        f for f in os.listdir(good_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    print(f"[{category}] found {len(image_files)} good training images")

    all_embeddings = []
    batch = []

    def flush(batch_tensors):
        if not batch_tensors:
            return
        x = torch.stack(batch_tensors)
        emb = extractor(x)
        all_embeddings.append(emb.cpu().numpy())

    for fname in tqdm(image_files, desc=f"Extracting features [{category}]"):
        img_tensor = load_image(os.path.join(good_dir, fname))
        batch.append(img_tensor)
        if len(batch) == batch_size:
            flush(batch)
            batch = []
    flush(batch)

    embeddings = np.concatenate(all_embeddings, axis=0)  # (N, C, H, W)
    print(f"[{category}] embedding tensor shape: {embeddings.shape}")

    print(f"[{category}] fitting Gaussian per spatial location (this is the slow step)...")
    mean, inv_cov, hw = fit_gaussian(embeddings)

    os.makedirs(MODELS_ROOT, exist_ok=True)
    out_path = os.path.join(MODELS_ROOT, f"{category}_padim.pkl")
    save_stats(out_path, mean, inv_cov, hw)
    print(f"[{category}] saved stats -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=["bottle", "cable", "tile"])
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    extractor = FeatureExtractor()

    for cat in args.categories:
        train_category(cat, extractor)

    print("\nAll categories trained. Stats saved in models/")
