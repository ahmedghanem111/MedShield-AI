"""
Research baseline: Histogram + FFT high-freq ratio features + LogisticRegression.

Does not replace ResNet18; for comparison only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ensure_ai_cwd, load_manifest, set_seed
from dataset import split_frames

HF_RADIUS_FRAC = 0.35
HIST_BINS = 32
TARGET = 128


def load_gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L").resize((TARGET, TARGET)), dtype=np.float32)


def hist_features(gray: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(gray, bins=HIST_BINS, range=(0, 255), density=True)
    return hist.astype(np.float64)


def fft_hf_ratio(gray: np.ndarray) -> float:
    x = gray - gray.mean()
    spec = np.fft.fftshift(np.fft.fft2(x))
    mag = np.abs(spec)
    power = mag ** 2
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    high = r >= (HF_RADIUS_FRAC * r.max())
    return float(power[high].sum() / (power.sum() + 1e-12))


def featurize(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for _, row in tqdm(frame.iterrows(), total=len(frame), desc="features", leave=False):
        gray = load_gray(row.path)
        feat = np.concatenate([hist_features(gray), [fft_hf_ratio(gray), gray.mean() / 255.0]])
        xs.append(feat)
        ys.append(0 if row.label == "real" else 1)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-train", type=int, default=0, help="0 = all usable train rows")
    parser.add_argument("--out", type=str, default="models/baseline_hist_fft.json")
    args = parser.parse_args()

    ensure_ai_cwd()
    set_seed(42)
    Path("models").mkdir(exist_ok=True)

    manifest = load_manifest()
    train_df, val_df, test_df = split_frames(manifest)
    if args.max_train and args.max_train < len(train_df):
        train_df = train_df.sample(n=args.max_train, random_state=42)

    print(f"baseline features: train={len(train_df)} val={len(val_df)} test={len(test_df)}")
    X_train, y_train = featurize(train_df)
    X_val, y_val = featurize(val_df)
    X_test, y_test = featurize(test_df)

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)

    def pack(name, X, y):
        prob = clf.predict_proba(X)[:, 1]
        pred = (prob >= 0.5).astype(int)
        return {
            "split": name,
            "n": int(len(y)),
            "accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred)),
            "roc_auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else None,
        }

    report = {
        "model": "logistic_regression_hist32_fft_hf_mean",
        "feature_dim": int(X_train.shape[1]),
        "train": pack("train", X_train, y_train),
        "val": pack("val", X_val, y_val),
        "test": pack("test", X_test, y_test),
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("saved", args.out)


if __name__ == "__main__":
    main()
