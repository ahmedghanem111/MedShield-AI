import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

AI_ROOT = Path(__file__).resolve().parent.parent
os.chdir(AI_ROOT)

split_df = pd.read_csv("metadata/split.csv")
SEED = 42
SIGMA = 20
out_root = Path("fake")


def add_noise(img, rng):
    arr = np.asarray(img).astype(np.float32)
    noisy = arr + rng.normal(0, SIGMA, arr.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8))


def shift_gamma(img, gamma):
    arr = np.asarray(img).astype(np.float32) / 255.0
    out = np.clip(arr ** gamma, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))


def save_jpg(img, path):
    for _ in range(5):
        try:
            img.save(path, quality=95)
            return
        except PermissionError:
            time.sleep(0.3)
    img.save(path, quality=95)


rng = np.random.default_rng(SEED)
records = []
made = 0
for i, row in split_df.iterrows():
    src = Path(row.path)
    rel = src.relative_to("data")
    noise_path = out_root / "gaussian_noise" / rel
    gamma_path = out_root / "gamma" / rel
    noise_path.parent.mkdir(parents=True, exist_ok=True)
    gamma_path.parent.mkdir(parents=True, exist_ok=True)
    if noise_path.exists() and gamma_path.exists():
        gamma = float(rng.choice([0.55, 1.8]))
        with Image.open(noise_path) as done:
            shape = np.asarray(done).shape
        rng.normal(0, SIGMA, shape)
    else:
        img = Image.open(src).convert("RGB")
        gamma = float(rng.choice([0.55, 1.8]))
        save_jpg(add_noise(img, rng), noise_path)
        save_jpg(shift_gamma(img, gamma), gamma_path)
        made += 1
    records.append({
        "real_path": row.path,
        "fake_path": noise_path.as_posix(),
        "attack": "gaussian_noise",
        "param": SIGMA,
        "folder": row.folder,
        "split": row.split,
        "seed": SEED,
    })
    records.append({
        "real_path": row.path,
        "fake_path": gamma_path.as_posix(),
        "attack": "gamma",
        "param": gamma,
        "folder": row.folder,
        "split": row.split,
        "seed": SEED,
    })
    if (i + 1) % 1000 == 0:
        print(i + 1, "images, new", made, flush=True)

log = pd.DataFrame(records)
log.to_csv("metadata/attacks_noise_gamma.csv", index=False)
print("saved", len(log), "fake rows, new files", made, flush=True)
print(log.attack.value_counts().to_string(), flush=True)
print(log.groupby(["split", "attack"]).size().to_string(), flush=True)
