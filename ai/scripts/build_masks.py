"""
Ground-truth manipulation masks for the localizer.

tumor_removal already has exact masks (fake/tumor_removal_masks/). For the other
attacks the region was not logged, so the mask is where the fake differs from its
source: smoothed |fake - real| > DIFF_THR gray levels. JPEG (held out) is skipped.

Run from ai/:
  python scripts/build_masks.py
"""
from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ensure_ai_cwd, load_manifest
from localizer import mask_path_for

DIFF_THR = 8.0
DIFF_ATTACKS = ("gaussian_noise", "gamma", "copy_region", "lesion_insert")


def _gray(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def diff_mask(task: tuple[str, str, str]) -> tuple[str, float]:
    fake_path, real_path, out_path = task
    fake, real = _gray(fake_path), _gray(real_path)
    if fake.shape != real.shape:
        fake = np.asarray(Image.fromarray(fake.astype(np.uint8)).resize(real.shape[::-1], Image.BILINEAR), np.float32)
    d = ndi.gaussian_filter(np.abs(fake - real), sigma=2)
    m = ndi.binary_opening(d > DIFF_THR, iterations=1)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(m.astype(np.uint8) * 255).save(out)
    return fake_path, float(m.mean())


def main():
    ensure_ai_cwd()
    m = load_manifest()
    fakes = m[m.attack.isin(DIFF_ATTACKS)]
    tasks = [(r.path, r.source_path, mask_path_for(r.path, r.attack).as_posix()) for r in fakes.itertuples()]
    print(f"diff masks: {len(tasks)}", fakes.attack.value_counts().to_dict(), flush=True)

    t0, areas = time.time(), {}
    with Pool(6) as pool:
        for i, (path, area) in enumerate(pool.imap_unordered(diff_mask, tasks, chunksize=64), 1):
            areas[path] = area
            if i % 5000 == 0:
                print(f"{i}/{len(tasks)}  {(time.time() - t0) / 60:.1f} min", flush=True)

    fakes = fakes.assign(mask_area=fakes.path.map(areas))
    tr = m[m.attack == "tumor_removal"]
    missing = sum(not mask_path_for(p, "tumor_removal").exists() for p in tr.path)
    print("tumor_removal exact masks missing:", missing, flush=True)
    summary = fakes.groupby("attack").mask_area.describe(percentiles=[0.05, 0.5, 0.95])
    print(summary.round(4).to_string(), flush=True)
    print("empty masks per attack:", fakes[fakes.mask_area == 0].attack.value_counts().to_dict(), flush=True)
    print(f"done {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
