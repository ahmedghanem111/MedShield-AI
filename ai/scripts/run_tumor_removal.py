"""
tumor_removal attack: auto-locate the tumor and erase it with Big-LaMa inpainting.

Only tumor folders (glioma / meningioma / pituitary) are used. Each fake keeps the
split of its source image. Writes fake/tumor_removal/, fake/tumor_removal_masks/,
metadata/attacks_tumor_removal.csv, appends rows to metadata/manifest.csv
(original kept as metadata/manifest_v1.csv) and a preview grid in eda/.

Run from ai/:
  python scripts/run_tumor_removal.py --limit 24   # quick check, manifest untouched
  python scripts/run_tumor_removal.py --overwrite
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SEED, device, ensure_ai_cwd

ATTACK = "tumor_removal"
TUMOR_FOLDERS = ("glioma", "meningioma", "pituitary")
LAMA_PATH = Path("models/big-lama.pt")
LAMA_URL = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
MAX_SIDE = 1024
OUT_ROOT = Path("fake") / ATTACK
MASK_ROOT = Path("fake") / f"{ATTACK}_masks"


def brain_mask(g: np.ndarray) -> np.ndarray:
    m = g > max(15.0, 0.12 * float(g.max()))
    lab, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, lab, range(1, n + 1))
    m = lab == (int(np.argmax(sizes)) + 1)
    return ndi.binary_fill_holes(m)


def fallback_ellipse(inner: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = inner.shape
    ys, xs = np.nonzero(inner)
    k = int(rng.integers(0, len(ys)))
    cy, cx = ys[k], xs[k]
    s = min(h, w)
    ry, rx = rng.uniform(0.04, 0.10, size=2) * s
    theta = rng.uniform(0, np.pi)
    yy, xx = np.mgrid[0:h, 0:w]
    dy, dx = yy - cy, xx - cx
    u = dx * np.cos(theta) + dy * np.sin(theta)
    v = -dx * np.sin(theta) + dy * np.cos(theta)
    return ((u / rx) ** 2 + (v / ry) ** 2) <= 1.0


def tumor_mask(g: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, str]:
    """Largest bright anomaly inside the brain; random ellipse if none is plausible."""
    h, w = g.shape
    s_min = min(h, w)
    brain = brain_mask(g)
    inner = ndi.binary_erosion(brain, iterations=max(2, int(0.025 * s_min)))
    if inner.sum() < 0.05 * h * w:
        inner = brain
    if inner.sum() == 0:
        inner = np.ones_like(brain)

    smooth = ndi.gaussian_filter(g, sigma=max(1.5, s_min / 200))
    vals = smooth[inner]
    mu, sd = float(vals.mean()), float(vals.std()) + 1e-6
    cand = ndi.binary_opening(inner & (smooth > mu + 1.6 * sd), iterations=2)
    lab, n = ndi.label(cand)

    mask, source = None, "fallback"
    if n:
        idx = np.arange(1, n + 1)
        px = ndi.sum(cand, lab, idx)
        areas = px / max(inner.sum(), 1)
        means = ndi.mean(smooth, lab, idx)
        # scalp fat / neck tissue: close to the head outline and thin arc-shaped
        depth = ndi.mean(ndi.distance_transform_edt(brain), lab, idx)
        boxes = ndi.find_objects(lab)
        extent = np.array(
            [p / max((b[0].stop - b[0].start) * (b[1].stop - b[1].start), 1) for p, b in zip(px, boxes)]
        )
        ok = (areas >= 0.002) & (areas <= 0.15) & (depth >= 0.08 * s_min) & (extent >= 0.3)
        if ok.any():
            score = np.where(ok, areas * (means - mu), -np.inf)
            mask = lab == int(idx[int(np.argmax(score))])
            source = "auto"
    if mask is None:
        mask = fallback_ellipse(inner, rng)

    mask = ndi.binary_dilation(mask, iterations=max(3, int(0.012 * s_min)))
    mask = ndi.binary_fill_holes(mask) & ndi.binary_dilation(brain, iterations=3)
    return mask, source


class Lama:
    def __init__(self, path: Path, dev: torch.device):
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(LAMA_URL, str(path))
        self.dev = dev
        self.model = torch.jit.load(str(path), map_location=dev).eval()

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        scale = min(1.0, MAX_SIDE / max(h, w))
        img_in, m_in = rgb, mask
        if scale < 1.0:
            size = (int(w * scale), int(h * scale))
            img_in = np.asarray(Image.fromarray(rgb).resize(size, Image.BICUBIC))
            m_in = np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0
        ih, iw = m_in.shape
        ph, pw = (-ih) % 8, (-iw) % 8
        img_in = np.pad(img_in, ((0, ph), (0, pw), (0, 0)), mode="symmetric")
        m_in = np.pad(m_in, ((0, ph), (0, pw)), mode="symmetric")

        x = torch.from_numpy(img_in).permute(2, 0, 1)[None].float().div(255).to(self.dev)
        m = torch.from_numpy(m_in.astype(np.float32))[None, None].to(self.dev)
        out = self.model(x, m)[0].permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy()
        out = out[:ih, :iw]
        if scale < 1.0:
            out = np.asarray(Image.fromarray(out).resize((w, h), Image.BICUBIC))

        # MRI slices are grayscale; a colour tint in the fill would be a trivial shortcut.
        gray = out.astype(np.float32).mean(axis=2, keepdims=True).round().astype(np.uint8)
        result = rgb.copy()
        result[mask] = np.repeat(gray, 3, axis=2)[mask]
        return result


def save_jpg(img: Image.Image, path: Path) -> None:
    for _ in range(5):
        try:
            img.save(path, quality=95)
            return
        except PermissionError:
            time.sleep(0.3)
    img.save(path, quality=95)


def update_manifest(log: pd.DataFrame) -> None:
    manifest_path = Path("metadata/manifest.csv")
    backup = Path("metadata/manifest_v1.csv")
    if not backup.exists():
        shutil.copyfile(manifest_path, backup)
    freeze, freeze_backup = Path("metadata/data_freeze.json"), Path("metadata/data_freeze_v1.json")
    if freeze.exists() and not freeze_backup.exists():
        shutil.copyfile(freeze, freeze_backup)
    manifest = pd.read_csv(manifest_path)
    manifest = manifest[manifest.attack != ATTACK]
    rows = pd.DataFrame(
        {
            "path": log.fake_path,
            "label": "fake",
            "attack": ATTACK,
            "param": log.param,
            "folder": log.folder,
            "split": log.split,
            "seed": SEED,
            "held_out": False,
            "source_path": log.real_path,
            "donor_path": np.nan,
        }
    )
    manifest = pd.concat([manifest, rows[manifest.columns]], ignore_index=True)
    manifest.to_csv(manifest_path, index=False)
    print("manifest rows", len(manifest), "(backup:", backup.as_posix() + ")", flush=True)


def save_preview(log: pd.DataFrame, out: Path, per_class: int = 4) -> None:
    sample = log[log.split == "test"].sample(frac=1, random_state=SEED).groupby("folder").head(per_class)
    if sample.empty:
        sample = log.head(per_class * 3)
    fig, axes = plt.subplots(len(sample), 3, figsize=(9, 3 * len(sample)))
    axes = np.atleast_2d(axes)
    for r, (_, row) in enumerate(sample.iterrows()):
        real = np.asarray(Image.open(row.real_path).convert("RGB"))
        fake = np.asarray(Image.open(row.fake_path).convert("RGB"))
        mask = np.asarray(Image.open(row.mask_path)) > 0
        overlay = real.astype(np.float32) / 255
        overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array([1.0, 0.1, 0.1])
        for c, (im, title) in enumerate(
            [(real, f"real ({row.folder})"), (overlay, f"mask ({row.mask_source})"), (fake, ATTACK)]
        ):
            axes[r, c].imshow(im)
            axes[r, c].set_title(title, fontsize=9)
            axes[r, c].axis("off")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=90)
    plt.close(fig)
    print("preview", out.as_posix(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="process only N images; manifest is not touched")
    parser.add_argument("--overwrite", action="store_true", help="regenerate files that already exist")
    args = parser.parse_args()

    ensure_ai_cwd()
    split_df = pd.read_csv("metadata/split.csv")
    split_df = split_df[split_df.folder.isin(TUMOR_FOLDERS)].reset_index(drop=True)
    if args.limit:
        split_df = split_df.groupby("folder", group_keys=False).head(max(1, args.limit // 3))
    print(f"{ATTACK}: {len(split_df)} tumor images", split_df.split.value_counts().to_dict(), flush=True)

    dev = device()
    lama = Lama(LAMA_PATH, dev)
    print("LaMa on", dev, flush=True)

    records, made, t0 = [], 0, time.time()
    for i, row in enumerate(split_df.itertuples(index=False)):
        src = Path(row.path)
        rel = src.relative_to("data")
        fake_path = OUT_ROOT / rel
        mask_path = (MASK_ROOT / rel).with_suffix(".png")
        fake_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)

        rgb = np.asarray(Image.open(src).convert("RGB"))
        g = rgb.astype(np.float32).mean(axis=2)
        rng = np.random.default_rng(SEED + i)
        mask, source = tumor_mask(g, rng)

        if args.overwrite or not (fake_path.exists() and mask_path.exists()):
            save_jpg(Image.fromarray(lama(rgb, mask)), fake_path)
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            made += 1

        records.append(
            {
                "real_path": row.path,
                "fake_path": fake_path.as_posix(),
                "mask_path": mask_path.as_posix(),
                "attack": ATTACK,
                "param": round(float(mask.mean()), 5),
                "mask_source": source,
                "folder": row.folder,
                "split": row.split,
                "seed": SEED,
            }
        )
        if (i + 1) % 200 == 0:
            rate = (time.time() - t0) / (i + 1)
            eta = rate * (len(split_df) - i - 1) / 60
            print(f"{i + 1}/{len(split_df)}  new={made}  eta={eta:.1f} min", flush=True)

    log = pd.DataFrame(records)
    print("mask source:", log.mask_source.value_counts().to_dict(), flush=True)
    print("mask area frac: median", round(float(log.param.median()), 4), flush=True)
    save_preview(log, Path("eda") / f"{ATTACK}_samples.png")

    if args.limit:
        print(f"limit run: {made} new files, manifest unchanged", flush=True)
        return
    log.to_csv(f"metadata/attacks_{ATTACK}.csv", index=False)
    update_manifest(log)
    print(f"done {len(log)} rows, new files {made}, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
