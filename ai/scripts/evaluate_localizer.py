"""
Evaluate the manipulation localizer -> metrics/metrics_localizer.json + plots.

Decision threshold is fixed on val so that SPEC of val reals are called real,
then applied unchanged to test (and JPEG held-out).

Run from ai/:
  python scripts/evaluate_localizer.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SEED, device, ensure_ai_cwd, load_manifest, set_seed
from evaluate import bootstrap_auc_ci
from localizer import LOC_SIZE, MEAN, STD, LocDataset, load_localizer, mask_path_for, predict

SPEC = 0.90
SCORES = ("cls", "mask_topk", "combined")


def run(model, frame, dev, workers, batch_size):
    loader = DataLoader(LocDataset(frame), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)
    return predict(model, loader, dev)


def detection_block(pred: dict, score: str, thr: float, folders: np.ndarray) -> dict:
    y, s, attack = pred["label"], pred[score], pred["attack"]
    real = y == 0
    auc, lo, hi = bootstrap_auc_ci(y, s)
    yhat = s >= thr
    tp, fp = int((yhat & ~real).sum()), int((yhat & real).sum())
    fn, tn = int((~yhat & ~real).sum()), int((~yhat & real).sum())
    block = {
        "n": int(len(y)),
        "roc_auc": auc,
        "roc_auc_ci95": [lo, hi] if lo is not None else None,
        "accuracy": (tp + tn) / len(y),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "confusion_matrix": {"labels": ["real", "fake"], "matrix": [[tn, fp], [fn, tp]]},
        "real_acc": float((~yhat[real]).mean()),
        "real_acc_by_folder": {f: float((~yhat[real & (folders == f)]).mean()) for f in sorted(set(folders[real]))},
        "per_attack": {},
    }
    for a in sorted(set(attack[~real])):
        m = attack == a
        iou = pred["iou"][m]
        block["per_attack"][a] = {
            "n": int(m.sum()),
            "detected": float(yhat[m].mean()),
            "auc_vs_reals": float(roc_auc_score(y[real | m], s[real | m])),
            "pixel_iou": float(np.nanmean(iou)) if np.isfinite(iou).any() else None,
        }
    return block


def save_heatmaps(model, frame: pd.DataFrame, dev, out: Path, score: str) -> None:
    import torch

    from localizer import image_scores, preprocess

    rows = pd.concat(
        [
            frame[frame.attack == "tumor_removal"].sample(n=6, random_state=SEED),
            frame[frame.attack == "lesion_insert"].sample(n=2, random_state=SEED),
            frame[frame.attack == "copy_region"].sample(n=2, random_state=SEED),
            frame[frame.label == "real"].sample(n=2, random_state=SEED),
        ]
    )
    fig, axes = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
    mean, std = np.array(MEAN), np.array(STD)
    for r, (_, row) in enumerate(rows.iterrows()):
        x = preprocess(Image.open(row.path))
        with torch.no_grad():
            mask_logits, cls_logit = model(x[None].to(dev))
            prob = float(image_scores(mask_logits, cls_logit)[score][0])
            heat = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()
        img = np.clip(x.permute(1, 2, 0).numpy() * std + mean, 0, 1)
        mp = mask_path_for(row.path, row.attack)
        gt = (
            np.asarray(Image.open(mp).convert("L").resize((LOC_SIZE, LOC_SIZE), Image.NEAREST)) > 127
            if mp is not None and mp.exists()
            else np.zeros((LOC_SIZE, LOC_SIZE), bool)
        )
        heat_up = np.asarray(Image.fromarray((heat * 255).astype(np.uint8)).resize((LOC_SIZE, LOC_SIZE), Image.BILINEAR)) / 255
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(f"{row.label}/{row.attack}", fontsize=9)
        axes[r, 1].imshow(img)
        axes[r, 1].imshow(np.ma.masked_where(~gt, gt), cmap="autumn", alpha=0.5)
        axes[r, 1].set_title("ground-truth region", fontsize=9)
        axes[r, 2].imshow(img)
        axes[r, 2].imshow(heat_up, cmap="jet", alpha=0.45, vmin=0, vmax=1)
        axes[r, 2].set_title(f"predicted  p(fake)={prob:.2f}", fontsize=9)
        for a in axes[r]:
            a.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/localizer.pt")
    parser.add_argument("--out", type=str, default="metrics/metrics_localizer.json")
    parser.add_argument("--artifacts-dir", type=str, default="metrics/localizer")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    ensure_ai_cwd()
    set_seed(SEED)
    art = Path(args.artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    dev = device()
    model, ckpt = load_localizer(args.checkpoint, dev)
    print("loaded", args.checkpoint, "best_epoch", ckpt.get("best_epoch"), flush=True)

    manifest = load_manifest()
    use = manifest[~manifest.held_out]
    val_df, test_df = use[use.split == "val"], use[use.split == "test"]
    jpeg_df = pd.concat([test_df[test_df.label == "real"], manifest[manifest.attack == "jpeg"]], ignore_index=True)
    folder_of = manifest.set_index("path").folder

    val = run(model, val_df, dev, args.workers, args.batch_size)
    val_auc = {s: float(roc_auc_score(val["label"], val[s])) for s in SCORES}
    tr = val["attack"] == "tumor_removal"
    real = val["label"] == 0
    val_auc_tr = {s: float(roc_auc_score(val["label"][real | tr], val[s][real | tr])) for s in SCORES}
    score = max(SCORES, key=lambda s: val_auc[s] + val_auc_tr[s])
    thr = float(np.quantile(val[score][real], SPEC))
    print("val auc", val_auc, "val auc tumor_removal", val_auc_tr, f"-> score={score} thr={thr:.4f}", flush=True)

    test = run(model, test_df, dev, args.workers, args.batch_size)
    jpeg = run(model, jpeg_df, dev, args.workers, args.batch_size)
    test_folders = folder_of.loc[test["path"]].to_numpy()
    jpeg_folders = folder_of.loc[jpeg["path"]].to_numpy()

    test_block = detection_block(test, score, thr, test_folders)
    jpeg_block = detection_block(jpeg, score, thr, jpeg_folders)
    test_auc_by_score = {s: float(roc_auc_score(test["label"], test[s])) for s in SCORES}

    fig, ax = plt.subplots(figsize=(5, 4))
    y, s = test["label"], test[score]
    for name, m in [("all attacks", np.ones_like(y, bool)), ("tumor_removal", (y == 0) | (test["attack"] == "tumor_removal"))]:
        fpr, tpr, _ = roc_curve(y[m], s[m])
        ax.plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y[m], s[m]):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Localizer test ROC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(art / "roc_curve.png", dpi=140)
    plt.close(fig)
    save_heatmaps(model, test_df, dev, art / "heatmaps.png", score)

    report = {
        "checkpoint": args.checkpoint,
        "arch": ckpt.get("arch"),
        "loc_size": ckpt.get("loc_size"),
        "best_epoch": ckpt.get("best_epoch"),
        "image_score": score,
        "threshold": thr,
        "threshold_rule": f"quantile {SPEC} of val real scores (val real acc = {SPEC})",
        "val_auc_by_score": val_auc,
        "val_auc_tumor_removal_by_score": val_auc_tr,
        "test_auc_by_score": test_auc_by_score,
        "test": test_block,
        "jpeg_held_out": jpeg_block,
        "artifacts": {"roc_curve": (art / "roc_curve.png").as_posix(), "heatmaps": (art / "heatmaps.png").as_posix()},
        "notes": [
            "Masks: exact for tumor_removal, smoothed |fake - source| > 8 for other attacks.",
            "pixel_iou is per image at mask prob > 0.5 on the 1/4-resolution mask grid.",
            "JPEG is never trained on; evaluated as JPEG fakes vs test reals.",
        ],
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    brief = {
        "score": score,
        "threshold": round(thr, 4),
        "test_auc": round(test_block["roc_auc"], 4),
        "real_acc": round(test_block["real_acc"], 4),
        "real_acc_by_folder": {k: round(v, 3) for k, v in test_block["real_acc_by_folder"].items()},
        "per_attack": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in test_block["per_attack"].items()},
        "jpeg_auc": round(jpeg_block["roc_auc"], 4),
        "jpeg_detected": round(jpeg_block["per_attack"]["jpeg"]["detected"], 4),
    }
    print(json.dumps(brief, indent=2), flush=True)
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
