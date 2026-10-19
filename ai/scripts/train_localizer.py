"""
Train the manipulation localizer (pixel mask + image-level real/fake head).

Each epoch draws a balanced sample: 1/2 reals, 1/4 tumor_removal, 1/4 other attacks,
so "call everything fake" is no longer a cheap way to lower the loss.
Best epoch = mean of val AUC (all attacks) and val AUC (tumor_removal vs reals).

Run from ai/:
  python scripts/build_masks.py                      # once
  python scripts/train_localizer.py --smoke
  python scripts/train_localizer.py --epochs 6
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import SEED, device, ensure_ai_cwd, load_manifest, set_seed
from localizer import ARCH, LOC_SIZE, MASK_STRIDE, TOPK_FRAC, LocDataset, Localizer, predict


def balanced_sampler(frame, num_samples: int) -> WeightedRandomSampler:
    real = (frame.label == "real").to_numpy()
    tr = (frame.attack == "tumor_removal").to_numpy()
    other = ~real & ~tr
    w = np.where(real, 0.5 / real.sum(), np.where(tr, 0.25 / max(tr.sum(), 1), 0.25 / max(other.sum(), 1)))
    gen = torch.Generator().manual_seed(SEED)
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples, replacement=True, generator=gen)


def loss_fn(mask_logits, cls_logit, mask, y):
    bce_pix = F.binary_cross_entropy_with_logits(mask_logits, mask)
    fake = y > 0.5
    if fake.any():
        p = torch.sigmoid(mask_logits).flatten(1)[fake]
        t = mask.flatten(1)[fake]
        dice = (1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)).mean()
    else:
        dice = mask_logits.new_zeros(())
    bce_cls = F.binary_cross_entropy_with_logits(cls_logit, y)
    return bce_pix + dice + 0.5 * bce_cls, {"pix": bce_pix.item(), "dice": dice.item(), "cls": bce_cls.item()}


def val_summary(pred: dict) -> dict:
    y, s, attack = pred["label"], pred["combined"], pred["attack"]
    real = y == 0
    tr = attack == "tumor_removal"
    out = {
        "auc_all": float(roc_auc_score(y, s)),
        "auc_tumor_removal": float(roc_auc_score(y[real | tr], s[real | tr])) if tr.any() else float("nan"),
        "real_acc@0.5": float((s[real] < 0.5).mean()),
        "tumor_removal_caught@0.5": float((s[tr] >= 0.5).mean()) if tr.any() else float("nan"),
        "tumor_removal_iou": float(np.nanmean(pred["iou"][tr])) if tr.any() else float("nan"),
    }
    out["select"] = 0.5 * (out["auc_all"] + out["auc_tumor_removal"])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4, help="decoder/heads; encoder uses lr/3")
    parser.add_argument("--samples-per-epoch", type=int, default=14000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--init-from", type=str, default="models/best_model.pt", help="ResNet18 classifier for the encoder")
    parser.add_argument("--out", type=str, default="models/localizer.pt")
    parser.add_argument("--history", type=str, default="models/localizer_history.json")
    parser.add_argument("--smoke", action="store_true", help="1 short epoch on a small val subset")
    args = parser.parse_args()

    ensure_ai_cwd()
    set_seed(SEED)
    manifest = load_manifest()
    use = manifest[~manifest.held_out]
    train_df = use[use.split == "train"]
    val_df = use[use.split == "val"]
    epochs, n_per_epoch, out_path = args.epochs, args.samples_per_epoch, Path(args.out)
    if args.smoke:
        epochs, n_per_epoch, out_path = 1, 320, Path("models/localizer_smoke.pt")
        val_df = val_df.groupby("attack").head(40)
    print(f"train rows={len(train_df)} val rows={len(val_df)} epochs={epochs} samples/epoch={n_per_epoch}", flush=True)

    loader_kw = dict(num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(
        LocDataset(train_df, train=True),
        batch_size=args.batch_size,
        sampler=balanced_sampler(train_df, n_per_epoch),
        drop_last=True,
        **loader_kw,
    )
    val_loader = DataLoader(LocDataset(val_df), batch_size=args.batch_size * 2, shuffle=False, **loader_kw)

    dev = device()
    model = Localizer()
    if args.init_from:
        init = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_resnet_encoder(init["state_dict"])
        print("encoder from", args.init_from, flush=True)
    model.to(dev)
    print("device:", dev, flush=True)

    enc = [p for n, p in model.named_parameters() if n.startswith(("stem.", "layer"))]
    dec = [p for n, p in model.named_parameters() if not n.startswith(("stem.", "layer"))]
    opt = AdamW([{"params": enc, "lr": args.lr / 3}, {"params": dec, "lr": args.lr}], weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=max(epochs, 1))

    history, best, best_state, t0 = [], -1.0, None, time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        sums, n = {"loss": 0.0, "pix": 0.0, "dice": 0.0, "cls": 0.0}, 0
        for x, mask, y, _, _ in tqdm(train_loader, desc=f"train {epoch}", leave=False, mininterval=30):
            x, mask, y = x.to(dev, non_blocking=True), mask.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            mask_logits, cls_logit = model(x)
            loss, parts = loss_fn(mask_logits, cls_logit, mask, y)
            loss.backward()
            opt.step()
            sums["loss"] += loss.item()
            for k, v in parts.items():
                sums[k] += v
            n += 1
        sched.step()
        va = val_summary(predict(model, val_loader, dev))
        row = {"epoch": epoch, **{f"train_{k}": v / max(n, 1) for k, v in sums.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)
        print(
            f"epoch {epoch}/{epochs}  loss={row['train_loss']:.4f}  "
            f"val auc_all={va['auc_all']:.4f} auc_tr={va['auc_tumor_removal']:.4f}  "
            f"real_acc={va['real_acc@0.5']:.3f} tr_caught={va['tumor_removal_caught@0.5']:.3f}  "
            f"tr_iou={va['tumor_removal_iou']:.3f}  ({(time.time() - t0) / 60:.1f} min)",
            flush=True,
        )
        if best_state is None or va["select"] >= best:
            best = va["select"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_row = row

    elapsed = time.time() - t0
    torch.save(
        {
            "arch": ARCH,
            "loc_size": LOC_SIZE,
            "mask_stride": MASK_STRIDE,
            "topk_frac": TOPK_FRAC,
            "label_map": {"real": 0, "fake": 1},
            "seed": SEED,
            "init_from": args.init_from or None,
            "epochs_ran": epochs,
            "best_epoch": best_row["epoch"],
            "best_val": {k: v for k, v in best_row.items() if k.startswith("val_")},
            "elapsed_sec": elapsed,
            "state_dict": best_state,
        },
        out_path,
    )
    if not args.smoke:
        Path(args.history).write_text(json.dumps({"history": history, "elapsed_sec": elapsed}, indent=2), encoding="utf-8")
    print(f"saved {out_path}  best_epoch={best_row['epoch']}  elapsed={elapsed / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
