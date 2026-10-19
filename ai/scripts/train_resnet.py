"""
ResNet18 binary real/fake trainer.

Examples (from ai/):
  python scripts/train_resnet.py --smoke
  python scripts/train_resnet.py --epochs 8 --batch-size 32
  python scripts/train_resnet.py --init-from models/best_model.pt --epochs 3 --lr 3e-5 \
      --out models/best_model_ft.pt --history models/train_history_ft.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import device, ensure_ai_cwd, freeze_snapshot, load_manifest, set_seed
from dataset import make_loader, split_frames


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


@torch.no_grad()
def evaluate(model, loader, dev) -> dict:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y, _ in loader:
        x = x.to(dev)
        y = y.to(dev)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += float(loss.item()) * len(y)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        n += len(y)
    return {"loss": total_loss / max(n, 1), "acc": correct / max(n, 1), "n": n}


def train_one_epoch(model, loader, opt, criterion, dev) -> dict:
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y, _ in tqdm(loader, desc="train", leave=False):
        x = x.to(dev)
        y = y.to(dev)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
        total_loss += float(loss.item()) * len(y)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        n += len(y)
    return {"loss": total_loss / max(n, 1), "acc": correct / max(n, 1), "n": n}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="1 epoch, small subset, write smoke checkpoint")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--smoke-n", type=int, default=512, help="per-split max rows for smoke")
    parser.add_argument("--out", type=str, default="models/best_model.pt")
    parser.add_argument("--history", type=str, default="models/train_history.json")
    parser.add_argument("--init-from", type=str, default="", help="fine-tune from this checkpoint")
    args = parser.parse_args()

    ensure_ai_cwd()
    set_seed(42)
    Path("models").mkdir(exist_ok=True)

    freeze = freeze_snapshot()
    print("data freeze written:", freeze["by_split_label"])

    manifest = load_manifest()
    train_df, val_df, test_df = split_frames(manifest)

    if args.smoke:
        train_df = train_df.sample(n=min(args.smoke_n, len(train_df)), random_state=42)
        val_df = val_df.sample(n=min(args.smoke_n // 4, len(val_df)), random_state=42)
        epochs = 1
        out_path = Path("models/smoke_checkpoint.pt")
        print(f"SMOKE: train={len(train_df)} val={len(val_df)} epochs=1")
    else:
        epochs = args.epochs
        out_path = Path(args.out)
        print(f"FULL: train={len(train_df)} val={len(val_df)} test={len(test_df)} epochs={epochs}")

    train_loader = make_loader(train_df, train=True, batch_size=args.batch_size)
    val_loader = make_loader(val_df, train=False, batch_size=args.batch_size)

    dev = device()
    print("device:", dev)
    if args.init_from:
        model = build_model(pretrained=False)
        init_ckpt = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(init_ckpt["state_dict"])
        print("init from", args.init_from, "best_val_acc", init_ckpt.get("best_val_acc"))
    else:
        model = build_model(pretrained=True)
    model = model.to(dev)
    criterion = nn.CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=max(epochs, 1))

    history = []
    best_val = -1.0
    best_state = None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(model, train_loader, opt, criterion, dev)
        va = evaluate(model, val_loader, dev)
        sched.step()
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_acc": tr["acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
        }
        history.append(row)
        print(
            f"epoch {epoch}/{epochs}  "
            f"train_loss={tr['loss']:.4f} acc={tr['acc']:.4f}  "
            f"val_loss={va['loss']:.4f} acc={va['acc']:.4f}"
        )
        if va["acc"] >= best_val:
            best_val = va["acc"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0
    assert best_state is not None
    model.load_state_dict(best_state)

    payload = {
        "model": "resnet18",
        "num_classes": 2,
        "img_size": 224,
        "label_map": {"real": 0, "fake": 1},
        "seed": 42,
        "smoke": bool(args.smoke),
        "epochs_ran": epochs,
        "init_from": args.init_from or None,
        "attacks_in_train": sorted(train_df.attack.unique().tolist()),
        "best_val_acc": best_val,
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "elapsed_sec": elapsed,
        "state_dict": best_state,
    }
    torch.save(payload, out_path)
    hist_path = Path(args.history if not args.smoke else "models/smoke_history.json")
    hist_path.write_text(
        json.dumps({"history": history, "best_val_acc": best_val, "elapsed_sec": elapsed}, indent=2),
        encoding="utf-8",
    )
    print(f"saved {out_path}  best_val_acc={best_val:.4f}  elapsed={elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
