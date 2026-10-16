"""
Final evaluation for frozen ResNet18:
ROC, confusion matrix, per-attack accuracy, AUC 95% bootstrap CI,
JPEG held-out, Grad-CAM samples -> metrics/metrics.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torchvision import models, transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IMG_SIZE, device, ensure_ai_cwd, load_manifest, set_seed
from dataset import make_loader


def build_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def load_checkpoint(path: str | Path, dev: torch.device) -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model()
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev)
    model.eval()
    return model, ckpt


@torch.no_grad()
def collect_preds(model, frame: pd.DataFrame, batch_size: int, dev: torch.device):
    loader = make_loader(frame, train=False, batch_size=batch_size, shuffle=False)
    probs, labels, attacks, paths = [], [], [], []
    for x, y, meta in tqdm(loader, desc="eval", leave=False):
        x = x.to(dev)
        logits = model(x)
        p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        probs.append(p)
        labels.append(y.numpy())
        attacks.extend(meta["attack"])
        paths.extend(meta["path"])
    return (
        np.concatenate(probs),
        np.concatenate(labels),
        np.asarray(attacks),
        np.asarray(paths),
    )


def bootstrap_auc_ci(y_true, y_prob, n_boot: int = 1000, seed: int = 42, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return None, None, None
    base = float(roc_auc_score(y_true, y_prob))
    scores = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(roc_auc_score(yt, yp))
    if not scores:
        return base, None, None
    lo, hi = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return base, float(lo), float(hi)


def metrics_block(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    auc, lo, hi = bootstrap_auc_ci(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "roc_auc_ci95": [lo, hi] if lo is not None else None,
        "confusion_matrix": {"labels": ["real", "fake"], "matrix": cm},
    }


def per_attack_accuracy(y_true, y_prob, attacks, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    out = {}
    for attack in sorted(set(attacks.tolist())):
        mask = attacks == attack
        if mask.sum() == 0:
            continue
        # for attack==none (reals) and fakes of that attack
        yt, yp = y_true[mask], y_pred[mask]
        out[attack] = {
            "n": int(mask.sum()),
            "accuracy": float(accuracy_score(yt, yp)),
        }
    return out


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]
        score.backward()
        grads = self.gradients[0]  # C,H,W
        acts = self.activations[0]
        weights = grads.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(
            cam[None, None], size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
        )[0, 0]
        return cam.cpu().numpy()


def save_gradcam_grid(model, frame: pd.DataFrame, out_dir: Path, n: int = 4, dev=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    cam_engine = GradCAM(model, model.layer4[-1])
    tfm = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    sample = frame.sample(n=min(n, len(frame)), random_state=42)
    paths = []
    for i, (_, row) in enumerate(sample.iterrows()):
        img = Image.open(row.path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(dev)
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            heat = cam_engine(x, class_idx=1 if row.label == "fake" else 0)
        disp = np.asarray(img.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(disp)
        axes[0].set_title(f"{row.label}/{row.attack}")
        axes[0].axis("off")
        axes[1].imshow(disp)
        axes[1].imshow(heat, cmap="jet", alpha=0.45)
        axes[1].set_title("Grad-CAM")
        axes[1].axis("off")
        fig.tight_layout()
        out = out_dir / f"gradcam_{i}_{row.label}_{row.attack}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        paths.append(out.as_posix())
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="models/best_model.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", type=str, default="metrics/metrics.json")
    args = parser.parse_args()

    ensure_ai_cwd()
    set_seed(42)
    Path("metrics").mkdir(exist_ok=True)
    Path("metrics/gradcam").mkdir(parents=True, exist_ok=True)

    dev = device()
    model, ckpt = load_checkpoint(args.checkpoint, dev)
    print("loaded", args.checkpoint, "best_val_acc", ckpt.get("best_val_acc"))

    manifest = load_manifest()
    use = manifest[~manifest.held_out]
    test_df = use[use.split == "test"]
    jpeg_df = manifest[(manifest.attack == "jpeg") | ((manifest.split == "test") & (manifest.label == "real"))]
    # held-out protocol: jpeg fakes vs same-split test reals
    jpeg_fakes = manifest[manifest.attack == "jpeg"]
    test_reals = use[(use.split == "test") & (use.label == "real")]
    jpeg_eval = pd.concat([test_reals, jpeg_fakes], ignore_index=True)

    probs, labels, attacks, paths = collect_preds(model, test_df, args.batch_size, dev)
    overall = metrics_block(labels, probs)
    by_attack = per_attack_accuracy(labels, probs, attacks)

    # also accuracy when grouping fake attacks only + reals
    fake_mask = labels == 1
    attack_detect = {}
    for attack in sorted(set(attacks[fake_mask].tolist())):
        # reals + this attack's fakes
        m = (labels == 0) | (attacks == attack)
        attack_detect[attack] = metrics_block(labels[m], probs[m])

    jpeg_probs, jpeg_labels, jpeg_attacks, _ = collect_preds(model, jpeg_eval, args.batch_size, dev)
    jpeg_metrics = metrics_block(jpeg_labels, jpeg_probs)

    # plots
    fpr, tpr, _ = roc_curve(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC={overall['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Test ROC (usable attacks)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("metrics/roc_curve.png", dpi=140)
    plt.close(fig)

    cm = np.array(overall["confusion_matrix"]["matrix"])
    fig, ax = plt.subplots(figsize=(4, 3.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["real", "fake"])
    ax.set_yticks([0, 1], ["real", "fake"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_title("Confusion matrix (test)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig("metrics/confusion_matrix.png", dpi=140)
    plt.close(fig)

    gradcam_paths = save_gradcam_grid(
        model,
        test_df[test_df.label == "fake"],
        Path("metrics/gradcam"),
        n=6,
        dev=dev,
    )

    report = {
        "checkpoint": args.checkpoint,
        "model": ckpt.get("model", "resnet18"),
        "seed": 42,
        "label_map": {"real": 0, "fake": 1},
        "test_overall": overall,
        "per_attack_row_accuracy": by_attack,
        "per_fake_attack_vs_reals": attack_detect,
        "jpeg_held_out": jpeg_metrics,
        "baseline_ref": "models/baseline_hist_fft.json",
        "artifacts": {
            "roc_curve": "metrics/roc_curve.png",
            "confusion_matrix": "metrics/confusion_matrix.png",
            "gradcam": gradcam_paths,
        },
        "notes": [
            "Main test metrics exclude held_out JPEG rows.",
            "jpeg_held_out evaluates JPEG fakes against test reals only.",
            "AUC CI is 95% bootstrap with 1000 resamples.",
        ],
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("test_overall", "jpeg_held_out", "per_attack_row_accuracy")}, indent=2))
    print("saved", args.out)


if __name__ == "__main__":
    main()
