"""
Manipulation localizer: ResNet18 encoder + light FPN decoder (mask at 1/4 res)
plus an image-level real/fake head. Shared by train / evaluate / demo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import models, transforms

ARCH = "resnet18_fpn_localizer"
LOC_SIZE = 384
MASK_STRIDE = 4
TOPK_FRAC = 0.005
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

_normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def mask_path_for(path: str, attack: str) -> Path | None:
    """fake/<attack>/<rel>.jpg -> its ground-truth mask png (None for reals / unknown)."""
    p = Path(path)
    if attack in ("none", "jpeg") or p.parts[0] != "fake":
        return None
    rel = Path(*p.parts[2:]).with_suffix(".png")
    if attack == "tumor_removal":
        return Path("fake") / "tumor_removal_masks" / rel
    return Path("fake") / "masks" / attack / rel


def preprocess(img: Image.Image, size: int = LOC_SIZE) -> torch.Tensor:
    return _normalize(img.convert("RGB").resize((size, size), Image.BILINEAR))


class LocDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, train: bool = False, size: int = LOC_SIZE):
        self.frame = frame.reset_index(drop=True)
        self.train = train
        self.size = size
        self.msize = size // MASK_STRIDE

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        img = Image.open(row.path).convert("RGB").resize((self.size, self.size), Image.BILINEAR)
        mpath = mask_path_for(row.path, row.attack)
        if mpath is not None and mpath.exists():
            m = np.asarray(Image.open(mpath).convert("L").resize((self.size, self.size), Image.NEAREST)) > 127
            m = m.reshape(self.msize, MASK_STRIDE, self.msize, MASK_STRIDE).mean(axis=(1, 3))
        else:
            m = np.zeros((self.msize, self.msize))
        if self.train and np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            m = m[:, ::-1]
        x = _normalize(img)
        y = torch.tensor(1.0 if row.label == "fake" else 0.0)
        mask = torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32))[None]
        return x, mask, y, str(row.attack), str(row.path)


class Localizer(nn.Module):
    def __init__(self, ch: int = 64):
        super().__init__()
        r = models.resnet18(weights=None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.lat = nn.ModuleList([nn.Conv2d(c, ch, 1) for c in (64, 128, 256, 512)])
        self.head = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, 1, 1),
        )
        self.cls = nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c1 = self.layer1(self.stem(x))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        p = self.lat[3](c4)
        for lat, c in zip((self.lat[2], self.lat[1], self.lat[0]), (c3, c2, c1)):
            p = F.interpolate(p, size=c.shape[-2:], mode="bilinear", align_corners=False) + lat(c)
        mask_logits = self.head(p)
        cls_logit = self.cls(F.adaptive_avg_pool2d(c4, 1).flatten(1)).squeeze(1)
        return mask_logits, cls_logit

    def load_resnet_encoder(self, state_dict: dict) -> None:
        """Reuse a ResNet18 classifier checkpoint (conv1/bn1/layer1-4) as the encoder."""
        mapped = {}
        for k, v in state_dict.items():
            if k.startswith("fc."):
                continue
            if k.startswith("conv1."):
                k = "stem.0." + k[len("conv1."):]
            elif k.startswith("bn1."):
                k = "stem.1." + k[len("bn1."):]
            mapped[k] = v
        missing, unexpected = self.load_state_dict(mapped, strict=False)
        assert not unexpected, unexpected
        assert all(k.startswith(("lat.", "head.", "cls.")) for k in missing), missing


def image_scores(mask_logits: torch.Tensor, cls_logit: torch.Tensor) -> dict[str, torch.Tensor]:
    """Image-level fake probability from the cls head, the mask (top-k mean), and their mean."""
    pm = torch.sigmoid(mask_logits).flatten(1)
    k = max(1, int(TOPK_FRAC * pm.shape[1]))
    topk = pm.topk(k, dim=1).values.mean(dim=1)
    cls = torch.sigmoid(cls_logit)
    return {"cls": cls, "mask_topk": topk, "combined": 0.5 * (cls + topk)}


@torch.no_grad()
def predict(model: Localizer, loader, dev: torch.device) -> dict[str, np.ndarray]:
    """Per-image scores plus pixel IoU (mask prob > 0.5) for fakes that have a mask."""
    model.eval()
    out = {"cls": [], "mask_topk": [], "combined": [], "label": [], "attack": [], "path": [], "iou": []}
    for x, mask, y, attack, path in loader:
        mask_logits, cls_logit = model(x.to(dev, non_blocking=True))
        for k, v in image_scores(mask_logits, cls_logit).items():
            out[k].append(v.cpu().numpy())
        pred = (torch.sigmoid(mask_logits) > 0.5).flatten(1).cpu()
        gt = (mask > 0.5).flatten(1)
        inter = (pred & gt).sum(1).float()
        union = (pred | gt).sum(1).float()
        out["iou"].append(torch.where(union > 0, inter / union.clamp(min=1), torch.nan).numpy())
        out["label"].append(y.numpy())
        out["attack"].extend(attack)
        out["path"].extend(path)
    return {k: np.concatenate(v) if k not in ("attack", "path") else np.asarray(v) for k, v in out.items()}


def load_localizer(path: str | Path, dev: torch.device) -> tuple[Localizer, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = Localizer()
    model.load_state_dict(ckpt["state_dict"])
    return model.to(dev).eval(), ckpt
