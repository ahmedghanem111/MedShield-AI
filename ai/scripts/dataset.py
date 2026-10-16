from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import IMG_SIZE, NUM_WORKERS, SEED


LABEL_MAP = {"real": 0, "fake": 1}


def build_transforms(train: bool) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


class ManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, train: bool = False):
        self.frame = frame.reset_index(drop=True)
        self.tfm = build_transforms(train=train)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        img = Image.open(row.path).convert("RGB")
        x = self.tfm(img)
        y = LABEL_MAP[str(row.label)]
        meta = {
            "path": row.path,
            "attack": str(row.attack),
            "split": str(row.split),
            "held_out": bool(row.held_out),
        }
        return x, y, meta


def make_loader(
    frame: pd.DataFrame,
    train: bool,
    batch_size: int = 32,
    shuffle: bool | None = None,
) -> DataLoader:
    if shuffle is None:
        shuffle = train
    ds = ManifestDataset(frame, train=train)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )


def split_frames(manifest: pd.DataFrame):
    """Return train/val/test frames for main learning (no held-out JPEG)."""
    use = manifest[~manifest.held_out].copy()
    return (
        use[use.split == "train"],
        use[use.split == "val"],
        use[use.split == "test"],
    )
