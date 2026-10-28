"""Aligned HR/LR pairs from the DIV2K and Flickr2K author releases.

The loader never resizes or synthesizes LR. It selects aligned crops from the
published x4 image pairs and applies only synchronized flip/rotation.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _tensor(image: Image.Image):
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255)


class OfficialPairs(Dataset):
    def __init__(self, root: str | Path, split: str,
                 include_flickr: bool = True, center_crop: bool = True):
        self.root = Path(root)
        self.split = split
        self.center_crop = center_crop
        if split not in ("train", "valid"):
            raise ValueError(split)
        div = "train" if split == "train" else "valid"
        hr_dir = self.root / f"DIV2K_{div}_HR"
        lr_dir = self.root / f"DIV2K_{div}_LR_bicubic" / "X4"
        self.pairs = [(p, lr_dir / f"{p.stem}x4.png")
                      for p in sorted(hr_dir.glob("*.png"))]
        expected = 800 if split == "train" else 100
        if len(self.pairs) != expected:
            raise RuntimeError(f"DIV2K {split}: expected {expected} HR, got {len(self.pairs)}")
        if split == "train" and include_flickr:
            flickr_hr = self.root / "Flickr2K" / "Flickr2K_HR"
            flickr_lr = self.root / "Flickr2K" / "Flickr2K_LR_bicubic" / "X4"
            flickr_pairs = [(p, flickr_lr / f"{p.stem}x4.png")
                            for p in sorted(flickr_hr.glob("*.png"))]
            if len(flickr_pairs) != 2650:
                raise RuntimeError(f"Flickr2K: expected 2650 HR, got {len(flickr_pairs)}")
            self.pairs.extend(flickr_pairs)
        missing = [str(lr) for _, lr in self.pairs if not lr.is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} official LR files missing; first: {missing[:3]}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        hr_path, lr_path = self.pairs[index]
        with Image.open(hr_path) as image:
            hr = image.convert("RGB")
        with Image.open(lr_path) as image:
            lr = image.convert("RGB")
        width = min(lr.width, hr.width // 4)
        height = min(lr.height, hr.height // 4)
        if width < 64 or height < 64:
            raise ValueError(f"Pair too small: {hr_path}, {lr_path}")
        if abs(hr.width - lr.width * 4) > 3 or abs(hr.height - lr.height * 4) > 3:
            raise ValueError(f"Pair dimensions disagree: {hr_path}, {lr_path}")
        if self.split == "train":
            x = random.randint(0, width - 64)
            y = random.randint(0, height - 64)
        elif self.center_crop:
            x = (width - 64) // 2
            y = (height - 64) // 2
        else:
            x = y = None
        if x is not None:
            lr = lr.crop((x, y, x + 64, y + 64))
            hr = hr.crop((x * 4, y * 4, x * 4 + 256, y * 4 + 256))
        if self.split == "train":
            if random.random() < 0.5:
                lr = lr.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                hr = hr.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                lr = lr.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                hr = hr.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            turns = random.randrange(4)
            if turns:
                lr = lr.rotate(turns * 90, expand=True)
                hr = hr.rotate(turns * 90, expand=True)
        return {"lr": _tensor(lr), "hr": _tensor(hr), "id": hr_path.stem}
