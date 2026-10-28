"""Save signed wavelet coefficients and intermediate IDWT images."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from model import analyze


def _save(image: torch.Tensor, path: Path, signed_scale: float | None = None):
    image = image.detach().float().cpu().clamp(-1e6, 1e6)
    if signed_scale is not None:
        image = image / signed_scale * 0.25 + 0.5
    image = image.clamp(0, 1)
    image = (image.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(image).save(path)


def save_snapshot(root, step: int, lr, hr, image, coeff=None, stages=None):
    out = Path(root) / "artifacts" / f"step_{step:06d}"
    out.mkdir(parents=True, exist_ok=True)
    lr, hr, image = lr[0], hr[0], image[0]
    _save(hr, out / "HR.png")
    _save(image, out / "reconstruction_or_SR.png")
    _save(F.interpolate(lr[None], size=hr.shape[-2:], mode="bicubic",
                        align_corners=False)[0], out / "LR_bicubic_display.png")
    gt = analyze(hr[None])
    for level in (1, 2, 3):
        for band in ("LL", "LH", "HL", "HH"):
            target = gt[level][band][0]
            scale = 2 ** level if band == "LL" else {1: .125, 2: .25, 3: .4}[level]
            _save(target / scale if band == "LL" else target,
                  out / f"GT_{band}{level}.png",
                  None if band == "LL" else scale)
    if coeff is not None:
        _save(coeff["LL3"][0] / 8, out / "pred_LL3.png")
        for level in (3, 2, 1):
            for band in ("LH", "HL", "HH"):
                _save(coeff[f"{band}{level}"][0], out / f"pred_{band}{level}.png",
                      {1: .125, 2: .25, 3: .4}[level])
    if stages is not None:
        _save(stages[3][0] / 4, out / "IDWT_LL2.png")
        _save(stages[2][0] / 2, out / "IDWT_LL1.png")
        _save(stages[1][0], out / "IDWT_HR.png")
