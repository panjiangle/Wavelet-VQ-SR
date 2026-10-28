"""Tiled full-image validation on published DIV2K x4 pairs.

Identical 64px LR tiles, 16px LR overlap and cosine blending are used for both
FeMaSR B0 and the wavelet predictor. Frequent training validation separately
uses fixed 64/256 center crops.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from skimage.metrics import structural_similarity
import lpips
from DISTS_pytorch import DISTS
import pyiqa

from model import WaveletSRPredictor
from paired_data import OfficialPairs
from train import baseline, load_prior, y_channel


def starts(length: int, tile: int = 64, stride: int = 48):
    if length < tile:
        raise ValueError(f"LR dimension {length} is smaller than tile {tile}")
    points = list(range(0, length - tile + 1, stride))
    if points[-1] != length - tile:
        points.append(length - tile)
    return points


@torch.no_grad()
def tiled_sr(lr, model, teacher, variant, device):
    h, w = lr.shape[-2:]
    output = torch.zeros(3, h * 4, w * 4, dtype=torch.float32)
    weight_sum = torch.zeros(1, h * 4, w * 4, dtype=torch.float32)
    ramp = torch.hann_window(256, periodic=False).clamp_min(0.05)
    weight = (ramp[:, None] * ramp[None, :])[None]
    for y in starts(h):
        for x in starts(w):
            patch = lr[:, y:y + 64, x:x + 64][None].to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if variant == "baseline":
                    sr = model(patch)[0]
                else:
                    sr = model(patch, teacher)["image"]
            sr = sr[0].float().clamp(0, 1).cpu()
            ys, xs = y * 4, x * 4
            output[:, ys:ys + 256, xs:xs + 256] += sr * weight
            weight_sum[:, ys:ys + 256, xs:xs + 256] += weight
    return output / weight_sum.clamp_min(1e-8)


def metric(sr, hr):
    ys = y_channel(sr[None])[0, 0, 4:-4, 4:-4]
    yh = y_channel(hr[None])[0, 0, 4:-4, 4:-4]
    mse = torch.mean((ys - yh).square()).item()
    psnr = -10 * math.log10(max(mse, 1e-12))
    ssim = structural_similarity(ys.numpy(), yh.numpy(), data_range=1.0)
    return psnr, ssim


def save_png(tensor, path):
    image = (tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(image).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "wavelet", "wavelet_noband"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    teacher = load_prior(args.teacher, args.variant, device)
    model = baseline("b") if args.variant == "baseline" else WaveletSRPredictor()
    state = torch.load(args.model, map_location="cpu", weights_only=False)
    if state["variant"] != args.variant or state["stage"] != "b":
        raise ValueError("Wrong Stage B checkpoint")
    model.load_state_dict(state["model"])
    model = model.to(device).eval()
    lpips_model = lpips.LPIPS(net="alex").to(device).eval()
    dists_model = DISTS().to(device).eval()
    musiq_model = pyiqa.create_metric("musiq", device="cuda")
    dataset = OfficialPairs(args.data, "valid", center_crop=False)
    rows = []
    for index in range(min(args.limit, len(dataset))):
        item = dataset[index]
        lr, hr = item["lr"], item["hr"]
        h = min(lr.shape[-2], hr.shape[-2] // 4)
        w = min(lr.shape[-1], hr.shape[-1] // 4)
        lr = lr[..., :h, :w]
        hr = hr[..., :h * 4, :w * 4]
        sr = tiled_sr(lr, model, teacher, args.variant, device)
        psnr, ssim = metric(sr, hr)
        sr_gpu = sr[None].to(device)
        hr_gpu = hr[None].to(device)
        with torch.no_grad():
            lpips_alex = float(lpips_model(sr_gpu * 2 - 1, hr_gpu * 2 - 1).mean())
            dists = float(dists_model(sr_gpu, hr_gpu).mean())
            musiq = float(musiq_model(sr_gpu).mean())
        rows.append({"id": item["id"], "psnr_y": psnr, "ssim_y": ssim,
                     "lpips_alex": lpips_alex, "dists": dists, "musiq": musiq})
        print(f"{index + 1}/{min(args.limit, len(dataset))} {item['id']} "
              f"PSNR-Y={psnr:.4f} SSIM-Y={ssim:.5f} "
              f"LPIPS={lpips_alex:.5f} DISTS={dists:.5f} "
              f"MUSIQ={musiq:.2f}", flush=True)
        if index < 5:
            save_png(sr, args.out / f"{item['id']}_SR.png")
            save_png(hr, args.out / f"{item['id']}_HR.png")
    with (args.out / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "id", "psnr_y", "ssim_y", "lpips_alex", "dists", "musiq"))
        writer.writeheader()
        writer.writerows(rows)
    print("MEAN " + " ".join(
        f"{key}={np.mean([row[key] for row in rows]):.5f}"
        for key in ("psnr_y", "ssim_y", "lpips_alex", "dists", "musiq")),
        flush=True)


if __name__ == "__main__":
    main()
