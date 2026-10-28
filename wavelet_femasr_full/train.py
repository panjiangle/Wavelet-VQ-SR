"""Common paired-data training harness for FeMaSR B0 and wavelet VQ-SR.

Both variants see the same published HR/LR pairs, augmentations, image count,
effective batch and validation code. Checkpoints carry model and optimizer state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import lpips
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity

from basicsr.archs.femasr_arch import FeMaSRNet

from model import WaveletSRPredictor, WaveletTokenizer, normalized_band_loss
from paired_data import OfficialPairs
from visual import save_snapshot


CODEBOOK_PARAMS = [[32, 1024, 512]]


def baseline(stage):
    return FeMaSRNet(
        in_channel=3, codebook_params=CODEBOOK_PARAMS, gt_resolution=256,
        norm_type="gn", act_type="silu", use_semantic_loss=False,
        LQ_stage=stage == "b", scale_factor=4, use_quantize=True,
        codebook_data_init=stage == "a",
        dead_code_reset_interval=5 if stage == "a" else 0,
        dead_code_reset_until=100 if stage == "a" else 0)


def load_prior(path, variant, device):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if obj.get("variant") != variant or obj.get("stage") != "a":
        raise ValueError("Stage B teacher must be a matching Stage A checkpoint")
    model = baseline("a") if variant == "baseline" else WaveletTokenizer(
        use_band_input=variant == "wavelet")
    model.load_state_dict(obj["model"])
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def init_student_from_prior(student, prior):
    current = student.state_dict()
    compatible = {key: value for key, value in prior.state_dict().items()
                  if key in current and current[key].shape == value.shape}
    student.load_state_dict(compatible, strict=False)
    for name, module in student.named_children():
        if name in {"quantize_group", "decoder_group", "after_quant_group", "out_conv"}:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
    return len(compatible)


def y_channel(rgb):
    # MATLAB/BasicSR limited-range Y, float RGB in [0,1].
    weights = rgb.new_tensor([65.481, 128.553, 24.966]).view(1, 3, 1, 1)
    return (rgb * weights).sum(dim=1, keepdim=True) / 255 + 16 / 255


def scores(sr, hr, metric_model):
    sr = sr.float().clamp(0, 1)
    hr = hr.float().clamp(0, 1)
    ys = y_channel(sr)[..., 4:-4, 4:-4]
    yh = y_channel(hr)[..., 4:-4, 4:-4]
    mse = F.mse_loss(ys, yh).item()
    psnr = -10 * math.log10(max(mse, 1e-12))
    ssim = structural_similarity(
        ys[0, 0].detach().cpu().numpy(),
        yh[0, 0].detach().cpu().numpy(), data_range=1.0)
    perceptual = float(metric_model(sr * 2 - 1, hr * 2 - 1).mean())
    return psnr, ssim, perceptual


def build_result(variant, stage, model, teacher, lr, hr, update):
    if variant == "baseline":
        if stage == "a":
            image, vq_loss, _, indices = model(hr)
            return {"image": image, "vq_loss": vq_loss, "indices": indices}
        with torch.no_grad():
            _, _, _, target_indices = teacher(hr)
        image, code_loss, _, indices = model(lr, target_indices)
        return {"image": image, "code_loss": code_loss, "indices": indices}
    if stage == "a":
        return model(hr, update=update)
    with torch.no_grad():
        target = teacher(hr, update=False)
    result = model(lr, teacher)
    result["teacher"] = target
    return result


def training_loss(variant, stage, result, hr, metric_model):
    image_loss = F.l1_loss(result["image"].float(), hr)
    if stage == "a":
        perceptual = metric_model(result["image"].float() * 2 - 1, hr * 2 - 1).mean()
        loss = image_loss + perceptual + result["vq_loss"]
        parts = {"image": image_loss, "perceptual": perceptual,
                 "vq": result["vq_loss"]}
        if variant != "baseline":
            band = normalized_band_loss(result["coeff"], result["targets"])
            loss = loss + 0.2 * band
            parts["band"] = band
    else:
        if variant == "baseline":
            code = result["code_loss"]
            loss = image_loss + code
            parts = {"image": image_loss, "code": code}
        else:
            teacher = result["teacher"]
            code = sum(F.smooth_l1_loss(z.float(), q.detach().float())
                       for z, q in zip(result["z"], teacher["codes"])) / 4
            band = normalized_band_loss(result["coeff"], teacher["targets"])
            loss = image_loss + code + 0.2 * band
            parts = {"image": image_loss, "code": code, "band": band}
    return loss, {key: float(value.detach()) for key, value in parts.items()}


@torch.no_grad()
def validate(args, model, teacher, loader, device, metric_model, step, save_images):
    model.eval()
    metrics = []
    per_image = []
    histograms = None
    for n, batch in enumerate(loader):
        lr = batch["lr"].to(device, non_blocking=True)
        hr = batch["hr"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            result = build_result(args.variant, args.stage, model, teacher,
                                  lr, hr, update=False)
        metric = scores(result["image"], hr, metric_model)
        metrics.append(metric)
        per_image.append({"step": step, "id": batch["id"][0],
                          "psnr_y": metric[0], "ssim_y": metric[1],
                          "lpips_alex": metric[2]})
        indices = result.get("indices", [])
        if histograms is None:
            histograms = [torch.zeros(1024, dtype=torch.long) for _ in indices]
        for histogram, index_map in zip(histograms, indices):
            histogram += torch.bincount(index_map.detach().cpu().flatten(), minlength=1024)
        if n == 0 and save_images:
            save_snapshot(args.out, step, lr, hr, result["image"],
                          result.get("coeff"), result.get("stages"))
    values = np.asarray(metrics)
    means = values.mean(axis=0)
    usage = []
    for histogram in histograms or []:
        probabilities = histogram.float() / histogram.sum().clamp_min(1)
        nonzero = probabilities[probabilities > 0]
        usage.append({"used": int((histogram > 0).sum()),
                      "perplexity": float(torch.exp(-(nonzero * nonzero.log()).sum()))})
    return ({"psnr_y": float(means[0]), "ssim_y": float(means[1]),
             "lpips_alex": float(means[2]), "count": len(metrics)},
            usage, per_image)


def write_row(path, header, row):
    existed = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        if not existed:
            writer.writeheader()
        writer.writerow(row)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(path, variant, stage, step, model, optimizer, args, best_psnr):
    payload = {"variant": variant, "stage": stage, "step": step,
               "model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "best_psnr": best_psnr, "args": vars(args),
               "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all(),
               "numpy_rng": np.random.get_state(),
               "python_rng": random.getstate()}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("baseline", "wavelet", "wavelet_noband"), required=True)
    parser.add_argument("--stage", choices=("a", "b"), required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--val-every", type=int, default=5000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--skip-validation", action="store_true",
                        help="Throughput profiling only; do not use for formal runs")
    parser.add_argument("--save-points", type=int, nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--div2k-only", action="store_true",
                        help="Smoke only; formal runs require DF2K")
    args = parser.parse_args()
    if args.stage == "b" and not args.teacher:
        parser.error("Stage B requires --teacher from a matching Stage A run")
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")
    train_set = OfficialPairs(args.data, "train", include_flickr=not args.div2k_only)
    valid_set = OfficialPairs(args.data, "valid", center_crop=True)
    # Keep image order and worker crop/augmentation seeds identical across
    # architectures even when model initialization consumes different RNG draws.
    train_generator = torch.Generator().manual_seed(args.seed + 10000)
    valid_generator = torch.Generator().manual_seed(args.seed + 20000)
    train_loader = DataLoader(train_set, batch_size=args.microbatch, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              drop_last=True, persistent_workers=args.workers > 0,
                              generator=train_generator)
    valid_loader = DataLoader(valid_set, batch_size=1, shuffle=False,
                              num_workers=min(args.workers, 2), pin_memory=True,
                              generator=valid_generator)
    if args.variant == "baseline":
        model = baseline(args.stage)
    else:
        model = (WaveletTokenizer(use_band_input=args.variant == "wavelet")
                 if args.stage == "a" else WaveletSRPredictor())
    teacher = load_prior(args.teacher, args.variant, device) if args.stage == "b" else None
    if args.variant == "baseline" and args.stage == "b":
        copied = init_student_from_prior(model, teacher)
        print(f"Baseline Stage B copied {copied} compatible prior tensors", flush=True)
    model = model.to(device)
    if args.variant == "baseline" and args.stage == "a":
        codebook_weights = [book.embedding.weight for book in model.quantize_group]
        codebook_ids = {id(weight) for weight in codebook_weights}
        ordinary = [parameter for parameter in model.parameters()
                    if parameter.requires_grad and id(parameter) not in codebook_ids]
        groups = [{"params": ordinary, "lr": 1e-4},
                  {"params": codebook_weights, "lr": 1e-3}]
    else:
        groups = [{"params": [parameter for parameter in model.parameters()
                              if parameter.requires_grad], "lr": 1e-4}]
    optimizer = torch.optim.Adam(groups, betas=(0.9, 0.99))
    metric_model = lpips.LPIPS(net="alex").to(device).eval()
    for parameter in metric_model.parameters():
        parameter.requires_grad_(False)
    start = 0
    best_psnr = -float("inf")
    if args.resume:
        old = torch.load(args.resume, map_location="cpu", weights_only=False)
        if old["variant"] != args.variant or old["stage"] != args.stage:
            raise ValueError("Resume variant/stage mismatch")
        model.load_state_dict(old["model"])
        optimizer.load_state_dict(old["optimizer"])
        start = int(old["step"])
        best_psnr = float(old.get("best_psnr", best_psnr))
        torch.set_rng_state(old["torch_rng"])
        torch.cuda.set_rng_state_all(old["cuda_rng"])
        np.random.set_state(old["numpy_rng"])
        random.setstate(old["python_rng"])
    manifest = vars(args).copy()
    manifest["data"] = str(args.data)
    manifest["out"] = str(args.out)
    manifest["teacher"] = str(args.teacher) if args.teacher else None
    manifest["resume"] = str(args.resume) if args.resume else None
    manifest["train_pairs"] = len(train_set)
    manifest["valid_pairs"] = len(valid_set)
    manifest["trainable_parameters"] = sum(p.numel() for p in model.parameters()
                                           if p.requires_grad)
    manifest["code_sha256"] = {
        name: sha256_file(Path(__file__).with_name(name))
        for name in ("model.py", "paired_data.py", "train.py", "visual.py")
    }
    manifest["baseline_arch_sha256"] = sha256_file(
        Path(__file__).parent.parent / "FeMaSR_work" / "basicsr" / "archs" /
        "femasr_arch.py")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False), flush=True)
    iterator = iter(train_loader)
    t0 = time.monotonic()
    for step in range(start + 1, args.steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        parts = {}
        for _ in range(args.accumulation):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            lr = batch["lr"].to(device, non_blocking=True)
            hr = batch["hr"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = build_result(args.variant, args.stage, model, teacher,
                                      lr, hr, update=True)
                loss, parts = training_loss(args.variant, args.stage,
                                            result, hr, metric_model)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Nonfinite loss at step {step}")
            (loss / args.accumulation).backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            print(f"step={step} loss={float(loss):.6f} parts={parts} "
                  f"elapsed_s={time.monotonic() - t0:.1f} "
                  f"peak_GiB={torch.cuda.max_memory_allocated()/2**30:.2f}", flush=True)
        should_validate = (step == 1 or step % args.val_every == 0
                           or step in args.save_points or step == args.steps)
        if should_validate and not args.skip_validation:
            result, usage, per_image = validate(
                args, model, teacher, valid_loader, device, metric_model,
                step, save_images=True)
            print(f"VALID step={step} {result}", flush=True)
            print(f"USAGE step={step} {usage}", flush=True)
            with (args.out / "usage.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"step": step, "books": usage}) + "\n")
            write_row(args.out / "validation.csv",
                      ["step", "psnr_y", "ssim_y", "lpips_alex", "count"],
                      {"step": step, **result})
            for image_row in per_image:
                write_row(args.out / "validation_per_image.csv",
                          ["step", "id", "psnr_y", "ssim_y", "lpips_alex"],
                          image_row)
            improved = result["psnr_y"] > best_psnr
            if improved:
                best_psnr = result["psnr_y"]
            checkpoint(args.out / "last.pt", args.variant, args.stage,
                       step, model, optimizer, args, best_psnr)
            if improved:
                checkpoint(args.out / "best.pt", args.variant, args.stage,
                           step, model, optimizer, args, best_psnr)
            if step in args.save_points or step == args.steps:
                checkpoint(args.out / f"step_{step:06d}.pt", args.variant,
                           args.stage, step, model, optimizer, args, best_psnr)
    print("TRAINING COMPLETE", flush=True)


if __name__ == "__main__":
    main()
