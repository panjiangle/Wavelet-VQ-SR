"""Small GPU checks before any long training job."""

import torch
from torch.nn import functional as F

from model import (WaveletSRPredictor, WaveletTokenizer, analyze, dwt2,
                   idwt2, normalized_band_loss)


def main():
    torch.manual_seed(20261018)
    device = torch.device("cuda")
    image = torch.rand(1, 3, 256, 256, device=device)
    ll, details = dwt2(image)
    assert (idwt2(ll, details) - image).abs().max().item() < 1e-5
    levels = analyze(image)
    assert [levels[i]["LL"].shape[-1] for i in (1, 2, 3)] == [128, 64, 32]

    tokenizer = WaveletTokenizer().to(device).train()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = tokenizer(image)
        assert result["image"].shape == image.shape
        assert [item.shape[-1] for item in result["codes"]] == [16, 16, 32, 64]
        loss = F.l1_loss(result["image"].float(), image) + result["vq_loss"]
        loss = loss + 0.2 * normalized_band_loss(result["coeff"], result["targets"])
    loss.backward()
    assert any(p.grad is not None for p in tokenizer.hr_encoder.parameters())
    assert any(p.grad is not None for p in tokenizer.decoder.parameters())
    print("Stage A forward/backward OK", float(loss),
          torch.cuda.max_memory_allocated() / 2**30, "GiB", flush=True)

    tokenizer.zero_grad(set_to_none=True)
    tokenizer.eval()
    for parameter in tokenizer.parameters():
        parameter.requires_grad_(False)
    predictor = WaveletSRPredictor().to(device).train()
    lr = F.interpolate(image, size=(64, 64), mode="bicubic", align_corners=False)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = predictor(lr, tokenizer)
        assert output["image"].shape == image.shape
        assert [item.shape[-1] for item in output["indices"]] == [16, 16, 32, 64]
        loss_b = F.l1_loss(output["image"].float(), image)
    loss_b.backward()
    assert all(p.grad is None for p in tokenizer.parameters())
    assert any(p.grad is not None for p in predictor.lr_encoder.parameters())
    print("Stage B forward/backward OK", float(loss_b),
          torch.cuda.max_memory_allocated() / 2**30, "GiB", flush=True)


if __name__ == "__main__":
    main()
