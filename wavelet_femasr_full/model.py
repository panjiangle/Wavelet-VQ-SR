"""FeMaSR-encoder wavelet VQ tokenizer and LR predictor.

This is a new experimental architecture, not the author's FeMaSR model or SIT.
The FeMaSR encoder implementation is imported from a pinned clean checkout.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from basicsr.archs.femasr_arch import MultiScaleEncoder
from basicsr.archs.fema_utils import ResBlock


BANDS = ("LL3", "D3", "D2", "D1")
TOKEN_SIZES = (16, 16, 32, 64)
COEFF_SIZES = (32, 32, 64, 128)
BAND_SCALES = (8.0, 0.4, 0.25, 0.125)
CHANNELS = {8: 256, 16: 256, 32: 256, 64: 256, 128: 128, 256: 64, 512: 32}


def dwt2(x: torch.Tensor):
    if x.shape[-2] % 2 or x.shape[-1] % 2:
        raise ValueError("Haar DWT requires even height and width")
    a, b = x[..., ::2, ::2], x[..., ::2, 1::2]
    c, d = x[..., 1::2, ::2], x[..., 1::2, 1::2]
    return ((a + b + c + d) * 0.5,
            ((-a - b + c + d) * 0.5,
             (-a + b - c + d) * 0.5,
             (a - b - c + d) * 0.5))


def idwt2(ll: torch.Tensor, detail):
    lh, hl, hh = detail
    if not (ll.shape == lh.shape == hl.shape == hh.shape):
        raise ValueError("Haar IDWT band shapes do not match")
    a = (ll - lh - hl + hh) * 0.5
    b = (ll - lh + hl - hh) * 0.5
    c = (ll + lh - hl - hh) * 0.5
    d = (ll + lh + hl + hh) * 0.5
    out = ll.new_empty(*ll.shape[:-2], ll.shape[-2] * 2, ll.shape[-1] * 2)
    out[..., ::2, ::2], out[..., ::2, 1::2] = a, b
    out[..., 1::2, ::2], out[..., 1::2, 1::2] = c, d
    return out


def analyze(x: torch.Tensor):
    levels = {}
    ll = x
    for level in (1, 2, 3):
        ll, detail = dwt2(ll)
        levels[level] = dict(zip(("LL", "LH", "HL", "HH"), (ll, *detail)))
    return levels


def band_targets(levels):
    return (levels[3]["LL"], *(torch.cat(
        [levels[level][key] for key in ("LH", "HL", "HH")], dim=1)
        for level in (3, 2, 1)))


class EMACodebook(nn.Module):
    def __init__(self, size: int = 1024, dim: int = 128, decay: float = 0.99):
        super().__init__()
        self.size, self.dim, self.decay = size, dim, decay
        self.register_buffer("embedding", torch.zeros(size, dim))
        self.register_buffer("ema_count", torch.zeros(size))
        self.register_buffer("ema_sum", torch.zeros(size, dim))
        self.register_buffer("initialized", torch.tensor(False))

    @torch.no_grad()
    def _initialize(self, flat):
        chosen = flat[torch.randint(flat.shape[0], (self.size,), device=flat.device)]
        self.embedding.copy_(chosen)
        self.ema_sum.copy_(chosen)
        self.ema_count.fill_(1.0)
        self.initialized.fill_(True)

    @torch.no_grad()
    def _update(self, flat, idx):
        counts = torch.bincount(idx, minlength=self.size).to(self.ema_count)
        sums = torch.zeros_like(self.ema_sum)
        sums.index_add_(0, idx, flat.detach().to(sums))
        self.ema_count.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.ema_sum.mul_(self.decay).add_(sums, alpha=1 - self.decay)
        smoothed = (self.ema_count + 1e-5) / (self.ema_count.sum() + self.size * 1e-5)
        smoothed = smoothed * self.ema_count.sum()
        self.embedding.copy_(self.ema_sum / smoothed[:, None].clamp_min(1e-4))
        stale = self.ema_count < 0.02
        if stale.any():
            fresh = flat[torch.randint(flat.shape[0], (int(stale.sum()),), device=flat.device)]
            fresh = fresh.to(self.embedding)
            self.embedding[stale] = fresh
            self.ema_sum[stale] = fresh
            self.ema_count[stale] = 1.0

    def forward(self, z: torch.Tensor, update: bool = True):
        b, c, h, w = z.shape
        if c != self.dim:
            raise ValueError(f"Expected {self.dim} channels, got {c}")
        flat = z.permute(0, 2, 3, 1).reshape(-1, c)
        if self.training and update and not bool(self.initialized):
            self._initialize(flat.detach())
        # fp32 distance computation keeps assignments stable under bf16 training.
        values = flat.float()
        centers = self.embedding.float()
        distance = values.square().sum(1, keepdim=True) + centers.square().sum(1)[None]
        distance = distance - 2 * values @ centers.T
        idx = distance.argmin(dim=1)
        q = F.embedding(idx, self.embedding).to(z.dtype)
        q = q.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        if self.training and update:
            self._update(flat, idx)
        commitment = F.mse_loss(z.float(), q.detach().float())
        return z + (q - z).detach(), idx.view(b, 1, h, w), commitment


class BandStem(nn.Module):
    def __init__(self, input_channels: int, dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, 128, 3, stride=2, padding=1),
            ResBlock(128, 128, "gn", "silu"),
            nn.Conv2d(128, dim, 1),
        )

    def forward(self, x):
        return self.body(x)


class CoeffHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: float):
        super().__init__()
        self.scale = scale
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            ResBlock(128, 128, "gn", "silu"),
            ResBlock(128, 128, "gn", "silu"),
            ResBlock(128, 128, "gn", "silu"),
            nn.Conv2d(128, out_channels, 3, padding=1),
        )

    def forward(self, x, size):
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        return self.body(x) * self.scale


class WaveletDecoder(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.ll_head = CoeffHead(dim, 3, BAND_SCALES[0])
        self.detail_heads = nn.ModuleList([
            CoeffHead(dim * 2 + 3, 9, BAND_SCALES[i]) for i in (1, 2, 3)
        ])

    def forward(self, codes):
        if len(codes) != 4:
            raise ValueError("Expected four quantized feature grids")
        ll3 = self.ll_head(codes[0], COEFF_SIZES[0])
        coeff = {"LL3": ll3}
        stages = {}
        ll = ll3
        for i, level in enumerate((3, 2, 1), start=1):
            size = COEFF_SIZES[i]
            current = F.interpolate(codes[i], size=(size, size), mode="bilinear", align_corners=False)
            previous = F.interpolate(codes[i - 1], size=(size, size), mode="bilinear", align_corners=False)
            context = torch.cat((current, previous, ll), dim=1)
            detail = self.detail_heads[i - 1](context, size)
            parts = detail.chunk(3, dim=1)
            coeff.update({f"{name}{level}": part
                          for name, part in zip(("LH", "HL", "HH"), parts)})
            ll = idwt2(ll, parts)
            stages[level] = ll
        return ll, coeff, stages


class WaveletTokenizer(nn.Module):
    def __init__(self, dim: int = 128, codebook_size: int = 1024,
                 use_band_input: bool = True):
        super().__init__()
        self.dim, self.codebook_size = dim, codebook_size
        self.use_band_input = use_band_input
        self.hr_encoder = MultiScaleEncoder(
            3, 4, input_res=256, channel_query_dict=CHANNELS,
            norm_type="gn", act_type="silu", LQ_stage=False)
        self.band_stems = nn.ModuleList([
            BandStem(3 if i == 0 else 9, dim) for i in range(4)
        ])
        self.fusions = nn.ModuleList([
            nn.Sequential(nn.Conv2d(dim + 256 + (dim if i else 0), dim, 1),
                          ResBlock(dim, dim, "gn", "silu"))
            for i in range(4)
        ])
        self.books = nn.ModuleList([EMACodebook(codebook_size, dim) for _ in range(4)])
        self.decoder = WaveletDecoder(dim)

    def encode(self, hr: torch.Tensor, update: bool = True):
        if hr.shape[-2:] != (256, 256):
            raise ValueError("Stage A currently trains on 256x256 HR patches")
        levels = analyze(hr)
        targets = band_targets(levels)
        if self.use_band_input:
            encoder_inputs = targets
        else:
            # Same grids, channels, codebooks, decoder and supervision, but no
            # explicit DWT band is fed into the HR encoder. This isolates the
            # value of spectral input from the larger token/parameter budget.
            encoder_inputs = tuple(
                F.interpolate(hr, size=(size, size), mode="area").repeat(
                    1, 1 if i == 0 else 3, 1, 1)
                for i, size in enumerate(COEFF_SIZES))
        hr_features = self.hr_encoder(hr)
        # MultiScaleEncoder returns one item per downsampling block (not in_conv).
        # Its four HR outputs are 128,64,32,16 in that order.
        feature_by_size = {16: hr_features[3], 32: hr_features[2], 64: hr_features[1]}
        z_list, q_list, idx_list, losses = [], [], [], []
        for i, encoder_input in enumerate(encoder_inputs):
            scale = BAND_SCALES[i] if self.use_band_input else 1.0
            band_feature = self.band_stems[i](encoder_input / scale)
            parts = [band_feature, feature_by_size[TOKEN_SIZES[i]]]
            if i:
                parts.append(F.interpolate(q_list[-1], size=band_feature.shape[-2:], mode="nearest"))
            z = self.fusions[i](torch.cat(parts, dim=1))
            q, idx, commitment = self.books[i](z, update=update)
            z_list.append(z)
            q_list.append(q)
            idx_list.append(idx)
            losses.append(commitment)
        return {"z": z_list, "codes": q_list, "indices": idx_list,
                "vq_loss": torch.stack(losses).mean(), "targets": targets}

    def decode(self, codes):
        return self.decoder(codes)

    def forward(self, hr: torch.Tensor, update: bool = True):
        result = self.encode(hr, update=update)
        image, coeff, stages = self.decode(result["codes"])
        result.update(image=image, coeff=coeff, stages=stages)
        return result


class WaveletSRPredictor(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.lr_encoder = MultiScaleEncoder(
            3, 2, input_res=64, channel_query_dict=CHANNELS,
            norm_type="gn", act_type="silu", LQ_stage=True,
            input_resolution=(16, 16),
            embed_dim=256, blk_depth=4, num_heads=8, window_size=8)
        self.token_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256 + (dim if i else 0), 128, 1),
                          ResBlock(128, 128, "gn", "silu"),
                          ResBlock(128, 128, "gn", "silu"),
                          nn.Conv2d(128, dim, 1)) for i in range(4)
        ])

    def forward(self, lr: torch.Tensor, tokenizer: WaveletTokenizer):
        if lr.shape[-2:] != (64, 64):
            raise ValueError("Stage B currently trains on 64x64 LR patches")
        features = self.lr_encoder(lr)[-3:]
        feature_by_size = dict(zip((16, 32, 64), features))
        z_list, q_list, idx_list = [], [], []
        for i, size in enumerate(TOKEN_SIZES):
            parts = [feature_by_size[size]]
            if i:
                parts.append(F.interpolate(q_list[-1], size=(size, size), mode="nearest"))
            z = self.token_heads[i](torch.cat(parts, dim=1))
            q, idx, _ = tokenizer.books[i](z, update=False)
            z_list.append(z)
            q_list.append(q)
            idx_list.append(idx)
        image, coeff, stages = tokenizer.decode(q_list)
        return {"image": image, "coeff": coeff, "stages": stages,
                "z": z_list, "codes": q_list, "indices": idx_list}


def normalized_band_loss(coeff, target_bands):
    names = ("LL3", "D3", "D2", "D1")
    values = [coeff["LL3"]] + [torch.cat(
        [coeff[f"{key}{level}"] for key in ("LH", "HL", "HH")], dim=1)
        for level in (3, 2, 1)]
    return sum(F.smooth_l1_loss(pred / scale, target / scale)
               for pred, target, scale in zip(values, target_bands, BAND_SCALES)) / len(names)
