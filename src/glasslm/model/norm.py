"""Root Mean Square Layer Normalization (RMSNorm)."""

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Rescales each token vector by its root-mean-square magnitude.

    Why RMSNorm instead of LayerNorm?
    LayerNorm subtracts the mean and divides by standard deviation. RMSNorm skips
    subtracting the mean and just divides by the RMS size of the vector:
        y = (x / sqrt(mean(x^2) + eps)) * weight
    In practice it trains just as well while being simpler and faster on GPUs.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        # Learnable per-feature scale initialized to 1 so the layer starts as pure normalization.
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, dim)
        orig_dtype = x.dtype

        # Upcast to float32 before squaring/averaging so bf16/fp16 doesn't lose precision or overflow.
        x_f32 = x.float()
        rms_inv = torch.rsqrt(x_f32.pow(2).mean(dim=-1, keepdim=True) + self.eps)  # (B, T, 1)
        normed = (x_f32 * rms_inv).to(orig_dtype)  # (B, T, dim)

        return normed * self.weight  # (B, T, dim)
