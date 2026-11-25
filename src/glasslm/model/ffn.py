"""SwiGLU Feed-Forward Network (FFN)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from glasslm.config import ModelConfig


class SwiGLUFFN(nn.Module):
    """Gated Feed-Forward layer: w_down(SiLU(w_gate(x)) * w_up(x)).

    Why SwiGLU instead of a standard 2-layer ReLU MLP?
    After attention mixes information *between* tokens, each token is processed
    independently here. SwiGLU splits the hidden expansion into two parallel
    projections: `w_gate` (passed through smooth SiLU activation) acts like a
    learned dimmer switch controlling how much of `w_up` passes through.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.w_gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w_up = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.w_down = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        gated = F.silu(self.w_gate(x)) * self.w_up(x)  # (B, T, d_ff)
        return self.w_down(self.dropout(gated))  # (B, T, d_model)
