"""Grouped-Query Attention (GQA) with RoPE and dual SDPA / eager inspection paths."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from glasslm.config import ModelConfig
from glasslm.model.rope import apply_rope


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat Key/Value heads so each group of Query heads has matching K/V heads.

    In GQA, we store fewer K/V heads (e.g., 2) than Q heads (e.g., 8). Before
    computing Q @ K^T, each K/V head is logically shared across n_rep = 4 Q heads.

    Args:
        x: (B, n_kv_heads, T, head_dim)
        n_rep: number of query heads per KV head (n_heads // n_kv_heads)
    Returns:
        (B, n_heads, T, head_dim)
    """
    if n_rep == 1:
        return x
    bsz, n_kv_heads, seq_len, head_dim = x.shape
    # Expand without copying until reshape so memory overhead stays minimal.
    x = x[:, :, None, :, :].expand(bsz, n_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(bsz, n_kv_heads * n_rep, seq_len, head_dim)


class GroupedQueryAttention(nn.Module):
    """Causal Grouped-Query Attention with fast SDPA and inspectable eager path."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep = config.n_rep
        self.head_dim = config.head_dim
        self.dropout = config.dropout

        # No biases in modern Llama/Qwen-style attention projections.
        self.w_q = nn.Linear(config.d_model, self.n_heads * self.head_dim, bias=False)
        self.w_k = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.w_v = nn.Linear(config.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.w_o = nn.Linear(self.n_heads * self.head_dim, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        return_attn_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run causal self-attention.

        Args:
            x: (B, T, d_model)
            cos: (max_seq_len, head_dim)
            sin: (max_seq_len, head_dim)
            return_attn_weights: if True, uses the explicit eager path so callers
                (and the future Glass-Box UI) can inspect the (B, n_heads, T, T) attention map.
        """
        bsz, seq_len, _ = x.shape

        # 1. Project input vectors into Queries, Keys, and Values, then split into heads.
        q = self.w_q(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = self.w_k(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, n_kv_heads, T, head_dim)
        v = self.w_v(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)  # (B, n_kv_heads, T, head_dim)

        # 2. Rotate Q and K with RoPE so attention scores naturally encode relative token distance.
        q, k = apply_rope(q, k, cos, sin)

        # 3. Expand K and V heads to match Q heads (GQA -> full head count).
        k = repeat_kv(k, self.n_rep)  # (B, n_heads, T, head_dim)
        v = repeat_kv(v, self.n_rep)  # (B, n_heads, T, head_dim)

        attn_weights: torch.Tensor | None = None
        if not return_attn_weights:
            # Fast path: PyTorch SDPA automatically picks FlashAttention / memory-efficient kernels on CUDA.
            drop_p = self.dropout if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=drop_p, is_causal=True
            )  # (B, n_heads, T, head_dim)
        else:
            # Eager glass-box path: computes the exact same math explicitly so we can return attn_weights.
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale  # (B, n_heads, T, T)

            # Causal mask: forbid position t from looking at future tokens > t.
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=q.device, dtype=torch.float32),
                diagonal=1,
            )  # (T, T)
            scores = scores + causal_mask

            attn_weights = F.softmax(scores, dim=-1)  # (B, n_heads, T, T)
            if self.training and self.dropout > 0.0:
                attn_weights = F.dropout(attn_weights, p=self.dropout)

            out = torch.matmul(attn_weights.to(v.dtype), v)  # (B, n_heads, T, head_dim)

        # 4. Merge heads back into d_model and apply output projection.
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.n_heads * self.head_dim)  # (B, T, d_model)
        return self.w_o(out), attn_weights
