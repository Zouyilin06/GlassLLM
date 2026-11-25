"""Rotary Positional Embeddings (RoPE).

Instead of adding a fixed position vector to token embeddings at the start of the
model, RoPE rotates pairs of features in Query (Q) and Key (K) vectors by an angle
proportional to each token's position.

Why rotation works so well:
When Q at position m and K at position n are multiplied together in attention,
rotating Q by m*theta and K by n*theta makes their dot product depend ONLY on the
relative distance (m - n) between the two tokens.
"""

import torch


def precompute_rope_freqs(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cosine and sine rotation tables up to max_seq_len.

    Returns:
        cos: (max_seq_len, head_dim)
        sin: (max_seq_len, head_dim)
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}.")

    # Different feature pairs rotate at different speeds (wavelengths from fast to slow),
    # just like hands on a clock tracking both fine-grained and long-range distance.
    half_indices = torch.arange(0, head_dim, 2, dtype=torch.float32)  # (head_dim // 2,)
    inv_freq = 1.0 / (theta ** (half_indices / head_dim))  # (head_dim // 2,)

    positions = torch.arange(max_seq_len, dtype=torch.float32)  # (max_seq_len,)
    # Outer product gives angle m * freq_i for every position m and frequency i.
    freqs = torch.outer(positions, inv_freq)  # (max_seq_len, head_dim // 2)

    # Duplicate across halves so it matches rotate_half(x) cleanly: [theta_0..theta_k, theta_0..theta_k].
    emb = torch.cat((freqs, freqs), dim=-1)  # (max_seq_len, head_dim)
    return torch.cos(emb), torch.sin(emb)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Pair first half of head_dim with negated second half: (-x2, x1).

    For a 2D pair (x1, x2), rotating by angle phi is:
        x1' = x1 * cos(phi) - x2 * sin(phi)
        x2' = x2 * cos(phi) + x1 * sin(phi)
    Writing rotate_half([x1, x2]) = [-x2, x1] lets us express that 2D rotation
    for all pairs in one vectorized expression: x * cos + rotate_half(x) * sin.
    """
    # x: (..., head_dim)
    half = x.shape[-1] // 2
    x1 = x[..., :half]  # (..., head_dim // 2)
    x2 = x[..., half:]  # (..., head_dim // 2)
    return torch.cat((-x2, x1), dim=-1)  # (..., head_dim)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embeddings to Query and Key tensors.

    Args:
        q: (B, n_heads, T, head_dim)
        k: (B, n_kv_heads, T, head_dim)
        cos: (max_seq_len, head_dim) or (T, head_dim)
        sin: (max_seq_len, head_dim) or (T, head_dim)
    """
    seq_len = q.shape[2]
    # Slice to current sequence length T and reshape to broadcast over (B, heads, T, head_dim).
    cos_t = cos[:seq_len].unsqueeze(0).unsqueeze(0).to(dtype=q.dtype, device=q.device)  # (1, 1, T, head_dim)
    sin_t = sin[:seq_len].unsqueeze(0).unsqueeze(0).to(dtype=q.dtype, device=q.device)  # (1, 1, T, head_dim)

    q_rot = (q * cos_t) + (rotate_half(q) * sin_t)  # (B, n_heads, T, head_dim)
    k_rot = (k * cos_t) + (rotate_half(k) * sin_t)  # (B, n_kv_heads, T, head_dim)
    return q_rot, k_rot
