"""Unit tests verifying core GlassLM math, shapes, causality, and generation."""

import math

import torch

from glasslm.config import ModelConfig
from glasslm.model.attention import GroupedQueryAttention
from glasslm.model.norm import RMSNorm
from glasslm.model.rope import apply_rope, precompute_rope_freqs
from glasslm.model.transformer import GlassLM


def _small_config() -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=192,
        max_seq_len=32,
        dropout=0.0,
        tie_embeddings=True,
    )


def test_rmsnorm_shape_and_unit_rms() -> None:
    norm = RMSNorm(dim=64, eps=1e-6)
    x = torch.randn(2, 8, 64, dtype=torch.float32) * 5.0 + 3.0
    out = norm(x)

    assert out.shape == x.shape
    assert out.dtype == x.dtype
    rms = torch.sqrt(out.pow(2).mean(dim=-1))
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_rope_relative_position_property() -> None:
    """Dot product of RoPE(q, m) and RoPE(k, n) must depend only on (m - n)."""
    torch.manual_seed(0)
    head_dim = 16
    cos, sin = precompute_rope_freqs(head_dim=head_dim, max_seq_len=32)

    q_vec = torch.randn(head_dim)
    k_vec = torch.randn(head_dim)

    # Repeat the exact same q_vec and k_vec at every sequence position.
    q = q_vec.view(1, 1, 1, head_dim).expand(1, 1, 32, head_dim).clone()
    k = k_vec.view(1, 1, 1, head_dim).expand(1, 1, 32, head_dim).clone()

    q_rot, k_rot = apply_rope(q, k, cos, sin)

    # Distance m - n = 5 - 2 = 3 vs 10 - 7 = 3 must give identical dot products.
    dot_2_5 = torch.dot(q_rot[0, 0, 5], k_rot[0, 0, 2])
    dot_7_10 = torch.dot(q_rot[0, 0, 10], k_rot[0, 0, 7])
    assert torch.allclose(dot_2_5, dot_7_10, atol=1e-5)

    # Different relative offset (m - n = 15) should produce a clearly different dot product.
    dot_2_17 = torch.dot(q_rot[0, 0, 17], k_rot[0, 0, 2])
    assert not torch.allclose(dot_2_5, dot_2_17, atol=1e-2)


def test_gqa_sdpa_matches_eager() -> None:
    torch.manual_seed(1)
    cfg = _small_config()
    attn = GroupedQueryAttention(cfg).eval()
    cos, sin = precompute_rope_freqs(cfg.head_dim, cfg.max_seq_len)

    x = torch.randn(2, 10, cfg.d_model)
    out_sdpa, weights_none = attn(x, cos, sin, return_attn_weights=False)
    out_eager, weights = attn(x, cos, sin, return_attn_weights=True)

    assert weights_none is None
    assert weights is not None
    assert weights.shape == (2, cfg.n_heads, 10, 10)
    assert torch.allclose(out_sdpa, out_eager, atol=1e-5)

    # Attention rows must sum to 1 and place zero probability on future tokens.
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)
    upper_tri = torch.triu(weights, diagonal=1)
    assert torch.allclose(upper_tri, torch.zeros_like(upper_tri), atol=1e-7)


def test_causal_mask_no_future_leak() -> None:
    """Changing future tokens must NEVER alter logits at earlier positions."""
    torch.manual_seed(2)
    cfg = _small_config()
    model = GlassLM(cfg).eval()

    ids_a = torch.randint(0, cfg.vocab_size, (2, 12))
    ids_b = ids_a.clone()
    # Mutate tokens from index 6 onward.
    ids_b[:, 6:] = (ids_b[:, 6:] + 17) % cfg.vocab_size

    logits_a = model(ids_a)["logits"]
    logits_b = model(ids_b)["logits"]

    assert torch.allclose(logits_a[:, :6, :], logits_b[:, :6, :], atol=1e-6)


def test_forward_loss_gradients_and_weight_tying() -> None:
    torch.manual_seed(3)
    cfg = _small_config()
    model = GlassLM(cfg)

    assert model.lm_head.weight is model.tok_emb.weight

    x = torch.randint(0, cfg.vocab_size, (2, 16))
    y = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(x, targets=y)
    loss = out["loss"]

    assert out["logits"].shape == (2, 16, cfg.vocab_size)
    assert loss is not None and torch.isfinite(loss)
    # Initial cross-entropy on random logits should be near ln(vocab_size) = ~4.85.
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 1.0

    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Missing gradient on {name}"
        assert torch.isfinite(param.grad).all(), f"Non-finite gradient on {name}"
        assert param.grad.abs().sum() > 0, f"Zero gradient on {name}"


def test_intermediates_for_logit_lens() -> None:
    cfg = _small_config()
    model = GlassLM(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 8))

    out = model(x, return_intermediates=True)
    hidden_states = out["hidden_states"]
    attn_weights = out["attn_weights"]

    assert len(hidden_states) == cfg.n_layers
    assert len(attn_weights) == cfg.n_layers
    assert all(h.shape == (2, 8, cfg.d_model) for h in hidden_states)
    assert all(a.shape == (2, cfg.n_heads, 8, 8) for a in attn_weights)


def test_generate_shapes_and_top_probs() -> None:
    torch.manual_seed(4)
    cfg = _small_config()
    model = GlassLM(cfg).eval()
    prompt = torch.randint(0, cfg.vocab_size, (1, 5))

    out_ids, top_probs = model.generate(
        prompt,
        max_new_tokens=7,
        temperature=0.8,
        top_k=10,
        top_p=0.9,
        return_token_probs=True,
    )

    assert out_ids.shape == (1, 12)
    assert len(top_probs) == 7
    for step_top5 in top_probs:
        assert len(step_top5) == 5
        for tid, prob in step_top5:
            assert 0 <= tid < cfg.vocab_size
            assert 0.0 <= prob <= 1.0
