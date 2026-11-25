"""Full GlassLM Decoder-Only Transformer model."""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from glasslm.config import ModelConfig
from glasslm.model.attention import GroupedQueryAttention
from glasslm.model.ffn import SwiGLUFFN
from glasslm.model.norm import RMSNorm
from glasslm.model.rope import precompute_rope_freqs


class TransformerBlock(nn.Module):
    """One pre-norm Transformer decoder layer: Attention + SwiGLU with residual connections."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        return_attn_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # x: (B, T, d_model)
        # Pre-norm + residual connection keeps gradient flow clean across deep stacks.
        attn_out, attn_weights = self.attn(
            self.attn_norm(x), cos, sin, return_attn_weights=return_attn_weights
        )
        x = x + attn_out  # (B, T, d_model)
        x = x + self.ffn(self.ffn_norm(x))  # (B, T, d_model)
        return x, attn_weights


class GlassLM(nn.Module):
    """Modern Llama-3 / Qwen-style Small Language Model built from scratch."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token embedding table maps each integer token ID -> d_model vector.
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()

        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm_f = RMSNorm(config.d_model, eps=config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying shares the token embedding matrix with the output vocabulary head.
        # On small models (8k vocab, 512 dim), this saves ~4.2M parameters and acts as regularization.
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # Precompute RoPE sin/cos tables once; non-persistent so checkpoints stay lean.
        cos, sin = precompute_rope_freqs(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)  # (max_seq_len, head_dim)
        self.register_buffer("rope_sin", sin, persistent=False)  # (max_seq_len, head_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with GPT-style normal scaling for stable early training."""
        residual_std = 0.02 / math.sqrt(2 * self.config.n_layers)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Scale residual output projections down so initial residual variance doesn't explode.
                std = residual_std if (name.endswith(".w_o") or name.endswith(".w_down")) else 0.02
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        """Return total unique trainable parameter count (handles tied weights automatically)."""
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_intermediates: bool = False,
    ) -> dict[str, Any]:
        """Forward pass through the Transformer.

        Args:
            input_ids: (B, T) integer token IDs
            targets: optional (B, T) next-token IDs for cross-entropy training loss
            return_intermediates: if True, returns per-layer hidden states (for Logit Lens)
                and per-layer attention maps (for the Attention Explorer).
        """
        _, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length ({seq_len}) exceeds max_seq_len ({self.config.max_seq_len})."
            )

        x = self.emb_dropout(self.tok_emb(input_ids))  # (B, T, d_model)

        hidden_states: list[torch.Tensor] | None = [] if return_intermediates else None
        attn_weights_list: list[torch.Tensor] | None = [] if return_intermediates else None

        for block in self.blocks:
            x, attn_weights = block(
                x,
                self.rope_cos,
                self.rope_sin,
                return_attn_weights=return_intermediates,
            )
            if return_intermediates:
                assert hidden_states is not None and attn_weights_list is not None
                hidden_states.append(x)
                assert attn_weights is not None
                attn_weights_list.append(attn_weights)

        x = self.norm_f(x)  # (B, T, d_model)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss: torch.Tensor | None = None
        if targets is not None:
            # Compute cross-entropy in float32 for numerical stability under bf16 autocast.
            loss = F.cross_entropy(
                logits.float().view(-1, self.config.vocab_size),
                targets.view(-1),
            )

        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": hidden_states,
            "attn_weights": attn_weights_list,
        }

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 64,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
        return_token_probs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[list[tuple[int, float]]]]:
        """Autoregressively generate new tokens given a prompt tensor (B, T).

        When return_token_probs=True, also returns the top-5 candidate (token_id, prob)
        distribution for batch index 0 at every generated step.
        """
        was_training = self.training
        self.eval()
        try:
            generated = input_ids
            step_top_probs: list[list[tuple[int, float]]] = []

            for _ in range(max_new_tokens):
                # Crop context window to the model's maximum sequence length.
                ctx = generated[:, -self.config.max_seq_len :]  # (B, <= max_seq_len)
                logits = self.forward(ctx)["logits"][:, -1, :].float()  # (B, vocab_size)

                if temperature <= 0.0:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)  # (B, 1)
                else:
                    scaled = logits / temperature
                    if top_k > 0:
                        k = min(top_k, scaled.shape[-1])
                        kth_vals = torch.topk(scaled, k, dim=-1).values[:, -1:]
                        scaled = scaled.masked_fill(scaled < kth_vals, float("-inf"))

                    if 0.0 < top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
                        sorted_probs = F.softmax(sorted_logits, dim=-1)
                        cum_probs = torch.cumsum(sorted_probs, dim=-1)
                        # Remove tokens once cumulative probability exceeds top_p, always keeping top-1.
                        remove_mask = (cum_probs - sorted_probs) > top_p
                        sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
                        scaled = torch.full_like(scaled, float("-inf")).scatter_(
                            -1, sorted_indices, sorted_logits
                        )

                    probs = F.softmax(scaled, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

                if return_token_probs:
                    top5_probs, top5_ids = torch.topk(probs[0], k=min(5, probs.shape[-1]))
                    step_top_probs.append(
                        [(int(tid), float(p)) for tid, p in zip(top5_ids.tolist(), top5_probs.tolist())]
                    )

                generated = torch.cat((generated, next_token), dim=1)

                if eos_token_id is not None and torch.all(next_token == eos_token_id):
                    break

            if return_token_probs:
                return generated, step_top_probs
            return generated
        finally:
            if was_training:
                self.train()
