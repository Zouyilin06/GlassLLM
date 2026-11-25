"""Configuration dataclasses and YAML loader for GlassLM."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """Hyperparameters defining the GlassLM Transformer architecture."""

    vocab_size: int = 8192
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: int = 2
    d_ff: int = 1408
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})."
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})."
            )
        if self.head_dim % 2 != 0:
            raise ValueError(
                f"head_dim ({self.head_dim}) must be even so RoPE can rotate pairs of features."
            )

    @property
    def head_dim(self) -> int:
        """Dimension of each attention head."""
        return self.d_model // self.n_heads

    @property
    def n_rep(self) -> int:
        """How many query heads share each key/value head in GQA."""
        return self.n_heads // self.n_kv_heads


@dataclass
class TrainConfig:
    """Hyperparameters and paths for training GlassLM."""

    data_dir: str = "data/tinystories"
    tokenizer_path: str = "data/tinystories/tokenizer.json"
    out_dir: str = "checkpoints/base_30m"
    batch_size: int = 16
    grad_accum_steps: int = 4
    max_steps: int = 10000
    warmup_steps: int = 300
    eval_interval: int = 250
    eval_batches: int = 20
    log_interval: int = 25
    lr: float = 6e-4
    min_lr: float = 6e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    bf16: bool = True
    num_workers: int = 0
    seed: int = 42
    autosave_minutes: float = 5.0


@dataclass
class GlassLMConfig:
    """Top-level configuration pairing model and training settings."""

    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def load_config(path: str | Path) -> GlassLMConfig:
    """Load a GlassLMConfig from a YAML file."""
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    model_cfg = ModelConfig(**raw.get("model", {}))
    train_cfg = TrainConfig(**raw.get("train", {}))
    return GlassLMConfig(model=model_cfg, train=train_cfg)
