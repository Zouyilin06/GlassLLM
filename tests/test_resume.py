"""Unit tests for atomic checkpoint autosave, Ctrl+C handling, and exact resume."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from glasslm.config import GlassLMConfig, ModelConfig, TrainConfig
from glasslm.model.transformer import GlassLM
from glasslm.train.trainer import Trainer


def _tiny_config(out_dir: Path, max_steps: int) -> GlassLMConfig:
    model_cfg = ModelConfig(
        vocab_size=64,
        d_model=32,
        n_layers=2,
        n_heads=2,
        n_kv_heads=1,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
        tie_embeddings=True,
    )
    train_cfg = TrainConfig(
        data_dir=str(out_dir),
        tokenizer_path=str(out_dir / "tokenizer.json"),
        out_dir=str(out_dir),
        batch_size=4,
        grad_accum_steps=1,
        max_steps=max_steps,
        warmup_steps=2,
        eval_interval=2,
        eval_batches=2,
        log_interval=2,
        lr=1e-3,
        min_lr=1e-4,
        weight_decay=0.01,
        bf16=False,
        num_workers=0,
        seed=123,
        autosave_minutes=0.0,
    )
    return GlassLMConfig(model=model_cfg, train=train_cfg)


def _synthetic_loaders(vocab_size: int = 64, seq_len: int = 16) -> tuple[DataLoader, DataLoader]:
    torch.manual_seed(0)
    x = torch.randint(0, vocab_size, (16, seq_len))
    y = torch.randint(0, vocab_size, (16, seq_len))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=4, shuffle=False), DataLoader(ds, batch_size=4, shuffle=False)


def test_save_and_resume_exact_continuation(tmp_path: Path) -> None:
    train_loader, val_loader = _synthetic_loaders()
    cfg1 = _tiny_config(tmp_path, max_steps=4)
    trainer1 = Trainer(GlassLM(cfg1.model), cfg1, train_loader, val_loader)
    metrics1 = trainer1.train()

    assert metrics1["step"] == 4.0
    latest_path = tmp_path / "latest.pt"
    assert latest_path.exists()
    assert not list(tmp_path.glob("*.tmp"))

    # Must load cleanly with weights_only=True (same mode used by generate.py and resume).
    ckpt = torch.load(latest_path, map_location="cpu", weights_only=True)
    assert ckpt["step"] == 4
    assert "optimizer_state" in ckpt
    assert "best_val_loss" in ckpt

    # Resume with a fresh model/trainer up to max_steps=8.
    cfg2 = _tiny_config(tmp_path, max_steps=8)
    trainer2 = Trainer(GlassLM(cfg2.model), cfg2, train_loader, val_loader)
    loaded_step = trainer2.load_checkpoint(latest_path)
    assert loaded_step == 4
    assert len(trainer2.optimizer.state) > 0

    metrics2 = trainer2.train()
    assert metrics2["step"] == 8.0
    ckpt_after = torch.load(latest_path, map_location="cpu", weights_only=True)
    assert ckpt_after["step"] == 8


class _InterruptAfterBatches:
    """Wraps a DataLoader and raises KeyboardInterrupt once a target batch count is reached."""

    def __init__(self, loader: DataLoader, interrupt_at_batch: int) -> None:
        self.loader = loader
        self.interrupt_at_batch = interrupt_at_batch
        self.calls = 0

    def __iter__(self):
        for batch in self.loader:
            self.calls += 1
            if self.calls == self.interrupt_at_batch:
                raise KeyboardInterrupt
            yield batch


def test_keyboard_interrupt_triggers_autosave(tmp_path: Path) -> None:
    base_loader, val_loader = _synthetic_loaders()
    # grad_accum_steps=1, so batch 3 triggers right after step 2 completes.
    interrupt_loader = _InterruptAfterBatches(base_loader, interrupt_at_batch=3)

    cfg = _tiny_config(tmp_path, max_steps=5)
    cfg.train.eval_interval = 100  # ensure latest.pt is written solely by the interrupt handler
    trainer = Trainer(GlassLM(cfg.model), cfg, interrupt_loader, val_loader)  # type: ignore[arg-type]
    metrics = trainer.train()

    assert metrics["step"] == 2.0
    latest_path = tmp_path / "latest.pt"
    assert latest_path.exists()
    assert not list(tmp_path.glob("*.tmp"))

    ckpt = torch.load(latest_path, map_location="cpu", weights_only=True)
    assert ckpt["step"] == 2

    # Verify the interrupted checkpoint resumes cleanly to max_steps=5.
    resumed_trainer = Trainer(GlassLM(cfg.model), cfg, base_loader, val_loader)
    assert resumed_trainer.load_checkpoint(latest_path) == 2
    resumed_metrics = resumed_trainer.train()
    assert resumed_metrics["step"] == 5.0
