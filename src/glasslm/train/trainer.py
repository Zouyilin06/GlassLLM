"""PyTorch training loop with bf16 mixed precision, gradient accumulation, and eval."""

from dataclasses import asdict
import math
import os
from pathlib import Path
import time
from typing import Iterator

import torch
from torch.utils.data import DataLoader

from glasslm.config import GlassLMConfig
from glasslm.model.transformer import GlassLM
from glasslm.tokenizer.bpe import GlassTokenizer
from glasslm.train.lr_schedule import get_cosine_lr


def _infinite_batches(
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield batches endlessly across epochs so training is driven by optimizer steps."""
    while True:
        for batch in loader:
            yield batch


class Trainer:
    """Self-contained trainer tuned for single-GPU consumer hardware (RTX 4060 8GB)."""

    def __init__(
        self,
        model: GlassLM,
        config: GlassLMConfig,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        tokenizer: GlassTokenizer | None = None,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer

        torch.manual_seed(config.train.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.train.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.use_bf16 = bool(
            config.train.bf16
            and self.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )

        self.out_dir = Path(config.train.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.start_step: int = 0
        self.best_val_loss: float = float("inf")

        # Apply weight decay only to 2D+ weight matrices, never to 1D RMSNorm scales.
        decay_params = []
        no_decay_params = []
        for p in self.model.parameters():
            if not p.requires_grad:
                continue
            if p.ndim >= 2:
                decay_params.append(p)
            else:
                no_decay_params.append(p)

        param_groups = [
            {"params": decay_params, "weight_decay": config.train.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.train.lr,
            betas=(0.9, 0.95),
            fused=(self.device.type == "cuda"),
        )

    @torch.no_grad()
    def evaluate(self) -> float:
        """Compute average cross-entropy loss over up to eval_batches validation batches."""
        self.model.eval()
        losses: list[float] = []
        for i, (x, y) in enumerate(self.val_loader):
            if i >= self.config.train.eval_batches:
                break
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.use_bf16,
            ):
                loss = self.model(x, targets=y)["loss"]
            assert loss is not None
            losses.append(float(loss.item()))

        self.model.train()
        return sum(losses) / max(1, len(losses))

    def _save_checkpoint(
        self,
        filename: str,
        step: int,
        val_loss: float,
        best_val_loss: float,
    ) -> Path:
        ckpt_path = self.out_dir / filename
        tmp_path = ckpt_path.with_suffix(".pt.tmp")
        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "model_config": asdict(self.config.model),
            "step": step,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
            ),
        }
        torch.save(payload, tmp_path)
        os.replace(tmp_path, ckpt_path)
        return ckpt_path

    def load_checkpoint(self, ckpt_path: str | Path) -> int:
        """Restore model, optimizer, step counter, and RNG states for exact resume."""
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state"])

        self.start_step = int(ckpt.get("step", 0))
        self.best_val_loss = float(
            ckpt.get("best_val_loss", ckpt.get("val_loss", float("inf")))
        )

        # RNG restore APIs require a CPU uint8 tensor even when the checkpoint came from CUDA.
        if ckpt.get("rng_state") is not None:
            torch.set_rng_state(ckpt["rng_state"].detach().to(device="cpu", dtype=torch.uint8))
        if self.device.type == "cuda" and ckpt.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state(
                ckpt["cuda_rng_state"].detach().to(device="cpu", dtype=torch.uint8),
                self.device,
            )
        return self.start_step

    def train(self) -> dict[str, float]:
        """Run the full step-based training loop and return final metrics."""
        tcfg = self.config.train
        if self.start_step >= tcfg.max_steps:
            print(
                f"Checkpoint is already at step {self.start_step}/{tcfg.max_steps}. "
                "Pass a higher --max-steps to train further."
            )
            return {
                "train_loss": float("nan"),
                "val_loss": self.best_val_loss,
                "best_val_loss": self.best_val_loss,
                "step": float(self.start_step),
            }

        train_iter = _infinite_batches(self.train_loader)
        completed_steps = self.start_step
        best_val_loss = self.best_val_loss
        last_train_loss = float("nan")
        last_val_loss = self.best_val_loss

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        t_interval = time.perf_counter()
        last_save_time = t_interval
        tokens_in_interval = 0

        try:
            for step in range(self.start_step, tcfg.max_steps):
                lr = get_cosine_lr(
                    step=step,
                    warmup_steps=tcfg.warmup_steps,
                    max_steps=tcfg.max_steps,
                    max_lr=tcfg.lr,
                    min_lr=tcfg.min_lr,
                )
                for group in self.optimizer.param_groups:
                    group["lr"] = lr

                accum_loss = 0.0
                for _ in range(tcfg.grad_accum_steps):
                    x, y = next(train_iter)
                    x = x.to(self.device, non_blocking=True)
                    y = y.to(self.device, non_blocking=True)
                    tokens_in_interval += x.numel()

                    with torch.amp.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.use_bf16,
                    ):
                        loss = self.model(x, targets=y)["loss"]
                    assert loss is not None
                    scaled_loss = loss / float(tcfg.grad_accum_steps)
                    scaled_loss.backward()
                    accum_loss += float(loss.item()) / float(tcfg.grad_accum_steps)

                if tcfg.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), tcfg.grad_clip)

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                completed_steps = step + 1
                last_train_loss = accum_loss

                step_num = completed_steps
                is_last_step = step_num == tcfg.max_steps

                if (
                    step_num == self.start_step + 1
                    or step_num % tcfg.log_interval == 0
                    or is_last_step
                ):
                    elapsed = max(1e-6, time.perf_counter() - t_interval)
                    tok_per_sec = tokens_in_interval / elapsed
                    vram_str = ""
                    if self.device.type == "cuda":
                        peak_mb = torch.cuda.max_memory_allocated(self.device) / (1024**2)
                        vram_str = f" | peak_vram={peak_mb:.0f}MB"
                    print(
                        f"step {step_num:5d}/{tcfg.max_steps} | "
                        f"train_loss={last_train_loss:.4f} | lr={lr:.2e} | "
                        f"{tok_per_sec:,.0f} tok/s{vram_str}"
                    )
                    t_interval = time.perf_counter()
                    tokens_in_interval = 0

                saved_this_step = False
                if step_num % tcfg.eval_interval == 0 or is_last_step:
                    last_val_loss = self.evaluate()
                    ppl = math.exp(min(last_val_loss, 20.0))
                    print(
                        f"[eval] step {step_num:5d} | val_loss={last_val_loss:.4f} | ppl={ppl:.2f}"
                    )
                    if last_val_loss < best_val_loss:
                        best_val_loss = last_val_loss
                        best_path = self._save_checkpoint(
                            "best.pt", step_num, last_val_loss, best_val_loss
                        )
                        print(f"[ckpt] saved new best checkpoint -> {best_path}")
                    self._save_checkpoint("latest.pt", step_num, last_val_loss, best_val_loss)
                    last_save_time = time.perf_counter()
                    saved_this_step = True

                    if self.tokenizer is not None:
                        prompt_ids = torch.tensor(
                            [self.tokenizer.encode("Once upon a time", add_bos=True)],
                            dtype=torch.long,
                            device=self.device,
                        )
                        sample_ids = self.model.generate(
                            prompt_ids,
                            max_new_tokens=32,
                            temperature=0.8,
                            top_k=40,
                            eos_token_id=self.tokenizer.eos_id,
                        )
                        assert isinstance(sample_ids, torch.Tensor)
                        sample_text = self.tokenizer.decode(sample_ids[0].tolist())
                        print(f"[sample] {sample_text!r}")

                if (
                    not saved_this_step
                    and tcfg.autosave_minutes > 0
                    and (time.perf_counter() - last_save_time) >= tcfg.autosave_minutes * 60.0
                ):
                    ckpt_path = self._save_checkpoint(
                        "latest.pt", step_num, last_val_loss, best_val_loss
                    )
                    last_save_time = time.perf_counter()
                    print(f"[autosave] step {step_num:5d} | saved -> {ckpt_path}")

        except KeyboardInterrupt:
            self.optimizer.zero_grad(set_to_none=True)
            ckpt_path = self._save_checkpoint(
                "latest.pt", completed_steps, last_val_loss, best_val_loss
            )
            print(
                f"\n[interrupt] Training paused at step {completed_steps}/{tcfg.max_steps}. "
                f"Autosaved -> {ckpt_path}\n"
                "Resume later with `--resume`."
            )

        self.best_val_loss = best_val_loss
        return {
            "train_loss": last_train_loss,
            "val_loss": last_val_loss,
            "best_val_loss": best_val_loss,
            "step": float(completed_steps),
        }
