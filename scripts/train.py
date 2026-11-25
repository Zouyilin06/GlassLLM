"""Train GlassLM from a YAML config file."""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from glasslm.config import load_config
from glasslm.model.transformer import GlassLM
from glasslm.tokenizer.bpe import GlassTokenizer
from glasslm.train.dataset import MemmapTokenDataset
from glasslm.train.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GlassLM.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config file.")
    parser.add_argument("--max-steps", type=int, default=None, help="Optional override for max_steps.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume from latest.pt in out_dir (no value) or from an explicit checkpoint path.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps

    data_dir = Path(cfg.train.data_dir)
    tokenizer = GlassTokenizer(cfg.train.tokenizer_path)
    if tokenizer.vocab_size != cfg.model.vocab_size:
        print(
            f"Note: adjusting model.vocab_size from {cfg.model.vocab_size} "
            f"to tokenizer.vocab_size={tokenizer.vocab_size}"
        )
        cfg.model.vocab_size = tokenizer.vocab_size

    train_ds = MemmapTokenDataset(data_dir / "train.bin", seq_len=cfg.model.max_seq_len)
    val_ds = MemmapTokenDataset(data_dir / "val.bin", seq_len=cfg.model.max_seq_len)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=pin_memory,
        drop_last=True if len(train_ds) >= cfg.train.batch_size else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=pin_memory,
    )

    model = GlassLM(cfg.model)
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Device: {gpu_name} | bf16={cfg.train.bf16 and torch.cuda.is_available()}")
    print(
        f"GlassLM parameters: {model.num_parameters():,} | "
        f"train_windows={len(train_ds):,} | val_windows={len(val_ds):,}"
    )

    trainer = Trainer(
        model=model,
        config=cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
    )

    if args.resume is not None:
        if args.resume == "auto":
            ckpt_path = Path(cfg.train.out_dir) / "latest.pt"
            if ckpt_path.exists():
                loaded_step = trainer.load_checkpoint(ckpt_path)
                print(
                    f"[resume] loaded {ckpt_path} "
                    f"(continuing from step {loaded_step}/{cfg.train.max_steps})"
                )
            else:
                print(f"[resume] no checkpoint at {ckpt_path}; starting fresh from step 0")
        else:
            ckpt_path = Path(args.resume)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
            loaded_step = trainer.load_checkpoint(ckpt_path)
            print(
                f"[resume] loaded {ckpt_path} "
                f"(continuing from step {loaded_step}/{cfg.train.max_steps})"
            )

    metrics = trainer.train()
    print(f"Training complete. Best val_loss={metrics['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()
