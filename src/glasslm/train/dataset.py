"""Memory-mapped binary token datasets for fast, low-RAM LLM training."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from glasslm.tokenizer.bpe import GlassTokenizer


def write_tokens_bin(
    texts: Iterable[str],
    tokenizer: GlassTokenizer,
    out_path: str | Path,
) -> int:
    """Encode documents with <bos> ... <eos> boundaries into a flat uint16 binary file."""
    if tokenizer.vocab_size > 65535:
        raise ValueError(
            f"vocab_size ({tokenizer.vocab_size}) exceeds uint16 max (65535)."
        )

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    total_tokens = 0
    chunk_buffer: list[int] = []
    flush_threshold = 500_000

    with out_file.open("wb") as f:
        for text in texts:
            text = text.strip()
            if not text:
                continue
            ids = tokenizer.encode(text, add_bos=True, add_eos=True)
            chunk_buffer.extend(ids)
            total_tokens += len(ids)

            if len(chunk_buffer) >= flush_threshold:
                np.asarray(chunk_buffer, dtype=np.uint16).tofile(f)
                chunk_buffer.clear()

        if chunk_buffer:
            np.asarray(chunk_buffer, dtype=np.uint16).tofile(f)

    return total_tokens


class MemmapTokenDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Slices a flat uint16 token file into (x, y) next-token prediction windows.

    Using np.memmap keeps RAM usage near zero even on multi-hundred-MB corpora:
    the OS pages in only the chunks currently read by the DataLoader.
    """

    def __init__(
        self,
        bin_path: str | Path,
        seq_len: int,
        stride: int | None = None,
    ) -> None:
        self.bin_path = Path(bin_path)
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len

        self.tokens = np.memmap(self.bin_path, dtype=np.uint16, mode="r")
        n_tokens = len(self.tokens)
        if n_tokens < seq_len + 1:
            raise ValueError(
                f"Dataset {self.bin_path} has only {n_tokens} tokens, "
                f"need at least seq_len + 1 ({seq_len + 1})."
            )
        self.num_samples = (n_tokens - (seq_len + 1)) // self.stride + 1

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        # Grab seq_len + 1 tokens so target y is x shifted right by 1 position.
        window = torch.from_numpy(
            self.tokens[start : start + self.seq_len + 1].astype(np.int64)
        )
        x = window[:-1]  # (seq_len,)
        y = window[1:]   # (seq_len,)
        return x, y
