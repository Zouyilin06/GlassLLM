"""Unit tests for BPE tokenizer and memory-mapped next-token dataset."""

from pathlib import Path

import torch

from glasslm.tokenizer.bpe import train_bpe_tokenizer
from glasslm.train.dataset import MemmapTokenDataset, write_tokens_bin


def test_bpe_roundtrip_and_memmap_dataset(tmp_path: Path) -> None:
    texts = [
        "Once upon a time, a little cat sat on a warm red mat.",
        "The little cat liked to chase bright butterflies in the garden.",
        "Every sunny morning, Sam and Lily walked down to the sparkling river.",
        "They found a wooden boat and sailed across the quiet blue water.",
    ] * 10

    tok_path = tmp_path / "tokenizer.json"
    tok = train_bpe_tokenizer(texts, vocab_size=256, save_path=tok_path)

    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2

    sample = "Once upon a time, a little cat smiled."
    ids = tok.encode(sample, add_bos=True, add_eos=True)
    assert ids[0] == tok.bos_id
    assert ids[-1] == tok.eos_id
    assert tok.decode(ids) == sample

    bin_path = tmp_path / "train.bin"
    total_tokens = write_tokens_bin(texts, tok, bin_path)
    assert total_tokens > 64

    ds = MemmapTokenDataset(bin_path, seq_len=16)
    assert len(ds) >= 2
    x, y = ds[0]
    assert x.shape == (16,)
    assert y.shape == (16,)
    assert x.dtype == torch.long
    # Target y must equal input x shifted left by 1 (next-token prediction).
    assert torch.equal(y[:-1], x[1:])
