"""BPE tokenizer training and inference wrapper around HuggingFace `tokenizers`."""

from collections.abc import Iterable, Sequence
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

# Fixed special token IDs: <pad>=0, <bos>=1, <eos>=2, <unk>=3.
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class GlassTokenizer:
    """Thin, readable wrapper over a trained ByteLevel BPE tokenizer."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._tok = Tokenizer.from_file(str(self.path))

        self.pad_id = self._tok.token_to_id("<pad>")
        self.bos_id = self._tok.token_to_id("<bos>")
        self.eos_id = self._tok.token_to_id("<eos>")
        self.unk_id = self._tok.token_to_id("<unk>")

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        """Convert text into a list of integer token IDs."""
        ids = self._tok.encode(text).ids
        if add_bos and self.bos_id is not None:
            ids = [self.bos_id] + ids
        if add_eos and self.eos_id is not None:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        """Convert a sequence of token IDs back into readable text."""
        return self._tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def id_to_token(self, idx: int) -> str | None:
        """Return the raw subword string for a token ID (useful for debugging/UI)."""
        return self._tok.id_to_token(idx)


def train_bpe_tokenizer(
    texts: Iterable[str],
    vocab_size: int,
    save_path: str | Path,
    min_frequency: int = 2,
) -> GlassTokenizer:
    """Train a Byte-Level BPE tokenizer on an iterable of strings and save as JSON."""
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    # ByteLevel handles arbitrary Unicode without <unk> and preserves spaces cleanly.
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=False,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)

    out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    return GlassTokenizer(out_path)
