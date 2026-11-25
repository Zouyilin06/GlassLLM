"""Prepare BPE tokenizer and train/val .bin token files for GlassLM."""

import argparse
from pathlib import Path
import random

from glasslm.tokenizer.bpe import train_bpe_tokenizer
from glasslm.train.dataset import write_tokens_bin


def _build_demo_stories(seed: int = 42) -> list[str]:
    """Generate ~180 varied mini stories offline so the smoke test needs zero downloads."""
    rng = random.Random(seed)
    characters = [
        ("Lily", "she", "her"),
        ("Sam", "he", "his"),
        ("Milo", "he", "his"),
        ("Nora", "she", "her"),
        ("Pip", "he", "his"),
        ("Maya", "she", "her"),
    ]
    companions = [
        "a small brown dog",
        "a curious orange cat",
        "a sleepy little owl",
        "a tiny green frog",
        "a fluffy white rabbit",
    ]
    places = [
        "a sunny garden",
        "a quiet forest",
        "a cozy blue house",
        "a sparkling river",
        "a windy hill",
    ]
    objects = [
        "a shiny red ball",
        "a wooden toy boat",
        "a warm blueberry muffin",
        "a golden brass key",
        "a painted kite",
    ]
    actions = [
        "shared it with a friend and laughed all afternoon",
        "carefully carried it home before the sun went down",
        "built a little game around it and smiled brightly",
        "kept it safe inside a woven basket by the door",
    ]

    stories: list[str] = []
    for _ in range(180):
        name, subj, poss = rng.choice(characters)
        companion = rng.choice(companions)
        place = rng.choice(places)
        obj = rng.choice(objects)
        action = rng.choice(actions)
        story = (
            f"Once upon a time, {name} walked into {place} with {companion}. "
            f"Under a tall oak tree, {subj} found {obj} resting on the soft grass. "
            f"{name} showed {poss} friend the treasure, and together they {action}. "
            f"Everyone felt happy and peaceful as the stars came out."
        )
        stories.append(story)
    return stories


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare tokenizer and token bins for GlassLM.")
    parser.add_argument(
        "--source",
        choices=["demo", "tinystories"],
        default="demo",
        help="'demo' uses built-in stories offline; 'tinystories' downloads roneneldan/TinyStories.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="BPE vocabulary size (default: 512 for demo, 8192 for tinystories).",
    )
    parser.add_argument(
        "--max-stories",
        type=int,
        default=None,
        help="Optional cap on number of stories (useful for fast TinyStories experiments).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for tokenizer.json, train.bin, and val.bin.",
    )
    args = parser.parse_args()

    if args.source == "demo":
        out_dir = args.out_dir or Path("data/demo")
        vocab_size = args.vocab_size or 512
        stories = _build_demo_stories()
        if args.max_stories is not None:
            stories = stories[: args.max_stories]
        split_idx = max(1, int(0.85 * len(stories)))
        train_texts = stories[:split_idx]
        val_texts = stories[split_idx:]
    else:
        from datasets import load_dataset

        out_dir = args.out_dir or Path("data/tinystories")
        vocab_size = args.vocab_size or 8192
        print("Loading roneneldan/TinyStories from HuggingFace...")
        ds = load_dataset("roneneldan/TinyStories")
        train_texts = [row["text"] for row in ds["train"]]
        val_texts = [row["text"] for row in ds["validation"]]
        if args.max_stories is not None:
            train_texts = train_texts[: args.max_stories]
            val_texts = val_texts[: max(200, args.max_stories // 10)]

    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / "tokenizer.json"

    print(f"Training BPE tokenizer (vocab_size={vocab_size}) on {len(train_texts):,} stories...")
    tokenizer = train_bpe_tokenizer(train_texts, vocab_size=vocab_size, save_path=tok_path)
    print(f"Saved tokenizer -> {tok_path} (actual vocab={tokenizer.vocab_size})")

    train_tokens = write_tokens_bin(train_texts, tokenizer, out_dir / "train.bin")
    val_tokens = write_tokens_bin(val_texts, tokenizer, out_dir / "val.bin")
    print(f"Wrote train.bin ({train_tokens:,} tokens) and val.bin ({val_tokens:,} tokens) in {out_dir}")


if __name__ == "__main__":
    main()
