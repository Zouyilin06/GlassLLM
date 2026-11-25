"""Generate text from a trained GlassLM checkpoint with optional top-token inspection."""

import argparse
from pathlib import Path

import torch

from glasslm.config import ModelConfig
from glasslm.model.transformer import GlassLM
from glasslm.tokenizer.bpe import GlassTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a GlassLM checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to .pt checkpoint.")
    parser.add_argument("--tokenizer", type=Path, required=True, help="Path to tokenizer.json.")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Input prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature (0 = greedy).")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k filtering (0 = disabled).")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus top-p filtering.")
    parser.add_argument(
        "--show-top-tokens",
        action="store_true",
        help="Print the top-5 candidate token probabilities for the first 10 generated steps.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_cfg = ModelConfig(**ckpt["model_config"])

    model = GlassLM(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tokenizer = GlassTokenizer(args.tokenizer)
    prompt_ids = tokenizer.encode(args.prompt, add_bos=True)
    input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    result = model.generate(
        input_tensor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=tokenizer.eos_id,
        return_token_probs=args.show_top_tokens,
    )

    if args.show_top_tokens:
        out_ids, step_probs = result
        assert isinstance(out_ids, torch.Tensor)
    else:
        assert isinstance(result, torch.Tensor)
        out_ids = result
        step_probs = []

    generated_ids = out_ids[0].tolist()
    print("=== Generated Text ===")
    print(tokenizer.decode(generated_ids))

    if args.show_top_tokens and step_probs:
        new_ids = generated_ids[len(prompt_ids) :]
        print("\n=== Glass-Box Sneak Peek: Top-5 Candidates (First 10 Steps) ===")
        for step_idx, (chosen_id, top5) in enumerate(zip(new_ids[:10], step_probs[:10]), start=1):
            chosen_tok = repr(tokenizer.decode([chosen_id], skip_special_tokens=False))
            candidates = ", ".join(
                f"{repr(tokenizer.decode([tid], skip_special_tokens=False))}: {prob:.1%}"
                for tid, prob in top5
            )
            print(f"step {step_idx:2d} | picked {chosen_tok:<12} | top5 -> {candidates}")


if __name__ == "__main__":
    main()
