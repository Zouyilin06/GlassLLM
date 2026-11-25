# GlassLM

A modern ~30M-parameter Small Language Model built from scratch in pure PyTorch—tuned to train locally on an **RTX 4060 (8GB VRAM)** and paired with glass-box inspection tools so you can see *how* the model predicts every token.

## Architecture (Llama-3 / Qwen Style)

```text
Input Text
   │
   ▼  Byte-Level BPE Tokenizer (vocab = 8,192)
   ▼  Token Embeddings (tied with LM Head)
   │
   ▼  TransformerBlock × 8
      ├── RMSNorm (pre-norm)
      ├── RoPE + Grouped-Query Attention (8 Q heads, 2 KV heads, SDPA / eager glass-box mode)
      ├── RMSNorm (pre-norm)
      └── SwiGLU Feed-Forward Network (d_ff = 1,408)
   │
   ▼  Final RMSNorm -> LM Head -> Next-Token Probabilities
```

### RTX 4060 8GB Profile (`configs/base_30m.yaml`)
- **Parameters:** ~28.3M (with tied embeddings)
- **Precision:** `bfloat16` mixed precision + PyTorch `SDPA` (FlashAttention backend)
- **Effective Batch Size:** `16 × 4 grad_accum = 64` sequences (`seq_len = 512` → 32,768 tokens/step)

---

## Quickstart

### 1. Install dependencies (CUDA PyTorch via `uv`)
```bash
uv sync
```

### 2. Run unit tests
```bash
uv run pytest -v
```

### 3. Instant offline smoke test (runs in ~15 seconds)
```bash
uv run python scripts/prepare_data.py --source demo
uv run python scripts/train.py --config configs/tiny_smoke.yaml
uv run python scripts/generate.py --checkpoint checkpoints/tiny_smoke/best.pt --tokenizer data/demo/tokenizer.json --prompt "Once upon a time" --max-new-tokens 30 --show-top-tokens
```

### 4. Full TinyStories training on RTX 4060
```bash
uv run python scripts/prepare_data.py --source tinystories
uv run python scripts/train.py --config configs/base_30m.yaml
```
*(Want a faster 15-minute run first? Add `--max-stories 50000` to `prepare_data.py` and `--max-steps 2000` to `train.py`.)*

### 5. Pause & Resume Training (Autosave)
Training autosaves `latest.pt` every `autosave_minutes` (default `5.0`), on every `eval_interval`, and immediately if you press `Ctrl+C`. To pick up right where you left off (including optimizer state and learning rate schedule):
```bash
uv run python scripts/train.py --config configs/base_30m.yaml --resume
```

---

## How to Read This Repo (Recommended Learning Order)

Every file in `src/glasslm/model/` includes plain-English comments and tensor shape annotations:

1. [`src/glasslm/model/norm.py`](src/glasslm/model/norm.py) — `RMSNorm`: keeping activation magnitudes stable
2. [`src/glasslm/model/rope.py`](src/glasslm/model/rope.py) — `RoPE`: encoding relative token distance via 2D rotations
3. [`src/glasslm/model/attention.py`](src/glasslm/model/attention.py) — `GroupedQueryAttention`: sharing KV heads + causal masking + SDPA vs. eager mode
4. [`src/glasslm/model/ffn.py`](src/glasslm/model/ffn.py) — `SwiGLUFFN`: gated feed-forward computation
5. [`src/glasslm/model/transformer.py`](src/glasslm/model/transformer.py) — `GlassLM`: stacking blocks, weight tying, cross-entropy loss, and sampling
6. [`src/glasslm/train/trainer.py`](src/glasslm/train/trainer.py) — `Trainer`: `bf16` autocast, gradient accumulation, cosine LR, and checkpointing

---

## Roadmap

- [x] **Milestone 1:** Core Llama-style Transformer, BPE tokenizer, memmap dataset, `bf16` trainer, math & causal unit tests, CLI generation with top-5 token inspector
- [ ] **Milestone 2:** Static KV-cache + speedup benchmark, Logit Lens + attention hooks, interactive FastAPI Glass-Box Web UI
- [ ] **Milestone 3:** Post-training lab (LoRA & DPO from scratch)
- [ ] **Milestone 4:** INT8/INT4 weight quantization & speculative decoding
