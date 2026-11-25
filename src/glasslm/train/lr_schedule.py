"""Learning rate schedules for LLM training."""

import math


def get_cosine_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Linear warmup followed by cosine decay down to min_lr."""
    if warmup_steps > 0 and step < warmup_steps:
        return max_lr * float(step + 1) / float(warmup_steps)
    if step >= max_steps:
        return min_lr

    decay_steps = max(1, max_steps - warmup_steps)
    progress = float(step - warmup_steps) / float(decay_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (max_lr - min_lr)
