import math

def cosine_lr_with_warmup(optimizer, step, total_steps, warmup_steps, peak_lr, ):
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be >= 0 and < total_steps")

    step = min(max(step, 0), total_steps - 1)

    if warmup_steps > 0 and step < warmup_steps:
        lr = peak_lr * (step + 1) / warmup_steps
    else:
        decay_steps = total_steps - warmup_steps

        if decay_steps <= 1:
            progress = 1.0
        else:
            progress = (step - warmup_steps) / (decay_steps - 1)

        progress = min(max(progress, 0.0), 1.0)

        lr = peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

    for group in optimizer.param_groups:
        group["lr"] = lr

    return lr