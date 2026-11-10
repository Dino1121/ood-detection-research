import torch


def energy_score(logits, temperature=1.0):
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    return temperature * torch.logsumexp(
        logits / temperature,
        dim=1
    )
