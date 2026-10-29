import torch

from utils.optimizers import LARS, SAM

def build_optimizer(
    name,
    parameters,
    learning_rate,
    weight_decay,
    momentum=0.9,
    eta=0.001,
    sam_rho=0.05,
):
    name = name.lower()

    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    elif name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    elif name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    elif name == "lars":
        return LARS(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            eta=eta,
            weight_decay_filter=True,
            lars_adaptation_filter=True,
        )

    elif name == "sam":
        return SAM(
            parameters,
            torch.optim.SGD,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            rho=sam_rho,
        )

    raise ValueError(f"Unsupported optimizer: {name}")
