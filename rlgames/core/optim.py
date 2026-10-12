"""Optimizer construction."""

from dataclasses import dataclass

import optax


@dataclass
class OptimConfig:
    name: str = "adamw"  # adamw | sgd | muon
    lr: float = 3e-4
    weight_decay: float = 0.0
    b1: float = 0.9
    b2: float = 0.999
    max_grad_norm: float = 1.0  # 0 disables clipping


def make_optimizer(cfg: OptimConfig) -> optax.GradientTransformation:
    if cfg.name == "adamw":
        opt = optax.adamw(cfg.lr, b1=cfg.b1, b2=cfg.b2, weight_decay=cfg.weight_decay)
    elif cfg.name == "sgd":
        opt = optax.sgd(cfg.lr)
    elif cfg.name == "muon":
        # Muon on 2-D weight matrices, Adam on everything else (biases).
        opt = optax.contrib.muon(
            cfg.lr, adam_b1=cfg.b1, adam_b2=cfg.b2, weight_decay=cfg.weight_decay
        )
    else:
        raise ValueError(f"unknown optimizer {cfg.name}")
    if cfg.max_grad_norm > 0:
        opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), opt)
    return opt
