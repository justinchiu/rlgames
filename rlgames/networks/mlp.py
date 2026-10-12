"""Policy-value MLP over a game's observation vector."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class MLPConfig:
    hidden: int = 128
    depth: int = 2


def _dense(key: jax.Array, n_in: int, n_out: int, scale: float = 1.0) -> dict:
    w = jax.random.normal(key, (n_in, n_out)) * scale / jnp.sqrt(n_in)
    return {"w": w, "b": jnp.zeros(n_out)}


def init_mlp(key: jax.Array, obs_dim: int, num_actions: int, cfg: MLPConfig) -> dict:
    keys = jax.random.split(key, cfg.depth + 2)
    sizes = [obs_dim] + [cfg.hidden] * cfg.depth
    return {
        "torso": [_dense(keys[i], sizes[i], sizes[i + 1]) for i in range(cfg.depth)],
        # Small output layers so the initial policy is close to uniform and values near 0.
        "policy": _dense(keys[-2], sizes[-1], num_actions, scale=0.01),
        "value": _dense(keys[-1], sizes[-1], 1, scale=0.01),
    }


def apply_mlp(params: dict, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Returns (logits[..., num_actions], value[...]). Logits are not masked."""
    x = obs
    for layer in params["torso"]:
        x = jax.nn.relu(x @ layer["w"] + layer["b"])
    logits = x @ params["policy"]["w"] + params["policy"]["b"]
    value = (x @ params["value"]["w"] + params["value"]["b"])[..., 0]
    return logits, value


def mlp_policy_fn(params, obs, legal, player) -> jax.Array:
    """`PolicyFn` for `rlgames.core.env.make_rollout`."""
    return apply_mlp(params, obs)[0]
