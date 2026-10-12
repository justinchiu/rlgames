"""Self-play training loop.

    uv run python scripts/train/selfplay_main.py --game kuhn_poker --save_dir exps/kuhn

Writes to `save_dir`:
    cfg.json        the full config
    metrics.jsonl   one line per evaluation
    latest.pkl      the most recent checkpoint (the play server loads this)
    ckpt_<i>.pkl    periodic checkpoints
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field

import jax
import jax.numpy as jnp
import pyrallis

from rlgames import utils
from rlgames.core.env import make_rollout
from rlgames.core.evaluation import make_versus_eval, network_exploitability, tabulate_policy
from rlgames.core.learner import Learner, LearnerConfig
from rlgames.core.optim import OptimConfig, make_optimizer
from rlgames.core.tree import enumerate_tree
from rlgames.games import make
from rlgames.networks.mlp import MLPConfig, apply_mlp, init_mlp, mlp_policy_fn

# Games with more nodes than this are evaluated only against a random opponent.
MAX_EXACT_NODES = 1_000_000
# Games with at most this many information sets log their full policy (e.g. matrix games).
MAX_LOGGED_INFOSETS = 4


@dataclass
class MainConfig:
    game: str = "kuhn_poker"
    game_kwargs: dict = field(default_factory=dict)
    seed: int = 1
    num_envs: int = 1024
    num_iters: int = 2000
    eval_every: int = 20
    save_every: int = 20
    keep_every: int = 500
    num_eval_envs: int = 2048
    save_dir: str = "exps/dev"
    # Skip learning; useful for checking the pipeline before implementing the learner.
    dry_run: bool = False
    network: MLPConfig = field(default_factory=MLPConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    learner: LearnerConfig = field(default_factory=LearnerConfig)


def make_policy_stats():
    @jax.jit
    def policy_stats(params, batch):
        logits, _ = apply_mlp(params, batch.obs)
        log_probs = jax.nn.log_softmax(jnp.where(batch.legal, logits, -jnp.inf))
        probs = jnp.exp(log_probs)
        entropy = -jnp.where(batch.legal, probs * log_probs, 0.0).sum(-1)
        n = jnp.maximum(batch.mask.sum(), 1)
        return jnp.where(batch.mask, entropy, 0.0).sum() / n

    return policy_stats


def run_training(cfg: MainConfig):
    os.makedirs(cfg.save_dir, exist_ok=True)
    with open(os.path.join(cfg.save_dir, "cfg.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(json.dumps(asdict(cfg), indent=2))

    game = make(cfg.game, **cfg.game_kwargs)
    key = jax.random.key(cfg.seed)
    key, init_key = jax.random.split(key)
    params = init_mlp(init_key, game.obs_dim, game.num_actions, cfg.network)

    learner = Learner(cfg.learner, make_optimizer(cfg.optim))
    opt_state = learner.init(params)

    rollout = make_rollout(game, cfg.num_envs, mlp_policy_fn)
    versus_random = make_versus_eval(game, cfg.num_eval_envs, mlp_policy_fn)
    policy_stats = make_policy_stats()

    tree = None
    try:
        tree = enumerate_tree(game, max_nodes=MAX_EXACT_NODES)
        print(f"{game.name}: {tree.num_nodes} nodes, {tree.num_infosets} infosets; "
              "logging exact exploitability")
    except ValueError:
        print(f"{game.name}: too large for exact exploitability; logging return vs random")

    metrics = utils.MetricsWriter(cfg.save_dir)
    start = time.time()
    for it in range(cfg.num_iters + 1):
        key, rollout_key, update_key, eval_key = jax.random.split(key, 4)
        out = rollout(rollout_key, params)

        record = {}
        if it % cfg.eval_every == 0:
            record = {
                "iter": it,
                "games": it * cfg.num_envs,
                "time": time.time() - start,
                "game_length": float(out.length.mean()),
                "p0_return": float(out.returns[:, 0].mean()),
                "entropy": float(policy_stats(params, out.batch)),
                "vs_random": versus_random(eval_key, params),
            }
            if tree is not None:
                record["exploitability"] = network_exploitability(tree, params)
                if tree.num_infosets <= MAX_LOGGED_INFOSETS:
                    record["policy"] = tabulate_policy(tree, params).tolist()

        if it % cfg.save_every == 0:
            ckpt = {"iter": it, "game": cfg.game, "game_kwargs": cfg.game_kwargs,
                    "network": asdict(cfg.network), "params": params}
            utils.save_pickle(os.path.join(cfg.save_dir, "latest.pkl"), ckpt)
            if it % cfg.keep_every == 0:
                utils.save_pickle(os.path.join(cfg.save_dir, f"ckpt_{it}.pkl"), ckpt)

        if it < cfg.num_iters and not cfg.dry_run:
            params, opt_state, learner_metrics = learner.update(
                params, opt_state, out.batch, update_key
            )
            if record:
                record.update({k: float(v) for k, v in learner_metrics.items()})

        if record:
            metrics.write(record)
            summary = ", ".join(
                f"{k}={v:.4g}" for k, v in record.items() if isinstance(v, (int, float))
            )
            print(summary, flush=True)


if __name__ == "__main__":
    run_training(pyrallis.parse(config_class=MainConfig))
