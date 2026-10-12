import json
import os
import random
import tempfile
import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import optax

from rlgames import utils
from rlgames.core.env import make_rollout
from rlgames.core.learner import Learner
from rlgames.core.optim import OptimConfig
from rlgames.games import GAMES, make
from rlgames.networks.mlp import MLPConfig, init_mlp, mlp_policy_fn
from scripts.play.server import App
from scripts.train.selfplay_main import MainConfig, run_training


def weight_decay_update(self, params, opt_state, batch, key):
    """Placeholder update that only exercises the optimizer plumbing: it shrinks the
    weights toward zero and ignores the batch."""
    loss, grads = jax.value_and_grad(
        lambda p: sum(jnp.sum(x**2) for x in jax.tree.leaves(p))
    )(params)
    updates, opt_state = self.optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates), opt_state, {"loss": loss}


class PipelineTest(unittest.TestCase):
    def test_rollout_batch(self):
        game = make("leduc_poker")
        params = init_mlp(jax.random.key(0), game.obs_dim, game.num_actions, MLPConfig())
        out = make_rollout(game, 64, mlp_policy_fn)(jax.random.key(1), params)
        batch = out.batch
        self.assertEqual(batch.obs.shape, (game.max_game_length * 64, game.obs_dim))
        mask = batch.mask
        # Every decision took a legal action, and outcomes match the game returns.
        self.assertTrue(bool(batch.legal[mask, batch.action[mask]].all()))
        expected = out.returns[batch.env, jnp.maximum(batch.player, 0)]
        self.assertTrue(bool(jnp.allclose(jnp.where(mask, expected, 0.0), batch.outcome)))
        self.assertTrue(bool((batch.log_prob[mask] <= 0).all()))

    def test_training_writes_run(self):
        for optim in ("adamw", "sgd", "muon"):
            with tempfile.TemporaryDirectory() as save_dir, \
                    mock.patch.object(Learner, "update", weight_decay_update):
                cfg = MainConfig(game="kuhn_poker", num_envs=32, num_iters=4, eval_every=2,
                                 save_every=2, num_eval_envs=32, save_dir=save_dir,
                                 optim=OptimConfig(name=optim))
                run_training(cfg)
                with open(os.path.join(save_dir, "metrics.jsonl")) as f:
                    records = [json.loads(line) for line in f]
                self.assertEqual([r["iter"] for r in records], [0, 2, 4])
                self.assertIn("loss", records[1])
                self.assertIn("exploitability", records[0])
                ckpt = utils.load_pickle(os.path.join(save_dir, "latest.pkl"))
                self.assertEqual(ckpt["iter"], 4)

    def test_server_plays_every_game(self):
        with tempfile.TemporaryDirectory() as exps_dir:
            for name in GAMES:
                cfg = MainConfig(game=name, num_envs=8, num_iters=0, num_eval_envs=8,
                                 save_dir=os.path.join(exps_dir, name), dry_run=True)
                run_training(cfg)
            app = App(exps_dir)
            self.assertEqual(len(app.runs()), len(GAMES))
            for name in GAMES:
                for seat in (0, 1):
                    state = app.new_game(name, seat, greedy=False)
                    while not state["terminal"]:
                        self.assertTrue(state["actions"], name)
                        state = app.act(state["session"], random.choice(state["actions"])["id"])
                    self.assertIn("result", state)


if __name__ == "__main__":
    unittest.main()
