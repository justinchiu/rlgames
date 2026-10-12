"""Your learning algorithm goes here.

The training loop (`scripts/train/selfplay_main.py`) repeats:

    rollout = play `num_envs` complete games, both seats using the current params
    params, opt_state, metrics = learner.update(params, opt_state, rollout.batch, key)

and periodically evaluates, logs and checkpoints. Everything except `update` is
already wired up, so this file is the only place an algorithm needs to live.

What you get in `batch` (see `rlgames.core.env.Batch`), one row per game step,
flattened over [max_game_length * num_envs]:

    batch.obs       float32[B, obs_dim]  observation of the player to act
    batch.legal     bool[B, A]           legal actions at that step
    batch.action    int32[B]             action that was taken
    batch.player    int32[B]             who took it (0 or 1)
    batch.log_prob  float32[B]           log-probability of the action when it was sampled
    batch.mask      bool[B]              True for real player decisions; False rows are
                                         chance steps or padding and must be ignored
    batch.outcome   float32[B]           final return of that game for the player who acted
    batch.step      int32[B]             position of the step within its game
    batch.env       int32[B]             which game the step came from

Useful pieces:

    from rlgames.networks.mlp import apply_mlp
    logits, value = apply_mlp(params, batch.obs)       # [B, A], [B]
    logits = jnp.where(batch.legal, logits, -jnp.inf)  # mask illegal actions
    jax.nn.log_softmax(logits)                         # log-probabilities

    loss, grads = jax.value_and_grad(your_loss)(params, ...)
    updates, opt_state = self.optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)

Wrap the math in a function decorated with `jax.jit` so it compiles once.
Return any scalars you want plotted in `metrics`; they show up on the dashboard.
"""

from dataclasses import dataclass

import jax
import optax

from rlgames.core.env import Batch


@dataclass
class LearnerConfig:
    """Hyperparameters for your algorithm. Fields added here become command-line
    flags, e.g. a field `temperature` is set with `--learner.temperature 0.1`."""

    example_coef: float = 0.0


class Learner:
    def __init__(self, cfg: LearnerConfig, optimizer: optax.GradientTransformation):
        self.cfg = cfg
        self.optimizer = optimizer

    def init(self, params) -> optax.OptState:
        """Called once before training. Set up any extra state you need here too,
        e.g. `self.reference_params = params`."""
        return self.optimizer.init(params)

    def update(
        self, params, opt_state: optax.OptState, batch: Batch, key: jax.Array
    ) -> tuple[object, optax.OptState, dict]:
        """One learning step on a fresh batch of self-play games.

        Returns (new_params, new_opt_state, metrics), where metrics maps names to
        scalars to log.
        """
        raise NotImplementedError(
            "Implement Learner.update in rlgames/core/learner.py, "
            "or pass --dry_run true to run the pipeline without learning."
        )
