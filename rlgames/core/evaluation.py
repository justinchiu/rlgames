"""Evaluation of a network policy: exact exploitability on small games and average
return against a fixed opponent on any game."""

import jax
import jax.numpy as jnp
import numpy as np

from rlgames.core.env import PolicyFn, make_rollout, uniform_policy_fn
from rlgames.core.exploitability import exploitability
from rlgames.core.tree import GameTree
from rlgames.games.base import Game
from rlgames.networks.mlp import apply_mlp


def tabulate_policy(tree: GameTree, params) -> np.ndarray:
    """float[num_infosets, num_actions] action probabilities of the network at every
    information set of the tree."""
    logits, _ = apply_mlp(params, jnp.asarray(tree.infoset_obs))
    logits = jnp.where(tree.infoset_legal, logits, -jnp.inf)
    return np.asarray(jax.nn.softmax(logits), dtype=np.float64)


def network_exploitability(tree: GameTree, params) -> float:
    return exploitability(tree, tabulate_policy(tree, params))


def make_versus_eval(game: Game, num_envs: int, policy_fn: PolicyFn, opponent_fn: PolicyFn = None):
    """Returns `evaluate(key, params) -> float`, the average return of `policy_fn`
    against `opponent_fn` (uniform random by default), averaged over both seats."""
    opponent_fn = opponent_fn or uniform_policy_fn
    rollouts = []
    for seat in (0, 1):

        def seated(params, obs, legal, player, seat=seat):
            ours = policy_fn(params, obs, legal, player)
            theirs = opponent_fn(None, obs, legal, player)
            return jnp.where((player == seat)[:, None], ours, theirs)

        rollouts.append(make_rollout(game, num_envs, seated))

    def evaluate(key, params) -> float:
        k0, k1 = jax.random.split(key)
        r0 = rollouts[0](k0, params).returns[:, 0].mean()
        r1 = rollouts[1](k1, params).returns[:, 1].mean()
        return float((r0 + r1) / 2)

    return evaluate
