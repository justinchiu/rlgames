"""Batched self-play rollouts.

`make_rollout` builds a jitted function that plays `num_envs` complete games in
parallel from the root, sampling chance outcomes from the game and player
actions from a policy, and returns every player decision as a flat `Batch`.
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from rlgames.games.base import Game
from rlgames.utils import select
from rlgames.utils.constants import CHANCE

# policy_fn(params, obs[N, obs_dim], legal[N, A], player[N]) -> logits[N, A].
# Illegal actions are masked out by the rollout, so the logits there do not matter.
PolicyFn = Callable[[object, jax.Array, jax.Array, jax.Array], jax.Array]


class Batch(NamedTuple):
    """One row per step of every game, flattened to [max_game_length * num_envs].

    Only rows with `mask == True` are player decisions; the others are chance
    steps or padding after the game ended.
    """

    obs: jax.Array  # float32[B, obs_dim], observation of the player to act
    legal: jax.Array  # bool[B, A]
    action: jax.Array  # int32[B]
    player: jax.Array  # int32[B], who acted
    log_prob: jax.Array  # float32[B], log-probability of `action` under the acting policy
    mask: jax.Array  # bool[B], True for player decisions
    outcome: jax.Array  # float32[B], final return of the game for the player who acted
    step: jax.Array  # int32[B], position of the step within its game
    env: jax.Array  # int32[B], which game the step belongs to


class Rollout(NamedTuple):
    batch: Batch
    returns: jax.Array  # float32[num_envs, 2], final returns of each game
    length: jax.Array  # int32[num_envs], steps taken in each game


def make_rollout(game: Game, num_envs: int, policy_fn: PolicyFn):
    """Returns a jitted `rollout(key, params) -> Rollout`."""

    def rollout(key: jax.Array, params) -> Rollout:
        init = jax.tree.map(lambda x: jnp.broadcast_to(x, (num_envs,) + x.shape), game.init())

        def play_step(state, key):
            chance_key, policy_key = jax.random.split(key)
            obs = jax.vmap(game.observe)(state, jnp.maximum(state.player, 0))
            logits = policy_fn(params, obs, state.legal, state.player)
            logits = jnp.where(state.legal, logits, -jnp.inf)
            log_probs = jax.nn.log_softmax(logits)
            decision = (state.player >= 0) & ~state.terminal
            # Sanitize rows without legal actions (chance, terminal) so sampling is defined.
            log_probs = jnp.where(decision[:, None], log_probs, 0.0)
            player_action = jax.random.categorical(policy_key, log_probs)
            chance_action = jax.random.categorical(chance_key, jnp.log(state.chance))
            action = jnp.where(state.player == CHANCE, chance_action, player_action)

            next_state = jax.vmap(game.step)(state, action)
            next_state = select(state.terminal, state, next_state)
            record = dict(
                obs=obs,
                legal=state.legal,
                action=action.astype(jnp.int32),
                player=state.player,
                log_prob=jnp.take_along_axis(log_probs, player_action[:, None], 1)[:, 0],
                mask=decision,
                active=~state.terminal,
            )
            return next_state, record

        keys = jax.random.split(key, game.max_game_length)
        final, rec = jax.lax.scan(play_step, init, keys)

        t, n = game.max_game_length, num_envs
        player = jnp.maximum(rec["player"], 0)
        outcome = jnp.take_along_axis(final.returns[None], player[..., None], 2)[..., 0]
        batch = Batch(
            obs=rec["obs"].reshape(t * n, -1),
            legal=rec["legal"].reshape(t * n, -1),
            action=rec["action"].reshape(-1),
            player=rec["player"].reshape(-1),
            log_prob=jnp.where(rec["mask"], rec["log_prob"], 0.0).reshape(-1),
            mask=rec["mask"].reshape(-1),
            outcome=jnp.where(rec["mask"], outcome, 0.0).reshape(-1),
            step=jnp.broadcast_to(jnp.arange(t)[:, None], (t, n)).reshape(-1),
            env=jnp.broadcast_to(jnp.arange(n)[None], (t, n)).reshape(-1),
        )
        return Rollout(batch=batch, returns=final.returns, length=rec["active"].sum(0))

    return jax.jit(rollout)


def uniform_policy_fn(params, obs, legal, player) -> jax.Array:
    return jnp.zeros(legal.shape, jnp.float32)
