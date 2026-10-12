"""Two-player zero-sum matrix games, played as player 0 then player 1.

Player 1 does not observe player 0's action, so this is equivalent to a
simultaneous-move game.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rlgames.utils.constants import TERMINAL


class MatrixState(NamedTuple):
    player: jax.Array
    legal: jax.Array
    chance: jax.Array
    terminal: jax.Array
    returns: jax.Array
    action0: jax.Array
    action1: jax.Array


class MatrixGame:
    obs_dim = 1
    max_game_length = 2

    def __init__(self, payoffs, name: str = "matrix"):
        """`payoffs[i, j]` is player 0's payoff when the players pick i and j."""
        payoffs = np.asarray(payoffs, dtype=np.float32)
        if payoffs.ndim != 2 or payoffs.shape[0] != payoffs.shape[1]:
            raise ValueError("payoff matrix must be square")
        self.name = name
        self.payoffs = jnp.asarray(payoffs)
        self.num_actions = payoffs.shape[0]

    def init(self) -> MatrixState:
        n = self.num_actions
        return MatrixState(
            player=jnp.int32(0),
            legal=jnp.ones(n, bool),
            chance=jnp.zeros(n, jnp.float32),
            terminal=jnp.bool_(False),
            returns=jnp.zeros(2, jnp.float32),
            action0=jnp.int32(-1),
            action1=jnp.int32(-1),
        )

    def step(self, state: MatrixState, action: jax.Array) -> MatrixState:
        first = state.player == 0
        payoff = self.payoffs[state.action0, action]
        return state._replace(
            player=jnp.where(first, 1, TERMINAL).astype(jnp.int32),
            legal=state.legal & first,
            terminal=~first,
            returns=jnp.where(first, state.returns, jnp.stack([payoff, -payoff])),
            action0=jnp.where(first, action, state.action0).astype(jnp.int32),
            action1=jnp.where(first, state.action1, action).astype(jnp.int32),
        )

    def observe(self, state: MatrixState, player) -> jax.Array:
        # Neither player has private information before acting.
        return jnp.zeros(1, jnp.float32)


def matching_pennies() -> MatrixGame:
    return MatrixGame([[1, -1], [-1, 1]], name="matching_pennies")


def rock_paper_scissors() -> MatrixGame:
    return MatrixGame([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], name="rock_paper_scissors")


def biased_rock_paper_scissors() -> MatrixGame:
    """RPS where wins with rock or scissors pay 2, so the Nash equilibrium is
    (1/4, 1/2, 1/4) over rock, paper, scissors."""
    return MatrixGame([[0, -1, 2], [1, 0, -1], [-2, 1, 0]], name="biased_rock_paper_scissors")
