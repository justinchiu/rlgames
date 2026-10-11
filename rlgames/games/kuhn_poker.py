"""Kuhn poker.

Three cards (J < Q < K), ante 1, one card each, one betting round where a bet
is 1 chip. Actions are 0 = pass (check or fold) and 1 = bet (bet or call).
The value of the game for player 0 is -1/18.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rlgames.utils import one_hot, select, uniform_over
from rlgames.utils.constants import CHANCE, TERMINAL

N_CARD = 3
N_ACTION = 3  # chance deals one of 3 cards; players use actions 0 and 1
MAX_MOVES = 3


class KuhnState(NamedTuple):
    player: jax.Array
    legal: jax.Array
    chance: jax.Array
    terminal: jax.Array
    returns: jax.Array
    cards: jax.Array  # int32[2], -1 until dealt
    history: jax.Array  # int32[MAX_MOVES], -1 for moves not yet made
    num_moves: jax.Array
    contrib: jax.Array  # float32[2], chips in the pot


class KuhnPoker:
    name = "kuhn_poker"
    num_actions = N_ACTION
    obs_dim = N_CARD + MAX_MOVES * 2

    def init(self) -> KuhnState:
        return KuhnState(
            player=jnp.int32(CHANCE),
            legal=jnp.zeros(N_ACTION, bool),
            chance=uniform_over(jnp.ones(N_CARD, bool)),
            terminal=jnp.bool_(False),
            returns=jnp.zeros(2, jnp.float32),
            cards=jnp.full(2, -1, jnp.int32),
            history=jnp.full(MAX_MOVES, -1, jnp.int32),
            num_moves=jnp.int32(0),
            contrib=jnp.ones(2, jnp.float32),
        )

    def step(self, state: KuhnState, action: jax.Array) -> KuhnState:
        return select(
            state.player == CHANCE, self._deal(state, action), self._act(state, action)
        )

    def _deal(self, state: KuhnState, card: jax.Array) -> KuhnState:
        seat = jnp.where(state.cards[0] < 0, 0, 1)
        cards = state.cards.at[seat].set(card)
        done = seat == 1
        remaining = jnp.ones(N_CARD, bool).at[card].set(False)
        return state._replace(
            player=jnp.where(done, 0, CHANCE).astype(jnp.int32),
            legal=jnp.where(done, jnp.array([True, True, False]), False),
            chance=jnp.where(done, 0.0, uniform_over(remaining)),
            cards=cards,
        )

    def _act(self, state: KuhnState, action: jax.Array) -> KuhnState:
        p = state.player
        facing = state.contrib[p] < state.contrib[1 - p]
        contrib = state.contrib.at[p].add(action.astype(jnp.float32))
        fold = (action == 0) & facing
        call = (action == 1) & facing
        check_check = (action == 0) & ~facing & (state.num_moves == 1)
        terminal = fold | call | check_check

        p0_wins = state.cards[0] > state.cards[1]
        showdown = jnp.where(p0_wins, contrib[1], -contrib[0])
        p0_return = jnp.where(fold, jnp.where(p == 0, -contrib[0], contrib[1]), showdown)

        return state._replace(
            player=jnp.where(terminal, TERMINAL, 1 - p).astype(jnp.int32),
            legal=state.legal & ~terminal,
            terminal=terminal,
            returns=jnp.where(terminal, jnp.stack([p0_return, -p0_return]), 0.0),
            history=state.history.at[state.num_moves].set(action),
            num_moves=state.num_moves + 1,
            contrib=contrib,
        )

    def observe(self, state: KuhnState, player) -> jax.Array:
        card = one_hot(state.cards[player], N_CARD)
        history = one_hot(state.history, 2).reshape(-1)
        return jnp.concatenate([card, history])
