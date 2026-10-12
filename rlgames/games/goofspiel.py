"""Goofspiel.

Each player holds cards 1..K and K prize cards are revealed one at a time in
random order. For each prize both players bid a card simultaneously; the higher
bid wins the prize's value and equal bids discard it. Bids are revealed after
each round. Sequentialized as chance (prize), player 0 (bid), player 1 (bid).
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rlgames.utils import one_hot, select, uniform_over
from rlgames.utils.constants import CHANCE, TERMINAL


class GoofspielState(NamedTuple):
    player: jax.Array
    legal: jax.Array
    chance: jax.Array
    terminal: jax.Array
    returns: jax.Array
    hands: jax.Array  # bool[2, K], cards still held
    prizes: jax.Array  # int32[K], prize revealed in each round, -1 for future rounds
    bids: jax.Array  # int32[2, K], bid in each round, -1 for future rounds
    round: jax.Array  # number of completed rounds
    points: jax.Array  # float32[2]


class Goofspiel:
    def __init__(self, num_cards: int = 4, win_loss: bool = True):
        """With `win_loss` the return is the sign of the point difference, otherwise
        the point difference itself."""
        self.num_cards = num_cards
        self.win_loss = win_loss
        self.name = f"goofspiel_{num_cards}"
        self.num_actions = num_cards
        self.obs_dim = 3 * num_cards * num_cards
        self.max_game_length = 3 * num_cards

    def init(self) -> GoofspielState:
        k = self.num_cards
        return GoofspielState(
            player=jnp.int32(CHANCE),
            legal=jnp.zeros(k, bool),
            chance=uniform_over(jnp.ones(k, bool)),
            terminal=jnp.bool_(False),
            returns=jnp.zeros(2, jnp.float32),
            hands=jnp.ones((2, k), bool),
            prizes=jnp.full(k, -1, jnp.int32),
            bids=jnp.full((2, k), -1, jnp.int32),
            round=jnp.int32(0),
            points=jnp.zeros(2, jnp.float32),
        )

    def step(self, state: GoofspielState, action: jax.Array) -> GoofspielState:
        return select(
            state.player == CHANCE,
            self._reveal(state, action),
            select(state.player == 0, self._bid0(state, action), self._bid1(state, action)),
        )

    def _reveal(self, state: GoofspielState, prize: jax.Array) -> GoofspielState:
        return state._replace(
            player=jnp.int32(0),
            legal=state.hands[0],
            chance=jnp.zeros(self.num_cards, jnp.float32),
            prizes=state.prizes.at[state.round].set(prize),
        )

    def _bid0(self, state: GoofspielState, card: jax.Array) -> GoofspielState:
        return state._replace(
            player=jnp.int32(1),
            legal=state.hands[1],
            hands=state.hands.at[0, card].set(False),
            bids=state.bids.at[0, state.round].set(card),
        )

    def _bid1(self, state: GoofspielState, card: jax.Array) -> GoofspielState:
        bid0 = state.bids[0, state.round]
        value = (state.prizes[state.round] + 1).astype(jnp.float32)
        points = state.points.at[0].add(jnp.where(bid0 > card, value, 0.0))
        points = points.at[1].add(jnp.where(card > bid0, value, 0.0))
        hands = state.hands.at[1, card].set(False)
        round = state.round + 1
        terminal = round == self.num_cards

        diff = points[0] - points[1]
        p0_return = jnp.sign(diff) if self.win_loss else diff
        prizes_left = one_hot(state.prizes, self.num_cards).sum(0) == 0
        return state._replace(
            player=jnp.where(terminal, TERMINAL, CHANCE).astype(jnp.int32),
            legal=jnp.zeros(self.num_cards, bool),
            chance=jnp.where(terminal, 0.0, uniform_over(prizes_left)),
            terminal=terminal,
            returns=jnp.where(terminal, jnp.stack([p0_return, -p0_return]), 0.0),
            hands=hands,
            bids=state.bids.at[1, state.round].set(card),
            round=round,
            points=points,
        )

    def observe(self, state: GoofspielState, player) -> jax.Array:
        k = self.num_cards
        completed = jnp.arange(k) < state.round
        opponent_bids = jnp.where(completed, state.bids[1 - player], -1)
        return jnp.concatenate(
            [
                one_hot(state.prizes, k).reshape(-1),
                one_hot(state.bids[player], k).reshape(-1),
                one_hot(opponent_bids, k).reshape(-1),
            ]
        )
