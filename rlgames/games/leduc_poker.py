"""Leduc poker.

Six cards (two suits of J, Q, K), ante 1, one private card each. Two betting
rounds with a public card dealt between them; raises are 2 in the first round
and 4 in the second, with at most two raises per round. At showdown a pair with
the public card wins, otherwise the higher card wins, and equal ranks split.

Actions are 0 = fold, 1 = call (or check), 2 = raise (or bet). Folding is only
legal when facing a bet. The value of the game for player 0 is about -0.0856.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rlgames.utils import one_hot, select, uniform_over
from rlgames.utils.constants import CHANCE, TERMINAL

N_CARD = 6
N_RANK = 3
N_ACTION = 6  # chance deals one of 6 cards; players use actions 0, 1 and 2
N_PLAYER_ACTION = 3
FOLD, CALL, RAISE = 0, 1, 2
MAX_RAISES = 2
MAX_ROUND_MOVES = 4  # check, raise, raise, call
RAISE_SIZES = (2.0, 4.0)


class LeducState(NamedTuple):
    player: jax.Array
    legal: jax.Array
    chance: jax.Array
    terminal: jax.Array
    returns: jax.Array
    cards: jax.Array  # int32[2], -1 until dealt
    public: jax.Array  # int32, -1 until dealt
    round: jax.Array
    contrib: jax.Array  # float32[2], chips in the pot
    num_raises: jax.Array  # raises in the current round
    round_moves: jax.Array  # moves in the current round
    history: jax.Array  # int32[2, MAX_ROUND_MOVES], -1 for moves not yet made


def _legal(contrib: jax.Array, player: jax.Array, num_raises: jax.Array) -> jax.Array:
    facing = contrib[player] < contrib[1 - player]
    actions = jnp.array([facing, True, num_raises < MAX_RAISES])
    return jnp.zeros(N_ACTION, bool).at[:N_PLAYER_ACTION].set(actions)


def _remaining(cards: jax.Array, public: jax.Array) -> jax.Array:
    dealt = one_hot(cards, N_CARD).sum(0) + one_hot(public, N_CARD)
    return dealt == 0


class LeducPoker:
    name = "leduc_poker"
    num_actions = N_ACTION
    obs_dim = 2 * N_RANK + 2 * MAX_ROUND_MOVES * N_PLAYER_ACTION

    def init(self) -> LeducState:
        return LeducState(
            player=jnp.int32(CHANCE),
            legal=jnp.zeros(N_ACTION, bool),
            chance=uniform_over(jnp.ones(N_CARD, bool)),
            terminal=jnp.bool_(False),
            returns=jnp.zeros(2, jnp.float32),
            cards=jnp.full(2, -1, jnp.int32),
            public=jnp.int32(-1),
            round=jnp.int32(0),
            contrib=jnp.ones(2, jnp.float32),
            num_raises=jnp.int32(0),
            round_moves=jnp.int32(0),
            history=jnp.full((2, MAX_ROUND_MOVES), -1, jnp.int32),
        )

    def step(self, state: LeducState, action: jax.Array) -> LeducState:
        return select(
            state.player == CHANCE, self._deal(state, action), self._act(state, action)
        )

    def _deal(self, state: LeducState, card: jax.Array) -> LeducState:
        to_p0 = state.cards[0] < 0
        to_p1 = ~to_p0 & (state.cards[1] < 0)
        to_public = ~to_p0 & ~to_p1
        cards = jnp.where(to_p0, state.cards.at[0].set(card), state.cards)
        cards = jnp.where(to_p1, cards.at[1].set(card), cards)
        public = jnp.where(to_public, card, state.public)
        betting = ~to_p0
        return state._replace(
            player=jnp.where(betting, 0, CHANCE).astype(jnp.int32),
            legal=jnp.where(betting, _legal(state.contrib, jnp.int32(0), state.num_raises), False),
            chance=jnp.where(betting, 0.0, uniform_over(_remaining(cards, public))),
            cards=cards,
            public=public.astype(jnp.int32),
        )

    def _act(self, state: LeducState, action: jax.Array) -> LeducState:
        p = state.player
        raise_size = jnp.where(state.round == 0, RAISE_SIZES[0], RAISE_SIZES[1])
        top = state.contrib.max()
        new_contrib = jnp.where(action == RAISE, top + raise_size, top)
        contrib = jnp.where(action == FOLD, state.contrib, state.contrib.at[p].set(new_contrib))
        num_raises = state.num_raises + (action == RAISE)
        history = state.history.at[state.round, state.round_moves].set(action)

        fold = action == FOLD
        round_over = (action == CALL) & (state.round_moves >= 1)
        showdown = round_over & (state.round == 1)
        deal_public = round_over & (state.round == 0)
        terminal = fold | showdown

        ranks = state.cards // 2
        public_rank = state.public // 2
        strength = ranks + N_RANK * (ranks == public_rank)
        p0_showdown = jnp.sign(strength[0] - strength[1]) * contrib[0]
        p0_fold = jnp.where(p == 0, -contrib[0], contrib[1])
        p0_return = jnp.where(fold, p0_fold, p0_showdown)

        next_player = jnp.where(deal_public, CHANCE, 1 - p)
        next_player = jnp.where(terminal, TERMINAL, next_player)
        next_raises = jnp.where(deal_public, 0, num_raises)
        legal = jnp.where(
            terminal | deal_public, False, _legal(contrib, 1 - p, next_raises)
        )
        chance = jnp.where(deal_public, uniform_over(_remaining(state.cards, state.public)), 0.0)
        return state._replace(
            player=next_player.astype(jnp.int32),
            legal=legal,
            chance=chance,
            terminal=terminal,
            returns=jnp.where(terminal, jnp.stack([p0_return, -p0_return]), 0.0),
            round=state.round + deal_public,
            contrib=contrib,
            num_raises=next_raises.astype(jnp.int32),
            round_moves=jnp.where(deal_public, 0, state.round_moves + 1).astype(jnp.int32),
            history=history,
        )

    def observe(self, state: LeducState, player) -> jax.Array:
        # Suits never matter, so cards are encoded by rank. This merges histories that
        # differ only by suit into one information set, which leaves the equilibrium intact.
        private = one_hot(state.cards[player] // 2, N_RANK)
        public = one_hot(jnp.where(state.public >= 0, state.public // 2, -1), N_RANK)
        history = one_hot(state.history, N_PLAYER_ACTION).reshape(-1)
        return jnp.concatenate([private, public, history])
