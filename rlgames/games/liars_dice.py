"""Liar's dice.

Each player rolls `num_dice` private dice with `num_faces` faces. Players take
turns, starting with player 0, either raising the bid "at least q dice show
face f" or calling liar on the previous bid. Bids are ordered by quantity, then
face. On a liar call the dice are revealed: if the bid holds, the bidder wins,
otherwise the caller wins. No face is wild.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from rlgames.utils import one_hot, select, uniform_over
from rlgames.utils.constants import CHANCE, TERMINAL


class LiarsDiceState(NamedTuple):
    player: jax.Array
    legal: jax.Array
    chance: jax.Array
    terminal: jax.Array
    returns: jax.Array
    dice: jax.Array  # int32[2, num_dice], face index, -1 until rolled
    num_rolled: jax.Array
    bids: jax.Array  # bool[num_bids], bids made so far
    last_bid: jax.Array  # -1 before the first bid


class LiarsDice:
    def __init__(self, num_dice: int = 1, num_faces: int = 6):
        self.num_dice = num_dice
        self.num_faces = num_faces
        self.num_bids = 2 * num_dice * num_faces
        self.liar = self.num_bids  # action index of calling liar
        self.name = f"liars_dice_{num_dice}d{num_faces}"
        self.num_actions = max(num_faces, self.num_bids + 1)
        self.obs_dim = num_dice * num_faces + self.num_bids
        self.max_game_length = 2 * num_dice + self.num_bids + 1

    def init(self) -> LiarsDiceState:
        a = self.num_actions
        return LiarsDiceState(
            player=jnp.int32(CHANCE),
            legal=jnp.zeros(a, bool),
            chance=self._roll_probs(),
            terminal=jnp.bool_(False),
            returns=jnp.zeros(2, jnp.float32),
            dice=jnp.full((2, self.num_dice), -1, jnp.int32),
            num_rolled=jnp.int32(0),
            bids=jnp.zeros(self.num_bids, bool),
            last_bid=jnp.int32(-1),
        )

    def _roll_probs(self) -> jax.Array:
        faces = jnp.arange(self.num_actions) < self.num_faces
        return uniform_over(faces)

    def _legal(self, last_bid: jax.Array) -> jax.Array:
        actions = jnp.arange(self.num_actions)
        raise_bid = (actions > last_bid) & (actions < self.num_bids)
        return raise_bid | ((actions == self.liar) & (last_bid >= 0))

    def step(self, state: LiarsDiceState, action: jax.Array) -> LiarsDiceState:
        return select(
            state.player == CHANCE, self._roll(state, action), self._act(state, action)
        )

    def _roll(self, state: LiarsDiceState, face: jax.Array) -> LiarsDiceState:
        seat, die = jnp.divmod(state.num_rolled, self.num_dice)
        num_rolled = state.num_rolled + 1
        done = num_rolled == 2 * self.num_dice
        return state._replace(
            player=jnp.where(done, 0, CHANCE).astype(jnp.int32),
            legal=jnp.where(done, self._legal(state.last_bid), False),
            chance=jnp.where(done, 0.0, self._roll_probs()),
            dice=state.dice.at[seat, die].set(face),
            num_rolled=num_rolled,
        )

    def _act(self, state: LiarsDiceState, action: jax.Array) -> LiarsDiceState:
        p = state.player
        call = action == self.liar
        quantity = state.last_bid // self.num_faces + 1
        face = state.last_bid % self.num_faces
        bid_holds = (state.dice == face).sum() >= quantity
        # The previous bid was made by the other player.
        p_wins = jnp.where(bid_holds, -1.0, 1.0)
        p0_return = jnp.where(p == 0, p_wins, -p_wins)
        # The highest bid can only be answered by calling liar, so play always ends.
        return state._replace(
            player=jnp.where(call, TERMINAL, 1 - p).astype(jnp.int32),
            legal=jnp.where(call, False, self._legal(action)),
            terminal=call,
            returns=jnp.where(call, jnp.stack([p0_return, -p0_return]), 0.0),
            bids=jnp.where(call, state.bids, state.bids.at[action].set(True)),
            last_bid=jnp.where(call, state.last_bid, action).astype(jnp.int32),
        )

    def observe(self, state: LiarsDiceState, player) -> jax.Array:
        # Bids strictly increase and alternate starting with player 0, so the set of bids
        # made determines the whole bidding history.
        dice = one_hot(state.dice[player], self.num_faces).reshape(-1)
        return jnp.concatenate([dice, state.bids.astype(jnp.float32)])
