"""Interface shared by all games.

A game is a set of pure functions over a NamedTuple state, so `jax.jit` and
`jax.vmap` apply to all of them. Chance is explicit: at a chance node
`player == CHANCE` and `chance` holds the outcome probabilities, which lets the
exact tools in `rlgames.core.tree` enumerate every outcome.

Simultaneous moves are sequentialized: the second mover's observation omits
the first mover's action until both have acted.

Every state has these fields:
    player:   int32, 0 or 1 to act, CHANCE at chance nodes, TERMINAL at the end
    legal:    bool[num_actions], legal actions of the player to act
    chance:   float32[num_actions], outcome probabilities at chance nodes, else 0
    terminal: bool
    returns:  float32[2], payoffs at terminal states, returns[0] == -returns[1]
"""

from typing import Any, Protocol

import jax

State = Any


class Game(Protocol):
    name: str
    num_actions: int
    obs_dim: int

    def init(self) -> State:
        """The root state, usually a chance node."""
        ...

    def step(self, state: State, action: jax.Array) -> State:
        """Apply a player or chance action. Undefined on terminal states."""
        ...

    def observe(self, state: State, player: int | jax.Array) -> jax.Array:
        """float32[obs_dim] observation for `player`.

        For the acting player this encodes their information state with perfect
        recall, so (player, observation) identifies an information set.
        """
        ...
