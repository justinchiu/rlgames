"""m,n,k-games: two players alternate marking cells of a rows x cols board and the
first to get k in a row (horizontally, vertically or diagonally) wins. With
`gravity`, a move picks a column and the mark drops to the lowest empty cell.

Tic-tac-toe is (3, 3, 3) and Connect Four is (6, 7, 4) with gravity. These are
the perfect-information games in the package.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from rlgames.utils.constants import TERMINAL

EMPTY = -1


def winning_lines(rows: int, cols: int, k: int) -> np.ndarray:
    """int[L, k] flat cell indices of every line of k cells."""
    lines = []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for r in range(rows):
            for c in range(cols):
                cells = [(r + i * dr, c + i * dc) for i in range(k)]
                if all(0 <= rr < rows and 0 <= cc < cols for rr, cc in cells):
                    lines.append([rr * cols + cc for rr, cc in cells])
    return np.array(lines, dtype=np.int32)


class MNKState(NamedTuple):
    player: jax.Array
    legal: jax.Array
    chance: jax.Array
    terminal: jax.Array
    returns: jax.Array
    board: jax.Array  # int32[rows * cols], EMPTY or the owner's index
    num_moves: jax.Array


class MNKGame:
    def __init__(self, rows: int, cols: int, k: int, gravity: bool = False, name: str = ""):
        self.rows, self.cols, self.k, self.gravity = rows, cols, k, gravity
        self.name = name or f"mnk_{rows}_{cols}_{k}" + ("_gravity" if gravity else "")
        self.num_cells = rows * cols
        self.num_actions = cols if gravity else self.num_cells
        self.obs_dim = 2 * self.num_cells
        self.lines = jnp.asarray(winning_lines(rows, cols, k))

    def init(self) -> MNKState:
        return MNKState(
            player=jnp.int32(0),
            legal=jnp.ones(self.num_actions, bool),
            chance=jnp.zeros(self.num_actions, jnp.float32),
            terminal=jnp.bool_(False),
            returns=jnp.zeros(2, jnp.float32),
            board=jnp.full(self.num_cells, EMPTY, jnp.int32),
            num_moves=jnp.int32(0),
        )

    def _cell(self, board: jax.Array, action: jax.Array) -> jax.Array:
        if not self.gravity:
            return action
        # Row 0 is the top, so the lowest empty cell is the last empty one in the column.
        column = board.reshape(self.rows, self.cols)[:, action]
        row = self.rows - 1 - jnp.argmax((column == EMPTY)[::-1])
        return row * self.cols + action

    def _legal(self, board: jax.Array) -> jax.Array:
        empty = board == EMPTY
        if self.gravity:
            return empty.reshape(self.rows, self.cols)[0]
        return empty

    def step(self, state: MNKState, action: jax.Array) -> MNKState:
        p = state.player
        board = state.board.at[self._cell(state.board, action)].set(p)
        won = (board[self.lines] == p).all(1).any()
        num_moves = state.num_moves + 1
        terminal = won | (num_moves == self.num_cells)
        p0_return = jnp.where(won, jnp.where(p == 0, 1.0, -1.0), 0.0)
        return state._replace(
            player=jnp.where(terminal, TERMINAL, 1 - p).astype(jnp.int32),
            legal=self._legal(board) & ~terminal,
            terminal=terminal,
            returns=jnp.stack([p0_return, -p0_return]),
            board=board,
            num_moves=num_moves,
        )

    def observe(self, state: MNKState, player) -> jax.Array:
        own = state.board == player
        other = state.board == 1 - player
        return jnp.concatenate([own, other]).astype(jnp.float32)


def tic_tac_toe() -> MNKGame:
    return MNKGame(3, 3, 3, name="tic_tac_toe")


def connect_four() -> MNKGame:
    return MNKGame(6, 7, 4, gravity=True, name="connect_four")
