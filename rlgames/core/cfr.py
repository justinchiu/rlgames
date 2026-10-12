"""Your CFR+ goes here. tests/test_cfr.py checks it against the reference
implementation in rlgames/reference/cfr.py (spoiler: don't read it until you're stuck).

Run the tests with

    uv run pytest tests/test_cfr.py -v

They skip until a function stops raising NotImplementedError, so you can work on
one piece at a time: `regret_matching` first, then `CFRPlus`.

A tabular policy is float[num_infosets, num_actions]: row i is the action distribution
at information set i, zero on illegal actions. Useful arrays on `tree` (see
rlgames/core/tree.py): `infoset_legal`, `infoset_player`, and for every node `player`,
`terminal`, `infoset`, `children`. Useful helpers there: `node_policy`, `reach_probs`,
`child_values`, `expected_values`. `expected_values` gives player 0's value; player 1's is
its negation.

To match the reference exactly, follow these conventions:
- `iterate` updates player 0 and then player 1 (alternating updates). Each player's
  update uses the current policy at that moment, so player 1 sees player 0's new regrets.
- The average policy at iteration t (counting from 1) adds t**avg_power times the
  player's own reach probability times the current policy. If a player's own reach to an
  information set differs between its histories (transpositions in tic-tac-toe), sum it.
- Where regrets or the average weights are all zero, the policy is uniform over legal
  actions.
"""

from dataclasses import dataclass

import numpy as np

from rlgames.core.tree import GameTree


def regret_matching(regrets: np.ndarray, legal: np.ndarray) -> np.ndarray:
    """float[I, A] regrets and bool[I, A] legal actions -> float[I, A] policy."""
    raise NotImplementedError("Implement regret_matching in rlgames/core/cfr.py")


@dataclass
class CFRConfig:
    num_iters: int = 1000
    # Weight iteration t's policy by t**avg_power in the average policy (1 = linear).
    avg_power: float = 1.0


class CFRPlus:
    def __init__(self, tree: GameTree, cfg: CFRConfig = CFRConfig()):
        self.tree = tree
        self.cfg = cfg

    def current_policy(self) -> np.ndarray:
        """The policy this iteration plays."""
        raise NotImplementedError("Implement CFRPlus in rlgames/core/cfr.py")

    def average_policy(self) -> np.ndarray:
        """The weighted average of past current policies; this is what converges."""
        raise NotImplementedError("Implement CFRPlus in rlgames/core/cfr.py")

    def iterate(self):
        """One iteration: update player 0, then player 1."""
        raise NotImplementedError("Implement CFRPlus in rlgames/core/cfr.py")

    def solve(self, callback=None) -> np.ndarray:
        for _ in range(self.cfg.num_iters):
            self.iterate()
            if callback is not None:
                callback(self)
        return self.average_policy()
