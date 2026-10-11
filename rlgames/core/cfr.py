"""Tabular CFR+ on an enumerated game tree.

Used as a ground-truth equilibrium solver and as a reference curve for the
learning-dynamics experiments.
"""

from dataclasses import dataclass

import numpy as np

from rlgames.core.tree import GameTree, child_values, expected_values, node_policy, reach_probs


@dataclass
class CFRConfig:
    num_iters: int = 1000
    # Weight iteration t's policy by t^avg_power in the average policy (1 = linear).
    avg_power: float = 1.0


class CFRPlus:
    def __init__(self, tree: GameTree, cfg: CFRConfig = CFRConfig()):
        self.tree = tree
        self.cfg = cfg
        self.regrets = np.zeros((tree.num_infosets, tree.num_actions))
        self.policy_sum = np.zeros((tree.num_infosets, tree.num_actions))
        self.num_iters = 0

    def current_policy(self) -> np.ndarray:
        positive = np.where(self.tree.infoset_legal, np.maximum(self.regrets, 0), 0)
        total = positive.sum(1, keepdims=True)
        legal = self.tree.infoset_legal.astype(np.float64)
        uniform = legal / legal.sum(1, keepdims=True)
        return np.where(total > 0, positive / np.maximum(total, 1e-300), uniform)

    def average_policy(self) -> np.ndarray:
        total = self.policy_sum.sum(1, keepdims=True)
        legal = self.tree.infoset_legal.astype(np.float64)
        uniform = legal / legal.sum(1, keepdims=True)
        return np.where(total > 0, self.policy_sum / np.maximum(total, 1e-300), uniform)

    def iterate(self):
        """One iteration with alternating updates: player 0, then player 1."""
        self.num_iters += 1
        for player in (0, 1):
            self._update(player)

    def _update(self, player: int):
        tree = self.tree
        policy = self.current_policy()
        probs = node_policy(tree, policy)
        sign = 1.0 if player == 0 else -1.0
        values = sign * expected_values(tree, probs)
        reach_others = reach_probs(tree, probs, include=tree.player != player)
        reach_self = reach_probs(tree, probs, include=tree.player == player)

        ours = (tree.player == player) & ~tree.terminal
        nodes = np.nonzero(ours)[0]
        q = np.concatenate(
            [
                child_values(tree, values, tree.depth_slice(d))[ours[tree.depth_slice(d)]]
                for d in range(tree.max_depth)
            ]
        )
        infosets = tree.infoset[nodes]
        instant = reach_others[nodes, None] * (q - values[nodes, None])
        regret_delta = np.zeros_like(self.regrets)
        np.add.at(regret_delta, infosets, instant)
        mask = tree.infoset_player == player
        self.regrets[mask] = np.maximum(self.regrets[mask] + regret_delta[mask], 0)

        # Average policy weighted by own reach. With perfect recall every history in an
        # information set has the same own reach, and summing only rescales each set. Summing
        # also handles transpositions in perfect-information games, where one board is reached
        # by several move orders with different own reach.
        own_reach = np.zeros(tree.num_infosets)
        np.add.at(own_reach, infosets, reach_self[nodes])
        weight = self.num_iters**self.cfg.avg_power
        self.policy_sum[mask] += weight * own_reach[mask, None] * policy[mask]

    def solve(self, callback=None) -> np.ndarray:
        for _ in range(self.cfg.num_iters):
            self.iterate()
            if callback is not None:
                callback(self)
        return self.average_policy()
