"""Exact enumeration of small game trees.

The tree is expanded breadth-first with the game's own jitted `step`, one depth
at a time, and flattened into numpy arrays ordered by depth. Exploitability and
CFR (rlgames/reference/cfr.py) operate on these arrays.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from rlgames.games.base import Game


@dataclass
class GameTree:
    num_actions: int
    # Per node, ordered by depth. Node 0 is the root.
    player: np.ndarray  # int32[N]
    legal: np.ndarray  # bool[N, A]
    chance: np.ndarray  # float32[N, A]
    terminal: np.ndarray  # bool[N]
    returns: np.ndarray  # float32[N, 2]
    children: np.ndarray  # int64[N, A], -1 where there is no child
    infoset: np.ndarray  # int64[N], -1 at chance and terminal nodes
    depth_offsets: np.ndarray  # nodes at depth d are [offsets[d], offsets[d + 1])
    # Per information set.
    infoset_player: np.ndarray  # int32[I]
    infoset_legal: np.ndarray  # bool[I, A]
    infoset_obs: np.ndarray  # float32[I, obs_dim]

    @property
    def num_nodes(self) -> int:
        return len(self.player)

    @property
    def num_infosets(self) -> int:
        return len(self.infoset_player)

    @property
    def max_depth(self) -> int:
        return len(self.depth_offsets) - 1

    def depth_slice(self, depth: int) -> slice:
        return slice(self.depth_offsets[depth], self.depth_offsets[depth + 1])


def enumerate_tree(game: Game, max_nodes: int = 5_000_000) -> GameTree:
    num_actions = game.num_actions
    actions = jnp.arange(num_actions)

    @jax.jit
    def expand(states):
        children = jax.vmap(lambda s: jax.vmap(lambda a: game.step(s, a))(actions))(states)
        observer = jnp.maximum(states.player, 0)
        obs = jax.vmap(game.observe)(states, observer)
        return children, obs

    frontier = jax.tree.map(lambda x: x[None], game.init())
    levels = []
    num_nodes = 0
    while True:
        children, obs = expand(frontier)
        frontier_np = jax.tree.map(np.asarray, frontier)
        valid = (frontier_np.legal | (frontier_np.chance > 0)) & ~frontier_np.terminal[:, None]
        levels.append((frontier_np, np.asarray(obs), valid))
        num_nodes += len(valid)
        if num_nodes > max_nodes:
            raise ValueError(f"{game.name} has more than {max_nodes} nodes")
        if not valid.any():
            break
        rows, cols = np.nonzero(valid)
        frontier = jax.tree.map(lambda x: x[rows, cols], children)

    offsets = np.cumsum([0] + [len(v) for _, _, v in levels])
    child_ids = []
    for depth, (_, _, valid) in enumerate(levels):
        ids = np.full(valid.shape, -1, np.int64)
        ids[valid] = offsets[depth + 1] + np.arange(valid.sum())
        child_ids.append(ids)

    def cat(get):
        return np.concatenate([get(states) for states, _, _ in levels])

    player = cat(lambda s: s.player).astype(np.int32)
    legal = cat(lambda s: s.legal)
    terminal = cat(lambda s: s.terminal)
    obs = np.concatenate([o for _, o, _ in levels])
    depth = np.repeat(np.arange(len(levels)), np.diff(offsets))

    # An information set is a (player, observation) pair at a decision node.
    decision = (player >= 0) & ~terminal
    keys = np.concatenate([player[decision, None].astype(np.float32), obs[decision]], axis=1)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    infoset = np.full(len(player), -1, np.int64)
    infoset[decision] = inverse.reshape(-1)

    infoset_legal = np.zeros((len(unique_keys), num_actions), bool)
    infoset_legal[infoset[decision]] = legal[decision]
    if not (infoset_legal[infoset[decision]] == legal[decision]).all():
        raise ValueError(f"{game.name}: legal actions differ within an information set")
    infoset_depth = np.zeros(len(unique_keys), np.int64)
    infoset_depth[infoset[decision]] = depth[decision]
    if not (infoset_depth[infoset[decision]] == depth[decision]).all():
        # Exploitability and CFR process one depth at a time, which relies on this.
        raise ValueError(f"{game.name}: an information set spans several depths")

    return GameTree(
        num_actions=num_actions,
        player=player,
        legal=legal,
        chance=cat(lambda s: s.chance).astype(np.float32),
        terminal=terminal,
        returns=cat(lambda s: s.returns).astype(np.float32),
        children=np.concatenate(child_ids),
        infoset=infoset,
        depth_offsets=offsets,
        infoset_player=unique_keys[:, 0].astype(np.int32),
        infoset_legal=infoset_legal,
        infoset_obs=unique_keys[:, 1:],
    )


def node_policy(tree: GameTree, policy: np.ndarray) -> np.ndarray:
    """float[N, A] action probabilities at every node: the chance distribution at
    chance nodes, `policy[infoset]` at decision nodes and zeros at terminals."""
    probs = tree.chance.astype(np.float64)
    decision = tree.infoset >= 0
    probs[decision] = policy[tree.infoset[decision]]
    return probs


def reach_probs(tree: GameTree, probs: np.ndarray, include: np.ndarray) -> np.ndarray:
    """float[N] probability of reaching each node, counting only the actions of
    nodes where `include` is True (other actions count as probability 1)."""
    reach = np.zeros(tree.num_nodes)
    reach[0] = 1.0
    for depth in range(tree.max_depth):
        nodes = tree.depth_slice(depth)
        step = np.where(include[nodes, None], probs[nodes], 1.0)
        valid = tree.children[nodes] >= 0
        reach[tree.children[nodes][valid]] = (reach[nodes, None] * step)[valid]
    return reach


def child_values(tree: GameTree, values: np.ndarray, nodes: slice) -> np.ndarray:
    """float[n, A] values of the children of `nodes`, 0 where there is no child."""
    children = tree.children[nodes]
    return np.where(children >= 0, values[np.maximum(children, 0)], 0.0)


def expected_values(tree: GameTree, probs: np.ndarray) -> np.ndarray:
    """float[N] expected return for player 0 at every node when everyone plays `probs`."""
    values = tree.returns[:, 0].astype(np.float64)
    for depth in reversed(range(tree.max_depth)):
        nodes = tree.depth_slice(depth)
        inner = ~tree.terminal[nodes]
        expected = (probs[nodes] * child_values(tree, values, nodes)).sum(1)
        values[nodes] = np.where(inner, expected, values[nodes])
    return values


def uniform_policy(tree: GameTree) -> np.ndarray:
    legal = tree.infoset_legal.astype(np.float64)
    return legal / legal.sum(1, keepdims=True)
