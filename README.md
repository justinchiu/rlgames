# rlgames

Small two-player zero-sum games in JAX for studying RL training dynamics, with exact
exploitability for every game small enough to enumerate. Runs on CPU (Mac), GPU or TPU.

| Game | Information | Nodes | Infosets |
|---|---|---|---|
| `matching_pennies`, `rock_paper_scissors`, `biased_rock_paper_scissors` | simultaneous | 7–13 | 2 |
| `kuhn_poker` | imperfect | 58 | 12 |
| `leduc_poker` | imperfect | 9,457 | 288 |
| `goofspiel` (4 cards) | simultaneous | 68,245 | 34,952 |
| `liars_dice` (1 die, 6 faces) | imperfect | 294,883 | 24,576 |
| `tic_tac_toe` | perfect | 549,946 | 4,520 |
| `connect_four` | perfect | too large to enumerate | |

## Layout

- `rlgames/games/` — the games as pure functions over NamedTuple states (`jit`/`vmap`-able).
  `base.py` documents the interface.
- `rlgames/core/tree.py` — breadth-first enumeration of a game tree into numpy arrays.
- `rlgames/core/exploitability.py` — exact best responses, NashConv and exploitability.
- `rlgames/core/cfr.py` — tabular CFR+, the reference equilibrium solver.

## Usage

```python
from rlgames.games import make
from rlgames.core.tree import enumerate_tree, uniform_policy
from rlgames.core.exploitability import exploitability
from rlgames.core.cfr import CFRPlus, CFRConfig

tree = enumerate_tree(make("leduc_poker"))
print(exploitability(tree, uniform_policy(tree)))  # 2.3736
policy = CFRPlus(tree, CFRConfig(num_iters=1000)).solve()
print(exploitability(tree, policy))  # ~2e-4
```

A policy is a `float[num_infosets, num_actions]` array; for a neural policy, evaluate it on
`tree.infoset_obs` (masked by `tree.infoset_legal`).

```
uv run pytest
```
