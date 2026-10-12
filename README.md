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

## Workflow

New to these algorithms? [docs/algorithms.md](docs/algorithms.md) is a study guide: policy
gradient self-play, why it cycles, regularization and MMD, and CFR.

1. **Implement a learning algorithm** in `rlgames/core/learner.py`. `Learner.update` receives a
   batch of self-play games played by the current network and returns new parameters. The
   docstring there describes the batch fields and the JAX/optax plumbing. Algorithm
   hyperparameters go in `LearnerConfig` and become command-line flags.

2. **Train**:
   ```
   uv run python scripts/train/selfplay_main.py --game kuhn_poker --save_dir exps/kuhn
   ```
   Options include `--num_envs`, `--num_iters`, `--optim.name adamw|sgd|muon`,
   `--optim.lr`, `--network.hidden`, `--game_kwargs '{num_cards: 5}'` and `--learner.<field>`.
   Add `--dry_run true` to run everything except learning.

3. **Watch and play**, in a second terminal:
   ```
   uv run python scripts/play/server.py --exps_dir exps
   ```
   Open http://localhost:8000. The left side plots the run's metrics as they are written:
   exact exploitability on every game except Connect Four, return against a random
   opponent, policy entropy, anything your learner returns, and for matrix games the policy
   trajectory, which shows cycling directly. The right side plays you against the latest
   checkpoint; each new game picks up the newest one.

## Layout

- `rlgames/games/`: the games as pure functions over NamedTuple states (`jit`/`vmap`-able).
  `base.py` documents the interface.
- `rlgames/core/env.py`: batched self-play rollouts that return a flat `Batch` of decisions.
- `rlgames/core/learner.py`: the learning algorithm (yours to write).
- `rlgames/core/cfr.py`: tabular CFR+ (yours to write). `tests/test_cfr.py` compares it,
  iteration by iteration, to the reference implementation.
- `rlgames/core/optim.py`: optimizer construction (AdamW, SGD, Muon).
- `rlgames/core/evaluation.py`: exploitability of a network and return against an opponent.
- `rlgames/core/tree.py`, `exploitability.py`: exact tree enumeration and best responses.
- `rlgames/reference/`: **reference implementations (spoilers)** of algorithms you may want
  to write yourself, kept for checking your answers. Currently `cfr.py` (tabular CFR+).
  Nothing outside `tests/` imports them.
- `rlgames/networks/mlp.py`: the policy-value network.
- `rlgames/play/`: views of each game and the web page for the play server.
- `scripts/train/selfplay_main.py`, `scripts/play/server.py`: entry points.

## Exact evaluation

```python
from rlgames.games import make
from rlgames.core.tree import enumerate_tree, uniform_policy
from rlgames.core.exploitability import exploitability
from rlgames.reference.cfr import CFRPlus, CFRConfig

tree = enumerate_tree(make("leduc_poker"))
print(exploitability(tree, uniform_policy(tree)))  # 2.3736
policy = CFRPlus(tree, CFRConfig(num_iters=1000)).solve()
print(exploitability(tree, policy))  # ~2e-4
```

A tabular policy is a `float[num_infosets, num_actions]` array.
`rlgames.core.evaluation.tabulate_policy` builds one from network parameters.

## Tests

`tests/test_cfr.py` runs every check on both the reference and your `rlgames/core/cfr.py`;
checks on your code skip until it stops raising `NotImplementedError`:

```
uv run pytest tests/test_cfr.py -v
```

Everything:

```
uv run pytest
```
