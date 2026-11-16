# rlgames, OpenSpiel, and Ataraxos

Three codebases that sit next to each other in this workspace and cover overlapping ground:

- **rlgames** (this repo): small two-player zero-sum games in JAX, exact evaluation, and
  learning exercises (CFR+, PPO, search).
- **OpenSpiel** (`../open_spiel`): DeepMind's general library of games and algorithms for
  multi-agent RL and game theory.
- **Ataraxos** (`../stratego`): one large system that trains a Stratego agent by self-play
  RL and plays with test-time search.

Checked against the local checkouts: OpenSpiel at `4956843d`, Ataraxos at `92db29e`.
OpenSpiel's and this repo's CFR implementations were deliberately not read in detail;
they are references for the `CFRPlus` exercise and are compared only by their public
options.

## At a glance

| | rlgames | OpenSpiel | Ataraxos |
|---|---|---|---|
| Purpose | Learn and measure algorithms on games with exact ground truth | Broad research library: many games, many algorithms | Win at one large game |
| Games | 9 small games (matrix, Kuhn, Leduc, Goofspiel 4, Liar's dice 1d6, tic-tac-toe, Connect Four) | 80+ games, N-player, general-sum, mean-field | Stratego only |
| Game engine | Pure JAX functions over NamedTuple states | C++ core with `pyspiel` Python bindings; also a pure-JAX port of many games | C++/CUDA engine (`src/env`) exposed via pybind as `pystratego` |
| Batching | `jit` + `vmap` over thousands of games | One mutable `State` at a time (C++); batched in the JAX port | Thousands of games on the GPU inside the CUDA engine |
| Ground truth | Exact tree, best response, exploitability for every game except Connect Four | Exact best response / exploitability for small games | None exact; evaluation by head-to-head play |
| Learning framework | JAX + optax | NumPy/C++ tabular; JAX and PyTorch for neural methods | PyTorch + CUDA |
| Main algorithms | CFR+ (exercise), PPO self-play (exercise), rollout search (planned) | CFR family, MCCFR, Deep CFR, MMD, R-NaD, NFSP, PSRO, DQN, policy gradient, … | PPO self-play with a magnet, test-time search with a learned belief |
| Networks | MLP | Varies by algorithm | Transformers (policy/value, belief, setup arrangement) |

## Games and engines

### Rules that differ

The same names do not always mean the same game.

| Game | rlgames vs OpenSpiel |
|---|---|
| Kuhn poker | Same rules and action ids. Value −1/18. |
| Leduc poker | Same rules (raises 2 then 4, at most 2 per round) and action ids (fold 0, call 1, raise 2). rlgames encodes cards by rank only, which matches OpenSpiel's `suit_isomorphism=true`; OpenSpiel's default is `false`, so its information-set count is larger. Value ≈ −0.0856. |
| Goofspiel | rlgames uses 4 cards; OpenSpiel defaults to 13. Revealing bids each round matches OpenSpiel's default `imp_info=false`. OpenSpiel has native simultaneous moves; rlgames has player 0 bid first and hides that bid from player 1. |
| Liar's dice | **Different game.** OpenSpiel makes the highest face wild (`liars_dice.h`); rlgames has no wild face. Equilibria and exploitability numbers do not transfer. |
| Biased RPS | **Different game.** OpenSpiel's `matrix_brps` pays `[[0,−25,50],[25,0,−5],[−50,5,0]]`; rlgames uses `[[0,−1,2],[1,0,−1],[−2,1,0]]` with Nash (¼, ½, ¼). |
| Matching pennies, RPS, tic-tac-toe, Connect Four | Same. |

The rlgames docstring for biased RPS says "wins with rock or scissors pay 2"; the matrix
only pays 2 for rock beating scissors.

### Engine interfaces

| Concept | rlgames | OpenSpiel (C++ / `pyspiel`) | OpenSpiel JAX port | Ataraxos |
|---|---|---|---|---|
| Start | `game.init()` | `game.new_initial_state()` | `game.new_initial_state()` | `env.reset()` (batched) |
| Move | `game.step(state, a)` → new state | `state.apply_action(a)` (mutates) | `game.apply_action(state, a)` → new state | `env.apply_actions(actions)` (batched, in place) |
| Legal actions | `state.legal` (bool mask) | `state.legal_actions()` (list) | `legal_actions_mask(state)` | `current_legal_action_mask` |
| Chance | `state.player == CHANCE`, `state.chance` probabilities | `is_chance_node()`, `chance_outcomes()` | `chance_outcomes(state)` | Hidden piece setup; no explicit chance nodes in the search API |
| Player information | `observe(state, player)`: perfect-recall tensor | `information_state_string/tensor`, plus an observation API | `information_state_tensor` | `current_infostate_tensor` plus piece ids and move count |
| Payoffs | `state.returns[2]` | `state.returns()` | `returns(state)` | `current_reward_pl0` |

The OpenSpiel JAX port (`open_spiel/python/jax/games/`) uses the same functional design as
rlgames, with OpenSpiel's names and action ids. It is tested move for move against
`pyspiel` (`openspiel_equivalence*_test.py`) and covers Kuhn, Leduc, Liar's dice,
tic-tac-toe, and Connect Four among others, but not Goofspiel or the matrix games.

## Algorithms

### Tabular and exact

| | rlgames | OpenSpiel |
|---|---|---|
| Tree enumeration | `core/tree.py`: expands with the game's own jitted `step` under `vmap`, stores levels as NumPy arrays | Recursive traversal of `pyspiel` states |
| Best response, exploitability | `core/exploitability.py` (NumPy) | `python/algorithms/best_response.py`, `exploitability.py`; C++ versions too |
| CFR family | CFR+ only, as an exercise (`core/cfr.py`) | CFR, CFR+, Linear CFR, Discounted CFR, CFR-BR, outcome- and external-sampling MCCFR, Deep CFR, a JAX CFR |
| Regularized tree methods | None yet | `mmd_dilated.py` (magnetic mirror descent over the sequence form), R-NaD |

OpenSpiel's `CFRPlusSolver` sets `alternating_updates=True`, `regret_matching_plus=True`,
and `linear_averaging=True`. The rlgames `CFRPlus` docstring asks for the same three
(alternating P0-then-P1 updates, clipped regrets, average weighted by `t**avg_power` with
`avg_power = 1`), so exploitability curves should be directly comparable.

Ataraxos has no tabular or exact component: Stratego is far too large.

### Self-play RL

| | rlgames | Ataraxos |
|---|---|---|
| Learner | PPO stub (`core/learner.py`) for the user to write | PPO (`pyengine/core/rl.py`) |
| Loop | Collect a batch of self-play games, then update | `collect_and_train` (`rl.py:270`): collect rollouts, compute targets, train several epochs, clear the buffer. Synchronous, so `π_old` is exactly one collection round old. |
| Targets | To be declared (GAE etc.) | TD(λ) returns (`td_lambda = 0.8`) and GAE advantages (`gae_lambda = 0.5`) |
| Policy loss | — | Clipped ratio, `clip_range = 0.2` (`rl.py:547–551`) |
| Stay near the old policy | — | `kl_coef · KL(π ‖ π_old)`, `kl_coef = 0.1` (`rl.py:553`) |
| Magnet | — | `temperature · KL(π ‖ uniform)` (`rl.py:565–576`). The temperature is annealed: `0.05 / (step + 1)^0.3`, clamped to [0.001, 0.1] (`rl.py:500`, `power_schedule` at `rl.py:756`), so the magnet's bias shrinks over training. |
| Averaging | — | An EMA of the network weights with decay 0.999 (`rl.py:156`, updated at `rl.py:612`), saved with checkpoints and selectable for evaluation (`get_eval_model(ema=True)`). This is an average in weight space, not CFR's reach-weighted average policy. |
| Extra phases | — | A separate PPO-trained setup network chooses the initial piece arrangement (`arr_*` config, `arrangement_transformer.py`). |

OpenSpiel's RL algorithms (NFSP, policy gradient, PSRO, DQN, R-NaD) live in
`python/algorithms`, `python/jax`, and `python/pytorch`, and run on its `rl_environment`
wrapper rather than a batched JAX or CUDA engine.

### Test-time search

Only Ataraxos does search today. rlgames plans it as roadmap step 5.

1. **Hidden state.** A belief transformer, trained separately (`scripts/train/belief_main.py`),
   samples the opponent's unknown pieces (`search.py:281`, `_generate_env_state`).
2. **Action values.** For each legal action, play a few moves with the policy network for
   both sides and bootstrap with the value network, mixed by `td_lambda`
   (`search.py:180`, `estimate_q_values`).
3. **Policy.** One magnet step from the network's logits (`search.py:450`):
   `(logits + η·q + η·τ·log m) / (1 + η·τ)`.

Within one information set, sampling hidden states from the true posterior weights them
in proportion to `reach_opp`, the same weighting as CFR's counterfactual values but
normalized. Ataraxos's belief is learned, so the match is only as good as the belief model.

## How the pieces relate

| Idea | rlgames | OpenSpiel | Ataraxos |
|---|---|---|---|
| Information set | `(player, observe(state))` | `information_state_string` | Infostate tensor plus piece ids and move count, as one side sees them |
| Counterfactual value | Exact, from the tree (`cfv` in `abstractions.md`) | Exact in CFR; sampled in MCCFR | Not used |
| Action value at a decision | Exact `q(h, a)` from the tree | Exact or sampled | Rollouts plus value net, averaged over belief samples |
| Policy improvement | Regret matching (CFR+); PPO step | Regret matching, MMD, policy gradient, best response, … | PPO step during training; one magnet step at test time |
| What converges | CFR+'s average policy | Depends on algorithm | The current network, stabilized by the magnet and `KL(π ‖ π_old)`; an EMA of weights is kept |
| Exploitability | Exact | Exact for small games | Not computable; Elo-style evaluation instead |

## What each is good for here

- **rlgames** is where to learn and measure. Every number has an exact reference, so the
  bias and variance of sampling, neural approximation, and search can be isolated
  (`abstractions.md`, "Bias and variance experiments").
- **OpenSpiel** is a cross-check and a source of baselines. Useful next steps:
  - compare rlgames' exact game values, information-set counts, and exploitability on
    Kuhn and Leduc against OpenSpiel;
  - replay rlgames games against the OpenSpiel JAX port for Kuhn, Leduc, tic-tac-toe,
    and Connect Four;
  - after the CFR+ exercise passes, compare its curve with `CFRPlusSolver`, and read
    `mmd_dilated.py` next to Ataraxos's magnet.
- **Ataraxos** is the target design for the neural path (roadmap steps 4–6). Port its
  choices deliberately and record each difference in targets, sampling, regularization,
  or policy updates, as `CLAUDE.md` asks.
