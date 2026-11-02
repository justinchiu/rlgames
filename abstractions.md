# Abstractions for CFR, PPO, and rollout search

## Goal

Implement both CFR+ and a neural self-play pipeline with PPO and rollout search,
then study the bias and variance introduced by sampling and neural approximations.
Exact tabular methods on small games provide ground truth. They are a reference
backend, not a requirement to enumerate the tree in every algorithm.

This is a proposed design, not a description of interfaces already implemented.
Keep the existing exercises intact while introducing interfaces as needed.

## Related implementation: `../stratego`

Use the local Stratego repository as the reference for the neural self-play and
rollout-search path, including its naming conventions:

- [`../stratego/pyengine/core/rl.py`](../stratego/pyengine/core/rl.py): neural
  self-play training, advantage-based PPO losses, and value fitting.
- [`../stratego/pyengine/core/search.py`](../stratego/pyengine/core/search.py):
  `SearchBot`, hidden-state sampling, rollout action-value estimation,
  `compute_search_policy`, and regret-gap diagnostics.
- [`../stratego/pyengine/core/env.py`](../stratego/pyengine/core/env.py): batched
  environment interaction and player-perspective terminal rewards.

That implementation uses PyTorch/CUDA. Our neural implementation can use JAX
while retaining comparable algorithmic choices and terminology. It is not a CFR
implementation: our exact CFR+ solver supplies a separate small-game baseline.
When porting a component, document any differences in targets, sampling,
regularization, or policy updates so comparisons remain interpretable.

## Notation and conventions

Use function syntax in derivations. Keep terminology aligned with the inspected
Stratego code: `reward`, `q`, `value`, `advantages`, and action probabilities.
These names align with its code; they do not claim to reproduce a paper's notation.

```text
t                    learning iteration, not time within a game
h                    history / full game-tree node
I                    information set containing histories indistinguishable to a player
i                    player whose payoff is being evaluated
a                    action
sigma(t, i, I, a)     action probability
reward(i, z)         actual terminal payoff at terminal history z
q(t, i, h, a)        expected payoff after a at h, then current-policy continuation
value(t, i, h)       expected payoff under the current policy at h
r(t, i, I, a)        instantaneous counterfactual regret
R(T, i, I, a)        accumulated counterfactual regret
```

History-conditioned values and information-conditioned values are different
quantities. An information-conditioned value requires an explicit distribution
over compatible histories. Counterfactual weighting must not silently be replaced
by a normalized posterior over histories.

Always record payoff perspective. In two-player zero-sum games one player's
payoff is the negative of the other's; this is a game property, not a universal
assumption of the value interface.

## Shared components

| Component | Responsibility | Implementations / variants |
|---|---|---|
| Game | Initial state, legal actions, chance, transitions, observations, terminal rewards | Existing small games; later larger games |
| Policy | Map player information and legal mask to action probabilities | Tabular, neural, search-adjusted |
| Representation | Map histories to player information or model inputs | Exact infoset IDs, observation encoders, recurrent history encoders |
| Data source | Produce histories or trajectories under specified policies | Exact enumeration, on-policy sampling, replay |
| Value estimator | Estimate a declared payoff quantity for fixed policies | Exact tree evaluation, Monte Carlo, bootstrapped rollouts, neural prediction |
| Belief provider | Supply a distribution or samples over hidden histories given player information | Exact conditional distribution, learned belief, specified approximation |
| Target builder | Construct the quantities an algorithm learns from | Counterfactual regret increments, returns, advantages, search action values |
| Policy improvement | Convert learning signals into an improved policy or objective | Regret matching, PPO objective, regularized search update |
| Function approximator | Fit or represent a declared target | Tables, neural value/policy/regret models |
| Evaluator | Measure estimation error and strategic performance | Exact values, exploitability where applicable, sampled returns |

Separate representations from estimators: a neural network is a representation
and prediction mechanism, not a complete specification of its statistical target.
Likewise, a tabular estimator can still use noisy sampled targets.

## Interface sketches

These are conceptual signatures. Exact argument types can follow the existing
game and JAX batch APIs rather than requiring a class hierarchy.

```text
policy_probs(policy_state, player_info, legal) -> probabilities
collect(game, policy_profile, collection_config, key) -> trajectories
evaluate(estimator_state, policy_profile, query, key) -> estimate_result
build_targets(algorithm_state, data, estimates, config) -> targets
update(algorithm_state, data, targets, key) -> new_state, metrics
search(policy_profile, player_info, belief_provider, estimator, config, key)
    -> improved_probabilities, diagnostics
```

An estimation query/result should identify:

- The target: state value, action value, terminal return, or counterfactual quantity.
- The player perspective, policy snapshot, conditioning information, and horizon.
- Exact versus sampled computation, sample count, and bootstrap policy/model.
- Whether an output is a scalar expectation or a distribution over returns.

Separate random keys for trajectory sampling, hidden-state sampling, and model
initialization so experiments can vary one source of randomness at a time.

Full state may be used by exact evaluators and explicitly declared privileged
critics. Actor policies must only receive information available to the player.
Search must not reveal sampled hidden state to the acting policy accidentally.

## Preserve algorithm-specific semantics

Share games, policy queries, estimates, masking, data plumbing, and evaluation.
Do not force all algorithms into the same regret update or training batch.

### CFR+

- Traverse or sample histories with explicit self, opponent, and chance reach.
- Aggregate counterfactual regret increments by information set.
- Apply alternating player updates and CFR+ clipping of accumulated regrets.
- Produce the current policy through regret matching.
- Maintain a separate own-reach-weighted average policy.
- Keep tabular cumulative regrets, average-policy weights, and iteration count
  as algorithm state. A neural variant needs an explicit fitting/replay scheme;
  replacing a table with a network does not automatically preserve CFR guarantees.

### PPO self-play

- Collect trajectories with recorded behavior-policy action log probabilities.
- Build return and advantage targets using a declared estimator, such as GAE.
- Optimize the clipped current/behavior likelihood-ratio objective.
- Keep optimizer state, policy/value parameters, behavior snapshots, and target
  configuration explicit. Do not confuse game steps with learning iterations.

### Rollout search

- Start from a base policy and the current player's information.
- Sample compatible hidden states when necessary and estimate action values.
- Combine action values and base-policy logits using a specified regularized
  update, producing a search policy.
- Declare whether future rollout decisions use the base policy or further search.
- Record the policy actually used to collect training actions. If search selects
  those actions, its probabilities differ from the base network's probabilities.
  PPO behavior ratios must account for the chosen collection design.

The inspected Stratego search logs a best-action gap:

```text
regret_gap = max_a q(a) - sum_a policy(a) * q(a)
```

That diagnostic is not CFR's cumulative counterfactual regret. Use distinct names
in shared metrics to avoid treating them as interchangeable.

## Bias and variance experiments

First freeze a game, policy profile, query set, and exact target. Repeat estimation
under independent seeds. For a scalar target:

```text
truth       = exact evaluation of the declared target
mean_est    = mean over repeated estimates
bias        = mean_est - truth
variance    = mean((estimate - mean_est)**2)
mse         = mean((estimate - truth)**2)
            = bias**2 + variance
```

Here variance uses the empirical population denominator; distinguish it from an
unbiased sample-variance estimator when reporting uncertainty.

Vary sampling and approximation separately:

1. Exact traversal and exact continuation values: numerical reference.
2. Sampled trajectories with exact continuation values: isolate sampling effects.
3. Identical queries with neural continuation values: measure prediction error
   and its effect on targets while holding sampling conditions controlled.
4. Sampled hidden states with an exact versus learned belief: isolate belief error.
5. Retrained neural models across datasets and initializations: measure variability
   from data collection and fitting separately from inference-time sampling.
6. Full learning runs: measure how these errors affect policy quality and stability.

Use held-out queries and exact policy snapshots. A fixed network prediction at a
fixed input is deterministic; variance requires specifying what is randomized.
Report per-query errors and a declared aggregation distribution over queries.
Use paired queries/seeds when comparing estimators where practical.

For exact truth, match continuation policy, payoff perspective, conditioning,
discounting, and horizon. A bootstrapped finite-depth target is not automatically
the same quantity as a full-game expected payoff.

Track estimator accuracy separately from strategic quality: low value error does
not by itself imply low exploitability. Report exploitability on supported small
zero-sum games and sampled performance with uncertainty elsewhere.

## Backend and implementation direction

- Retain NumPy exact CFR/evaluation as a readable oracle for small games.
- Use JAX for neural training, batched simulation, sampled estimators, and compiled
  solver variants when useful. Backend conversion should happen at explicit boundaries.
- Use fixed-shape arrays and explicit masks in compiled paths. Typical policy,
  action-value, and legal-mask shapes are `[batch, actions]`; tabular arrays use
  `[num_infosets, actions]`.
- Keep illegal-action masking and zero-total normalization consistent across backends.
- Reuse the existing `Game`, rollout `Batch`, network, optimizer, and learner
  infrastructure where appropriate. Extend trajectory storage for the chosen
  advantage estimator rather than assuming final outcomes alone supply all inputs.
- Introduce shared interfaces incrementally as the second implementation needs them.
  Preserve distinct CFR and PPO orchestration while testing interchangeable value
  estimators against the same exact queries.

Near-term sequence: derive and implement exact CFR+, establish exact evaluation
queries, implement PPO and rollout search on the same games, then introduce sampled
and neural variants with controlled error experiments.
