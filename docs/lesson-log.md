# Lesson log

Where the CFR+ lessons stand, so a new session can pick up without rereading the chat.
Teaching process is in `CLAUDE.md` (intuition → math → implementation, one phase at a time).

## Current position (2026-11-15)

Roadmap step 1, `CFRPlus` in `rlgames/core/cfr.py`.

| Phase | Status |
|---|---|
| 1. Intuition | Done for regret matching. User predicted a new RPS case correctly and summarized it in their own words. |
| 2. Math | In progress. Regret definitions covered; CFR's per-infoset extension introduced. Kuhn hand-worked case not done yet. |
| 3. Implementation | Not started. `regret_matching` is already implemented and tested; `CFRPlus` is still a stub. |

## Covered so far

- Starting point: the user remembers policy gradient ("policy from the returns"), not CFR.
  Explanations build from PG: `q − v` is the advantage and also the instantaneous regret.
- Notation: `sigma` is the policy; reach is `reach`, `reach_self`, `reach_opp` (not `pi`).
  See `abstractions.md`.
- Self-play cycling, current vs average policy, exploitability. Biased RPS orbits page.
- Magnet / magnetic mirror descent, its bias (`π ∝ m·exp(q/τ)`), moving magnets
  (lagged, EMA, periodic reset), APMD (arXiv 2611.14467) and ProRL resets.
- Regret matching, phase 1: tally `R += q − v`, play `σ ∝ R⁺`, uniform when all ≤ 0.
  Worked RPS rounds vs opponent playing rock, rock, scissors: tally (−1, 1, −3) → (1, 1, −2),
  round 4 policy (½, ½, 0), round 5 all rock. Not a best response (uses whole history),
  not a gradient step (no learning rate), the average converges, not the current policy.
- Regret matching, phase 2: definitions of `q_t`, `v_t`, `r_t`, `R_T`, `Reg_T`; regret is
  the total shortfall vs the best *fixed* action in hindsight. Cited bound
  `Reg_T ≤ Δ·√(|A|·T)`; the orthogonality step `Σ R⁺(q − v) = 0` was shown but not
  absorbed and is optional.
- EMA on regrets: Discounted / Linear CFR exist; constant-ρ EMA forgets and loses the
  guarantee. Hedge = KL-proximal (TRPO-like) update, also no-regret; smoothing alone does
  not stop current-policy cycling.
- CFR intuition: regret matching at every information set, with `q` replaced by the
  counterfactual value (opponent and chance reach only). Kuhn example: holding Q facing a
  bet from an opponent who bets K always and J ⅓ → call and fold both −⅔.
- Ataraxos (`../stratego`): PPO self-play with a uniform-magnet KL and a KL to `π_old`
  (the policy that collected the batch, one collection round old); test-time search
  samples hidden states from a belief model, estimates `q` by bootstrapped rollouts, and
  takes one magnet step from the network logits. Not CFR.

## Open questions left with the user

- Why is weighting by your *own* reach a problem for a rarely reached infoset?
- Would playing the argmax of regret (fictitious-play-like) keep `R⁺ · r = 0`? (optional)
- Kuhn exercise on the Kuhn table page (Q1–6).
- Whether Ataraxos collects and trains asynchronously (`π_old` staleness): not checked.

## Next

1. Phase 2 hand case: compute `cfv` and regret for one Kuhn infoset by hand.
   State the purpose before asking; keep arithmetic to one small case.
2. Phase 3: map CFR+ to `rlgames/core/cfr.py` (regret and average tables
   `[num_infosets, actions]`, tree fields, alternating updates, `t**avg_power` weighting,
   clipping). The user writes `CFRPlus`; `tests/test_cfr.py` has 9 skipped tests that turn on.
3. After tests pass: read `rlgames/reference/cfr.py` and `../stratego` together and
   record differences in `abstractions.md`.

## Teaching notes

- State the point before asking for computation. The user pushed back on chains of
  fill-in arithmetic ("youre just asking me for random computation").
- Recap any numbers the user needs instead of expecting them to remember earlier rounds.
- Define abbreviations (RPS = rock-paper-scissors) and symbols (`v`) before using them.

## Pages

- `docs/regret-matching.html`: regret matching explainer (in repo).
- Kuhn poker table: https://claude.ai/artifact/VpaRD7HKUoEhSXpqDkX4pu
- Self-play orbits on biased RPS: https://claude.ai/artifact/6hFtirsTvEa61t3sMQUWtY
