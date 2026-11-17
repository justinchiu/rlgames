# Related work: why self-play cycles, and how to stop it

Background for `docs/selfplay-orbits.html` and the magnet in Ataraxos. In two-player zero-sum games,
simple learning rules (gradient ascent, multiplicative weights, regret matching) do not settle at a
Nash equilibrium. Their current policy cycles around it. There are three standard responses:
average the iterates, look ahead (optimism), or regularize toward a magnet.

Checked against the arXiv or publisher pages on 2026-11-17, unless marked otherwise.

## The problem: cycles

**Mertikopoulos, Papadimitriou, Piliouras. "Cycles in Adversarial Regularized Learning." SODA 2018.**
[arXiv:1709.02738](https://arxiv.org/abs/1709.02738)
- Setting: continuous-time follow-the-regularized-leader (FTRL), which includes gradient ascent and
  multiplicative weights, in zero-sum games with an interior equilibrium.
- Result: the dynamics are Poincaré recurrent. Almost every trajectory returns arbitrarily close
  to where it started. So the current policy never converges, for any choice of regularizer, and
  the result extends to zero-sum polymatrix (networked) games.
- Connection here: the closed orbits on the orbits page, in the small-step limit.

**Bailey, Piliouras. "Multiplicative Weights Update in Zero-Sum Games." EC 2018.**
[ACM](https://doi.org/10.1145/3219166.3219235)
- Setting: discrete-time multiplicative weights (Hedge) with a fixed step size.
- Result: with an interior equilibrium, the KL divergence from equilibrium does not decrease, so
  play drifts outward toward the boundary of the simplex. The same holds for other FTRL dynamics,
  including gradient descent, and with varying step sizes.
- Connection here: the outward spiral of the policy-gradient panel at large η. The time average
  still converges, but the day-to-day policy does not.

## Fix 1: average the iterates (no-regret)

These are the classical results the CFR+ exercise builds on. They give convergence of the
*average* policy only. I didn't recheck the citations for this section.

- **Hart, Mas-Colell. "A Simple Adaptive Procedure Leading to Correlated Equilibrium."
  Econometrica 2000.** Regret matching, and its `Reg_T ≤ Δ·√(|A|·T)` bound.
- **Zinkevich, Johanson, Bowling, Piccione. "Regret Minimization in Games with Incomplete
  Information." NeurIPS 2007.** CFR: regret matching at each information set with counterfactual
  values, plus the decomposition of total regret into per-infoset regrets.
- **Tammelin. "Solving Large Imperfect Information Games Using CFR+." arXiv:1407.5042, 2014.**
  CFR+: clipped regrets, alternating updates, and linearly weighted averaging. This is roadmap
  step 1.

## Fix 2: optimism (look one step ahead)

**Daskalakis, Ilyas, Syrgkanis, Zeng. "Training GANs with Optimism." ICLR 2018.**
[arXiv:1711.00141](https://arxiv.org/abs/1711.00141)
- Optimistic mirror descent uses the last gradient as a prediction of the next one and
  corrects for it.
- Result: in bilinear (matrix) zero-sum games, the *last iterate* of optimistic gradient
  descent converges to equilibrium, while plain gradient descent cycles. Unconstrained setting.
  The paper also introduces Optimistic Adam for GANs, with no guarantee there.

**Daskalakis, Panageas. "Last-Iterate Convergence: Zero-Sum Games and Constrained Min-Max
Optimization." ITCS 2019.** [arXiv:1807.04252](https://arxiv.org/abs/1807.04252)
- Extends this to the simplex: optimistic multiplicative weights (OMWU) converges in last iterate.
  The proof shows the KL divergence to the equilibrium decreases.
- (I checked the arXiv abstract, not the venue.)

## Fix 3: regularize toward a magnet

**Pérolat et al. "From Poincaré Recurrence to Convergence in Imperfect Information Games: Finding
Equilibrium via Regularization." ICML 2021.** [arXiv:2002.08456](https://arxiv.org/abs/2002.08456)
- Extends the recurrence result to imperfect-information (extensive-form) games.
- Adding a reward regularization term makes the dynamics converge, but to the equilibrium of the
  regularized game. Repeatedly re-centring the regularizer on the current policy recovers the
  original equilibrium. This is the idea behind R-NaD (regularized Nash dynamics), used in
  DeepMind's Stratego agent DeepNash.

**Sokota, D'Orazio, Kolter, Loizou, Lanctot, Mitliagkas, Brown, Kroer. "A Unified Approach to
Reinforcement Learning, Quantal Response Equilibria, and Two-Player Zero-Sum Games." ICLR 2023.**
[arXiv:2206.05825](https://arxiv.org/abs/2206.05825), code at
[github.com/ssokota/mmd](https://github.com/ssokota/mmd)
- Magnetic mirror descent (MMD) is a mirror-descent step plus a KL pull toward a magnet policy.
  It is the "magnet" panel on the orbits page.
- Result: the *current* policy converges linearly to a quantal response equilibrium (QRE) in
  extensive-form games. The authors call this the first linear convergence proof for a
  first-order solver. A QRE is the equilibrium of the regularized game, so it is biased away from
  Nash by an amount set by the temperature.
- Empirically, MMD is competitive with CFR as a tabular solver, and with PPO on Atari and MuJoCo
  as a deep RL algorithm.
- Implementations: OpenSpiel `python/algorithms/mmd_dilated.py`. Ataraxos uses the same idea as
  `temperature·KL(π‖uniform)` in its PPO loss (`../stratego/pyengine/core/rl.py:565–576`), with
  the temperature annealed toward zero (`rl.py:500`).

## How the three fixes compare

| Approach | What converges | To what | Cost |
|---|---|---|---|
| Averaging (regret matching, CFR+) | Average policy | Nash | Must store and use an average; the current policy keeps cycling |
| Optimism (OGD, OMWU) | Current policy | Nash | One extra gradient (or a stored previous one) per step; mainly tabular or bilinear guarantees |
| Magnet (R-NaD, MMD) | Current policy | Regularized equilibrium (QRE) | Biased by the temperature; anneal it or move the magnet to remove the bias |

For neural self-play, only the current network is easy to use, which is why Ataraxos and
DeepNash use magnets rather than averaging.

## Further reading

- Bailey, Nagarajan, Piliouras. Stochastic multiplicative weights in zero-sum games.
  [arXiv:2110.02134](https://arxiv.org/abs/2110.02134), 2021. Not read.
