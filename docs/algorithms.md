# Algorithms for two-player zero-sum games: a study guide

This explains the ideas behind the algorithms you might implement in this repo, in the
order that builds on itself. It has no code; the places your code goes are
`rlgames/core/learner.py` (anything trained from self-play games) and
`rlgames/core/cfr.py` (CFR+, checked by `tests/test_cfr.py`).

Suggested path:

1. Policy gradient self-play on matching pennies and rock-paper-scissors. Watch it cycle.
2. Fix the cycling: averaging, entropy regularization, then magnetic mirror descent.
3. Move to Kuhn and Leduc poker and see what changes with hidden information.
4. CFR+ on the exact tree, which gives the "right answer" curves to compare against.

---

## 0. The setting

Two players, one's gain is the other's loss. A **policy** maps what a player has seen
(its *information set*: own cards, public actions so far) to a distribution over legal
actions.

- **Best response** to a policy: the policy that wins the most against it. It
  is deterministic, and computing it needs the full game tree, which is why only the
  small games get exact numbers.
- **Exploitability**: how much a best-responding opponent wins against you, averaged
  over the two seats, relative to the game's value. It is zero exactly at a **Nash
  equilibrium**, and it is the main curve on the dashboard. Lower is better; it never goes
  negative.
- **Game value**: what player 0 expects to win when both play an equilibrium. Kuhn poker:
  −1/18 ≈ −0.056. Leduc: about −0.086. Symmetric games (matching pennies, RPS,
  tic-tac-toe): 0.

Why exploitability and not win rate? Win rate against a fixed opponent (the dashboard's
`vs_random`) rewards exploiting that opponent's specific mistakes. A policy can crush
random play and still be very exploitable. In zero-sum games the robust goal is low
exploitability.

Key fact: in zero-sum games **every Nash equilibrium gives the same value**, and
playing any equilibrium policy guarantees at least that value against anyone. That is
what makes "solve the game" well defined here, and it is not true in general-sum or
more-than-two-player games.

---

## 1. Policy gradient self-play

### The idea

Both seats use the same network. Play games, then nudge the policy toward actions that
led to good outcomes for the player who took them.

The **policy gradient theorem** (REINFORCE): the gradient of expected return is the
expectation of `return × ∇ log π(action | observation)`. So the loss to *minimize* for
one batch is the negative of the average, over real decisions, of
`outcome × log π(action)`.

In the batch:

- `batch.outcome` is already from the acting player's perspective, so the same formula
  serves both seats.
- `batch.mask` marks real decisions; leave out every other row.
- Compute `log π(action)` by running the network on `batch.obs`, masking illegal actions,
  taking a log-softmax and indexing with `batch.action`. Recompute it inside the loss;
  `batch.log_prob` is a constant recorded at sampling time, so gradients don't flow
  through it.

### Variance and baselines

The raw outcome is noisy. Subtracting a **baseline** b(observation) that doesn't depend
on the action keeps the gradient unbiased and lowers variance: use
`(outcome − b) × ∇ log π`. The network already has a value head (`apply_mlp` returns
`value`). Train it with a squared error toward `outcome`, and use its prediction, without
gradient (`jax.lax.stop_gradient`), as `b`. `outcome − b` is the **advantage**.

### On-policy

The gradient formula assumes the batch was sampled from the current policy. That holds
for one step per batch, which is what the training loop does. Taking several gradient
steps on the same batch needs a correction: importance weights
`π_new / π_old = exp(log π_new − batch.log_prob)`, usually clipped (that is PPO).

### What you will see: cycling

Run it on `matching_pennies`. Player 0 wins if the coins match, player 1 if they differ.
The only equilibrium is 50/50 for both.

Picture the two players' probabilities of Heads as a point in a square. The gradient
does not point toward the centre; it points *around* it. If P1 favours Heads, P0 shifts
toward Heads, which pushes P1 toward Tails, which pushes P0 toward Tails, and so on.
With infinitesimal steps the point orbits the equilibrium forever. With real step sizes it
slowly spirals *outward*.

The dashboard's policy-trajectory plot shows this directly (a circle or spiral in the
square; for RPS, a loop in the triangle). Exploitability oscillates instead of falling.
This is the same phenomenon as GAN training instability: simultaneous gradient play on
a min-max objective has rotational dynamics.

Things to try and watch:

- `rock_paper_scissors`, then `biased_rock_paper_scissors`, whose equilibrium
  (¼, ½, ¼) is not uniform, so "high entropy" alone does not get you there.
- Learning rate: larger learning rates make bigger orbits and faster outward spiral.
- `--optim.name sgd` vs `adamw` vs `muon`. Adam's normalisation changes the effective
  step size per parameter, which changes the shape of the orbit.

---

## 2. Fixing the cycling

### 2a. Average the iterates

Even when the current policy (the **last iterate**) cycles, the *time average* of the
policies can converge. For no-regret learners such as Hedge (multiplicative weights),
there is a theorem: in zero-sum games, the average policy's exploitability goes to zero.
Softmax policy gradient is close to Hedge on matrix games, and the average of the orbit
sits near the centre.

The catch with a neural network: "the average policy" is the average of *distributions*,
not of parameters. Averaging parameters does not average policies. You either keep a
second network trained to imitate the average (as Neural Fictitious Self-Play does) or
keep checkpoints and mix them. This is cumbersome, which motivates methods whose last
iterate converges.

### 2b. Entropy regularization

Add `−τ × entropy(π)` to the loss, i.e. reward randomness. The game is then
*regularized*: each player maximises return plus τ times entropy. Two things change:

- The regularized game has a unique equilibrium, called a **quantal response
  equilibrium** (QRE), and it is strictly inside the simplex.
- The dynamics now pull inward. The rotation is still there but the spiral contracts,
  and the last iterate converges.

The price: the QRE is not the Nash equilibrium. With τ large it is close to uniform; as τ
goes to 0 the QRE approaches Nash, but convergence slows. On biased RPS you can watch
the fixed point move from the centre toward (¼, ½, ¼) as you lower τ.

### 2c. Magnetic mirror descent (MMD)

MMD (Sokota et al., 2022, "A Unified Approach to Reinforcement Learning, Quantal Response
Equilibria, and Two-Player Zero-Sum Games") generalises entropy regularization: instead
of pulling toward uniform, pull toward a **magnet** policy ρ. Each step maximises:

    expected advantage under π
    − (1/η) × KL(π ‖ π_current)     (don't move too far in one step)
    − α × KL(π ‖ ρ)                 (stay near the magnet)

With ρ uniform the magnet term is entropy regularization. The paper's results:

- For α > 0, the last iterate converges (linearly, in the tabular setting) to the
  equilibrium of the game regularized toward ρ.
- In the softmax / tabular case the step has a closed form: the new policy is
  proportional to `[π_current × ρ^(αη) × exp(η × Q)]^(1/(1 + αη))`. Getting this
  formula from the objective above is a good exercise.
- **Moving magnet**: every so often, set ρ to the current policy (or move it slowly
  toward it). Each phase converges to a regularized equilibrium closer to Nash, so the
  sequence approaches the actual equilibrium.

As a deep RL loss it looks like policy gradient (often PPO-style) plus a KL penalty
toward the magnet network, which is a frozen copy of the parameters
(`self.magnet_params = params` in `Learner.init`, refreshed every N updates).

What to watch: spirals contract, last-iterate exploitability goes down (on the
dashboard you are plotting the network's current policy, so a falling curve means
last-iterate convergence), and the plateau level depends on α.

---

## 3. Hidden information: Kuhn and Leduc poker

Moving from matrix games to poker adds two things.

**Sequential decisions.** One game has several decisions, and `batch.outcome` is only
known at the end. Every decision in a game gets credit for the final outcome. That is
correct in expectation but noisy; the baseline matters more here.

**Information sets.** You see your card, not your opponent's. The equilibrium has to
**bluff** (bet weak hands sometimes) and **slow-play** (check strong hands sometimes),
because a deterministic policy leaks your card through your actions. In Kuhn poker, for
example, an equilibrium player 0 bets the Jack with some probability α ∈ [0, ⅓] and the
King with probability 3α. A policy that is greedy about its current estimates will not
find these mixed strategies; it oscillates between "always bluff" and "never bluff",
which is the cycling from section 1 again, now inside the game tree.

Kuhn has 12 information sets, so you can print the full policy (`tabulate_policy` in
`rlgames/core/evaluation.py`) and compare it to the known equilibrium family.

---

## 4. Counterfactual regret minimization (CFR)

CFR is not deep RL. It walks the whole game tree every iteration, keeps a table indexed
by information set, and comes with convergence guarantees. Its exploitability curves are
the reference your learning algorithms are trying to approach.

### Regret, in one decision

Suppose you face the same decision many times. Your **regret** for action a is: total
reward you *would* have got by always playing a, minus what you actually got.
**Regret matching**: play each action with probability proportional to its positive
cumulative regret (uniform if none is positive). Regret matching is a no-regret
algorithm: average regret goes to zero. And, from section 2a, if both players have no
regret, their *average* policies converge to Nash in zero-sum games.

### The counterfactual part

In a tree, a decision is reached with some probability that depends on everyone's
earlier actions, including your own. CFR's trick is to measure regret at each information
set **as if you had played to reach it**:

- The **counterfactual value** of an action at information set I sums over the histories
  h in I (the possible hidden states consistent with what you see). Each term is the value
  of taking the action at h and then following the current policy, weighted by the
  probability that **chance and the opponent** reach h. Your own probability of
  reaching h is left out.
- The instantaneous regret of action a at I is its counterfactual value minus the
  counterfactual value of the current policy at I, which is the policy-weighted average
  over actions.

The theorem: total regret in the whole game is bounded by the sum of these local
counterfactual regrets. So running regret matching independently at every information
set makes the overall average policy converge to Nash.

Why leave out your own reach? If your current policy rarely reaches I, you would
otherwise learn almost nothing there. And it is precisely the decisions you avoid that a
best-responding opponent could punish you for. Counterfactual weighting makes every
information set learn at full speed.

### The average policy

As with all no-regret methods, the **current** policy does not converge; the **average**
does. The average must be weighted by **your own** reach probability: an information set
you rarely reach should contribute less of each iteration's policy to the average. This
is the mirror image of the counterfactual weighting, and it is easy to mix the two up.

### CFR+

Three changes that make it converge much faster in practice:

1. **Regret matching+**: after each update, floor cumulative regrets at zero. Actions
   that were terrible once can recover quickly when they become good.
2. **Alternating updates**: update player 0, then player 1 *using player 0's new
   policy*, instead of both at once.
3. **Linear averaging**: weight iteration t's policy by t in the average, so early,
   bad iterations fade.

Vanilla CFR's exploitability falls roughly like 1/√T; CFR+ in practice closer to 1/T.

### Implementing it here

The tree is already enumerated into arrays (`rlgames/core/tree.py`), with helpers for the
pieces you need:

- `expected_values`: the value of every node under a policy
- `reach_probs`: reach probabilities that include only the actors you choose (chance,
  the opponent, or yourself)
- `child_values`: the values of a node's children, one per action

What is left to you is the CFR logic: which reach to use for regrets versus the average,
how to sum over the histories in an information set, and the CFR+ details.
`rlgames/core/cfr.py` lists the conventions that make your numbers match the reference
exactly.

Work in this order, running `uv run pytest tests/test_cfr.py -v` as you go:

1. `regret_matching`: two tests.
2. `current_policy`, `average_policy` and `iterate`. Rock-paper-scissors is the easiest
   comparison (one decision per player). Kuhn adds chance and sequential play. Goofspiel
   adds simultaneous moves. Tic-tac-toe adds **transpositions**: the same board reached
   by different move orders, which you have to sum over when computing own reach.
3. The convergence tests: Kuhn's value of −1/18 and biased RPS's (¼, ½, ¼).

When a comparison test fails, it tells you the iteration and whether the current or
average policy differs. If iteration 1's current policy is already wrong, the problem is
in the regret update; if only the average differs, look at its weighting.

---

## 5. Where this leads

- **Deep CFR / DREAM / ESCHER**: CFR with sampled trees and networks instead of tables,
  for games too large to enumerate.
- **R-NaD / MMD at scale**: last-iterate convergent regularized dynamics with neural
  networks. DeepMind's DeepNash for Stratego used R-NaD; Ataraxos's RL loss
  (`pyengine/core/rl.py` in the stratego repo) regularizes toward a uniform magnet.
- **Search** (AlphaZero, ReBeL, Student of Games): combine a learned value function with
  planning at play time. Perfect information (tic-tac-toe, Connect Four) is the easy
  case; with hidden information the search has to reason over beliefs about hidden
  state.
