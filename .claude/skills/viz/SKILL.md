---
name: viz
description: Build an interactive, standalone HTML page that shows an algorithm running (self-play dynamics, regret matching, CFR, magnets, PPO) or lets the user play or step through a game. Use when the user asks for a viz, visualization, interactive page, demo, or "something I can play with", or when a lesson needs an interactive component.
---

# Interactive lesson pages

Model pages: `docs/selfplay-orbits.html` (learners running side by side on biased RPS),
`docs/play.html` (every repo game against a bot), and `docs/kuhn-table.html`
(rules, playable hand, game tree, exercise). Read the closest one before starting.

## What makes these pages useful

- **Run the real algorithm live in the page.** Implement the actual update rule in JS
  and animate it. Don't draw a sketch of what it would do. Provide Play, Step, and
  Reset controls, an iteration counter, and sliders for the hyperparameters that matter
  (step size η, temperature τ, reset period K).
- **Compare variants in parallel, not behind a selector.** Small multiples with one panel
  per method, sharing the same inputs and start point, all updating together. The user
  asked for this explicitly.
- **Each panel states its math.** Show the objective, or "goal (not optimized directly)"
  when there is no objective, such as regret matching. Then show the update rule using the
  `abstractions.md` notation (`sigma`, `q`, `v`, `reach_opp`), and a one-line summary of
  what to look for.
- **Use the same inputs for code and math.** When comparing methods (for example PG and
  regret matching), write each update as a function of the same arguments (`pi`, `q`), so
  the only difference is the rule itself. Show the Python next to the formula.
- **Show the quantities the lesson is about.** Show the current policy and the average
  policy as separate traces. Show exploitability on a log-scale chart with hover. Show the
  Nash point, plus the magnet and its path when there is one.
- **Make it directly manipulable.** Clicking a plot sets the start point for every
  learner. Changing a slider takes effect immediately.
- **Make it self-contained.** Put the rules, definitions, payoff tables, and the exercise
  on the page, so it makes sense without the chat. End with a short "try this" or
  prediction prompt (see the teach skill: ask, don't tell).
- **For games, show the information set and the numbers behind decisions.** Show what
  the acting player knows and what is hidden ("the AI holds J or K"). Make the rules match
  `rlgames/games/*.py` exactly. At each decision, show:
  - the true history next to the player's observations
  - the histories in the information set, with their reaches, belief, and `q(h, a)`
  - per action: `cfv`, regret, and the PG, RM, and magnet updates

  Let the user review every decision after the game, for either player. `docs/play.html`
  does all of this generically via `obsToken`, a DFS over consistent histories, and a
  memoized expected-value recursion.

## Format

- Put a standalone `.html` in `docs/`: `<!doctype html>`, charset, viewport, inline CSS
  and JS, no build step. It must work when opened from disk. Google Fonts are optional;
  give a system-font fallback.
- Define colors as CSS variables on `:root`, with dark mode under
  `@media (prefers-color-scheme: dark)` (guarded by `:root:not([data-theme="light"])`) and
  `:root[data-theme="dark"]`. Give `body` an explicit background.
- Use a 16px side gutter, no horizontal scroll at phone width, and grids that collapse to
  one column.
- Draw plots in SVG or canvas by hand, with no chart libraries. Blue is the current
  policy, orange is the average, ★ is Nash, and ◆ is the magnet.
- Wrap any `localStorage` use in try/catch. Use it only for conveniences such as the last
  game picked.
- If the user wants a shareable link, publish the same file as an Artifact too. The repo
  copy is the source of truth.

## Verify before showing it

1. Run `node --check` on the extracted `<script>` contents.
2. Check that the numbers are right outside the page. Run the core update in a standalone
   node script and confirm the claims the page makes. Examples: PG orbits outward, the RM
   average converges, the magnet fixed point is `π ∝ m·exp(q/τ)`, the equilibrium bot
   is unexploitable.
3. For game rules, replay random JS games through the Python engines (`rlgames.games.make`)
   and compare the legal action sets and returns at every step. The check used for
   `docs/play.html` lives in the scratchpad history; recreate it if the rules change.
4. Say whether the page was viewed rendered. If not, tell the user it wasn't.

## After building

- Link the page from `docs/lesson-log.md` under Pages.
- Commit and push only when the user asks.
