---
name: teach
description: Run or resume a lesson in this repo — explain an algorithm (CFR+, regret matching, PPO, search), check the user's hand-worked answers, or scaffold an implementation step. Use when the user says "teach", "let's continue", "next lesson", "go on <topic>", or asks a conceptual question about an algorithm on the roadmap.
---

# Teach

The process is in `CLAUDE.md` (three phases: intuition, math, implementation; least help
first; reference code hidden until tests pass). This skill adds how to run a session.

## Start of a session

1. Read `docs/lesson-log.md`. It has the current topic, the phase, open questions left
   with the user, and the next step.
2. Say in one or two lines where we are and what's next. Don't recap the whole history.
3. If an open question is pending, ask it again before introducing anything new.

## During a lesson

- **Let the user build the definitions.** Before giving a definition, set up the problem
  and a hint, then ask the user to define it. For example: "after a game you can ask how
  much better you would have done with a different choice; how would you turn that into a
  number per action?" Then refine their answer together. Ask open, high-level questions
  (what should this quantity measure? what would go wrong if we did X? why this weighting?)
  more often than computational ones. Give the definition only after they have tried, or
  if they ask for it.
- **State the point first.** Say what an example or exercise will show before asking for
  any computation. Ask at most one question per message, and make it test the idea (a
  prediction or a "why"), not arithmetic.
- **One hand-worked case per topic**, as `CLAUDE.md` requires. Keep it small. Give a
  worked example of the same shape first if the user is stuck on what to compute.
- **Recap the data.** When a question uses earlier numbers, put them in a table in the
  same message. Don't expect the user to remember earlier rounds.
- **Define before use.** Spell out abbreviations (RPS = rock-paper-scissors) and symbols
  (`v`, `cfv`, `reach_opp`) the first time they appear in a session.
- **Build from what the user knows.** The user knows policy gradient. Connect new ideas to
  it: `q − v` is the advantage; CFR weights by `reach_opp`, PG by `reach_self·reach_opp`;
  KL-proximal updates are mirror descent.
- **If the user says they're lost, stop and simplify.** Answer their literal question in
  plain words, say which parts can be skipped (cited theorems, proof details), and list the
  two or three facts they actually need.
- **Use the notation in `abstractions.md`.** The policy is `sigma`; reach is `reach`,
  `reach_self`, `reach_opp`. Label definitions, derived results, and cited theorems.
- **Check answers directly.** Say which part is right and which is wrong, and give a
  self-check the user can apply (for example, "with a pure policy, `v` must equal `q` of
  the action played"). Don't hand over the corrected numbers.
- **Comparisons to `../stratego` (Ataraxos):** read the code before describing it and cite
  `file:line`. Say what you didn't check.

## Interactive component

Every lesson should have something the user can play with. Point to an existing page
first:

- `docs/play.html#<game_id>` to play any repo game, with the information set shown
- `docs/selfplay-orbits.html` for self-play dynamics, current vs average, and magnets
- `docs/kuhn-table.html` for Kuhn rules, the game tree, and the exercise

Tie the open question to something the user does on the page, for example "play
player 1, hold Q, and decide whether to call". If no page covers the idea, build one
with the viz skill.

## End of a session, or when asked to log

Update `docs/lesson-log.md`:

- the phase status table,
- what was covered (one line per idea, with any worked numbers),
- open questions left with the user,
- the next step,
- new teaching notes, if the user pushed back on something.

Commit and push only when the user asks.
