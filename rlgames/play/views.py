"""Human-readable views of game states for the play page.

Each view works on a state whose arrays have been converted to numpy, from the
perspective of one `viewer` seat, and never shows hidden information unless
`reveal` is set (the play server sets it once the game is over).
"""

import numpy as np

from rlgames.games.goofspiel import Goofspiel
from rlgames.games.kuhn_poker import KuhnPoker
from rlgames.games.leduc_poker import LeducPoker
from rlgames.games.liars_dice import LiarsDice
from rlgames.games.matrix import MatrixGame
from rlgames.games.mnk import EMPTY, MNKGame
from rlgames.utils.constants import CHANCE


def who(player: int, viewer: int) -> str:
    return "You" if player == viewer else "AI"


class View:
    def __init__(self, game):
        self.game = game

    def labels(self, state) -> list[str]:
        """Label of every action id at `state`."""
        return [str(a) for a in range(self.game.num_actions)]

    def event(self, state, action: int, viewer: int) -> str | None:
        """Description of `action` taken at `state`, or None if `viewer` cannot see it."""
        if int(state.player) == CHANCE:
            return None
        return f"{who(int(state.player), viewer)}: {self.labels(state)[action]}"

    def info(self, state, viewer: int, reveal: bool) -> list[list[str]]:
        """Rows of [label, value] for the info panel."""
        return []

    def grid(self, state) -> dict | None:
        return None


class MatrixView(View):
    NAMES = {
        "matching_pennies": ["Heads", "Tails"],
        "rock_paper_scissors": ["Rock", "Paper", "Scissors"],
        "biased_rock_paper_scissors": ["Rock", "Paper", "Scissors"],
    }

    def labels(self, state):
        return self.NAMES.get(self.game.name, super().labels(state))

    def event(self, state, action, viewer):
        # Moves are simultaneous: nobody sees the other's choice until the end.
        if int(state.player) != viewer:
            return None
        return f"You: {self.labels(state)[action]}"

    def info(self, state, viewer, reveal):
        names = self.labels(state)
        payoffs = np.asarray(self.game.payoffs)
        if viewer == 1:
            payoffs = -payoffs.T + 0.0  # avoid printing -0
        rows = [[f"Your payoff for {names[i]}", " / ".join(f"{v:+g}" for v in payoffs[i])]
                for i in range(len(names))]
        rows.insert(0, ["vs AI playing", " / ".join(names)])
        if reveal:
            moves = [int(state.action0), int(state.action1)]
            rows.append(["You played", names[moves[viewer]]])
            rows.append(["AI played", names[moves[1 - viewer]]])
        return rows


class KuhnView(View):
    CARDS = ["J", "Q", "K"]

    def labels(self, state):
        p = max(int(state.player), 0)
        facing = state.contrib[p] < state.contrib[1 - p]
        return (["Fold", "Call"] if facing else ["Check", "Bet"]) + ["-"]

    def event(self, state, action, viewer):
        if int(state.player) == CHANCE:
            seat = 0 if state.cards[0] < 0 else 1
            return f"You are dealt {self.CARDS[action]}" if seat == viewer else None
        return super().event(state, action, viewer)

    def info(self, state, viewer, reveal):
        def card(seat):
            c = int(state.cards[seat])
            return self.CARDS[c] if c >= 0 else "-"

        return [
            ["Your card", card(viewer)],
            ["AI card", card(1 - viewer) if reveal else "?"],
            ["Pot", f"{float(state.contrib.sum()):g}"],
            ["Your chips in", f"{float(state.contrib[viewer]):g}"],
        ]


class LeducView(View):
    def card(self, c: int) -> str:
        return "JQK"[c // 2] + "♠♥"[c % 2] if c >= 0 else "-"

    def labels(self, state):
        p = max(int(state.player), 0)
        facing = state.contrib[p] < state.contrib[1 - p]
        names = ["Fold", "Call", "Raise"] if facing else ["-", "Check", "Bet"]
        return names + ["-"] * (self.game.num_actions - 3)

    def event(self, state, action, viewer):
        if int(state.player) == CHANCE:
            if state.cards[0] < 0:
                return f"You are dealt {self.card(action)}" if viewer == 0 else None
            if state.cards[1] < 0:
                return f"You are dealt {self.card(action)}" if viewer == 1 else None
            return f"Public card: {self.card(action)}"
        return super().event(state, action, viewer)

    def info(self, state, viewer, reveal):
        return [
            ["Your card", self.card(int(state.cards[viewer]))],
            ["AI card", self.card(int(state.cards[1 - viewer])) if reveal else "?"],
            ["Public card", self.card(int(state.public))],
            ["Round", str(int(state.round) + 1)],
            ["Pot", f"{float(state.contrib.sum()):g}"],
            ["Your chips in", f"{float(state.contrib[viewer]):g}"],
        ]


class GoofspielView(View):
    def labels(self, state):
        return [f"Bid {a + 1}" for a in range(self.game.num_actions)]

    def event(self, state, action, viewer):
        p = int(state.player)
        if p == CHANCE:
            return f"Prize card: {action + 1}"
        if p == viewer:
            return f"You bid {action + 1}"
        if p == 1:
            # Player 1's bid completes the round, so both bids are revealed together.
            return f"AI bid {action + 1}"
        return "AI placed a hidden bid"

    def info(self, state, viewer, reveal):
        k = self.game.num_cards
        done = int(state.round)

        def cards(seq):
            return " ".join(str(int(c) + 1) if c >= 0 else "·" for c in seq)

        ai_bids = [state.bids[1 - viewer][r] if r < done or reveal else -1 for r in range(k)]
        return [
            ["Prizes", cards(state.prizes)],
            ["Your bids", cards(state.bids[viewer])],
            ["AI bids", cards(ai_bids)],
            ["Your hand", " ".join(str(c + 1) for c in range(k) if state.hands[viewer][c])],
            ["Points (you / AI)", f"{float(state.points[viewer]):g} / {float(state.points[1 - viewer]):g}"],
        ]


class LiarsDiceView(View):
    def bid(self, b: int) -> str:
        quantity, face = divmod(b, self.game.num_faces)
        return f"{quantity + 1} × {face + 1}"

    def labels(self, state):
        names = [self.bid(b) for b in range(self.game.num_bids)] + ["Liar!"]
        return names + ["-"] * (self.game.num_actions - len(names))

    def event(self, state, action, viewer):
        if int(state.player) == CHANCE:
            seat = int(state.num_rolled) // self.game.num_dice
            return f"You rolled {action + 1}" if seat == viewer else None
        return super().event(state, action, viewer)

    def info(self, state, viewer, reveal):
        def dice(seat):
            return " ".join(str(int(d) + 1) if d >= 0 else "-" for d in state.dice[seat])

        last = int(state.last_bid)
        return [
            ["Your dice", dice(viewer)],
            ["AI dice", dice(1 - viewer) if reveal else "?"],
            ["Current bid", self.bid(last) if last >= 0 else "none"],
            ["Bid format", "quantity × face, over both players' dice"],
        ]


class MNKView(View):
    MARKS = ["X", "O"]

    def labels(self, state):
        if self.game.gravity:
            return [f"Column {c + 1}" for c in range(self.game.cols)]
        return [f"Row {r + 1}, column {c + 1}" for r in range(self.game.rows)
                for c in range(self.game.cols)]

    def info(self, state, viewer, reveal):
        return [["You play", self.MARKS[viewer]], ["AI plays", self.MARKS[1 - viewer]]]

    def grid(self, state):
        cells = [self.MARKS[v] if v != EMPTY else "" for v in state.board.tolist()]
        return {
            "rows": self.game.rows,
            "cols": self.game.cols,
            "cells": cells,
            "gravity": self.game.gravity,
        }


def make_view(game) -> View:
    for game_cls, view_cls in [
        (MatrixGame, MatrixView),
        (KuhnPoker, KuhnView),
        (LeducPoker, LeducView),
        (Goofspiel, GoofspielView),
        (LiarsDice, LiarsDiceView),
        (MNKGame, MNKView),
    ]:
        if isinstance(game, game_cls):
            return view_cls(game)
    return View(game)
