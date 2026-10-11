from rlgames.games.base import Game, State  # noqa: F401
from rlgames.games.goofspiel import Goofspiel
from rlgames.games.kuhn_poker import KuhnPoker
from rlgames.games.leduc_poker import LeducPoker
from rlgames.games.liars_dice import LiarsDice
from rlgames.games.matrix import (
    biased_rock_paper_scissors,
    matching_pennies,
    rock_paper_scissors,
)
from rlgames.games.mnk import connect_four, tic_tac_toe

GAMES = {
    "matching_pennies": matching_pennies,
    "rock_paper_scissors": rock_paper_scissors,
    "biased_rock_paper_scissors": biased_rock_paper_scissors,
    "kuhn_poker": KuhnPoker,
    "leduc_poker": LeducPoker,
    "goofspiel": Goofspiel,
    "liars_dice": LiarsDice,
    "tic_tac_toe": tic_tac_toe,
    "connect_four": connect_four,
}


def make(name: str, **kwargs) -> Game:
    return GAMES[name](**kwargs)
