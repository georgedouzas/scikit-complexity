"""Define the game state spec of the minority game."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

from ....specs import BaseSpec
from ....types import IntArray


@dataclass(frozen=True)
class _GameStateSpec(BaseSpec):
    """History, round, and strategy tables of a run, which the game keeps in the storage of the model.

    Args:
        history:
            The code of the last outcomes.

        round:
            The number of rounds played.

        tables:
            The table of every strategy code met so far.
    """

    history: int
    round: int = 0
    tables: Mapping[float, IntArray] = field(default_factory=dict)
