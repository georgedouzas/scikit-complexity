"""Define the game state spec of the minority game."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass, field

from ....core.specs import BaseSpec
from ....core.types import IntArray


@dataclass(frozen=True)
class _GameStateSpec(BaseSpec):
    """History and strategy tables of a run, which the game keeps in the storage of the system.

    Args:
        history:
            The code of the last outcomes.

        tables:
            The table of every strategy code met so far.
    """

    history: int
    tables: Mapping[float, IntArray] = field(default_factory=dict)
