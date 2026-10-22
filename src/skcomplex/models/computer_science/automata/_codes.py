"""Rule codes of cellular automata: Wolfram numbers of elementary rules and the life-like rules."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
from collections.abc import Iterable
from dataclasses import dataclass

from ....core.model import OuterTotalistic, Table

ELEMENTARY_RULES = 256


@dataclass(frozen=True)
class Wolfram:
    """Elementary cellular automaton by its Wolfram number, on the left, centre, and right cells.

    A local rule takes it as its rule and reads it through `table()`.

    Args:
        number:
            The rule number from 0 to 255.

    Examples:
        >>> from skcomplex.models.computer_science.automata import Wolfram
        >>> Wolfram(30).table().mapping[(1, 0, 0)]
        1
    """

    number: int

    def table(self) -> Table:
        """Return the rule as a table of the eight patterns.

        Returns:
            table:
                The table.

        Raises:
            ValueError:
                The number is outside 0 to 255.
        """
        if not 0 <= self.number < ELEMENTARY_RULES:
            error_msg = f'A Wolfram rule number is between 0 and 255, but it is {self.number}.'
            raise ValueError(error_msg)
        patterns = itertools.product((0, 1), repeat=3)
        return Table(
            {pattern: (self.number >> (4 * pattern[0] + 2 * pattern[1] + pattern[2])) & 1 for pattern in patterns},
        )


def life(born: Iterable[int] = (3,), survive: Iterable[int] = (2, 3)) -> OuterTotalistic:
    """Return a life-like rule in the notation `B3/S23` of Conway's Game of Life.

    Args:
        born:
            The numbers of live neighbours that bring a dead cell to life.

        survive:
            The numbers of live neighbours that keep a live cell alive.

    Returns:
        rule:
            The outer totalistic rule.

    Examples:
        >>> from skcomplex.models.computer_science.automata import life
        >>> rule = life()
        >>> rule.rule[(0, 3)], rule.rule[(1, 2)], (1, 4) in rule.rule
        (1, 1, False)
    """
    return OuterTotalistic({**{(0, count): 1 for count in born}, **{(1, count): 1 for count in survive}})
