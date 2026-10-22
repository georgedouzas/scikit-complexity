"""Interpret the words of rewriting systems as line segments in two or three dimensions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from ..base import RewritingError

ACTIONS = frozenset(
    {
        'draw',
        'move',
        'left',
        'right',
        'reverse',
        'push',
        'pop',
        'pitch_down',
        'pitch_up',
        'roll_left',
        'roll_right',
    },
)
DEFAULT_SYMBOLS: dict[str, str] = {
    'F': 'draw',
    'f': 'move',
    '+': 'left',
    '-': 'right',
    '|': 'reverse',
    '[': 'push',
    ']': 'pop',
    '&': 'pitch_down',
    '^': 'pitch_up',
    '\\': 'roll_left',
    '/': 'roll_right',
}


@dataclass(frozen=True)
class WordGeometry:
    r"""Geometric interpretation of a word as line segments in two or three dimensions.

    A cursor walks the word: symbols mean `draw` (forward with a segment), `move` (forward without one), `left` and
    `right` (turn by the angle), `reverse` (turn by 180 degrees), `push` and `pop` (save and restore the position and
    heading), and in three dimensions `pitch_down`, `pitch_up`, `roll_left`, and `roll_right`. This is the turtle
    interpretation of Prusinkiewicz and Lindenmayer (1990), whose symbols are the default meanings: `F`, `f`, `+`, `-`,
    `|`, `[`, `]`, `&`, `^`, `\`, and `/`. A module's first parameter overrides the step of a move or the angle of a
    turn.

    Args:
        angle:
            The turning angle in degrees.

        step:
            The step at generation 0.

        contraction:
            The factor the step is multiplied by at every generation.

        symbols:
            Meanings of symbols that replace or extend the default ones.

        dimension:
            2 or 3.

        heading:
            The initial heading in degrees in two dimensions, measured from the x axis.

    Examples:
        >>> from skcomplex.core.viz import WordGeometry
        >>> WordGeometry(angle=60.0, contraction=1 / 3).meanings()['F']
        'draw'
    """

    angle: float = 90.0
    step: float = 1.0
    contraction: float = 1.0
    symbols: Mapping[str, str] | None = None
    dimension: int = 2
    heading: float = 0.0

    def meanings(self) -> dict[str, str]:
        """Return the meaning of every symbol the geometry interprets.

        Returns:
            meanings:
                A mapping from symbol to action.

        Raises:
            RewritingError:
                A meaning is not an action of the geometry, or the dimension is neither 2 nor 3.
        """
        meanings = {**DEFAULT_SYMBOLS, **(self.symbols or {})}
        unknown = sorted(set(meanings.values()) - ACTIONS)
        if unknown:
            error_msg = f'The word geometry has no action {unknown[0]!r}. Use one of {", ".join(sorted(ACTIONS))}.'
            raise RewritingError(error_msg)
        if self.dimension not in {2, 3}:
            error_msg = f'The word geometry has dimension {self.dimension}. Use 2 or 3.'
            raise RewritingError(error_msg)
        return meanings
