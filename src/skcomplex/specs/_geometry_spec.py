"""Define the geometry spec."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass

from ..errors import RewritingError
from ._base_spec import BaseSpec

TURTLE_SYMBOLS = {
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
class GeometrySpec(BaseSpec):
    r"""Spec of the geometric interpretation of a word as line segments in two or three dimensions.

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
            Meanings of symbols that replace or extend the default ones of `TURTLE_SYMBOLS`.

        dimension:
            2 or 3.

        heading:
            The initial heading in degrees in two dimensions, measured from the x axis.

    Examples:
        >>> from skcomplex.specs import GeometrySpec
        >>> GeometrySpec(angle=60.0, contraction=1 / 3).angle
        60.0
    """

    angle: float = 90.0
    step: float = 1.0
    contraction: float = 1.0
    symbols: Mapping[str, str] | None = None
    dimension: int = 2
    heading: float = 0.0

    def _check(self) -> None:
        """Raise an error for a meaning that is not an action, or a dimension that is neither 2 nor 3.

        Raises:
            RewritingError:
                A meaning is not an action of the geometry, or the dimension is neither 2 nor 3.
        """
        actions = set(TURTLE_SYMBOLS.values())
        unknown = sorted(set((self.symbols or {}).values()) - actions)
        if unknown:
            error_msg = f'The word geometry has no action {unknown[0]!r}. Use one of {", ".join(sorted(actions))}.'
            raise RewritingError(error_msg)
        if self.dimension not in {2, 3}:
            error_msg = f'The word geometry has dimension {self.dimension}. Use 2 or 3.'
            raise RewritingError(error_msg)
