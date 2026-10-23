"""Share the helpers of the compilation tests."""

from skcomplex.core.components import Element
from skcomplex.core.laws import Noise


def _pair(process: str = 'common', conventions: tuple[str, str] = ('ito', 'ito')) -> type[Element]:
    """Return an element whose two states take unit noise on one local process."""
    first, second = conventions
    return Element.define(
        'Pair',
        states={'x': 1.0, 'y': 1.0},
        parameters={'sigma': 0.5},
        laws=lambda s: [
            Noise(s.x, s.sigma * s.y, convention=first, process=process),
            Noise(s.y, s.sigma * s.x, convention=second, process=process),
        ],
    )
