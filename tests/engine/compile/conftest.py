"""Share the helpers of the compilation tests."""

from skcomplex.core.components import ElementComponent
from skcomplex.core.laws import NoiseLaw


def _pair(process: str = 'common', conventions: tuple[str, str] = ('ito', 'ito')) -> type[ElementComponent]:
    """Return an element whose two states take unit noise on one local process."""
    first, second = conventions
    return ElementComponent.define(
        'PairComponent',
        states={'x': 1.0, 'y': 1.0},
        parameters={'sigma': 0.5},
        laws=lambda s: [
            NoiseLaw(s.x, s.sigma * s.y, convention=first, process=process),
            NoiseLaw(s.y, s.sigma * s.x, convention=second, process=process),
        ],
    )
