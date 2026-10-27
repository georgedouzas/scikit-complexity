"""Share the helpers of the compilation tests."""

from skcomplex.core.components import Element
from skcomplex.core.declarations import parameter, state

from skcomplex.core.laws import NoiseLaw


def _pair(process: str = 'common', conventions: tuple[str, str] = ('ito', 'ito')) -> type[Element]:
    """Return an element whose two states take unit noise on one local process."""
    first, second = conventions
    return type(
        'PairElement',
        (Element,),
        {
            'x': state(1.0),
            'y': state(1.0),
            'sigma': parameter(0.5),
            'laws': lambda self, s: [
                NoiseLaw(s.x, s.sigma * s.y, convention=first, process=process),
                NoiseLaw(s.y, s.sigma * s.x, convention=second, process=process),
            ],
        },
    )
