"""Model jerk equations, third-order equations that are the simplest chaotic flows."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import abs_symbolic

from ...core.base import state
from ...core.model import Element, Rate, System, derivative
from ...core.runtime import register_model

Jerk = Element.define(
    'Jerk',
    states={'x': state(default=0.0, order=3, initial=(0.0, 1.0))},
    parameters={'A': 0.6},
    laws=lambda s: [Rate(s.x, -s.A * derivative(s.x, 2) - derivative(s.x, 1) + abs_symbolic(s.x) - 1)],
)


@register_model(
    field='nonlinear dynamics',
    tags=('chaos', 'ode', 'third order'),
    references=('Sprott 1997', 'Datseris and Parlitz 2022'),
    defaults={'until': 200.0, 'dt': 0.1},
)
def sprott_jerk(
    x: float = 0.0,
    velocity: float = 0.0,
    acceleration: float = 1.0,
    *,
    A: float = 0.6,
) -> System:
    """Return Sprott's jerk equation `x''' = -A x'' - x' + |x| - 1`, chaotic near A = 0.6.

    Args:
        x:
            The initial position.

        velocity:
            The initial first derivative.

        acceleration:
            The initial second derivative.

        A:
            The damping of the second derivative.

    Returns:
        system:
            The model, with the state `jerk.x` of order 3 and its derivatives `jerk.x_d1` and `jerk.x_d2`.

    Examples:
        >>> from skcomplex.models.nonlinear import sprott_jerk
        >>> sprott_jerk(A=0.5).elements[0].A
        0.5
    """
    return System(elements=[Jerk('jerk', x=x, x_d1=velocity, x_d2=acceleration, A=A)])
