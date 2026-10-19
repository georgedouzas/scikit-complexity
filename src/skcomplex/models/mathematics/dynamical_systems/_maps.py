"""Dynamical systems models: chaotic flows and maps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.model import Element, System, Update
from ....core.runtime import register_model

LogisticMap = Element.define(
    'LogisticMap',
    states={'x': 0.2},
    parameters={'r': 3.7},
    laws=lambda s: [Update(s.x, s.r * s.x * (1 - s.x))],
)

HenonMap = Element.define(
    'HenonMap',
    states={'x': 0.0, 'y': 0.0},
    parameters={'a': 1.4, 'b': 0.3},
    laws=lambda s: [Update(s.x, 1 - s.a * s.x**2 + s.y), Update(s.y, s.b * s.x)],
)


@register_model(
    field='mathematics/dynamical_systems',
    tags=('chaos', 'discrete map', 'period doubling'),
    references=('May 1976', 'Satz 2022'),
    defaults={'until': 200, 'dt': 1},
)
def logistic_map(x: float = 0.2, *, r: float = 3.7) -> System:
    """Return the logistic map `x_{t+1} = r x_t (1 - x_t)`, which reaches chaos through period doubling as `r` grows.

    Args:
        x:
            The initial value, between 0 and 1.

        r:
            The growth rate, between 0 and 4.

    Returns:
        system:
            The model, with the state `map.x`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import logistic_map
        >>> logistic_map(r=3.2).elements[0].r
        3.2
    """
    return System(elements=[LogisticMap('map', x=x, r=r)])


@register_model(
    field='mathematics/dynamical_systems',
    tags=('chaos', 'discrete map', 'strange attractor'),
    references=('Hénon 1976', 'Datseris and Parlitz 2022'),
    defaults={'until': 1000, 'dt': 1},
)
def henon_map(x: float = 0.0, y: float = 0.0, *, a: float = 1.4, b: float = 0.3) -> System:
    """Return the Hénon map `x' = 1 - a x² + y`, `y' = b x`, with a fractal strange attractor for its defaults.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        a:
            The nonlinearity.

        b:
            The contraction.

    Returns:
        system:
            The model, with the states `map.x` and `map.y`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import henon_map
        >>> henon_map(a=1.2).elements[0].a
        1.2
    """
    return System(elements=[HenonMap('map', x=x, y=y, a=a, b=b)])
