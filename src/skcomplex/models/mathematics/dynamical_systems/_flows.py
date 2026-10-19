"""Dynamical systems models: chaotic flows and maps."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from sage.all__sagemath_symbolics import abs_symbolic

from ....core.base import state
from ....core.model import Element, Rate, System, derivative
from ....core.runtime import register_model

Lorenz = Element.define(
    'Lorenz',
    states={'x': 1.0, 'y': 1.0, 'z': 1.0},
    parameters={'sigma': 10.0, 'rho': 28.0, 'beta': 8 / 3},
    laws=lambda s: [
        Rate(s.x, s.sigma * (s.y - s.x)),
        Rate(s.y, s.x * (s.rho - s.z) - s.y),
        Rate(s.z, s.x * s.y - s.beta * s.z),
    ],
)

Rossler = Element.define(
    'Rossler',
    states={'x': 1.0, 'y': 1.0, 'z': 1.0},
    parameters={'a': 0.2, 'b': 0.2, 'c': 5.7},
    laws=lambda s: [
        Rate(s.x, -s.y - s.z),
        Rate(s.y, s.x + s.a * s.y),
        Rate(s.z, s.b + s.z * (s.x - s.c)),
    ],
)


@register_model(
    field='mathematics/dynamical_systems',
    tags=('chaos', 'ode', 'strange attractor'),
    references=('Lorenz 1963', 'Datseris and Parlitz 2022'),
    defaults={'until': 50.0, 'dt': 0.01},
)
def lorenz(
    x: float = 1.0,
    y: float = 1.0,
    z: float = 1.0,
    *,
    sigma: float = 10.0,
    rho: float = 28.0,
    beta: float = 8 / 3,
) -> System:
    """Return the Lorenz-63 model of convection, chaotic on a butterfly-shaped attractor for its standard parameters.

    Args:
        x:
            The initial convection intensity.

        y:
            The initial temperature difference between rising and falling currents.

        z:
            The initial deviation of the vertical temperature profile.

        sigma:
            The Prandtl number.

        rho:
            The Rayleigh number relative to its critical value.

        beta:
            The geometric factor.

    Returns:
        system:
            The model, with the states `lorenz.x`, `lorenz.y`, and `lorenz.z`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import lorenz
        >>> lorenz(rho=20.0).elements[0].rho
        20.0
    """
    return System(elements=[Lorenz('lorenz', x=x, y=y, z=z, sigma=sigma, rho=rho, beta=beta)])


@register_model(
    field='mathematics/dynamical_systems',
    tags=('chaos', 'ode', 'strange attractor'),
    references=('Rössler 1976',),
    defaults={'until': 200.0, 'dt': 0.05},
)
def rossler(
    x: float = 1.0,
    y: float = 1.0,
    z: float = 1.0,
    *,
    a: float = 0.2,
    b: float = 0.2,
    c: float = 5.7,
) -> System:
    """Return the Rössler model, a chaotic flow with one folded band for its standard parameters.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial third coordinate.

        a:
            The feedback of the second coordinate on itself.

        b:
            The constant drive of the third coordinate.

        c:
            The threshold of the third coordinate's growth.

    Returns:
        system:
            The model, with the states `rossler.x`, `rossler.y`, and `rossler.z`.

    Examples:
        >>> from skcomplex.models.mathematics.dynamical_systems import rossler
        >>> rossler(c=4.0).elements[0].c
        4.0
    """
    return System(elements=[Rossler('rossler', x=x, y=y, z=z, a=a, b=b, c=c)])


Jerk = Element.define(
    'Jerk',
    states={'x': state(default=0.0, order=3, initial=(0.0, 1.0))},
    parameters={'A': 0.6},
    laws=lambda s: [Rate(s.x, -s.A * derivative(s.x, 2) - derivative(s.x, 1) + abs_symbolic(s.x) - 1)],
)


@register_model(
    field='mathematics/dynamical_systems',
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
        >>> from skcomplex.models.mathematics.dynamical_systems import sprott_jerk
        >>> sprott_jerk(A=0.5).elements[0].A
        0.5
    """
    return System(elements=[Jerk('jerk', x=x, x_d1=velocity, x_d2=acceleration, A=A)])
