"""Dynamical systems systems: chaotic flows and singular growth."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.components.system import System
from ..._catalog import register_system
from ..common.flows import LorenzElement
from ._blow_up import BlowUpElement
from ._jerk import JerkElement
from ._rossler import RosslerElement


@register_system(
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
    """Return the Lorenz-63 system of convection, chaotic on a butterfly-shaped attractor for its standard parameters.

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
            The system, with the states `lorenz.x`, `lorenz.y`, and `lorenz.z`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import lorenz
        >>> lorenz(rho=20.0).elements[0].rho
        20.0
    """
    return System(elements=[LorenzElement('lorenz', x=x, y=y, z=z, sigma=sigma, rho=rho, beta=beta)])


@register_system(
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
    """Return the Rössler system, a chaotic flow with one folded band for its standard parameters.

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
            The system, with the states `rossler.x`, `rossler.y`, and `rossler.z`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import rossler
        >>> rossler(c=4.0).elements[0].c
        4.0
    """
    return System(elements=[RosslerElement('rossler', x=x, y=y, z=z, a=a, b=b, c=c)])


@register_system(
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
            The system, with the state `jerk.x` of order 3 and its derivatives `jerk.x_d1` and `jerk.x_d2`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import sprott_jerk
        >>> sprott_jerk(A=0.5).elements[0].A
        0.5
    """
    return System(elements=[JerkElement('jerk', x=x, x_d1=velocity, x_d2=acceleration, A=A)])


@register_system(
    field='mathematics/dynamical_systems',
    tags=('ode', 'singularity', 'hyperbolic growth'),
    references=('Forrest 2018',),
    defaults={'until': 0.9, 'dt': 0.01},
)
def finite_time_blow_up(x: float = 1.0, *, a: float = 1.0, p: float = 2.0) -> System:
    """Return hyperbolic growth `x' = a x^p`, which diverges at the finite time `x0^(1 - p) / (a (p - 1))` for p > 1.

    Args:
        x:
            The initial value, positive.

        a:
            The growth coefficient.

        p:
            The growth exponent.

    Returns:
        system:
            The system, with the state `growth.x`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import finite_time_blow_up
        >>> finite_time_blow_up(p=3.0).elements[0].p
        3.0
    """
    return System(elements=[BlowUpElement('growth', x=x, a=a, p=p)])
