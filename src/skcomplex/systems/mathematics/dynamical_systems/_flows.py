"""Dynamical systems systems: chaotic and conservative flows, delayed feedback, and singular growth."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ....core.components.system import System
from ..._catalog import register_system
from ..common.flows import LorenzElement
from ._blow_up import BlowUpElement
from ._hyperchaotic_rossler import HyperchaoticRosslerElement
from ._ikeda import IkedaElement
from ._jerk import JerkElement
from ._langford import LangfordElement
from ._mixed_flow import MixedFlowElement
from ._nose_hoover import NoseHooverElement
from ._rossler import RosslerElement
from ._thomas import ThomasElement


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


@register_system(
    field='mathematics/dynamical_systems',
    tags=('chaos', 'ode', 'hyperchaos', 'strange attractor'),
    references=('Rössler 1979', 'Dilão 2023'),
    defaults={'until': 300.0, 'dt': 0.05},
)
def hyperchaotic_rossler(
    x: float = -10.0,
    y: float = -6.0,
    z: float = 0.0,
    w: float = 10.0,
    *,
    a: float = 0.25,
    b: float = 3.0,
    c: float = 0.05,
    d: float = 0.5,
) -> System:
    """Return Rössler's four-dimensional flow, hyperchaotic with two positive Lyapunov exponents for its defaults.

    The flow `x' = -(y + z)`, `y' = x + a y + w`, `z' = b + x z`, `w' = c w - d z` stretches the attractor in two
    directions at once, with the exponents near 0.11 and 0.02 for the parameters of Dilão (2023, Exercise 4.18).

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial third coordinate.

        w:
            The initial fourth coordinate.

        a:
            The feedback of the second coordinate on itself.

        b:
            The constant drive of the third coordinate.

        c:
            The growth of the fourth coordinate.

        d:
            The damping of the fourth coordinate by the third.

    Returns:
        system:
            The system, with the states `rossler.x`, `rossler.y`, `rossler.z`, and `rossler.w`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import hyperchaotic_rossler
        >>> hyperchaotic_rossler(c=0.1).elements[0].c
        0.1
    """
    return System(elements=[HyperchaoticRosslerElement('rossler', x=x, y=y, z=z, w=w, a=a, b=b, c=c, d=d)])


@register_system(
    field='mathematics/dynamical_systems',
    tags=('chaos', 'ode', 'conservative', 'thermostat', 'kam'),
    references=('Hoover 1995', 'Datseris and Parlitz 2022'),
    defaults={'until': 200.0, 'dt': 0.05},
)
def nose_hoover(x: float = 0.0, y: float = 0.1, z: float = 0.0) -> System:
    """Return the Nosé-Hoover oscillator `x' = y`, `y' = y z - x`, `z' = 1 - y²`, with tori next to a chaotic sea.

    The flow conserves volume on average and its state space is mixed. From `(0, 1.549934227822898, 0)` the orbit is
    periodic, from `(0, 1, 0)` it winds on a torus around that orbit, and from the default `(0, 0.1, 0)` it is chaotic
    (Datseris and Parlitz 2022, Exercise 2.11). Since `z' = 1 - y²`, every bounded orbit has `<y²> = 1`.

    Args:
        x:
            The initial position.

        y:
            The initial momentum.

        z:
            The initial friction of the thermostat.

    Returns:
        system:
            The system, with the states `oscillator.x`, `oscillator.y`, and `oscillator.z`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import nose_hoover
        >>> nose_hoover(y=1.0).elements[0].y
        1.0
    """
    return System(elements=[NoseHooverElement('oscillator', x=x, y=y, z=z)])


@register_system(
    field='mathematics/dynamical_systems',
    tags=('chaos', 'ode', 'conservative', 'strange attractor', 'torus'),
    references=('Sprott 2014', 'Datseris and Parlitz 2022'),
    defaults={'until': 200.0, 'dt': 0.05},
)
def sprott_mixed_flow(x: float = 2.0, y: float = 0.0, z: float = 0.0) -> System:
    """Return Sprott's flow with a conservative torus and a dissipative chaotic attractor side by side.

    The flow is `x' = y + 2xy + xz`, `y' = 1 - 2x² + yz`, `z' = x - x² - y²`, whose volumes change at the rate
    `2(y + z)`. From `(1, 0, 0)` the orbit winds on a 2-torus where this rate averages to zero, and from the default
    `(2, 0, 0)` it falls onto a chaotic attractor where it averages below zero (Datseris and Parlitz 2022, Exercises
    3.17 and 8.16).

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial third coordinate.

    Returns:
        system:
            The system, with the states `flow.x`, `flow.y`, and `flow.z`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import sprott_mixed_flow
        >>> sprott_mixed_flow(1.0).elements[0].x
        1.0
    """
    return System(elements=[MixedFlowElement('flow', x=x, y=y, z=z)])


@register_system(
    field='mathematics/dynamical_systems',
    tags=('chaos', 'ode', 'torus', 'strange attractor'),
    references=('Langford 1984', 'Dilão 2023'),
    defaults={'until': 200.0, 'dt': 0.05},
)
def langford(
    x: float = 0.1,
    y: float = 0.0,
    z: float = 0.0,
    *,
    a: float = 0.95,
    b: float = 0.7,
    c: float = 0.6,
    d: float = 3.5,
    e: float = 0.25,
    f: float = 0.1,
) -> System:
    """Return the Langford flow, a rotation whose torus breaks into a strange attractor for the defaults of Dilão.

    The flow is `x' = (z - b) x - d y`, `y' = d x + (z - b) y`, and
    `z' = c + a z - z³ / 3 - (x² + y²)(1 + e z) + f z x³`, with the reference parameters of Dilão (2023, Exercise
    3.22). Without `f` the flow is symmetric under rotations about the `z` axis.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial height.

        a:
            The linear growth of the height.

        b:
            The height above which the rotation grows.

        c:
            The constant drive of the height.

        d:
            The angular frequency of the rotation.

        e:
            The coupling of the radius and the height.

        f:
            The term that breaks the rotational symmetry.

    Returns:
        system:
            The system, with the states `langford.x`, `langford.y`, and `langford.z`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import langford
        >>> langford(f=0.0).elements[0].f
        0.0
    """
    return System(elements=[LangfordElement('langford', x=x, y=y, z=z, a=a, b=b, c=c, d=d, e=e, f=f)])


@register_system(
    field='mathematics/dynamical_systems',
    tags=('chaos', 'ode', 'cyclic symmetry', 'strange attractor'),
    references=('Thomas 1999', 'Dilão 2023'),
    defaults={'until': 500.0, 'dt': 0.1},
)
def thomas(x: float = 0.1, y: float = 0.0, z: float = 0.0, *, a: float = 0.18) -> System:
    """Return the Thomas flow `x' = -a x + sin y` with its cyclic permutations, chaotic for `a = 0.18`.

    The flow is symmetric under the cyclic permutation of its coordinates. As the damping `a` falls the fixed points
    give way to limit cycles and then to chaos, and at `a = 0` the orbits wander through a lattice of cells, Thomas's
    labyrinth chaos (Dilão 2023, Exercise 3.23).

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial third coordinate.

        a:
            The damping of every coordinate.

    Returns:
        system:
            The system, with the states `thomas.x`, `thomas.y`, and `thomas.z`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import thomas
        >>> thomas(a=0.2).elements[0].a
        0.2
    """
    return System(elements=[ThomasElement('thomas', x=x, y=y, z=z, a=a)])


@register_system(
    field='mathematics/dynamical_systems',
    tags=('delay', 'dde', 'hopf bifurcation', 'oscillation'),
    references=('Ikeda and Matsumoto 1987', 'Dilão 2023'),
    defaults={'until': 100.0, 'dt': 0.1},
)
def ikeda(x: float = 2.0, *, a: float = 1.0, tau: float = 2.0) -> System:
    """Return the Ikeda delay equation `x' = a sin x(t - tau)` in the form of Dilão (2023, Exercise 6.4).

    The rest points `x = k π` alternate in stability for `a > 0`: those of odd `k` attract while `a tau < π / 2`, and
    lose stability to an oscillation in a Hopf bifurcation at `a tau = π / 2`. The history before the start is the
    initial value.

    Args:
        x:
            The initial value, also its history.

        a:
            The gain of the delayed feedback.

        tau:
            The delay.

    Returns:
        system:
            The system, with the state `ikeda.x`.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import ikeda
        >>> ikeda(tau=1.0).elements[0].tau
        1.0
    """
    return System(elements=[IkedaElement('ikeda', x=x, a=a, tau=tau)])
