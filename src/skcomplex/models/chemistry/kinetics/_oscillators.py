"""Oscillating and autocatalytic reactions in well-mixed reactors."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.base import Domain, state
from ....core.components import Element
from ....core.laws import Rate
from ....core.runtime import register_model
from ....core.solvers import ScipyODE
from ....core.system import System

BrusselatorReaction = Element.define(
    'BrusselatorReaction',
    states={'u': state(default=1.0, domain=Domain.NONNEGATIVE), 'v': state(default=1.0, domain=Domain.NONNEGATIVE)},
    parameters={'a': 1.0, 'b': 3.0},
    laws=lambda s: [
        Rate(s.u, s.a - (s.b + 1) * s.u + s.u**2 * s.v),
        Rate(s.v, s.b * s.u - s.u**2 * s.v),
    ],
)

Oregonator = Element.define(
    'Oregonator',
    states={
        'x': state(default=0.1, domain=Domain.NONNEGATIVE),
        'y': state(default=0.1, domain=Domain.NONNEGATIVE),
        'z': state(default=0.1, domain=Domain.NONNEGATIVE),
    },
    parameters={'epsilon': 0.04, 'delta': 4e-4, 'q': 8e-4, 'f': 1.0},
    laws=lambda s: [
        Rate(s.x, (s.q * s.y - s.x * s.y + s.x * (1 - s.x)) / s.epsilon),
        Rate(s.y, (-s.q * s.y - s.x * s.y + s.f * s.z) / s.delta),
        Rate(s.z, s.x - s.z),
    ],
)

CubicAutocatalysis = Element.define(
    'CubicAutocatalysis',
    states={'a': state(default=1.0, domain=Domain.NONNEGATIVE), 'b': state(default=0.0, domain=Domain.NONNEGATIVE)},
    parameters={'F': 0.04, 'k': 0.05},
    laws=lambda s: [
        Rate(s.a, -s.a * s.b**2 + s.F * (1 - s.a)),
        Rate(s.b, s.a * s.b**2 - (s.F + s.k) * s.b),
    ],
)


@register_model(
    field='chemistry/kinetics',
    tags=('ode', 'oscillation', 'limit cycle', 'autocatalysis'),
    references=('Prigogine and Lefever 1968', 'Datseris and Parlitz 2022', 'Trzaska 2023'),
    defaults={'until': 50.0, 'dt': 0.1},
)
def brusselator_ode(u: float = 1.0, v: float = 1.0, *, a: float = 1.0, b: float = 3.0) -> System:
    """Return the well-mixed Brusselator, which oscillates on a limit cycle when `b > 1 + a**2`.

    The concentrations follow `du/dt = a - (b + 1) u + u^2 v` and `dv/dt = b u - u^2 v`, with the fixed point
    `(a, b / a)`.

    Args:
        u:
            The initial concentration of the autocatalytic intermediate.

        v:
            The initial concentration of the second intermediate.

        a:
            The constant supply of the first reactant.

        b:
            The constant supply of the second reactant.

    Returns:
        system:
            The model, with the concentrations `reactor.u` and `reactor.v`.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import brusselator_ode
        >>> brusselator_ode(b=1.5).elements[0].b
        1.5
    """
    return System(components=[BrusselatorReaction('reactor', u=u, v=v, a=a, b=b)])


@register_model(
    field='chemistry/kinetics',
    tags=('ode', 'oscillation', 'relaxation oscillation', 'stiff', 'Belousov-Zhabotinsky'),
    references=('Field and Noyes 1974', 'Field, Körös, and Noyes 1972', 'Tyson 1985', 'Trzaska 2023'),
    defaults={'until': 30.0, 'dt': 0.05, 'solver': ScipyODE(method='LSODA', rtol=1e-6, atol=1e-9)},
)
def oregonator(
    x: float = 0.1,
    y: float = 0.1,
    z: float = 0.1,
    *,
    epsilon: float = 0.04,
    delta: float = 4e-4,
    q: float = 8e-4,
    f: float = 1.0,
) -> System:
    """Return the scaled Oregonator, the reduction of the FKN mechanism of the Belousov-Zhabotinsky reaction.

    The scaled concentrations of bromous acid `x`, bromide `y`, and the oxidized catalyst `z` follow
    `epsilon dx/dt = q y - x y + x (1 - x)`, `delta dy/dt = -q y - x y + f z`, and `dz/dt = x - z`. With the standard
    parameters the reaction settles on a relaxation limit cycle, and the small `epsilon` and `delta` make it stiff.

    Args:
        x:
            The initial scaled concentration of bromous acid.

        y:
            The initial scaled concentration of bromide.

        z:
            The initial scaled concentration of the oxidized catalyst.

        epsilon:
            The ratio of the time scales of bromous acid and the catalyst.

        delta:
            The ratio of the time scales of bromide and the catalyst.

        q:
            The ratio of rate constants that sets the lowest bromous acid concentration.

        f:
            The stoichiometric factor of bromide production from the catalyst.

    Returns:
        system:
            The model, with the concentrations `reactor.x`, `reactor.y`, and `reactor.z`.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import oregonator
        >>> oregonator(f=0.8).elements[0].f
        0.8
    """
    return System(components=[Oregonator('reactor', x=x, y=y, z=z, epsilon=epsilon, delta=delta, q=q, f=f)])


@register_model(
    field='chemistry/kinetics',
    tags=('ode', 'autocatalysis', 'bistability', 'reactor'),
    references=('Gray and Scott 1983', 'Trzaska 2023'),
    defaults={'until': 500.0, 'dt': 1.0},
)
def cubic_autocatalysis(a: float = 0.5, b: float = 0.5, *, F: float = 0.04, k: float = 0.05) -> System:
    """Return cubic autocatalysis `A + 2B -> 3B` with the decay `B -> C` in a continuously fed reactor.

    The dimensionless concentrations follow `da/dt = -a b^2 + F (1 - a)` and `db/dt = a b^2 - (F + k) b`. The
    washout state `(1, 0)` is always stable, and a reacting state coexists with it when `F + k < sqrt(F) / 2`.

    Args:
        a:
            The initial concentration of the reactant.

        b:
            The initial concentration of the autocatalyst.

        F:
            The feed rate, the inverse residence time.

        k:
            The decay rate of the autocatalyst.

    Returns:
        system:
            The model, with the concentrations `reactor.a` and `reactor.b`.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import cubic_autocatalysis
        >>> cubic_autocatalysis(k=0.06).elements[0].k
        0.06
    """
    return System(components=[CubicAutocatalysis('reactor', a=a, b=b, F=F, k=k)])
