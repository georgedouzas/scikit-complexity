"""Oscillating and autocatalytic reactions in well-mixed reactors."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace

from sage.all__sagemath_symbolics import Expression

from ....core.interactions import ReactionInteraction, RelaxationInteraction
from ....core.system import System
from ....engine.solvers import ScipyODESolver
from ..._catalog import register_model
from ..common.kinetics import BrusselatorReactionElement, CubicAutocatalysisInteraction
from ._cubic_autocatalysis import CubicAutocatalysisElement
from ._oregonator import OregonatorElement


def _get_residence(s: SimpleNamespace) -> Expression:
    """Return the residence time `1 / F` of the continuously fed reactor."""
    return 1 / s.target.F


def _get_x_scale(s: SimpleNamespace) -> Expression:
    """Return the stoichiometric coefficient `1 / epsilon` of the scaled bromous acid."""
    return 1 / s.target.epsilon


def _get_y_scale(s: SimpleNamespace) -> Expression:
    """Return the stoichiometric coefficient `1 / delta` of the scaled bromide."""
    return 1 / s.target.delta


def _get_y_yield(s: SimpleNamespace) -> Expression:
    """Return the stoichiometric coefficient `f / delta` of the scaled bromide made from the oxidized catalyst."""
    return s.target.f / s.target.delta


def _get_bromate_bromide(s: SimpleNamespace) -> Expression:
    """Return the propensity `q y` of the reaction of bromate with bromide."""
    return s.target.q * s.target.y


def _get_bromous_bromide(s: SimpleNamespace) -> Expression:
    """Return the propensity `x y` of the reaction of bromous acid with bromide."""
    return s.target.x * s.target.y


def _get_disproportionation(s: SimpleNamespace) -> Expression:
    """Return the propensity `x^2` of the disproportionation of bromous acid."""
    return s.target.x**2


@register_model(
    field='chemistry/kinetics',
    tags=('ode', 'oscillation', 'limit cycle', 'autocatalysis'),
    references=('Prigogine and Lefever 1968', 'Datseris and Parlitz 2022', 'Trzaska 2023'),
    defaults={'until': 50.0, 'dt': 0.1},
)
def brusselator_ode(u: float = 1.0, v: float = 1.0, *, a: float = 1.0, b: float = 3.0) -> System:
    """Return the well-mixed Brusselator, which oscillates on a limit cycle when `b > 1 + a**2`.

    The reactions `A -> U`, `B + U -> V + D`, `2U + V -> 3U`, and `U -> E` with the buffered reactants `A` and `B`
    give `du/dt = a - (b + 1) u + u^2 v` and `dv/dt = b u - u^2 v`, with the fixed point `(a, b / a)`.

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
    return System(
        elements=[BrusselatorReactionElement('reactor', u=u, v=v, a=a, b=b)],
        interactions=[
            ReactionInteraction('feed', target='reactor', products={'u': 1}, modifiers={'a': 1}),
            ReactionInteraction(
                'conversion',
                target='reactor',
                reactants={'u': 1},
                products={'v': 1},
                modifiers={'b': 1},
            ),
            CubicAutocatalysisInteraction('autocatalysis', target='reactor', substrate='v', autocatalyst='u'),
            ReactionInteraction('decay', target='reactor', reactants={'u': 1}),
        ],
    )


@register_model(
    field='chemistry/kinetics',
    tags=('ode', 'oscillation', 'relaxation oscillation', 'stiff', 'Belousov-Zhabotinsky'),
    references=('Field and Noyes 1974', 'Field, Körös, and Noyes 1972', 'Tyson 1985', 'Trzaska 2023'),
    defaults={'until': 30.0, 'dt': 0.05, 'solver': ScipyODESolver(method='LSODA', rtol=1e-6, atol=1e-9)},
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
    `epsilon dx/dt = q y - x y + x (1 - x)`, `delta dy/dt = -q y - x y + f z`, and `dz/dt = x - z`. The five reactions
    of the mechanism, bromate with bromide `Y -> X` at `q y`, bromous acid with bromide `X + Y -> 0` at `x y`, the
    autocatalysis `X -> 2X + Z` at `x`, the disproportionation `2X -> 0` at `x^2`, and the reduction of the catalyst
    `Z -> f Y` at `z`, change `x` and `y` by stoichiometric coefficients scaled by `1 / epsilon` and `1 / delta`. With
    the standard parameters the reaction settles on a relaxation limit cycle, and the small `epsilon` and `delta` make
    it stiff.

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
    return System(
        elements=[OregonatorElement('reactor', x=x, y=y, z=z, epsilon=epsilon, delta=delta, q=q, f=f)],
        interactions=[
            ReactionInteraction(
                'bromate_bromide',
                target='reactor',
                reactants={'y': _get_y_scale},
                products={'x': _get_x_scale},
                propensity=_get_bromate_bromide,
            ),
            ReactionInteraction(
                'bromous_bromide',
                target='reactor',
                reactants={'x': _get_x_scale, 'y': _get_y_scale},
                propensity=_get_bromous_bromide,
            ),
            ReactionInteraction(
                'autocatalysis',
                target='reactor',
                products={'x': _get_x_scale, 'z': 1},
                modifiers={'x': 1},
            ),
            ReactionInteraction(
                'disproportionation',
                target='reactor',
                reactants={'x': _get_x_scale},
                propensity=_get_disproportionation,
            ),
            ReactionInteraction(
                'catalyst_reduction',
                target='reactor',
                reactants={'z': 1},
                products={'y': _get_y_yield},
            ),
        ],
    )


@register_model(
    field='chemistry/kinetics',
    tags=('ode', 'autocatalysis', 'bistability', 'reactor'),
    references=('Gray and Scott 1983', 'Trzaska 2023'),
    defaults={'until': 500.0, 'dt': 1.0},
)
def cubic_autocatalysis(a: float = 0.5, b: float = 0.5, *, F: float = 0.04, k: float = 0.05) -> System:
    """Return cubic autocatalysis `A + 2B -> 3B` with the decay `B -> C` in a continuously fed reactor.

    The flow through the reactor relaxes the reactant to its feed concentration 1 and washes out the autocatalyst at
    the rate `F`, so the dimensionless concentrations follow `da/dt = -a b^2 + F (1 - a)` and
    `db/dt = a b^2 - (F + k) b`. The washout state `(1, 0)` is always stable, and a reacting state coexists with it
    when `F + k < sqrt(F) / 2`.

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
    return System(
        elements=[CubicAutocatalysisElement('reactor', a=a, b=b, F=F, k=k)],
        interactions=[
            CubicAutocatalysisInteraction('autocatalysis', target='reactor', substrate='a', autocatalyst='b'),
            ReactionInteraction('decay', target='reactor', reactants={'b': 1}, modifiers={'k': 1}),
            RelaxationInteraction('feed', target='reactor', state='a', goal=1.0, time=_get_residence),
            RelaxationInteraction('outflow', target='reactor', state='b', goal=0.0, time=_get_residence),
        ],
    )
