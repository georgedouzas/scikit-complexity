"""Oscillating and autocatalytic reactions in well-mixed reactors."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element
from ....core.interactions import ReactionInteraction, RelaxationInteraction
from ....core.laws import BaseLaw, RateLaw
from ....core.system import System
from ....engine.solvers import ScipyODESolver
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model


def _get_feed(s: SimpleNamespace) -> Expression:
    """Return the propensity `a` of `A -> U`, the constant supply of the first reactant."""
    return s.products[0].a


def _get_conversion(s: SimpleNamespace) -> Expression:
    """Return the propensity `b u` of `B + U -> V + D`."""
    reactor = s.reactants[0]
    return reactor.b * reactor.u


def _get_decay(s: SimpleNamespace) -> Expression:
    """Return the propensity `k b` of the decay `B -> C` of the autocatalyst."""
    reactor = s.reactants[0]
    return reactor.k * reactor.b


def _get_residence(s: SimpleNamespace) -> Expression:
    """Return the residence time `1 / F` of the continuously fed reactor."""
    return 1 / s.target.F


class BrusselatorReactionElement(Element):
    """Well-mixed reactor of the Brusselator, the concentrations `u` and `v` of its two intermediates.

    The four reactions of `brusselator_ode` change the concentrations and read the supplies `a` and `b` of the two
    reactants from the reactor.

    Args:
        label:
            The label of the reactor.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        u:
            The initial concentration of the autocatalytic intermediate.

        v:
            The initial concentration of the second intermediate.

        a:
            The constant supply of the first reactant.

        b:
            The constant supply of the second reactant.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import BrusselatorReactionElement
        >>> BrusselatorReactionElement('reactor').b
        3.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        v: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        a: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        b: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 3.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.v = v
        self.a = a
        self.b = b


class OregonatorElement(Element):
    """Reactor of the scaled Oregonator, the scaled concentrations of bromous acid, bromide, and the oxidized catalyst.

    The scaled concentrations follow `epsilon dx/dt = q y - x y + x (1 - x)`, `delta dy/dt = -q y - x y + f z`, and
    `dz/dt = x - z`. The time scales `epsilon` and `delta` divide the changes of `x` and `y` by every reaction, which
    the coefficients of mass action cannot hold, so the reactor keeps its own rate laws.

    Args:
        label:
            The label of the reactor.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

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

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import OregonatorElement
        >>> OregonatorElement('reactor').epsilon
        0.04
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        y: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        z: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.04,
        delta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.0004,
        q: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0008,
        f: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.epsilon = epsilon
        self.delta = delta
        self.q = q
        self.f = f

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the scaled rate equations of bromous acid, bromide, and the oxidized catalyst.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The rate laws of the three concentrations.
        """
        return [
            RateLaw(s.x, (s.q * s.y - s.x * s.y + s.x * (1 - s.x)) / s.epsilon),
            RateLaw(s.y, (-s.q * s.y - s.x * s.y + s.f * s.z) / s.delta),
            RateLaw(s.z, s.x - s.z),
        ]


class CubicAutocatalysisElement(Element):
    """Continuously fed reactor of cubic autocatalysis, the concentrations of the reactant `a` and the autocatalyst `b`.

    The reactions and the flows of `cubic_autocatalysis` change the concentrations and read the feed rate `F` and the
    decay rate `k` from the reactor.

    Args:
        label:
            The label of the reactor.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        a:
            The initial concentration of the reactant.

        b:
            The initial concentration of the autocatalyst.

        F:
            The feed rate, the inverse residence time.

        k:
            The decay rate of the autocatalyst.

    Examples:
        >>> from skcomplex.models.chemistry.kinetics import CubicAutocatalysisElement
        >>> CubicAutocatalysisElement('reactor').F
        0.04
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        a: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        b: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        F: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.04,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.a = a
        self.b = b
        self.F = F
        self.k = k


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
    concentration_u, concentration_v = ('reactor', 'u'), ('reactor', 'v')
    return System(
        elements=[BrusselatorReactionElement('reactor', u=u, v=v, a=a, b=b)],
        interactions=[
            ReactionInteraction('feed', products={concentration_u: 1}, propensity=_get_feed),
            ReactionInteraction(
                'conversion',
                reactants={concentration_u: 1},
                products={concentration_v: 1},
                propensity=_get_conversion,
            ),
            ReactionInteraction(
                'autocatalysis',
                reactants={concentration_u: 2, concentration_v: 1},
                products={concentration_u: 3},
            ),
            ReactionInteraction('decay', reactants={concentration_u: 1}),
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
    return System(elements=[OregonatorElement('reactor', x=x, y=y, z=z, epsilon=epsilon, delta=delta, q=q, f=f)])


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
    reactant, autocatalyst = ('reactor', 'a'), ('reactor', 'b')
    return System(
        elements=[CubicAutocatalysisElement('reactor', a=a, b=b, F=F, k=k)],
        interactions=[
            ReactionInteraction('autocatalysis', reactants={reactant: 1, autocatalyst: 2}, products={autocatalyst: 3}),
            ReactionInteraction('decay', reactants={autocatalyst: 1}, propensity=_get_decay),
            RelaxationInteraction('feed', target='reactor', state='a', goal=1.0, time=_get_residence),
            RelaxationInteraction('outflow', target='reactor', state='b', goal=0.0, time=_get_residence),
        ],
    )
