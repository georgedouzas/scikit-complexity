"""Oscillating and autocatalytic reactions in well-mixed reactors."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.interactions import ReactionInteraction, RelaxationInteraction
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import ScipyODESolver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..common.kinetics import BrusselatorReactionElement, CubicAutocatalysisInteraction


class CubicAutocatalysisElement(Element):
    """Continuously fed reactor of cubic autocatalysis.

    It holds the concentrations of the reactant `a` and the autocatalyst `b`.

    The reactions and the flows of `CubicAutocatalysisSystem` change the concentrations and read the feed rate `F` and
    the decay rate `k` from the reactor.

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
        >>> from skcomplex.systems.chemistry.kinetics import CubicAutocatalysisElement
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


class OregonatorElement(Element):
    """Reactor of the scaled Oregonator.

    It holds the scaled concentrations of bromous acid, bromide, and the oxidized catalyst.

    The scaled concentrations follow `epsilon dx/dt = q y - x y + x (1 - x)`, `delta dy/dt = -q y - x y + f z`, and
    `dz/dt = x - z`. The reactions of `OregonatorSystem` change the concentrations, with the time scales `epsilon` and
    `delta` dividing the stoichiometric coefficients of `x` and `y`, and read the parameters from the reactor.

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
        >>> from skcomplex.systems.chemistry.kinetics import OregonatorElement
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


class BrusselatorODESystem(BaseSystem):
    """The well-mixed Brusselator.

    It oscillates on a limit cycle when `b > 1 + a**2`.

    The reactions `A -> U`, `B + U -> V + D`, `2U + V -> 3U`, and `U -> E` with the buffered reactants `A` and `B`
    give `du/dt = a - (b + 1) u + u^2 v` and `dv/dt = b u - u^2 v`, with the fixed point `(a, b / a)`. The
    concentrations are `reactor.u` and `reactor.v`.

    Args:
        u:
            The initial concentration of the autocatalytic intermediate.

        v:
            The initial concentration of the second intermediate.

        a:
            The constant supply of the first reactant.

        b:
            The constant supply of the second reactant.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import BrusselatorODESystem
        >>> BrusselatorODESystem(b=1.5).prepare().system_.elements[0].b
        1.5
    """

    field = 'chemistry/kinetics'
    topics = ('ode', 'oscillation', 'limit cycle', 'autocatalysis')
    references = ('Prigogine and Lefever 1968', 'Datseris and Parlitz 2022', 'Trzaska 2023')
    until = 50.0
    dt = 0.1

    def __init__(
        self,
        u: float = 1.0,
        v: float = 1.0,
        *,
        a: float = 1.0,
        b: float = 3.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.u = u
        self.v = v
        self.a = a
        self.b = b

    def _build_system(self) -> System:
        """Return the reactor and the four reactions of the Brusselator."""
        return System(
            self.label,
            elements=[BrusselatorReactionElement('reactor', u=self.u, v=self.v, a=self.a, b=self.b)],
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


class OregonatorSystem(BaseSystem):
    """The scaled Oregonator of the Belousov-Zhabotinsky reaction.

    It is the reduction of the FKN mechanism of the reaction.

    The scaled concentrations of bromous acid `x`, bromide `y`, and the oxidized catalyst `z` follow
    `epsilon dx/dt = q y - x y + x (1 - x)`, `delta dy/dt = -q y - x y + f z`, and `dz/dt = x - z`. The five reactions
    of the mechanism, bromate with bromide `Y -> X` at `q y`, bromous acid with bromide `X + Y -> 0` at `x y`, the
    autocatalysis `X -> 2X + Z` at `x`, the disproportionation `2X -> 0` at `x^2`, and the reduction of the catalyst
    `Z -> f Y` at `z`, change `x` and `y` by stoichiometric coefficients scaled by `1 / epsilon` and `1 / delta`. With
    the standard parameters the reaction settles on a relaxation limit cycle, and the small `epsilon` and `delta` make
    it stiff. The concentrations are `reactor.x`, `reactor.y`, and `reactor.z`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import OregonatorSystem
        >>> OregonatorSystem(f=0.8).prepare().system_.elements[0].f
        0.8
    """

    field = 'chemistry/kinetics'
    topics = ('ode', 'oscillation', 'relaxation oscillation', 'stiff', 'Belousov-Zhabotinsky')
    references = ('Field and Noyes 1974', 'Field, Körös, and Noyes 1972', 'Tyson 1985', 'Trzaska 2023')
    until = 30.0
    dt = 0.05
    solver = ScipyODESolver(method='LSODA', rtol=1e-6, atol=1e-9)

    def __init__(
        self,
        x: float = 0.1,
        y: float = 0.1,
        z: float = 0.1,
        *,
        epsilon: float = 0.04,
        delta: float = 4e-4,
        q: float = 8e-4,
        f: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.x = x
        self.y = y
        self.z = z
        self.epsilon = epsilon
        self.delta = delta
        self.q = q
        self.f = f

    @staticmethod
    def _get_x_scale(s: SimpleNamespace) -> Expression:
        """Return the stoichiometric coefficient `1 / epsilon` of the scaled bromous acid."""
        return 1 / s.target.epsilon

    @staticmethod
    def _get_y_scale(s: SimpleNamespace) -> Expression:
        """Return the stoichiometric coefficient `1 / delta` of the scaled bromide."""
        return 1 / s.target.delta

    @staticmethod
    def _get_y_yield(s: SimpleNamespace) -> Expression:
        """Return the stoichiometric coefficient `f / delta` of the scaled bromide.

        It is the bromide made from the oxidized catalyst.
        """
        return s.target.f / s.target.delta

    @staticmethod
    def _get_bromate_bromide(s: SimpleNamespace) -> Expression:
        """Return the propensity `q y` of the reaction of bromate with bromide."""
        return s.target.q * s.target.y

    @staticmethod
    def _get_bromous_bromide(s: SimpleNamespace) -> Expression:
        """Return the propensity `x y` of the reaction of bromous acid with bromide."""
        return s.target.x * s.target.y

    @staticmethod
    def _get_disproportionation(s: SimpleNamespace) -> Expression:
        """Return the propensity `x^2` of the disproportionation of bromous acid."""
        return s.target.x**2

    def _build_system(self) -> System:
        """Return the reactor and the five reactions of the Oregonator."""
        reactor = OregonatorElement(
            'reactor',
            x=self.x,
            y=self.y,
            z=self.z,
            epsilon=self.epsilon,
            delta=self.delta,
            q=self.q,
            f=self.f,
        )
        return System(
            self.label,
            elements=[reactor],
            interactions=[
                ReactionInteraction(
                    'bromate_bromide',
                    target='reactor',
                    reactants={'y': self._get_y_scale},
                    products={'x': self._get_x_scale},
                    propensity=self._get_bromate_bromide,
                ),
                ReactionInteraction(
                    'bromous_bromide',
                    target='reactor',
                    reactants={'x': self._get_x_scale, 'y': self._get_y_scale},
                    propensity=self._get_bromous_bromide,
                ),
                ReactionInteraction(
                    'autocatalysis',
                    target='reactor',
                    products={'x': self._get_x_scale, 'z': 1},
                    modifiers={'x': 1},
                ),
                ReactionInteraction(
                    'disproportionation',
                    target='reactor',
                    reactants={'x': self._get_x_scale},
                    propensity=self._get_disproportionation,
                ),
                ReactionInteraction(
                    'catalyst_reduction',
                    target='reactor',
                    reactants={'z': 1},
                    products={'y': self._get_y_yield},
                ),
            ],
        )


class CubicAutocatalysisSystem(BaseSystem):
    """Cubic autocatalysis `A + 2B -> 3B` with the decay `B -> C`.

    It runs in a continuously fed reactor.

    The flow through the reactor relaxes the reactant to its feed concentration 1 and washes out the autocatalyst at
    the rate `F`, so the dimensionless concentrations follow `da/dt = -a b^2 + F (1 - a)` and
    `db/dt = a b^2 - (F + k) b`. The washout state `(1, 0)` is always stable, and a reacting state coexists with it
    when `F + k < sqrt(F) / 2`. The concentrations are `reactor.a` and `reactor.b`.

    Args:
        a:
            The initial concentration of the reactant.

        b:
            The initial concentration of the autocatalyst.

        F:
            The feed rate, the inverse residence time.

        k:
            The decay rate of the autocatalyst.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.chemistry.kinetics import CubicAutocatalysisSystem
        >>> CubicAutocatalysisSystem(k=0.06).prepare().system_.elements[0].k
        0.06
    """

    field = 'chemistry/kinetics'
    topics = ('ode', 'autocatalysis', 'bistability', 'reactor')
    references = ('Gray and Scott 1983', 'Trzaska 2023')
    until = 500.0
    dt = 1.0

    def __init__(
        self,
        a: float = 0.5,
        b: float = 0.5,
        *,
        F: float = 0.04,
        k: float = 0.05,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.a = a
        self.b = b
        self.F = F
        self.k = k

    @staticmethod
    def _get_residence(s: SimpleNamespace) -> Expression:
        """Return the residence time `1 / F` of the continuously fed reactor."""
        return 1 / s.target.F

    def _build_system(self) -> System:
        """Return the reactor, the autocatalysis, the decay, and the flows.

        The flows run through the reactor.
        """
        return System(
            self.label,
            elements=[CubicAutocatalysisElement('reactor', a=self.a, b=self.b, F=self.F, k=self.k)],
            interactions=[
                CubicAutocatalysisInteraction('autocatalysis', target='reactor', substrate='a', autocatalyst='b'),
                ReactionInteraction('decay', target='reactor', reactants={'b': 1}, modifiers={'k': 1}),
                RelaxationInteraction('feed', target='reactor', state='a', goal=1.0, time=self._get_residence),
                RelaxationInteraction('outflow', target='reactor', state='b', goal=0.0, time=self._get_residence),
            ],
        )
