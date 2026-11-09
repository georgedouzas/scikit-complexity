"""Probabilistic and mean-field systems.

The module covers master equations, particle exchange, and Landau relaxation.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, Expression, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw, JumpLaw, LocalRuleLaw
from ....core.components.spaces import BallNeighborhood, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.specs import DeclaredVariableSpec, ParameterSpec, StateSpec, VisualSpec
from ....core.types import Array
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


class MicrostateElement(Element):
    """State of a discrete master equation on a ring.

    The state has a probability `P` and a potential `U`.

    Every step the probability moves to each neighbouring state `B` from a state `A` with the Glauber probability
    `w / (1 + exp(beta (U_B - U_A)))`, so `P_A(t + 1) = P_A + sum_B T_AB P_B - sum_B T_BA P_A`, and the stationary
    distribution is `exp(-beta U) / Z`.

    Args:
        label:
            The label of the state or the lattice group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        P:
            The initial probability.

        U:
            The potential.

        beta:
            The inverse temperature.

        w:
            The attempt probability of a move to each neighbour, at most 1/2.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import MicrostateElement
        >>> MicrostateElement('state').w
        0.5
    """

    law_classes = (LocalRuleLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        P: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        U: Annotated[float, StateSpec()] = 0.0,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        w: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label, visual=visual)
        self.P = P
        self.U = U
        self.beta = beta
        self.w = w

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the master equation step.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            LocalRuleLaw(
                s.P,
                BallNeighborhood(n_dim=1, center='inline'),
                MicrostateElement._master_step,
                reads=(s.P, s.U),
            ),
        ]

    @staticmethod
    def _glauber(source: Array, destination: Array, beta: float | Array, w: float | Array) -> Array:
        """Return the probability of a move between two states.

        The probability is `w / (1 + exp(beta (U_destination - U_source)))`.
        """
        return w / (1 + np.exp(beta * (destination - source)))

    @staticmethod
    def _master_step(*around: Array, beta: float | Array, w: float | Array) -> Array:
        """Return the probabilities after one step of the master equation.

        The step moves probability to the two neighbouring states.

        The neighbourhood columns are the probabilities and then the potentials of the left state, the state, and the
        right state.
        """
        left, center, right, energy_left, energy, energy_right = around
        gain = (
            MicrostateElement._glauber(energy_left, energy, beta, w) * left
            + MicrostateElement._glauber(energy_right, energy, beta, w) * right
        )
        loss = (
            MicrostateElement._glauber(energy, energy_left, beta, w)
            + MicrostateElement._glauber(energy, energy_right, beta, w)
        ) * center
        return center + gain - loss


class ZoneElement(Element):
    """Small zone of a two-phase apparatus.

    The zone exchanges particles with the rest of the apparatus.

    Every one of the `Na` particles enters the zone at the rate `k p` and leaves it at the rate `k (1 - p)`, where
    `p = dV / V` is the volume fraction of the zone, so the number in the zone is binomial with mean `Na p`.

    Args:
        label:
            The label of the zone or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        N:
            The initial number of particles in the zone.

        Na:
            The number of particles in the apparatus and the zone.

        p:
            The volume fraction of the zone.

        k:
            The rate of exchange.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ZoneElement
        >>> ZoneElement('zone').Na
        100.0
    """

    law_classes = (JumpLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        N: Annotated[float, StateSpec(domain=NN)] = 0.0,
        Na: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        p: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        k: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.N = N
        self.Na = Na
        self.p = p
        self.k = k

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the entries into and exits from the zone.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            JumpLaw(s.k * s.p * (s.Na - s.N), {s.N: 1}),
            JumpLaw(s.k * (1 - s.p) * s.N, {s.N: -1}),
        ]


class DiscreteMasterEquationSystem(BaseSystem):
    """The discrete-time master equation of states on a ring.

    The equation relaxes to the Boltzmann distribution.

    The potential is `U = depth cos(2 pi i / n_states)`, the probability starts on the state of highest potential, and
    moves to neighbouring states with Glauber probabilities, which satisfy detailed balance. The system has the
    probabilities `states.P` and potentials `states.U`.

    Args:
        n_states:
            The number of states.

        beta:
            The inverse temperature.

        depth:
            The amplitude of the potential.

        w:
            The attempt probability of a move to each neighbour, at most 1/2.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import DiscreteMasterEquationSystem
        >>> DiscreteMasterEquationSystem(10).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'physics/statistical_mechanics'
    topics = ('master equation', 'probability', 'detailed balance', 'lattice')
    references = ('Jensen 2023', 'van Kampen 2007')
    until = 500
    dt = 10

    def __init__(
        self,
        n_states: int = 50,
        *,
        beta: float = 1.0,
        depth: float = 2.0,
        w: float = 0.5,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_states = n_states
        self.beta = beta
        self.depth = depth
        self.w = w

    def _build_system(self) -> CompositeSystem:
        """Return the discrete-time master equation of states on a ring.

        The equation relaxes to the Boltzmann distribution.
        """
        potential = self.depth * np.cos(2 * np.pi * np.arange(self.n_states) / self.n_states)
        probability = np.zeros(self.n_states)
        probability[0] = 1.0
        space = LatticeSpace(self.n_states)
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement(
                    'states',
                    element=MicrostateElement,
                    space=space,
                    values={'P': probability, 'U': potential, 'beta': self.beta, 'w': self.w},
                    on='sites',
                ),
            ],
        )


class TwoPhaseZoneExchangeSystem(BaseSystem):
    """Independent zones of a two-phase apparatus.

    Each zone exchanges particles with the rest of the apparatus.

    The zones start empty and their numbers of particles settle on the binomial distribution of mean `Na p` and variance
    `Na p (1 - p)`. The system has the numbers `zones.N`.

    Args:
        n_zones:
            The number of independent zones.

        Na:
            The number of particles in the apparatus and a zone.

        p:
            The volume fraction of a zone.

        k:
            The rate of exchange.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import TwoPhaseZoneExchangeSystem
        >>> TwoPhaseZoneExchangeSystem(5).prepare().system_.elements[0].prepare().size_
        5
    """

    field = 'physics/statistical_mechanics'
    topics = ('jump process', 'binomial', 'fluctuations', 'two-phase flow')
    references = ('Barsky 2020',)
    until = 20.0
    dt = 0.5

    def __init__(
        self,
        n_zones: int = 100,
        *,
        Na: float = 100.0,
        p: float = 0.1,
        k: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_zones = n_zones
        self.Na = Na
        self.p = p
        self.k = k

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        return CompositeSystem(
            self.label,
            elements=[
                GroupElement(
                    'zones',
                    element=ZoneElement,
                    size=self.n_zones,
                    values={'Na': self.Na, 'p': self.p, 'k': self.k},
                ),
            ],
        )


class LandauRelaxationSystem(DynamicalSystem):
    """The relaxation of a Landau order parameter.

    The order parameter relaxes to `sqrt(a (Tc - T) / b)` below `Tc` and to 0 above it.

    The relaxation time `1 / (Gamma a |T - Tc|)` diverges at the critical temperature, which is critical slowing down.
    The system has the order parameter `magnet.phi`.

    Args:
        phi:
            The initial order parameter.

        T:
            The temperature.

        Tc:
            The critical temperature.

        a:
            The coefficient of the quadratic term per unit temperature.

        b:
            The coefficient of the quartic term.

        h:
            The external field.

        Gamma:
            The kinetic coefficient.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import LandauRelaxationSystem
        >>> LandauRelaxationSystem(T=2.0).prepare().system_.elements[0].get_value('T')
        2.0
    """

    field = 'physics/statistical_mechanics'
    topics = ('ode', 'mean field', 'phase transition', 'critical slowing down')
    references = ('Landau 1937', 'Gros 2015')
    until = 20.0
    dt = 0.1

    def __init__(
        self,
        phi: float = 0.1,
        *,
        T: float = 0.5,
        Tc: float = 1.0,
        a: float = 1.0,
        b: float = 1.0,
        h: float = 0.0,
        Gamma: float = 1.0,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.phi = phi
        self.T = T
        self.Tc = Tc
        self.a = a
        self.b = b
        self.h = h
        self.Gamma = Gamma

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'magnet'

    def _get_variables(self) -> dict[str, DeclaredVariableSpec]:
        """Return the states and the parameters with their specs."""
        return {
            'phi': DeclaredVariableSpec(self.phi, StateSpec()),
            'a': DeclaredVariableSpec(self.a, ParameterSpec(domain=RealSet.open(0, oo))),
            'b': DeclaredVariableSpec(self.b, ParameterSpec(domain=RealSet.open(0, oo))),
            'T': DeclaredVariableSpec(self.T, ParameterSpec(domain=RealSet.closed_open(0, oo))),
            'Tc': DeclaredVariableSpec(self.Tc, ParameterSpec(domain=RealSet.open(0, oo))),
            'h': DeclaredVariableSpec(self.h, ParameterSpec()),
            'Gamma': DeclaredVariableSpec(self.Gamma, ParameterSpec(domain=RealSet.open(0, oo))),
        }

    def _get_equations(self) -> Equations:
        """Return the relaxation down the gradient of the free energy."""
        return {
            'phi': lambda s: -s.target.Gamma * self._force(s),
        }

    def _force(self, s: SimpleNamespace) -> Expression:
        """Return `force` from the namespace of an equation."""
        return s.target.a * (s.target.T - s.target.Tc) * s.target.phi + s.target.b * s.target.phi**3 - s.target.h
