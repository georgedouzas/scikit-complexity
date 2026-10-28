"""Probabilistic and mean-field models: master equations, particle exchange, and Landau relaxation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement, LatticeElement
from ....core.laws import BaseLaw, JumpLaw, LocalRuleLaw, RateLaw
from ....core.spaces import BoxNeighborhood, LatticeSpace
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ....core.system import System
from ..._catalog import register_model

type Array = npt.NDArray[np.float64]


def _glauber(source: Array, destination: Array, beta: Any, w: Any) -> Array:  # noqa: ANN401
    """Return the probability `w / (1 + exp(beta (U_destination - U_source)))` of a move between two states."""
    return w / (1 + np.exp(beta * (destination - source)))


def _master_step(  # noqa: PLR0917
    left: Array,
    center: Array,
    right: Array,
    energy_left: Array,
    energy: Array,
    energy_right: Array,
    *,
    beta: Any,  # noqa: ANN401
    w: Any,  # noqa: ANN401
) -> Array:
    """Return the probabilities after one step of the master equation with moves to the two neighbouring states."""
    gain = _glauber(energy_left, energy, beta, w) * left + _glauber(energy_right, energy, beta, w) * right
    loss = (_glauber(energy, energy_left, beta, w) + _glauber(energy, energy_right, beta, w)) * center
    return center + gain - loss


class MicrostateElement(Element):
    """State of a discrete master equation on a ring, with a probability `P` and a potential `U`.

    Every step the probability moves to each neighbouring state `B` from a state `A` with the Glauber probability
    `w / (1 + exp(beta (U_B - U_A)))`, so `P_A(t + 1) = P_A + sum_B T_AB P_B - sum_B T_BA P_A`, and the stationary
    distribution is `exp(-beta U) / Z`.

    Args:
        label:
            The label of the state or the lattice group.

        P:
            The initial probability.

        U:
            The potential.

        beta:
            The inverse temperature.

        w:
            The attempt probability of a move to each neighbour, at most 1/2.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import MicrostateElement
        >>> MicrostateElement('state').w
        0.5
    """

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
        return [LocalRuleLaw(s.P, BoxNeighborhood(), _master_step, reads=(s.P, s.U))]


class ZoneElement(Element):
    """Small zone of a two-phase apparatus that exchanges particles with the rest of it.

    Every one of the `Na` particles enters the zone at the rate `k p` and leaves it at the rate `k (1 - p)`, where
    `p = dV / V` is the volume fraction of the zone, so the number in the zone is binomial with mean `Na p`.

    Args:
        label:
            The label of the zone or the group.

        N:
            The initial number of particles in the zone.

        Na:
            The number of particles in the apparatus and the zone.

        p:
            The volume fraction of the zone.

        k:
            The rate of exchange.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import ZoneElement
        >>> ZoneElement('zone').Na
        100.0
    """

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


class OrderParameterElement(Element):
    """Order parameter of Landau theory relaxing down its free energy, `dphi/dt = -Gamma dF/dphi`.

    The free energy is `F = a (T - Tc) phi^2 / 2 + b phi^4 / 4 - h phi`.

    Args:
        label:
            The label of the order parameter.

        phi:
            The initial order parameter.

        a:
            The coefficient of the quadratic term per unit temperature.

        b:
            The coefficient of the quartic term.

        T:
            The temperature.

        Tc:
            The critical temperature.

        h:
            The external field.

        Gamma:
            The kinetic coefficient.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import OrderParameterElement
        >>> OrderParameterElement('magnet').Tc
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        phi: Annotated[float, StateSpec()] = 0.1,
        a: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        b: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        T: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        Tc: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        h: Annotated[float, ParameterSpec()] = 0.0,
        Gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.phi = phi
        self.a = a
        self.b = b
        self.T = T
        self.Tc = Tc
        self.h = h
        self.Gamma = Gamma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the relaxation down the gradient of the free energy.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        force = s.a * (s.T - s.Tc) * s.phi + s.b * s.phi**3 - s.h
        return [RateLaw(s.phi, -s.Gamma * force)]


@register_model(
    field='physics/statistical_mechanics',
    tags=('master equation', 'probability', 'detailed balance', 'lattice'),
    references=('Jensen 2023', 'van Kampen 2007'),
    defaults={'until': 500, 'dt': 10},
)
def discrete_master_equation(
    n_states: int = 50,
    *,
    beta: float = 1.0,
    depth: float = 2.0,
    w: float = 0.5,
) -> System:
    """Return the discrete-time master equation of states on a ring, which relaxes to the Boltzmann distribution.

    The potential is `U = depth cos(2 pi i / n_states)`, the probability starts on the state of highest potential,
    and moves to neighbouring states with Glauber probabilities, which satisfy detailed balance.

    Args:
        n_states:
            The number of states.

        beta:
            The inverse temperature.

        depth:
            The amplitude of the potential.

        w:
            The attempt probability of a move to each neighbour, at most 1/2.

    Returns:
        system:
            The model, with the probabilities `states.P` and potentials `states.U`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import discrete_master_equation
        >>> discrete_master_equation(10).elements[0].check().prepare().size_
        10
    """
    potential = depth * np.cos(2 * np.pi * np.arange(n_states) / n_states)
    probability = np.zeros(n_states)
    probability[0] = 1.0
    space = LatticeSpace(n_states)
    return System(
        elements=[
            LatticeElement(
                'states',
                element=MicrostateElement,
                space=space,
                values={'P': probability, 'U': potential, 'beta': beta, 'w': w},
            ),
        ],
    )


@register_model(
    field='physics/statistical_mechanics',
    tags=('jump process', 'binomial', 'fluctuations', 'two-phase flow'),
    references=('Barsky 2020',),
    defaults={'until': 20.0, 'dt': 0.5},
)
def two_phase_zone_exchange(
    n_zones: int = 100,
    *,
    Na: float = 100.0,
    p: float = 0.1,
    k: float = 1.0,
) -> System:
    """Return independent zones of a two-phase apparatus, each exchanging particles with the rest of the apparatus.

    The zones start empty and their numbers of particles settle on the binomial distribution of mean `Na p` and
    variance `Na p (1 - p)`.

    Args:
        n_zones:
            The number of independent zones.

        Na:
            The number of particles in the apparatus and a zone.

        p:
            The volume fraction of a zone.

        k:
            The rate of exchange.

    Returns:
        system:
            The model, with the numbers `zones.N`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import two_phase_zone_exchange
        >>> two_phase_zone_exchange(5).elements[0].check().prepare().size_
        5
    """
    return System(
        elements=[GroupElement('zones', element=ZoneElement, size=n_zones, values={'Na': Na, 'p': p, 'k': k})],
    )


@register_model(
    field='physics/statistical_mechanics',
    tags=('ode', 'mean field', 'phase transition', 'critical slowing down'),
    references=('Landau 1937', 'Gros 2015'),
    defaults={'until': 20.0, 'dt': 0.1},
)
def landau_relaxation(
    phi: float = 0.1,
    *,
    T: float = 0.5,
    Tc: float = 1.0,
    a: float = 1.0,
    b: float = 1.0,
    h: float = 0.0,
    Gamma: float = 1.0,
) -> System:
    """Return the relaxation of a Landau order parameter, to `sqrt(a (Tc - T) / b)` below `Tc` and to 0 above it.

    The relaxation time `1 / (Gamma a |T - Tc|)` diverges at the critical temperature, which is critical slowing
    down.

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

    Returns:
        system:
            The model, with the order parameter `magnet.phi`.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import landau_relaxation
        >>> landau_relaxation(T=2.0).elements[0].T
        2.0
    """
    return System(elements=[OrderParameterElement('magnet', phi=phi, a=a, b=b, T=T, Tc=Tc, h=h, Gamma=Gamma)])
