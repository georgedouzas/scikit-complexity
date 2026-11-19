"""Probabilistic and mean-field systems.

The module covers master equations, particle exchange, and Landau relaxation.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import JumpLaw, RateLaw, UpdateLaw
from ....core.components.spaces import BaseSpace, LatticeNeighborhood, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role

RING = LatticeNeighborhood(((-1,), (0,), (1,)))


class MicrostateElement(Element):
    """State of a discrete master equation on a ring, with the probability `P` and the potential `U`.

    Args:
        label:
            The label of the state or the ring of states.

        P:
            The initial probability.

        U:
            The potential.

        space:
            The ring whose sites hold one state each, or `None` for one state.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import MicrostateElement
        >>> MicrostateElement('state', U=1.5).U
        1.5
    """

    def __init__(
        self,
        label: str,
        *,
        P: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        U: object = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.P = P
        self.U = U


class GlauberMasterInteraction(Interaction):
    """Discrete master equation of states on a ring with Glauber moves between neighbours.

    Every step the probability moves to each neighbouring state `B` from a state `A` with the Glauber probability
    `w / (1 + exp(beta (U_B - U_A)))`, so `P_A(t + 1) = P_A + sum_B T_AB P_B - sum_B T_BA P_A`, and the stationary
    distribution is `exp(-beta U) / Z`. It is a molecule hopping between conformations, a walker in a landscape, or a
    queue whose length changes by one.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the ring of states.

        beta:
            The inverse temperature.

        w:
            The attempt probability of a move to each neighbour, at most 1/2.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import GlauberMasterInteraction
        >>> GlauberMasterInteraction('moves', target='states').w
        0.5
    """

    law_classes = (UpdateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        beta: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        w: Annotated[float, RealSet.closed_open(0, oo)] = 0.5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.beta = beta
        self.w = w

    @staticmethod
    def _get_glauber(source: Array, destination: Array, beta: float, w: float) -> Array:
        """Return the probability of a move between two states.

        The probability is `w / (1 + exp(beta (U_destination - U_source)))`.
        """
        return w / (1 + np.exp(beta * (destination - source)))

    def _get_master_step(self, *around: Array) -> Array:
        """Return the probabilities after one step of the master equation.

        The neighbourhood columns are the probabilities and then the potentials of the left state, the state, and the
        right state.
        """
        left, center, right, energy_left, energy, energy_right = around
        beta, w = self.beta, self.w
        gain = (
            self._get_glauber(energy_left, energy, beta, w) * left
            + self._get_glauber(energy_right, energy, beta, w) * right
        )
        loss = (
            self._get_glauber(energy, energy_left, beta, w) + self._get_glauber(energy, energy_right, beta, w)
        ) * center
        return center + gain - loss

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the master equation step of the probabilities of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context.target
            self.laws_ = [UpdateLaw(s.P, self._get_master_step, neighborhood=RING, reads=(s.P, s.U), label=self.label)]


class ZoneElement(Element):
    """Small zones of a two-phase apparatus, with the number `N` of particles in each.

    Args:
        label:
            The label of the zone or the group.

        N:
            The initial number of particles in the zone, or one number per zone.

        size:
            The number of zones, or `None` for one.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ZoneElement
        >>> ZoneElement('zone').N
        0.0
    """

    def __init__(self, label: str, *, N: object = 0.0, size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.N = N


class ZoneExchangeInteraction(Interaction):
    """Exchange of particles between a small zone and the rest of a two-phase apparatus.

    Every one of the `Na` particles enters the zone at the rate `k p` and leaves it at the rate `k (1 - p)`, where
    `p = dV / V` is the volume fraction of the zone, so the number in the zone is binomial with mean `Na p`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the zones, whose value `N` counts their particles.

        Na:
            The number of particles in the apparatus and the zone.

        p:
            The volume fraction of the zone.

        k:
            The rate of exchange.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import ZoneExchangeInteraction
        >>> ZoneExchangeInteraction('exchange', target='zones').Na
        100.0
    """

    law_classes = (JumpLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        Na: Annotated[float, RealSet.closed_open(0, oo)] = 100.0,
        p: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        k: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.Na = Na
        self.p = p
        self.k = k

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the entries into and exits from the zone.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, N = context, context.target.N
            self.laws_ = [
                JumpLaw(s.k * s.p * (s.Na - N), {N: 1}),
                JumpLaw(s.k * (1 - s.p) * N, {N: -1}),
            ]


class LandauRelaxationInteraction(Interaction):
    """Relaxation of an order parameter down the gradient of the Landau free energy.

    The order parameter follows `phi' = -Gamma (a (T - Tc) phi + b phi^3 - h)`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the magnet, whose value `phi` is the order parameter.

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

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import LandauRelaxationInteraction
        >>> LandauRelaxationInteraction('relaxation', target='magnet', T=2.0).T
        2.0
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        T: Annotated[float, RealSet.closed_open(0, oo)] = 0.5,
        Tc: Annotated[float, RealSet.open(0, oo)] = 1.0,
        a: Annotated[float, RealSet.open(0, oo)] = 1.0,
        b: Annotated[float, RealSet.open(0, oo)] = 1.0,
        h: float = 0.0,
        Gamma: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.T = T
        self.Tc = Tc
        self.a = a
        self.b = b
        self.h = h
        self.Gamma = Gamma

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the relaxation of the order parameter of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, phi = context, context.target.phi
            force = s.a * (s.T - s.Tc) * phi + s.b * phi**3 - s.h
            self.laws_ = [RateLaw(phi.diff(s.t), -s.Gamma * force)]


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import DiscreteMasterEquationSystem
        >>> DiscreteMasterEquationSystem(10).prepare().system_.elements[0].space.prepare().size_
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_states = n_states
        self.beta = beta
        self.depth = depth
        self.w = w

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the ring of states with all the probability on the state of highest potential.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        potential = self.depth * np.cos(2 * np.pi * np.arange(self.n_states) / self.n_states)
        probability = np.zeros(self.n_states)
        probability[0] = 1.0
        states = MicrostateElement('states', P=probability, U=potential, space=LatticeSpace(self.n_states))
        moves = GlauberMasterInteraction('moves', target='states', beta=self.beta, w=self.w)
        self.system_ = System(self.label, elements=[states], interactions=[moves])


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import TwoPhaseZoneExchangeSystem
        >>> TwoPhaseZoneExchangeSystem(5).prepare().system_.elements[0].size
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_zones = n_zones
        self.Na = Na
        self.p = p
        self.k = k

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the zones and their exchange of particles.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[ZoneElement('zones', size=self.n_zones)],
            interactions=[ZoneExchangeInteraction('exchange', target='zones', Na=self.Na, p=self.p, k=self.k)],
        )


class LandauRelaxationSystem(BaseSystem):
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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.statistical_mechanics import LandauRelaxationSystem
        >>> LandauRelaxationSystem(T=2.0).prepare().system_.interactions[0].T
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.phi = phi
        self.T = T
        self.Tc = Tc
        self.a = a
        self.b = b
        self.h = h
        self.Gamma = Gamma

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the magnet and the relaxation of its order parameter.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        relaxation = LandauRelaxationInteraction(
            'relaxation',
            target='magnet',
            T=self.T,
            Tc=self.Tc,
            a=self.a,
            b=self.b,
            h=self.h,
            Gamma=self.Gamma,
        )
        self.system_ = System(
            self.label,
            elements=[Element('magnet', values={'phi': self.phi})],
            interactions=[relaxation],
        )
