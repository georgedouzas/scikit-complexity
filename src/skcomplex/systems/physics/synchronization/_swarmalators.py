"""Swarmalators.

Oscillators swarm, and their phases and positions couple to each other.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import cos, oo, sin, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import GroupElement
from ....core.components.interactions import PairInteraction
from ....core.components.spaces import ContinuousSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.components.types import Target
from ....core.engine.solvers import RK4Solver
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ..common.oscillators import SitedOscillatorElement


class SwarmalatorElement(SitedOscillatorElement):
    """Phase oscillator whose place `x` is a position.

    The position moves by the rates of interactions.

    Args:
        label:
            The label of the swarmalator or the group.

        visual:
            How the swarmalator is drawn, or `None` for `default_visual_`.

        theta:
            The initial phase.

        omega:
            The natural frequency.

        x:
            The initial position. A number applies to every component.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import SwarmalatorElement
        >>> SwarmalatorElement('unit', x=[0.0, 1.0]).x
        [0.0, 1.0]
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        theta: float = 0.0,
        omega: float = 1.0,
        x: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual, theta=theta, omega=omega, x=x)


class SwarmalatorCouplingInteraction(PairInteraction):
    """Coupling of two swarmalators.

    The coupling acts through their separation `r` and their phase difference `dtheta`.

    Each swarmalator moves toward the other at the speed `(A + J cos(dtheta)) / N`, away from it at `B / (N |r|)`, and
    turns its phase toward the other's at the rate `K sin(dtheta) / (N |r|)`, as in O'Keeffe, Ceron, and Petersen. It
    is a pair coupling of the positions and the phases jointly.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the group of the first members.

        second:
            The path of the group of the second members.

        cutoff:
            The distance beyond which pairs do not interact, large for the all-to-all system.

        position:
            The name of the vector state that holds the positions of the members.

        A:
            The strength of the attraction.

        B:
            The strength of the repulsion.

        J:
            The influence of the phase similarity on the attraction.

        K:
            The phase coupling, synchronizing when positive.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        n_units:
            The number of swarmalators `N` that normalizes the coupling.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import SwarmalatorCouplingInteraction
        >>> SwarmalatorCouplingInteraction('coupling', first='units', second='units', J=0.5).J
        0.5
    """

    def __init__(
        self,
        label: str,
        first: Annotated[str, RoleSpec()],
        second: Annotated[str, RoleSpec()],
        cutoff: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1000000.0,
        position: str = 'x',
        *,
        A: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        B: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        J: Annotated[float, ParameterSpec()] = 1.0,
        K: Annotated[float, ParameterSpec()] = 0.0,
        period: float | None = None,
        n_units: int = 1,
    ) -> None:
        super().__init__(
            label,
            first,
            second,
            cutoff=cutoff,
            position=position,
            write=(position, 'theta'),
            function=self._get_terms,
            period=period,
        )
        self.A = A
        self.B = B
        self.J = J
        self.K = K
        self.n_units = n_units

    def _get_terms(self, s: SimpleNamespace) -> tuple[Target, Target]:
        """Return the move and phase turn of the first member of a pair toward the second."""
        separation = getattr(s.second, self.position) - getattr(s.first, self.position)
        r = sqrt(separation.dot_product(separation))
        difference = s.second.theta - s.first.theta
        move = ((s.A + s.J * cos(difference)) / r - s.B / r**2) * separation / self.n_units
        return move, s.K * sin(difference) / (self.n_units * r)


class SwarmalatorsSystem(BaseSystem):
    """Swarmalators.

    The phases of identical oscillators set their attraction and their positions their coupling.

    Every swarmalator `i` moves by `(1 / N) sum_j [(x_j - x_i) / |x_j - x_i| (A + J cos(theta_j - theta_i)) - B (x_j -
    x_i) / |x_j - x_i|^2]` and turns by `(K / N) sum_j sin(theta_j - theta_i) / |x_j - x_i|`. With `J > 0` like phases
    attract. With `K = 0` and `J = 1` the swarm settles into a static phase wave, a disk whose phase winds once around
    its center, so one of the order parameters `S± = |mean(exp(i (phi ± theta)))|` of the spatial angle `phi` and the
    phase approaches 1. With `K > 0` the phases synchronize into a uniform disk, and with `K < 0` and small `J` they
    stay asynchronous. The system has the positions `units.x` and the phases `units.theta`.

    Args:
        n_units:
            The number of swarmalators `N`.

        J:
            The influence of the phase similarity on the attraction.

        K:
            The phase coupling.

        A:
            The strength of the attraction.

        B:
            The strength of the repulsion.

        random_state:
            The seed or generator of the initial positions in the square of side 2 and of the initial phases.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import SwarmalatorsSystem
        >>> SwarmalatorsSystem(10, K=-0.1).prepare().system_.interactions[0].K
        -0.1
    """

    field = 'physics/synchronization'
    topics = ('swarmalators', 'synchronization', 'swarming', 'pair interaction', 'ode')
    references = ("O'Keeffe, Ceron, and Petersen 2017", 'te Vrugt and Wittkowski 2024')
    until = 100.0
    dt = 1.0
    solver = RK4Solver(dt=0.1)

    def __init__(
        self,
        n_units: int = 100,
        *,
        J: float = 1.0,
        K: float = 0.0,
        A: float = 1.0,
        B: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_units = n_units
        self.J = J
        self.K = K
        self.A = A
        self.B = B
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system, built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        units = GroupElement(
            'units',
            element=SwarmalatorElement,
            size=self.n_units,
            values={
                'x': rng.uniform(-1.0, 1.0, (self.n_units, 2)),
                'theta': rng.uniform(-np.pi, np.pi, self.n_units),
                'omega': 0.0,
            },
            visual=VisualSpec('points', {'position': 'x', 'color_by': 'theta', 'radius': 0.03}),
        )
        coupling = SwarmalatorCouplingInteraction(
            'coupling',
            first='units',
            second='units',
            A=self.A,
            B=self.B,
            J=self.J,
            K=self.K,
            n_units=self.n_units,
        )
        return CompositeSystem(self.label, elements=[units], interactions=[coupling], space=ContinuousSpace(n_dim=2))
