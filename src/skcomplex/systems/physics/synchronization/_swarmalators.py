"""Swarmalators.

Oscillators swarm, and their phases and positions couple to each other.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import cos, oo, sin, sqrt
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.laws.utils import neighbors
from ....core.components.spaces import BaseSpace, ContinuousSpace, RadiusNeighborhood
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, RK4Solver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.oscillators import SitedOscillatorElement


class SwarmalatorElement(SitedOscillatorElement):
    """Phase oscillator whose place `x` is a position.

    The position moves by the rates of interactions, and the phase wraps around the circle.

    Args:
        label:
            The label of the swarmalator or the group.

        theta:
            The initial phase.

        omega:
            The natural frequency.

        x:
            The initial position, one row per member or shared.

        size:
            The number of swarmalators, or `None` for one.

        space:
            The continuous space of the swarmalators, or `None` for none.

        values:
            Further values by name, or `None`.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import SwarmalatorElement
        >>> SwarmalatorElement('unit', x=[0.0, 1.0]).x
        [0.0, 1.0]
    """

    visual = VisualSpec('points', {'position': 'x', 'color_by': 'theta', 'radius': 0.03})


class SwarmalatorCouplingInteraction(Interaction):
    """Coupling of swarmalators through their separations `r` and their phase differences `dtheta`.

    Each swarmalator moves toward every other at the speed `(A + J cos(dtheta)) / N`, away from it at `B / (N |r|)`,
    and turns its phase toward the other's at the rate `K sin(dtheta) / (N |r|)`, as in O'Keeffe, Ceron, and Petersen.
    The sums run over the others within the cutoff, as fireflies that swarm or sperm cells that swim and beat together.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of swarmalators.

        cutoff:
            The distance beyond which pairs do not interact, large for the all-to-all system.

        A:
            The strength of the attraction.

        B:
            The strength of the repulsion.

        J:
            The influence of the phase similarity on the attraction.

        K:
            The phase coupling, synchronizing when positive.

        n_units:
            The number of swarmalators `N` that normalizes the coupling.

    Examples:
        >>> from skcomplex.systems.physics.synchronization import SwarmalatorCouplingInteraction
        >>> SwarmalatorCouplingInteraction('coupling', target='units', J=0.5).J
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        cutoff: Annotated[float, RealSet.open(0, oo)] = 1000000.0,
        A: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        B: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        J: float = 1.0,
        K: float = 0.0,
        n_units: int = 1,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.cutoff = cutoff
        self.A = A
        self.B = B
        self.J = J
        self.K = K
        self.n_units = n_units

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the moves and the phase turns of the swarmalators toward the others.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            unit, other = s.target, s.target.other
            separation = other.x - unit.x
            r = sqrt(separation.dot_product(separation))
            difference = other.theta - unit.theta
            over = RadiusNeighborhood(self.cutoff)
            move = neighbors(((s.A + s.J * cos(difference)) / r - s.B / r**2) * separation / self.n_units, over=over)
            turn = neighbors(s.K * sin(difference) / (self.n_units * r), over=over)
            self.laws_ = [RateLaw(unit.x.diff(s.t), move), RateLaw(unit.theta.diff(s.t), turn)]


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_units = n_units
        self.J = J
        self.K = K
        self.A = A
        self.B = B
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of swarmalators in the plane.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        units = SwarmalatorElement(
            'units',
            x=rng.uniform(-1.0, 1.0, (self.n_units, 2)),
            theta=rng.uniform(-np.pi, np.pi, self.n_units),
            omega=0.0,
            size=self.n_units,
            space=ContinuousSpace(n_dim=2),
        )
        coupling = SwarmalatorCouplingInteraction(
            'coupling',
            target='units',
            A=self.A,
            B=self.B,
            J=self.J,
            K=self.K,
            n_units=self.n_units,
        )
        self.system_ = System(self.label, elements=[units], interactions=[coupling], solver=RK4Solver(dt=0.1))
