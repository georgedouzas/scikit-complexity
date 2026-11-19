"""Active polymers.

The polymers are chains of self-propelled beads bonded by springs.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import cos, oo, sin, sqrt
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element, Period
from ....core.components.interactions import Interaction
from ....core.components.laws import NoiseLaw, RateLaw
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, EulerMaruyamaSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role


class ActiveBeadElement(Element):
    """Beads of a chain with the coordinates `x` and `y` and the orientations `theta`.

    Args:
        label:
            The label of the chain.

        x:
            The initial first coordinates, one per bead or shared.

        y:
            The initial second coordinates, one per bead or shared.

        theta:
            The initial orientations, one per bead or shared.

        space:
            The lattice of the beads along the chain, or `None` for one bead.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ActiveBeadElement
        >>> ActiveBeadElement('bead', theta=1.0).theta
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        x: object = 0.0,
        y: object = 0.0,
        theta: Annotated[object, Period(2 * math.pi)] = 0.0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.x = x
        self.y = y
        self.theta = theta


class ActiveRouseInteraction(Interaction):
    """Rouse chain of active Brownian beads.

    The bead at `(x, y)` swims at the speed `v0` along its orientation `theta`, which diffuses with `D_r`, its position
    diffuses with `D_t`, and harmonic springs of stiffness `k` and zero rest length pull it towards its neighbours
    along the chain, `k (r_{i+1} + r_{i-1} - 2 r_i)`. The beads at the ends have one neighbour. It is a chain of
    self-propelled colloids, a filament driven by molecular motors, or a worm.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the chain, with the coordinates `x` and `y` and the orientations `theta`.

        v0:
            The self-propulsion speed.

        k:
            The stiffness of the bonds.

        D_t:
            The translational diffusion coefficient.

        D_r:
            The rotational diffusion coefficient.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ActiveRouseInteraction
        >>> ActiveRouseInteraction('rouse', target='chain').k
        10.0
    """

    law_classes = (NoiseLaw, RateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        v0: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        k: Annotated[float, RealSet.closed_open(0, oo)] = 10.0,
        D_t: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        D_r: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.v0 = v0
        self.k = k
        self.D_t = D_t
        self.D_r = D_r

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the propulsion, the bonds, and the noises of the beads of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context, context.t
            x, y, theta = s.target.x, s.target.y, s.target.theta
            self.laws_ = [
                RateLaw(x.diff(t), s.v0 * cos(theta) + s.k * laplacian(x)),
                RateLaw(y.diff(t), s.v0 * sin(theta) + s.k * laplacian(y)),
                NoiseLaw(x, sqrt(2 * s.D_t)),
                NoiseLaw(y, sqrt(2 * s.D_t)),
                NoiseLaw(theta, sqrt(2 * s.D_r)),
            ]


class ActivePolymerSystem(BaseSystem):
    """An active polymer, a Rouse chain of active Brownian beads.

    Each bead swims along its own orientation.

    Bead `i` follows `r_i' = v0 n_i + k (r_{i+1} + r_{i-1} - 2 r_i) + sqrt(2 D_t) xi_i`, with its orientation `n_i =
    (cos theta_i, sin theta_i)` diffusing with `D_r`, and the end beads bonded to one neighbour. The bonds cancel in the
    motion of the centre of mass, whose mean squared displacement is `(4 D_t t + 2 v0² (D_r t - 1 + exp(-D_r t)) / D_r²)
    / N`, that of an active Brownian particle divided by `N`. The activity swells the chain: a passive chain has the
    mean squared end-to-end distance `2 D_t (N - 1) / k` in the plane, and self-propulsion of persistence time `1 / D_r`
    adds to the bond fluctuations and stretches it further. The beads start on a straight line with unit spacing and
    random orientations. The system has the coordinates `chain.x` and `chain.y` and the orientations `chain.theta` of
    the beads.

    Args:
        n_beads:
            The number of beads.

        v0:
            The self-propulsion speed of every bead.

        k:
            The stiffness of the bonds.

        D_t:
            The translational diffusion coefficient.

        D_r:
            The rotational diffusion coefficient.

        random_state:
            The seed or generator of the initial orientations.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ActivePolymerSystem
        >>> ActivePolymerSystem(5).prepare().simulate(until=1.0).get_values('chain.x').shape[-1]
        5
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'sde', 'polymer', 'self-propulsion', 'rouse chain')
    references = ('Winkler and Gompper 2020', 'Osmanović and Rabin 2017', 'Eisenstecken et al. 2016')
    until = 50.0
    dt = 0.5

    def __init__(
        self,
        n_beads: int = 20,
        *,
        v0: float = 1.0,
        k: float = 10.0,
        D_t: float = 0.1,
        D_r: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_beads = n_beads
        self.v0 = v0
        self.k = k
        self.D_t = D_t
        self.D_r = D_r
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of an active polymer, a Rouse chain of active Brownian beads.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        angles = np.random.default_rng(self.random_state).uniform(0.0, 2 * np.pi, self.n_beads)
        chain = ActiveBeadElement(
            'chain',
            x=np.arange(self.n_beads, dtype=float),
            y=np.zeros(self.n_beads),
            theta=angles,
            space=LatticeSpace(self.n_beads, boundary='reflecting'),
        )
        rouse = ActiveRouseInteraction('rouse', target='chain', v0=self.v0, k=self.k, D_t=self.D_t, D_r=self.D_r)
        self.system_ = System(
            self.label,
            elements=[chain],
            interactions=[rouse],
            solver=EulerMaruyamaSolver(dt=0.005),
        )
