"""Active polymers.

The polymers are chains of self-propelled beads bonded by springs.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import cos, oo, pi, sin, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, FieldElement
from ....core.components.laws import BaseLaw, NoiseLaw, RateLaw
from ....core.components.laws.utils import laplacian
from ....core.components.spaces import LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import EulerMaruyamaSolver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class ActiveBeadElement(Element):
    """Bead of an active polymer.

    The bead is an active Brownian particle bonded to its neighbours along a chain by springs.

    The bead at `(x, y)` swims at the speed `v0` along its orientation `theta`, which diffuses with `D_r`, its position
    diffuses with `D_t`, and harmonic springs of stiffness `k` and zero rest length pull it towards its neighbours
    along the chain, `k (r_{i+1} + r_{i-1} - 2 r_i)`, the Rouse chain. The beads at the ends have one neighbour.

    Args:
        label:
            The label of the bead or the chain.

        visual:
            How the bead is drawn, or `None` for `default_visual_`.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        theta:
            The initial orientation.

        v0:
            The self-propulsion speed.

        k:
            The stiffness of the bonds.

        D_t:
            The translational diffusion coefficient.

        D_r:
            The rotational diffusion coefficient.

    Attributes:
        default_visual_:
            The visual of the bead when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ActiveBeadElement
        >>> ActiveBeadElement('bead').k
        10.0
    """

    law_classes = (
        NoiseLaw,
        RateLaw,
    )

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.0,
        y: Annotated[float, StateSpec()] = 0.0,
        theta: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
        v0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        k: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        D_t: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        D_r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.theta = theta
        self.v0 = v0
        self.k = k
        self.D_t = D_t
        self.D_r = D_r

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the propulsion, the bonds, and the noises of the bead.

        The noises act on its position and orientation.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [
            RateLaw(s.x, s.v0 * cos(s.theta) + s.k * laplacian(s.x)),
            RateLaw(s.y, s.v0 * sin(s.theta) + s.k * laplacian(s.y)),
            NoiseLaw(s.x, sqrt(2 * s.D_t)),
            NoiseLaw(s.y, sqrt(2 * s.D_t)),
            NoiseLaw(s.theta, sqrt(2 * s.D_r)),
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

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ActivePolymerSystem
        >>> ActivePolymerSystem(5).prepare().system_.elements[0].prepare().size_
        5
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'sde', 'polymer', 'self-propulsion', 'rouse chain')
    references = ('Winkler and Gompper 2020', 'Osmanović and Rabin 2017', 'Eisenstecken et al. 2016')
    until = 50.0
    dt = 0.5
    solver = EulerMaruyamaSolver(dt=0.005)

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
    ) -> None:
        super().__init__(label)
        self.n_beads = n_beads
        self.v0 = v0
        self.k = k
        self.D_t = D_t
        self.D_r = D_r
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the components of the system, built from its model parameters."""
        angles = np.random.default_rng(self.random_state).uniform(0.0, 2 * np.pi, self.n_beads)
        chain = FieldElement(
            'chain',
            element=ActiveBeadElement,
            space=LatticeSpace(self.n_beads, boundary='reflecting'),
            boundary={'x': 'zero-flux', 'y': 'zero-flux'},
            values={
                'x': np.arange(self.n_beads, dtype=float),
                'y': np.zeros(self.n_beads),
                'theta': angles,
                'v0': self.v0,
                'k': self.k,
                'D_t': self.D_t,
                'D_r': self.D_r,
            },
        )
        return CompositeSystem(self.label, elements=[chain])
