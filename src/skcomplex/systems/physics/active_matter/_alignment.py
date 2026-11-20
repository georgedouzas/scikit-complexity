"""Alignment of active Brownian particles.

A pairwise torque turns neighbours towards each other.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, I, exp, oo, sin
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import RateLaw
from ....core.components.laws.utils import neighbors
from ....core.components.spaces import BaseSpace, ContinuousSpace, RadiusNeighborhood
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, EulerMaruyamaSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ..common.particles import ActiveDiffusionInteraction, ActiveParticleElement, SelfPropulsionInteraction

HEADINGS = VisualSpec(
    'points',
    {'position': 'x', 'radius': 0.3, 'color_by': 'theta', 'color_range': (0.0, 2 * math.pi)},
)


class PolarAlignmentInteraction(Interaction):
    """Alignment torque `theta_i' += gamma sum_j sin(theta_j - theta_i)` between close particles.

    The sum runs over every neighbour `j` within the interaction range `cutoff`, so the torque turns each particle
    towards the headings of its neighbours. It is a bird turning with its flock, or a bacterium aligning with others
    in a swarm.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particles, with the positions `x` and the headings `theta`.

        gamma:
            The strength of the alignment.

        cutoff:
            The interaction range.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import PolarAlignmentInteraction
        >>> PolarAlignmentInteraction('align', target='swimmers', gamma=0.5).gamma
        0.5
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        gamma: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        cutoff: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.gamma = gamma
        self.cutoff = cutoff

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the torque on the heading of every particle of the target from its neighbours.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            theta = context.target.theta
            torque = neighbors(context.gamma * sin(context.target.other.theta - theta), RadiusNeighborhood(self.cutoff))
            self.laws_ = [RateLaw(theta.diff(context.t), torque)]


def _get_heading(member: SimpleNamespace) -> Expression:
    """Return the unit vector `exp(i theta)` of a member's heading.

    The vector lies in the complex plane.
    """
    return exp(I * member.theta)


def _get_polarization(s: SimpleNamespace) -> Expression:
    """Return the modulus of the mean heading, the polarization."""
    return abs(s.mean_heading)


class SwimmerElement(ActiveParticleElement):
    """Active particle drawn as a point coloured by its heading."""

    visual = HEADINGS


class AligningActiveBrownianParticlesSystem(BaseSystem):
    """Active Brownian particles in a periodic box whose headings align.

    A particle aligns with its neighbours within `d0`.

    The heading of particle `i` follows `dtheta_i = gamma sum_{|x_j - x_i| < d0} sin(theta_j - theta_i) dt + sqrt(2 D_r)
    dW`, and the particle swims at the constant `speed` along it. In the mean-field approximation of the Fokker-Planck
    equation, the disordered state is unstable above the density `rho_c = 2 D_r / (gamma pi d0^2)`, so at a fixed
    density `rho` the particles move together when `gamma` exceeds `2 D_r / (pi d0^2 rho)`. The polarization `|<exp(i
    theta)>|` is near 0 in the disordered state and near 1 in the ordered one. The system has the positions
    `swimmers.x`, the headings `swimmers.theta`, and the observables `mean_heading` and `polarization`.

    Args:
        n_particles:
            The number of particles.

        size:
            The side of the periodic box.

        speed:
            The self-propulsion speed.

        gamma:
            The strength of the alignment.

        D_r:
            The rotational diffusion coefficient.

        d0:
            The interaction range.

        D_t:
            The translational diffusion coefficient.

        random_state:
            The seed or generator of the uniform initial positions and headings.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import AligningActiveBrownianParticlesSystem
        >>> AligningActiveBrownianParticlesSystem(10, gamma=0.5).prepare().system_.interactions[0].gamma
        0.5
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'sde', 'flocking', 'alignment', 'pair interaction')
    references = ('Bertin 2021', 'Peruani, Deutsch, and Bär 2008', 'Chaté 2020')
    until = 100.0
    dt = 1.0
    observables = (
        Observable('mean_heading', _get_heading, group='swimmers'),
        Observable('polarization', _get_polarization),
    )

    def __init__(
        self,
        n_particles: int = 200,
        *,
        size: float = 20.0,
        speed: float = 1.0,
        gamma: float = 1.0,
        D_r: float = 0.1,
        d0: float = 1.0,
        D_t: float = 0.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.size = size
        self.speed = speed
        self.gamma = gamma
        self.D_r = D_r
        self.d0 = d0
        self.D_t = D_t
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of active Brownian particles in a periodic box whose headings align.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        swimmers = SwimmerElement(
            'swimmers',
            x=rng.uniform(0.0, self.size, (self.n_particles, 2)),
            theta=rng.uniform(0.0, 2 * math.pi, self.n_particles),
            size=self.n_particles,
            space=ContinuousSpace(extent=(self.size, self.size), boundary='periodic'),
        )
        self.system_ = System(
            self.label,
            elements=[swimmers],
            interactions=[
                PolarAlignmentInteraction('align', target='swimmers', gamma=self.gamma, cutoff=self.d0),
                ActiveDiffusionInteraction('diffusion', target='swimmers', D_t=self.D_t, D_r=self.D_r),
                SelfPropulsionInteraction('propulsion', target='swimmers', speed=self.speed),
            ],
            solver=EulerMaruyamaSolver(dt=0.05),
        )
