"""Alignment of active Brownian particles: a pairwise torque that turns neighbours towards each other."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import Expression, I, exp, oo, sin
from sage.sets.real_set import RealSet

from ....core.components.elements import GroupElement
from ....core.components.interactions import PairCouplingInteraction
from ....core.components.spaces import ContinuousSpace
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import EulerMaruyamaSolver
from ....core.specs import ObservableSpec, ParameterSpec, VisualSpec
from ..common.particles import ActiveParticleElement, SelfPropulsionInteraction

HEADINGS = VisualSpec(
    'points',
    {'position': 'x', 'radius': 0.3, 'color_by': 'theta', 'color_range': (0.0, 2 * math.pi)},
)


def _heading(member: SimpleNamespace) -> Expression:
    """Return the unit vector `exp(i theta)` of a member's heading in the complex plane."""
    return exp(I * member.theta)


def _polarization(s: SimpleNamespace) -> Expression:
    """Return the modulus of the mean heading, the polarization."""
    return abs(s.mean_heading)


class PolarAlignmentInteraction(PairCouplingInteraction):
    """Alignment torque between two particles closer than the cutoff, `theta_i' += gamma sin(theta_j - theta_i)`.

    Summed over every neighbour `j` within the interaction range `d0`, the cutoff, the torque turns each particle
    towards the headings of its neighbours.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        first:
            The path of the group or particle of the first members.

        second:
            The path of the group or particle of the second members.

        cutoff:
            The interaction range `d0`.

        position:
            The name of the vector state that holds the positions of the members.

        gamma:
            The strength of the alignment.

        period:
            The side of a periodic box, for minimum image distances, or `None` for the period of the space.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import PolarAlignmentInteraction
        >>> PolarAlignmentInteraction('align', first='swimmers', second='swimmers', gamma=0.5).gamma
        0.5
    """

    def __init__(
        self,
        label: str,
        first: str,
        second: str,
        cutoff: float = 1.0,
        position: str = 'x',
        *,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        period: float | None = None,
    ) -> None:
        super().__init__(
            label,
            first,
            second,
            cutoff=cutoff,
            position=position,
            write='theta',
            function=self._torque,
            period=period,
        )
        self.gamma = gamma

    @staticmethod
    def _torque(namespace: SimpleNamespace) -> Expression:
        """Return the torque `gamma sin(theta_j - theta_i)` on the first member of a pair."""
        return namespace.gamma * sin(namespace.second.theta - namespace.first.theta)


class AligningActiveBrownianParticlesSystem(BaseSystem):
    """Active Brownian particles in a periodic box whose headings align with the neighbours within `d0`.

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
    solver = EulerMaruyamaSolver(dt=0.05)
    observables = (
        ObservableSpec('mean_heading', _heading, group='swimmers'),
        ObservableSpec('polarization', _polarization),
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
    ) -> None:
        super().__init__(label)
        self.n_particles = n_particles
        self.size = size
        self.speed = speed
        self.gamma = gamma
        self.D_r = D_r
        self.d0 = d0
        self.D_t = D_t
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return active Brownian particles in a periodic box whose headings align with the neighbours within `d0`."""
        rng = np.random.default_rng(self.random_state)
        swimmers = GroupElement(
            'swimmers',
            element=ActiveParticleElement,
            size=self.n_particles,
            values={
                'x': rng.uniform(0.0, self.size, (self.n_particles, 2)),
                'theta': rng.uniform(0.0, 2 * math.pi, self.n_particles),
                'D_t': self.D_t,
                'D_r': self.D_r,
            },
            visual=HEADINGS,
        )
        alignment = PolarAlignmentInteraction(
            'align',
            first='swimmers',
            second='swimmers',
            cutoff=self.d0,
            period=self.size,
            gamma=self.gamma,
        )
        return System(
            self.label,
            elements=[swimmers],
            interactions=[alignment, SelfPropulsionInteraction('propulsion', target='swimmers', speed=self.speed)],
            space=ContinuousSpace(extent=(self.size, self.size), boundary='periodic'),
        )
