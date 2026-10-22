"""Alignment of active Brownian particles: a pairwise torque that turns neighbours towards each other."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np
from sage.all__sagemath_symbolics import I, exp, sin

from ....core.base import Domain, Expression, VisualSpec, parameter
from ....core.model import EuclideanSpace, Law, Observable, PairInteraction, PeriodicBox, Rate, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import EulerMaruyama
from ._particles import ActiveParticle

HEADINGS = VisualSpec(
    'points',
    {'position': 'x', 'radius': 0.3, 'color_by': 'theta', 'color_range': (0.0, 2 * math.pi)},
)


def _heading(member: Symbols) -> Expression:
    """Return the unit vector `exp(i theta)` of a member's heading in the complex plane."""
    return exp(I * member.theta)


def _polarization(s: Symbols) -> Expression:
    """Return the modulus of the mean heading, the polarization."""
    return abs(s.mean_heading)


class PolarAlignment(PairInteraction):
    """Alignment torque between two particles closer than the cutoff, `dtheta_i/dt = gamma sin(theta_j - theta_i)`.

    Summed over every neighbour `j` within the interaction range `d0`, the cutoff, the torque turns each particle
    towards the headings of its neighbours.

    Args:
        label:
            The label of the interaction.

        first:
            The path of the group or particle of the first members.

        second:
            The path of the group or particle of the second members.

        cutoff:
            The interaction range `d0`.

        period:
            The side of a periodic box, for minimum image distances, or `None` for an open space.

        gamma:
            The strength of the alignment.

    Examples:
        >>> from skcomplex.models.physics.active_matter import PolarAlignment
        >>> PolarAlignment('align', first='swimmers', second='swimmers', gamma=0.5).gamma
        0.5
    """

    gamma: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the opposite torques on the two members of a pair.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        torque = s.gamma * sin(s.second.theta - s.first.theta)
        return [Rate(s.first.theta, torque), Rate(s.second.theta, -torque)]


@register_model(
    field='physics/active_matter',
    tags=('active matter', 'sde', 'flocking', 'alignment', 'pair interaction'),
    references=('Bertin 2021', 'Peruani, Deutsch, and Bär 2008', 'Chaté 2020'),
    defaults={'until': 100.0, 'dt': 1.0, 'solver': EulerMaruyama(dt=0.05)},
    observables=lambda system: [
        Observable('mean_heading', _heading, group='swimmers'),
        Observable('polarization', _polarization),
    ],
)
def aligning_active_brownian_particles(
    n_particles: int = 200,
    *,
    size: float = 20.0,
    speed: float = 1.0,
    gamma: float = 1.0,
    D_r: float = 0.1,
    d0: float = 1.0,
    D_t: float = 0.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return active Brownian particles in a periodic box whose headings align with the neighbours within `d0`.

    The heading of particle `i` follows `dtheta_i = gamma sum_{|x_j - x_i| < d0} sin(theta_j - theta_i) dt +
    sqrt(2 D_r) dW`, and the particle swims at the constant `speed` along it. In the mean-field approximation of the
    Fokker-Planck equation, the disordered state is unstable above the density `rho_c = 2 D_r / (gamma pi d0^2)`, so
    at a fixed density `rho` the particles move together when `gamma` exceeds `2 D_r / (pi d0^2 rho)`. The
    polarization `|<exp(i theta)>|` is near 0 in the disordered state and near 1 in the ordered one.

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

    Returns:
        system:
            The model, with the positions `swimmers.x`, the headings `swimmers.theta`, and the observables
            `mean_heading` and `polarization`.

    Examples:
        >>> from skcomplex.models.physics.active_matter import aligning_active_brownian_particles
        >>> aligning_active_brownian_particles(10, gamma=0.5).interactions[1].gamma
        0.5
    """
    rng = np.random.default_rng(random_state)
    swimmers = ActiveParticle.group(
        'swimmers',
        n_particles,
        x=rng.uniform(0.0, size, (n_particles, 2)),
        theta=rng.uniform(0.0, 2 * math.pi, n_particles),
        speed=speed,
        D_t=D_t,
        D_r=D_r,
        visual=HEADINGS,
    )
    alignment = PolarAlignment('align', first='swimmers', second='swimmers', cutoff=d0, period=size, gamma=gamma)
    return System(
        components=[swimmers],
        interactions=[PeriodicBox('box', target='swimmers', size=size), alignment],
        space=EuclideanSpace(2),
    )
