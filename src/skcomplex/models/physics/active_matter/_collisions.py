"""Flocking by binary collisions: self-propelled particles that align their headings when they meet."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression

from ....core.elements import GroupElement
from ....core.spaces import ContinuousSpace
from ....core.system import System
from ....specs import ObservableSpec
from ..._catalog import register_model
from ._binary_collisions import BinaryCollisionsInteraction
from ._flocker import FlockerElement
from ._flocker_view import FlockerView


def _cosine(agent: FlockerView) -> float:
    """Return the first component of an agent's heading."""
    return math.cos(agent.theta)


def _sine(agent: FlockerView) -> float:
    """Return the second component of an agent's heading."""
    return math.sin(agent.theta)


def _polarization(s: SimpleNamespace) -> Expression:
    """Return the length of the mean heading, the polarization."""
    return (s.mean_cos**2 + s.mean_sin**2) ** 0.5


def _flocking_observables(system: System) -> list[ObservableSpec]:
    """Return the mean heading components and the polarization of the birds."""
    return [
        ObservableSpec('mean_cos', _cosine, group='birds'),
        ObservableSpec('mean_sin', _sine, group='birds'),
        ObservableSpec('polarization', _polarization),
    ]


@register_model(
    field='physics/active_matter',
    tags=('active matter', 'agents', 'flocking', 'collisions', 'boltzmann equation'),
    references=('Bertin, Droz, and Grégoire 2006', 'Bertin, Droz, and Grégoire 2009', 'Bertin 2021'),
    defaults={'until': 200, 'dt': 1},
    observables=_flocking_observables,
)
def binary_collision_flocking(
    n_particles: int = 400,
    *,
    size: float = 20.0,
    speed: float = 1.0,
    lam: float = 1.0,
    sigma: float = 0.3,
    d0: float = 0.5,
    substeps: int = 10,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return self-propelled particles that flock through aligning binary collisions, the Bertin-Droz-Grégoire model.

    Particles fly straight at a constant speed. Their headings jump by Gaussian angles in self-diffusion events at the
    rate `lam`, and two particles that come closer than `d0` both take their mean heading plus Gaussian noise. The
    hydrodynamic equations derived from the Boltzmann equation predict collective motion above the density
    `rho_t = pi lam (1 - exp(-sigma^2 / 2)) / (4 (exp(-sigma^2 / 2) - 2/3))`, for `speed = 1` and `d0 = 1/2`, so
    collective motion needs `sigma < sqrt(2 ln(3/2))`, about 0.90, at any density.

    Args:
        n_particles:
            The number of particles.

        size:
            The side of the periodic square.

        speed:
            The self-propulsion speed.

        lam:
            The rate of self-diffusion events.

        sigma:
            The standard deviation of the angular noise.

        d0:
            The collision distance.

        substeps:
            The number of substeps per unit of time.

        random_state:
            The seed or generator of the uniform initial positions and headings.

    Returns:
        system:
            The model, with the agent group `birds` and the observables `mean_cos`, `mean_sin`, and `polarization`.

    Examples:
        >>> from skcomplex.models.physics.active_matter import binary_collision_flocking
        >>> binary_collision_flocking(10).interactions[0].sigma
        0.3
    """
    rng = np.random.default_rng(random_state)
    birds = GroupElement(
        'birds',
        element=FlockerElement,
        size=n_particles,
        values={'theta': rng.uniform(0.0, 2 * math.pi, n_particles)},
        location=rng.uniform(0.0, size, (n_particles, 2)),
    )
    collisions = BinaryCollisionsInteraction(
        'collisions',
        target='birds',
        speed=speed,
        lam=lam,
        sigma=sigma,
        d0=d0,
        substeps=float(substeps),
    )
    return System(
        elements=[birds],
        interactions=[collisions],
        space=ContinuousSpace(extent=(size, size), boundary='periodic'),
    )
