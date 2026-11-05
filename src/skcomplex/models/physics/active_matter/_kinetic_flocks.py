"""Kinetic theory of flocking of Bertin, Droz, and Grégoire over position and orientation."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace

import numpy as np
from sage.all__sagemath_symbolics import Expression, cos, sin

from ....core.elements import DensityElement, FieldElement
from ....core.interactions import AdvectionInteraction
from ....core.laws.utils import coordinate
from ....core.spaces import LatticeSpace, PhaseSpace
from ....core.system import System
from ....engine.solvers import RK4Solver
from ....specs import AxisSpec
from ..._catalog import register_model
from ._angular_alignment import AngularAlignmentInteraction
from ._angular_diffusion import AngularDiffusionInteraction


@register_model(
    field='physics/active_matter',
    tags=('field', 'kinetic', 'phase space', 'flocking', 'pde'),
    references=('Bertin, Droz, and Grégoire 2006', 'Bertin, Droz, and Grégoire 2009'),
    defaults={'until': 0.5, 'dt': 0.1, 'solver': RK4Solver(dt=0.05)},
)
def bertin_droz_gregoire(
    positions: int = 4,
    angles: int = 16,
    *,
    density: float = 1.0,
    speed: float = 1.0,
    interaction_range: float = 0.5,
    tumbling: float = 1.0,
    noise: float = 0.5,
    perturbation: float = 0.01,
) -> System:
    """Return the kinetic theory of flocking, whose polar order grows above a transition density.

    Particles stream along their orientation, tumble at a rate by a wrapped normal noise, and align in binary collisions
    at their mean angle with the same noise. A homogeneous disordered state is unstable above the density
    `pi tumbling (1 - exp(-noise^2 / 2)) / (8 interaction_range speed (exp(-noise^2 / 2) - 2 / 3))`.

    Args:
        positions:
            The number of positions along each of the two periodic position axes of unit spacing.

        angles:
            The number of cells of the orientation axis.

        density:
            The density of the homogeneous state.

        speed:
            The speed of the particles.

        interaction_range:
            The interaction range.

        tumbling:
            The rate of the tumbling events.

        noise:
            The standard deviation of the noise of tumbling and alignment.

        perturbation:
            The relative amplitude of the initial polar perturbation.

    Returns:
        system:
            The model, with the density `flock.f`.

    Examples:
        >>> from skcomplex.models.physics.active_matter import bertin_droz_gregoire
        >>> bertin_droz_gregoire(2, 8).elements[0].prepare().size_
        32
    """
    axis = AxisSpec('theta', angles, 0.0, 2 * math.pi, periodic=True)
    space = PhaseSpace(LatticeSpace((positions, positions)), [axis]).prepare()
    theta = space.coordinates_['theta']
    profile = density * (1.0 + perturbation * np.cos(theta)) / (2 * math.pi)
    values = np.broadcast_to(profile, space.shape_).copy()
    flock = FieldElement(
        'flock',
        element=DensityElement,
        space=space,
        values={'f': values},
        discretization='finite-volume',
    )

    def drift(_: SimpleNamespace) -> list[Expression]:
        return [speed * cos(coordinate('theta')), speed * sin(coordinate('theta')), 0]

    return System(
        elements=[flock],
        interactions=[
            AdvectionInteraction('stream', target='flock', state='f', drift=drift),
            AngularDiffusionInteraction('tumble', candidate='flock', angles=angles, tumbling=tumbling, noise=noise),
            AngularAlignmentInteraction(
                'align',
                candidate='flock',
                field='flock',
                angles=angles,
                frequency=interaction_range * speed,
                noise=noise,
            ),
        ],
    )
