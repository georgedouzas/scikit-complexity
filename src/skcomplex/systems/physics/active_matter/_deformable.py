"""Deformable active particles: self-propelled particles whose shape and velocity drive each other."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.spaces import ContinuousSpace
from ....core.components.system import System
from ....core.engine.solvers import EulerMaruyamaSolver
from ..._catalog import register_system
from ._deformable_particle import DeformableParticleElement


@register_system(
    field='physics/active_matter',
    tags=('active matter', 'deformable particles', 'tensor', 'circular motion', 'bifurcation'),
    references=('Ohta and Ohkuma 2009', 'Ohta 2017', 'Tarama 2017'),
    defaults={'until': 100.0, 'dt': 0.25, 'solver': EulerMaruyamaSolver(dt=0.01)},
)
def deformable_active_particles(
    n_particles: int = 20,
    *,
    gamma: float = 1.0,
    a: float = 1.0,
    b: float = 1.0,
    kappa: float = 0.5,
    noise: float = 0.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return deformable self-propelled particles of Ohta and Ohkuma, which move straight or along circles.

    Every particle follows `x' = v`, `v' = gamma v - |v|² v - a S v`, and `S' = -kappa S + b (v v - |v|² I / 2)`, with
    the traceless symmetric deformation tensor `S`. A fast relaxing shape, large `kappa`, gives straight motion at the
    speed `sqrt(gamma / (1 + a b / (2 kappa)))`. For `a b (gamma - kappa / 2) > kappa²` a circular motion exists, at the
    speed `sqrt(gamma - kappa / 2)` with the angular velocity `sqrt(a b (gamma - kappa / 2) - kappa²) / 2`, its
    elongation turning with its velocity. For the defaults it exists below `kappa ≈ 0.78`, and particles started from
    small velocities reach it for `kappa <= 0.5`, while closer to the threshold straight motion stays stable too. The
    particles do not interact, and they start at random positions, small random velocities, and no deformation.

    Args:
        n_particles:
            The number of particles.

        gamma:
            The distance from the onset of self-propulsion.

        a:
            The coupling of the velocity to the deformation.

        b:
            The coupling of the deformation to the velocity.

        kappa:
            The relaxation rate of the deformation.

        noise:
            The amplitude of the noise of the velocity.

        random_state:
            The seed or generator of the initial positions and velocities.

    Returns:
        system:
            The system, with the positions `cells.x`, the velocities `cells.v`, and the deformations `cells.S`.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import deformable_active_particles
        >>> deformable_active_particles(5).elements[0].prepare().size_
        5
    """
    rng = np.random.default_rng(random_state)
    values = {
        'x': rng.uniform(0.0, 20.0, (n_particles, 2)),
        'v': rng.normal(0.0, 0.1, (n_particles, 2)),
        'S': np.zeros((n_particles, 2)),
        'gamma': gamma,
        'a': a,
        'b': b,
        'kappa': kappa,
        'noise': noise,
    }
    cells = GroupElement('cells', element=DeformableParticleElement, size=n_particles, values=values)
    return System(elements=[cells], space=ContinuousSpace(n_dim=2))
