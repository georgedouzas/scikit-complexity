"""Rigid Brownian bodies: axisymmetric particles in three dimensions and dimers of two spheres on a rod."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.components.elements import GroupElement
from ....core.components.spaces import ContinuousSpace
from ....core.components.system import System
from ....core.engine.solvers import EulerMaruyamaSolver
from ....core.errors import ParameterError
from ..._catalog import register_system
from ._anisotropic_particle import AnisotropicParticleElement
from ._dimer import DimerElement


@register_system(
    field='physics/soft_matter',
    tags=('brownian motion', 'sde', 'rotational diffusion', 'three dimensions', 'anisotropy', 'chirality'),
    references=('Perrin 1934', 'Han et al. 2006', 'Argun et al. 2021'),
    defaults={'until': 10.0, 'dt': 0.05, 'solver': EulerMaruyamaSolver(dt=0.001)},
)
def anisotropic_brownian_particles(
    n_particles: int = 200,
    *,
    D_par: float = 1.0,
    D_perp: float = 1.0,
    D_r: float = 0.75,
    D_spin: float = 0.75,
    D_c: float = 0.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return overdamped axisymmetric Brownian particles in three dimensions that translate and rotate.

    Every particle diffuses with `D_par` along its axis `u` and with `D_perp` across it, its axis diffuses on the unit
    sphere with `D_r`, so that `<u(t) . u(0)> = exp(-2 D_r t)`, and it spins about its axis with `D_spin`. The defaults
    are a Janus sphere of unit radius, `D_par = D_perp = D_t` and `D_r = 3 D_t / 4`, whose orientation is visible
    although its diffusion is isotropic. A prolate ellipsoid has `D_par > D_perp` from Perrin's factors, so its
    displacement over times shorter than `1 / D_r` is elongated along its axis. A chiral body, such as a spiral, has a
    coupling `D_c` of the translation along its axis and the rotation about it, `<dr_par dpsi> = 2 D_c dt`. These are
    Challenges 5.1 to 5.3 of Argun et al. The axis is held as a unit vector and rescaled to unit length after every
    step, and the particles start at the origin with axes drawn uniformly on the sphere.

    Args:
        n_particles:
            The number of particles.

        D_par:
            The translational diffusion coefficient along the axis.

        D_perp:
            The translational diffusion coefficient across the axis.

        D_r:
            The rotational diffusion coefficient of the axis.

        D_spin:
            The rotational diffusion coefficient about the axis.

        D_c:
            The coupling of the translation along the axis and the rotation about it.

        random_state:
            The seed or generator of the initial axes.

    Returns:
        system:
            The system, with the positions `particles.x`, `particles.y`, `particles.z`, the axes `particles.ux`,
            `particles.uy`, `particles.uz`, and the spins `particles.psi`.

    Raises:
        ParameterError:
            The coupling exceeds `sqrt(D_par D_spin)`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import anisotropic_brownian_particles
        >>> anisotropic_brownian_particles(5, D_par=1.5, D_perp=0.8).elements[0].values['D_perp']
        0.8
    """
    if D_c**2 > D_par * D_spin:
        error_msg = f'The coupling {D_c!r} exceeds sqrt(D_par D_spin), so the diffusion tensor is not positive.'
        raise ParameterError(error_msg)
    axes = np.random.default_rng(random_state).normal(size=(n_particles, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    values = {
        'ux': axes[:, 0],
        'uy': axes[:, 1],
        'uz': axes[:, 2],
        'D_par': D_par,
        'D_perp': D_perp,
        'D_r': D_r,
        'D_spin': D_spin,
        'D_c': D_c,
    }
    particles = GroupElement('particles', element=AnisotropicParticleElement, size=n_particles, values=values)
    return System(elements=[particles])


@register_system(
    field='physics/soft_matter',
    tags=('brownian motion', 'sde', 'colloidal molecule', 'rotational diffusion', 'active matter'),
    references=('Argun et al. 2021',),
    defaults={'until': 20.0, 'dt': 0.1, 'solver': EulerMaruyamaSolver(dt=0.005)},
)
def rigid_dimers(
    n_dimers: int = 100,
    *,
    L: float = 1.0,
    v0: float = 0.0,
    D_t: float = 0.1,
    D_r: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return colloidal molecules: pairs of Brownian or active spheres joined by rigid rods, free in the plane.

    Each sphere diffuses with `D_t`, and an active sphere also swims at the speed `v0` along its own orientation,
    which diffuses with `D_r`. The rod keeps the spheres at the distance `L`. Diffusing both spheres freely and then
    restoring their distance, the scheme of Problem 5.1 of Argun et al., gives in the limit of small steps a centre
    that diffuses with `D_t / 2` and a rod angle that diffuses with `2 D_t / L²`, so the rotational diffusion of the
    molecule falls as `L^-2` in this free-draining model. Active spheres, Problem 9.3, swim together along the mean of
    their orientations and turn the rod with the part of their difference across it, which raises the long-time
    diffusion of the centre to `D_t / 2 + v0² / (4 D_r)`. The dimers start at the origin with random rod angles and
    orientations.

    Args:
        n_dimers:
            The number of dimers.

        L:
            The length of the rod.

        v0:
            The swimming speed of each sphere, 0 for Brownian spheres.

        D_t:
            The translational diffusion coefficient of each sphere.

        D_r:
            The rotational diffusion coefficient of the orientation of each sphere.

        random_state:
            The seed or generator of the initial angles and orientations.

    Returns:
        system:
            The system, with the centres `dimers.c`, the rod angles `dimers.phi`, and the orientations `dimers.theta1`
            and `dimers.theta2`. The spheres sit at `c ± (L / 2) (cos phi, sin phi)`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import rigid_dimers
        >>> rigid_dimers(5, L=2.0).elements[0].values['L']
        2.0
    """
    rng = np.random.default_rng(random_state)
    angles = rng.uniform(0.0, 2 * np.pi, (3, n_dimers))
    values = {
        'c': np.zeros((n_dimers, 2)),
        'phi': angles[0],
        'theta1': angles[1],
        'theta2': angles[2],
        'L': L,
        'v0': v0,
        'D_t': D_t,
        'D_r': D_r,
    }
    dimers = GroupElement('dimers', element=DimerElement, size=n_dimers, values=values)
    return System(elements=[dimers], space=ContinuousSpace(n_dim=2))
