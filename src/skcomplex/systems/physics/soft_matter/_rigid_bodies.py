"""Rigid Brownian bodies.

The module covers axisymmetric particles in three dimensions and dimers of two spheres on a rod.

`AXES` holds the names of the coordinates of the position. `DIRECTIONS` holds the names of the components of the unit
vector along the axis.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import SR, cos, oo, sin, sqrt, vector
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element, Period
from ....core.components.interactions import Interaction
from ....core.components.laws import BoundaryLaw, NoiseLaw, RateLaw
from ....core.components.spaces import BaseSpace, ContinuousSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, EulerMaruyamaSolver
from ....core.errors import ParameterError
from ....core.specs import Role
from ....core.systems import BaseSystem, System

AXES = ('x', 'y', 'z')
DIRECTIONS = ('ux', 'uy', 'uz')


class AnisotropicParticleElement(Element):
    """Axisymmetric particles in three dimensions, with a position, a unit axis, and a spin about the axis.

    Args:
        label:
            The label of the particles.

        x:
            The initial first coordinate, one per particle or shared.

        y:
            The initial second coordinate, one per particle or shared.

        z:
            The initial third coordinate, one per particle or shared.

        ux:
            The initial first component of the axis, one per particle or shared.

        uy:
            The initial second component of the axis, one per particle or shared.

        uz:
            The initial third component of the axis, one per particle or shared.

        psi:
            The initial spin angle about the axis, one per particle or shared.

        size:
            The number of particles, or `None` for one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import AnisotropicParticleElement
        >>> AnisotropicParticleElement('janus').uz
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        x: object = 0.0,
        y: object = 0.0,
        z: object = 0.0,
        ux: object = 0.0,
        uy: object = 0.0,
        uz: object = 1.0,
        psi: object = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.x = x
        self.y = y
        self.z = z
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.psi = psi


class AnisotropicDiffusionInteraction(Interaction):
    """Diffusion of axisymmetric particles whose diffusion tensor turns with their axis.

    The position `r = (x, y, z)` diffuses with `D_par` along the unit axis `u = (ux, uy, uz)` and with `D_perp` across
    it, `dr = (sqrt(2 D_par) u u + sqrt(2 D_perp) (I - u u)) dW`. The axis diffuses on the unit sphere,
    `du = -2 D_r u dt + sqrt(2 D_r) (I - u u) dW_r` in the Itô form, and a boundary rescales it to unit length after
    every step against the drift of the discrete steps. The spin `psi` about the axis diffuses with `D_spin`, and its
    noise is correlated with the displacement along the axis through the coupling `D_c` of a chiral body, so that
    `<dr_par dpsi> = 2 D_c dt`, which needs `D_c² <= D_par D_spin`. It is a Janus colloid, an ellipsoid, or a
    bacterium shaped as a spiral.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particles.

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

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import AnisotropicDiffusionInteraction
        >>> AnisotropicDiffusionInteraction('diffusion', target='particles', D_perp=0.5).D_perp
        0.5
    """

    law_classes = (BoundaryLaw, NoiseLaw, RateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        D_par: Annotated[float, RealSet.open(0, oo)] = 1.0,
        D_perp: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        D_r: Annotated[float, RealSet.closed_open(0, oo)] = 0.75,
        D_spin: Annotated[float, RealSet.closed_open(0, oo)] = 0.75,
        D_c: float = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.D_par = D_par
        self.D_perp = D_perp
        self.D_r = D_r
        self.D_spin = D_spin
        self.D_c = D_c

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the translation and the rotation with their correlated noises, and the boundary of the axis.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context, context.t
            axis = [getattr(s.target, name) for name in DIRECTIONS]
            position = [getattr(s.target, name) for name in AXES]
            length = sqrt(sum(component**2 for component in axis))
            laws: list[BaseComponent] = []
            for i, (coordinate, component) in enumerate(zip(position, axis, strict=True)):
                laws.append(RateLaw(component.diff(t), -2 * s.D_r * component))
                for j, other in enumerate(axis):
                    projector = (1 if i == j else 0) - component * other
                    laws.append(NoiseLaw(coordinate, sqrt(2 * s.D_perp) * projector, process=f'translation_{j}'))
                    laws.append(NoiseLaw(coordinate, sqrt(2 * s.D_par) * component * other, process=f'translation_{j}'))
                    laws.append(NoiseLaw(component, sqrt(2 * s.D_r) * projector, process=f'rotation_{j}'))
                laws.append(BoundaryLaw(component, component / length))
            coupling = s.D_c / sqrt(s.D_par)
            for j, other in enumerate(axis):
                laws.append(NoiseLaw(s.target.psi, sqrt(2) * coupling * other, process=f'translation_{j}'))
            laws.append(NoiseLaw(s.target.psi, sqrt(2 * (s.D_spin - coupling**2)), process='spin'))
            self.laws_ = laws


class DimerElement(Element):
    """Dimers of two spheres joined by a rigid rod, described by the centre `c` and the angle `phi` of the rod.

    The spheres have the orientations `theta1` and `theta2`, along which active spheres swim, and which wrap around the
    circle.

    Args:
        label:
            The label of the dimers.

        c:
            The initial centre, one row per dimer or shared.

        phi:
            The initial angle of the rod, one per dimer or shared.

        theta1:
            The initial orientation of the first sphere, one per dimer or shared.

        theta2:
            The initial orientation of the second sphere, one per dimer or shared.

        size:
            The number of dimers, or `None` for one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import DimerElement
        >>> DimerElement('molecule', phi=1.0).phi
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        c: object = (0.0, 0.0),
        phi: object = 0.0,
        theta1: Annotated[object, Period(2 * math.pi)] = 0.0,
        theta2: Annotated[object, Period(2 * math.pi)] = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.c = c
        self.phi = phi
        self.theta1 = theta1
        self.theta2 = theta2


class RigidRodInteraction(Interaction):
    """Motion of two Brownian or active spheres joined by a rigid rod of length `L`.

    Each sphere diffuses with `D_t` and swims at the speed `v0` along its own orientation, `theta1` or `theta2`, which
    diffuse with `D_r`, and the rod keeps them at the distance `L`. Moving both spheres freely and then restoring
    their distance moves the centre `c` by the mean of their displacements and turns the rod by the part of their
    relative displacement across it, so in the limit of small steps `dc = v0 (n1 + n2) / 2 dt + sqrt(D_t) dW` and
    `dphi = v0 (n1 - n2) . e / L dt + 2 sqrt(D_t) / L dW`, with `e = (-sin phi, cos phi)` across the rod. It is a
    colloidal molecule, or a pair of swimming bacteria stuck together.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the dimers.

        L:
            The length of the rod.

        v0:
            The swimming speed of each sphere, 0 for Brownian spheres.

        D_t:
            The translational diffusion coefficient of each sphere.

        D_r:
            The rotational diffusion coefficient of the orientation of each sphere.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import RigidRodInteraction
        >>> RigidRodInteraction('rod', target='dimers', L=2.0).L
        2.0
    """

    law_classes = (NoiseLaw, RateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        L: Annotated[float, RealSet.open(0, oo)] = 1.0,
        v0: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
        D_t: Annotated[float, RealSet.closed_open(0, oo)] = 0.1,
        D_r: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.L = L
        self.v0 = v0
        self.D_t = D_t
        self.D_r = D_r

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the motion of the centre, the turning of the rod, and the diffusion of the orientations.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context, context.t
            c, phi, theta1, theta2 = s.target.c, s.target.phi, s.target.theta1, s.target.theta2
            first = vector(SR, [cos(theta1), sin(theta1)])
            second = vector(SR, [cos(theta2), sin(theta2)])
            across = vector(SR, [-sin(phi), cos(phi)])
            self.laws_ = [
                RateLaw(c, s.v0 * (first + second) / 2),
                NoiseLaw(c, sqrt(s.D_t)),
                RateLaw(phi.diff(t), s.v0 * (first - second).dot_product(across) / s.L),
                NoiseLaw(phi, 2 * sqrt(s.D_t) / s.L),
                NoiseLaw(theta1, sqrt(2 * s.D_r)),
                NoiseLaw(theta2, sqrt(2 * s.D_r)),
            ]


class AnisotropicBrownianParticlesSystem(BaseSystem):
    """Overdamped axisymmetric Brownian particles in three dimensions.

    The particles translate and rotate.

    Every particle diffuses with `D_par` along its axis `u` and with `D_perp` across it, its axis diffuses on the unit
    sphere with `D_r`, so that `<u(t) . u(0)> = exp(-2 D_r t)`, and it spins about its axis with `D_spin`. The defaults
    are a Janus sphere of unit radius, `D_par = D_perp = D_t` and `D_r = 3 D_t / 4`, whose orientation is visible
    although its diffusion is isotropic. A prolate ellipsoid has `D_par > D_perp` from Perrin's factors, so its
    displacement over times shorter than `1 / D_r` is elongated along its axis. A chiral body, such as a spiral, has a
    coupling `D_c` of the translation along its axis and the rotation about it, `<dr_par dpsi> = 2 D_c dt`. These are
    Challenges 5.1 to 5.3 of Argun et al. The axis is held as a unit vector and rescaled to unit length after every
    step, and the particles start at the origin with axes drawn uniformly on the sphere. The system has the positions
    `particles.x`, `particles.y`, `particles.z`, the axes `particles.ux`, `particles.uy`, `particles.uz`, and the spins
    `particles.psi`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Raises:
        ParameterError:
            The coupling exceeds `sqrt(D_par D_spin)`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import AnisotropicBrownianParticlesSystem
        >>> AnisotropicBrownianParticlesSystem(5, D_par=1.5, D_perp=0.8).prepare().system_.interactions[0].D_perp
        0.8
    """

    field = 'physics/soft_matter'
    topics = ('brownian motion', 'sde', 'rotational diffusion', 'three dimensions', 'anisotropy', 'chirality')
    references = ('Perrin 1934', 'Han et al. 2006', 'Argun et al. 2021')
    until = 10.0
    dt = 0.05

    def __init__(
        self,
        n_particles: int = 200,
        *,
        D_par: float = 1.0,
        D_perp: float = 1.0,
        D_r: float = 0.75,
        D_spin: float = 0.75,
        D_c: float = 0.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.D_par = D_par
        self.D_perp = D_perp
        self.D_r = D_r
        self.D_spin = D_spin
        self.D_c = D_c
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of overdamped axisymmetric Brownian particles in three dimensions.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.

        Raises:
            ParameterError:
                The coupling exceeds `sqrt(D_par D_spin)`.
        """
        super()._prepare(context)
        if self.D_c**2 > self.D_par * self.D_spin:
            error_msg = (
                f'The coupling {self.D_c!r} exceeds sqrt(D_par D_spin), so the diffusion tensor is not positive.'
            )
            raise ParameterError(error_msg)
        axes = np.random.default_rng(self.random_state).normal(size=(self.n_particles, 3))
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        particles = AnisotropicParticleElement(
            'particles',
            ux=axes[:, 0],
            uy=axes[:, 1],
            uz=axes[:, 2],
            size=self.n_particles,
        )
        diffusion = AnisotropicDiffusionInteraction(
            'diffusion',
            target='particles',
            D_par=self.D_par,
            D_perp=self.D_perp,
            D_r=self.D_r,
            D_spin=self.D_spin,
            D_c=self.D_c,
        )
        self.system_ = System(
            self.label,
            elements=[particles],
            interactions=[diffusion],
            solver=EulerMaruyamaSolver(dt=0.001),
        )


class RigidDimersSystem(BaseSystem):
    """Colloidal molecules.

    Pairs of Brownian or active spheres are joined by rigid rods, free in the plane.

    Each sphere diffuses with `D_t`, and an active sphere also swims at the speed `v0` along its own orientation, which
    diffuses with `D_r`. The rod keeps the spheres at the distance `L`. Diffusing both spheres freely and then restoring
    their distance, the scheme of Problem 5.1 of Argun et al., gives in the limit of small steps a centre that diffuses
    with `D_t / 2` and a rod angle that diffuses with `2 D_t / L²`, so the rotational diffusion of the molecule falls as
    `L^-2` in this free-draining model. Active spheres, Problem 9.3, swim together along the mean of their orientations
    and turn the rod with the part of their difference across it, which raises the long-time diffusion of the centre to
    `D_t / 2 + v0² / (4 D_r)`. The dimers start at the origin with random rod angles and orientations. The system has
    the centres `dimers.c`, the rod angles `dimers.phi`, and the orientations `dimers.theta1` and `dimers.theta2`. The
    spheres sit at `c ± (L / 2) (cos phi, sin phi)`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import RigidDimersSystem
        >>> RigidDimersSystem(5, L=2.0).prepare().system_.interactions[0].L
        2.0
    """

    field = 'physics/soft_matter'
    topics = ('brownian motion', 'sde', 'colloidal molecule', 'rotational diffusion', 'active matter')
    references = ('Argun et al. 2021',)
    until = 20.0
    dt = 0.1

    def __init__(
        self,
        n_dimers: int = 100,
        *,
        L: float = 1.0,
        v0: float = 0.0,
        D_t: float = 0.1,
        D_r: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_dimers = n_dimers
        self.L = L
        self.v0 = v0
        self.D_t = D_t
        self.D_r = D_r
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of pairs of Brownian or active spheres joined by rigid rods.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        angles = rng.uniform(0.0, 2 * np.pi, (3, self.n_dimers))
        dimers = DimerElement(
            'dimers',
            c=np.zeros((self.n_dimers, 2)),
            phi=angles[0],
            theta1=angles[1],
            theta2=angles[2],
            size=self.n_dimers,
        )
        rod = RigidRodInteraction('rod', target='dimers', L=self.L, v0=self.v0, D_t=self.D_t, D_r=self.D_r)
        self.system_ = System(
            self.label,
            elements=[dimers],
            interactions=[rod],
            space=ContinuousSpace(n_dim=2),
            solver=EulerMaruyamaSolver(dt=0.005),
        )
