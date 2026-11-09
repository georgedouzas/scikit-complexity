"""Rigid Brownian bodies.

The module covers axisymmetric particles in three dimensions and dimers of two spheres on a rod.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import SR, cos, oo, pi, sin, sqrt, vector
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.laws import BaseLaw, BoundaryLaw, NoiseLaw, RateLaw
from ....core.components.spaces import ContinuousSpace
from ....core.components.systems import BaseSystem, System
from ....core.engine.solvers import EulerMaruyamaSolver
from ....core.errors import ParameterError
from ....core.specs import ParameterSpec, StateSpec, VisualSpec

AXES = ('x', 'y', 'z')
"""The names of the coordinates of the position."""

DIRECTIONS = ('ux', 'uy', 'uz')
"""The names of the components of the unit vector along the axis."""


class AnisotropicParticleElement(Element):
    """Axisymmetric Brownian particle in three dimensions.

    Its diffusion tensor turns with its axis.

    The position `r = (x, y, z)` diffuses with `D_par` along the unit axis `u = (ux, uy, uz)` and with `D_perp` across
    it, `dr = (sqrt(2 D_par) u u + sqrt(2 D_perp) (I - u u)) dW`. The axis diffuses on the unit sphere,
    `du = -2 D_r u dt + sqrt(2 D_r) (I - u u) dW_r` in the Itô form, and a boundary rescales it to unit length after
    every step against the drift of the discrete steps. The spin `psi` about the axis diffuses with `D_spin`, and its
    noise is correlated with the displacement along the axis through the coupling `D_c` of a chiral body, so that
    `<dr_par dpsi> = 2 D_c dt`, which needs `D_c² <= D_par D_spin`.

    Args:
        label:
            The label of the particle or the group.

        visual:
            How the particle is drawn, or `None` for `default_visual_`.

        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        z:
            The initial third coordinate.

        ux:
            The initial first component of the axis.

        uy:
            The initial second component of the axis.

        uz:
            The initial third component of the axis.

        psi:
            The initial spin angle about the axis.

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

    Attributes:
        default_visual_:
            The visual of the particle when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import AnisotropicParticleElement
        >>> AnisotropicParticleElement('janus').uz
        1.0
    """

    law_classes = (
        BoundaryLaw,
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
        z: Annotated[float, StateSpec()] = 0.0,
        ux: Annotated[float, StateSpec()] = 0.0,
        uy: Annotated[float, StateSpec()] = 0.0,
        uz: Annotated[float, StateSpec()] = 1.0,
        psi: Annotated[float, StateSpec()] = 0.0,
        D_par: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        D_perp: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        D_r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.75,
        D_spin: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.75,
        D_c: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.z = z
        self.ux = ux
        self.uy = uy
        self.uz = uz
        self.psi = psi
        self.D_par = D_par
        self.D_perp = D_perp
        self.D_r = D_r
        self.D_spin = D_spin
        self.D_c = D_c

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the translation and the rotation with their correlated noises.

        The laws also hold the boundary of the axis.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        axis = [getattr(s, name) for name in DIRECTIONS]
        position = [getattr(s, name) for name in AXES]
        length = sqrt(sum(component**2 for component in axis))
        laws: list[BaseLaw] = []
        for i, (coordinate, component) in enumerate(zip(position, axis, strict=True)):
            laws.append(RateLaw(component, -2 * s.D_r * component))
            for j, other in enumerate(axis):
                projector = (1 if i == j else 0) - component * other
                laws.append(NoiseLaw(coordinate, sqrt(2 * s.D_perp) * projector, process=f'translation_{j}'))
                laws.append(NoiseLaw(coordinate, sqrt(2 * s.D_par) * component * other, process=f'translation_{j}'))
                laws.append(NoiseLaw(component, sqrt(2 * s.D_r) * projector, process=f'rotation_{j}'))
            laws.append(BoundaryLaw(component, component / length))
        coupling = s.D_c / sqrt(s.D_par)
        for j, other in enumerate(axis):
            laws.append(NoiseLaw(s.psi, sqrt(2) * coupling * other, process=f'translation_{j}'))
        laws.append(NoiseLaw(s.psi, sqrt(2 * (s.D_spin - coupling**2)), process='spin'))
        return laws


class DimerElement(Element):
    """Two Brownian or active spheres joined by a rigid rod.

    The dimer is described by the centre and the angle of the rod.

    Each sphere diffuses with `D_t` and swims at the speed `v0` along its own orientation, `theta1` or `theta2`, which
    diffuse with `D_r`, and the rod keeps them at the distance `L`. Moving both spheres freely and then restoring
    their distance moves the centre `c` by the mean of their displacements and turns the rod by the part of their
    relative displacement across it, so in the limit of small steps `dc = v0 (n1 + n2) / 2 dt + sqrt(D_t) dW` and
    `dphi = v0 (n1 - n2) . e / L dt + 2 sqrt(D_t) / L dW`, with `e = (-sin phi, cos phi)` across the rod.

    Args:
        label:
            The label of the dimer or the group.

        visual:
            How the dimer is drawn, or `None` for `default_visual_`.

        c:
            The initial centre.

        phi:
            The initial angle of the rod.

        theta1:
            The initial orientation of the first sphere.

        theta2:
            The initial orientation of the second sphere.

        L:
            The length of the rod.

        v0:
            The swimming speed of each sphere, 0 for Brownian spheres.

        D_t:
            The translational diffusion coefficient of each sphere.

        D_r:
            The rotational diffusion coefficient of the orientation of each sphere.

    Attributes:
        default_visual_:
            The visual of the dimer when it sets none.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import DimerElement
        >>> DimerElement('molecule').L
        1.0
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
        c: Annotated[npt.ArrayLike, StateSpec(vector=True, role='position')] = 0.0,
        phi: Annotated[float, StateSpec(role='direction')] = 0.0,
        theta1: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
        theta2: Annotated[float, StateSpec(period=2 * pi)] = 0.0,
        L: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        v0: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        D_t: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        D_r: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.c = c
        self.phi = phi
        self.theta1 = theta1
        self.theta2 = theta2
        self.L = L
        self.v0 = v0
        self.D_t = D_t
        self.D_r = D_r

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the motion of the centre, the turning of the rod, and the diffusion.

        The diffusion is that of the orientations.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        first = vector(SR, [cos(s.theta1), sin(s.theta1)])
        second = vector(SR, [cos(s.theta2), sin(s.theta2)])
        across = vector(SR, [-sin(s.phi), cos(s.phi)])
        return [
            RateLaw(s.c, s.v0 * (first + second) / 2),
            NoiseLaw(s.c, sqrt(s.D_t)),
            RateLaw(s.phi, s.v0 * (first - second).dot_product(across) / s.L),
            NoiseLaw(s.phi, 2 * sqrt(s.D_t) / s.L),
            NoiseLaw(s.theta1, sqrt(2 * s.D_r)),
            NoiseLaw(s.theta2, sqrt(2 * s.D_r)),
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

    Raises:
        ParameterError:
            The coupling exceeds `sqrt(D_par D_spin)`.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import AnisotropicBrownianParticlesSystem
        >>> AnisotropicBrownianParticlesSystem(5, D_par=1.5, D_perp=0.8).prepare().system_.elements[0].values['D_perp']
        0.8
    """

    field = 'physics/soft_matter'
    topics = ('brownian motion', 'sde', 'rotational diffusion', 'three dimensions', 'anisotropy', 'chirality')
    references = ('Perrin 1934', 'Han et al. 2006', 'Argun et al. 2021')
    until = 10.0
    dt = 0.05
    solver = EulerMaruyamaSolver(dt=0.001)

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
    ) -> None:
        super().__init__(label)
        self.n_particles = n_particles
        self.D_par = D_par
        self.D_perp = D_perp
        self.D_r = D_r
        self.D_spin = D_spin
        self.D_c = D_c
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return overdamped axisymmetric Brownian particles in three dimensions.

        The particles translate and rotate.
        """
        if self.D_c**2 > self.D_par * self.D_spin:
            error_msg = (
                f'The coupling {self.D_c!r} exceeds sqrt(D_par D_spin), so the diffusion tensor is not positive.'
            )
            raise ParameterError(error_msg)
        axes = np.random.default_rng(self.random_state).normal(size=(self.n_particles, 3))
        axes /= np.linalg.norm(axes, axis=1, keepdims=True)
        values = {
            'ux': axes[:, 0],
            'uy': axes[:, 1],
            'uz': axes[:, 2],
            'D_par': self.D_par,
            'D_perp': self.D_perp,
            'D_r': self.D_r,
            'D_spin': self.D_spin,
            'D_c': self.D_c,
        }
        particles = GroupElement('particles', element=AnisotropicParticleElement, size=self.n_particles, values=values)
        return System(self.label, elements=[particles])


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

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import RigidDimersSystem
        >>> RigidDimersSystem(5, L=2.0).prepare().system_.elements[0].values['L']
        2.0
    """

    field = 'physics/soft_matter'
    topics = ('brownian motion', 'sde', 'colloidal molecule', 'rotational diffusion', 'active matter')
    references = ('Argun et al. 2021',)
    until = 20.0
    dt = 0.1
    solver = EulerMaruyamaSolver(dt=0.005)

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
    ) -> None:
        super().__init__(label)
        self.n_dimers = n_dimers
        self.L = L
        self.v0 = v0
        self.D_t = D_t
        self.D_r = D_r
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return colloidal molecules.

        Pairs of Brownian or active spheres are joined by rigid rods, free in the plane.
        """
        rng = np.random.default_rng(self.random_state)
        angles = rng.uniform(0.0, 2 * np.pi, (3, self.n_dimers))
        values = {
            'c': np.zeros((self.n_dimers, 2)),
            'phi': angles[0],
            'theta1': angles[1],
            'theta2': angles[2],
            'L': self.L,
            'v0': self.v0,
            'D_t': self.D_t,
            'D_r': self.D_r,
        }
        dimers = GroupElement('dimers', element=DimerElement, size=self.n_dimers, values=values)
        return System(self.label, elements=[dimers], space=ContinuousSpace(n_dim=2))
