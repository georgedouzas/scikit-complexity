"""Deformable active particles.

The shape and velocity of self-propelled particles drive each other.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import SR, identity_matrix, matrix, oo, sqrt, vector
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import NoiseLaw, RateLaw
from ....core.components.spaces import BaseSpace, ContinuousSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, EulerMaruyamaSolver
from ....core.specs import Role
from ....core.systems import BaseSystem, System
from ..common.particles import POINTS


class DeformableParticleElement(Element):
    """Particles with the positions `x`, the velocities `v`, and the deformations `S`.

    The deformation is a traceless symmetric tensor `[[S_xx, S_xy], [S_xy, -S_xx]]`, given by its two independent
    components `(S_xx, S_xy)`.

    Args:
        label:
            The label of the particles.

        x:
            The initial positions, one row per particle or shared.

        v:
            The initial velocities, one row per particle or shared.

        S:
            The initial deformations, given by their two independent components, one row per particle or shared.

        size:
            The number of particles, or `None` for one.

        space:
            The continuous space of the particles, or `None` for none.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import DeformableParticleElement
        >>> DeformableParticleElement('cell', v=(1.0, 0.0)).v
        (1.0, 0.0)
    """

    visual = POINTS

    def __init__(
        self,
        label: str,
        *,
        x: object = (0.0, 0.0),
        v: object = (0.0, 0.0),
        S: object = (0.0, 0.0),
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.x = x
        self.v = v
        self.S = S


class ShapeVelocityCouplingInteraction(Interaction):
    """Coupling of the velocity and the deformation of self-propelled particles, the model of Ohta and Ohkuma.

    The particle moves with its velocity `v`, which follows `v' = gamma v - |v|² v - a S v`, and its deformation, the
    traceless symmetric tensor `S`, follows `S' = -kappa S + b (v v - |v|² I / 2)`. The motion stretches the particle
    along the velocity for `b > 0`, and the deformation turns the velocity back. Noise of amplitude `noise` may kick
    the velocity. It is a crawling cell, a droplet, or a camphor boat that changes shape as it moves.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particles, with the positions `x`, the velocities `v`, and the deformations `S`.

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

    Examples:
        >>> from skcomplex.systems.physics.active_matter import ShapeVelocityCouplingInteraction
        >>> ShapeVelocityCouplingInteraction('coupling', target='cells').kappa
        0.5
    """

    law_classes = (NoiseLaw, RateLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        gamma: float = 1.0,
        a: float = 1.0,
        b: float = 1.0,
        kappa: Annotated[float, RealSet.open(0, oo)] = 0.5,
        noise: Annotated[float, RealSet.closed_open(0, oo)] = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.gamma = gamma
        self.a = a
        self.b = b
        self.kappa = kappa
        self.noise = noise

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the motion, the velocity and its noise, and the deformation of the target.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s = context
            x, v, S = s.target.x, s.target.v, s.target.S
            tensor = matrix(SR, [[S[0], S[1]], [S[1], -S[0]]])
            speed = v.dot_product(v)
            stretch = v.outer_product(v) - speed / 2 * identity_matrix(SR, 2)
            deformation = -s.kappa * tensor + s.b * stretch
            self.laws_ = [
                RateLaw(x, v),
                RateLaw(v, s.gamma * v - speed * v - s.a * (tensor * v)),
                NoiseLaw(v, sqrt(2) * s.noise),
                RateLaw(S, vector(SR, [deformation[0, 0], deformation[0, 1]])),
            ]


class DeformableActiveParticlesSystem(BaseSystem):
    """Deformable self-propelled particles of Ohta and Ohkuma.

    The particles move straight or along circles.

    Every particle follows `x' = v`, `v' = gamma v - |v|² v - a S v`, and `S' = -kappa S + b (v v - |v|² I / 2)`, with
    the traceless symmetric deformation tensor `S`. A fast relaxing shape, large `kappa`, gives straight motion at the
    speed `sqrt(gamma / (1 + a b / (2 kappa)))`. For `a b (gamma - kappa / 2) > kappa²` a circular motion exists, at the
    speed `sqrt(gamma - kappa / 2)` with the angular velocity `sqrt(a b (gamma - kappa / 2) - kappa²) / 2`, its
    elongation turning with its velocity. For the defaults it exists below `kappa ≈ 0.78`, and particles started from
    small velocities reach it for `kappa <= 0.5`, while closer to the threshold straight motion stays stable too. The
    particles do not interact, and they start at random positions, small random velocities, and no deformation. The
    system has the positions `cells.x`, the velocities `cells.v`, and the deformations `cells.S`.

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

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.active_matter import DeformableActiveParticlesSystem
        >>> DeformableActiveParticlesSystem(5).prepare().system_.elements[0].size
        5
    """

    field = 'physics/active_matter'
    topics = ('active matter', 'deformable particles', 'tensor', 'circular motion', 'bifurcation')
    references = ('Ohta and Ohkuma 2009', 'Ohta 2017', 'Tarama 2017')
    until = 100.0
    dt = 0.25

    def __init__(
        self,
        n_particles: int = 20,
        *,
        gamma: float = 1.0,
        a: float = 1.0,
        b: float = 1.0,
        kappa: float = 0.5,
        noise: float = 0.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.gamma = gamma
        self.a = a
        self.b = b
        self.kappa = kappa
        self.noise = noise
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of deformable self-propelled particles of Ohta and Ohkuma.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        cells = DeformableParticleElement(
            'cells',
            x=rng.uniform(0.0, 20.0, (self.n_particles, 2)),
            v=rng.normal(0.0, 0.1, (self.n_particles, 2)),
            S=np.zeros((self.n_particles, 2)),
            size=self.n_particles,
            space=ContinuousSpace(n_dim=2),
        )
        coupling = ShapeVelocityCouplingInteraction(
            'coupling',
            target='cells',
            gamma=self.gamma,
            a=self.a,
            b=self.b,
            kappa=self.kappa,
            noise=self.noise,
        )
        self.system_ = System(
            self.label,
            elements=[cells],
            interactions=[coupling],
            solver=EulerMaruyamaSolver(dt=0.01),
        )
