"""Colored noise.

A linear Langevin equation is driven by Ornstein-Uhlenbeck noise of finite correlation time.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import NoiseLaw, RateLaw
from ....core.components.spaces import BaseSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, EulerMaruyamaSolver
from ....core.specs import Role
from ....core.systems import BaseSystem, System


class ColoredNoiseParticleElement(Element):
    """Particles with the velocities `v`, driven by the noise `xi`.

    Args:
        label:
            The label of the particles.

        v:
            The initial velocity, one per particle or shared.

        xi:
            The initial noise, one per particle or shared.

        size:
            The number of particles, or `None` for one.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ColoredNoiseParticleElement
        >>> ColoredNoiseParticleElement('particle', xi=0.5).xi
        0.5
    """

    def __init__(self, label: str, *, v: object = 0.0, xi: object = 0.0, size: int | None = None) -> None:
        super().__init__(label, size=size)
        self.v = v
        self.xi = xi


class ColoredNoiseInteraction(Interaction):
    """Friction and Ornstein-Uhlenbeck noise `xi` of finite correlation time on the velocity `v`.

    The velocity follows `dv/dt = -gamma v + xi` and the noise `tau dxi = -xi dt + sqrt(2 D) dW`. It is a colloid in an
    active bath, or a particle pushed by a fluctuating field with a memory.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particles.

        gamma:
            The friction coefficient.

        D:
            The noise intensity.

        tau:
            The correlation time of the noise.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ColoredNoiseInteraction
        >>> ColoredNoiseInteraction('langevin', target='particles', tau=0.5).tau
        0.5
    """

    law_classes = (RateLaw, NoiseLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        gamma: Annotated[float, RealSet.open(0, oo)] = 1.0,
        D: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        tau: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.gamma = gamma
        self.D = D
        self.tau = tau

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the rates of the velocity and the noise of the target, and the noise of the noise.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context, context.t
            v, xi = s.target.v, s.target.xi
            self.laws_ = [
                RateLaw(v.diff(t), -s.gamma * v + xi),
                RateLaw(xi.diff(t), -xi / s.tau),
                NoiseLaw(xi, sqrt(2 * s.D) / s.tau),
            ]


class ColoredNoiseLangevinSystem(BaseSystem):
    """Particles whose velocity is driven by Ornstein-Uhlenbeck noise `xi`.

    The velocity follows `dv/dt = -gamma v + xi`.

    The noise follows `tau dxi = -xi dt + sqrt(2 D) dW`, so its correlation is `<xi(t) xi(t')> = D / tau exp(-|t - t'| /
    tau)`, and it tends to the white noise of intensity `2 D` as `tau` goes to 0. The stationary velocity variance is
    `<v^2> = D / (gamma (1 + gamma tau))`, which is `D / gamma` in the white noise limit. The velocities start at 0 and
    the noise starts in its stationary state. Solver steps must be short against `tau`. The system has the velocities
    `particles.v` and the noise `particles.xi`.

    Args:
        n_particles:
            The number of independent particles.

        gamma:
            The friction coefficient.

        D:
            The noise intensity.

        tau:
            The correlation time of the noise.

        random_state:
            The seed or generator of the initial noise values.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ColoredNoiseLangevinSystem
        >>> ColoredNoiseLangevinSystem(10, tau=0.5).prepare().system_.interactions[0].tau
        0.5
    """

    field = 'physics/soft_matter'
    topics = ('noise', 'sde', 'colored noise', 'langevin equation')
    references = ('Bertin 2021', 'Hänggi and Jung 1995')
    until = 50.0
    dt = 0.1

    def __init__(
        self,
        n_particles: int = 500,
        *,
        gamma: float = 1.0,
        D: float = 1.0,
        tau: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_particles = n_particles
        self.gamma = gamma
        self.D = D
        self.tau = tau
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of particles whose velocity is driven by Ornstein-Uhlenbeck noise.

        Args:
            context:
                The observables to compile with the problem, or `None` for none.
        """
        super()._prepare(context)
        noise = np.random.default_rng(self.random_state).normal(0.0, math.sqrt(self.D / self.tau), self.n_particles)
        self.system_ = System(
            self.label,
            elements=[ColoredNoiseParticleElement('particles', xi=noise, size=self.n_particles)],
            interactions=[
                ColoredNoiseInteraction('langevin', target='particles', gamma=self.gamma, D=self.D, tau=self.tau),
            ],
            solver=EulerMaruyamaSolver(dt=0.01),
        )
