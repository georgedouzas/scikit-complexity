"""Colored noise.

A linear Langevin equation is driven by Ornstein-Uhlenbeck noise of finite correlation time.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo, sqrt
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import RateInteraction
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import EulerMaruyamaSolver
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..common.particles import RelaxationTimeInteraction


class ColoredNoiseParticleElement(Element):
    """Particle with the velocity `v`, driven by the noise `xi`.

    The noise has the correlation time `tau`.

    Args:
        label:
            The label of the particle.

        visual:
            How the particle is drawn, or `None` for none.

        v:
            The initial velocity.

        xi:
            The initial noise.

        gamma:
            The friction coefficient.

        D:
            The noise intensity.

        tau:
            The correlation time of the noise.

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ColoredNoiseParticleElement
        >>> ColoredNoiseParticleElement('particle', tau=0.5).tau
        0.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        v: Annotated[float, StateSpec()] = 0.0,
        xi: Annotated[float, StateSpec()] = 0.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.v = v
        self.xi = xi
        self.gamma = gamma
        self.D = D
        self.tau = tau


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

    Examples:
        >>> from skcomplex.systems.physics.soft_matter import ColoredNoiseLangevinSystem
        >>> ColoredNoiseLangevinSystem(10, tau=0.5).prepare().system_.elements[0].values['tau']
        0.5
    """

    field = 'physics/soft_matter'
    topics = ('noise', 'sde', 'colored noise', 'langevin equation')
    references = ('Bertin 2021', 'Hänggi and Jung 1995')
    until = 50.0
    dt = 0.1
    solver = EulerMaruyamaSolver(dt=0.01)

    def __init__(
        self,
        n_particles: int = 500,
        *,
        gamma: float = 1.0,
        D: float = 1.0,
        tau: float = 1.0,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_particles = n_particles
        self.gamma = gamma
        self.D = D
        self.tau = tau
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return particles whose velocity is driven by Ornstein-Uhlenbeck noise `xi`.

        The velocity follows `dv/dt = -gamma v + xi`.
        """
        noise = np.random.default_rng(self.random_state).normal(0.0, math.sqrt(self.D / self.tau), self.n_particles)
        particles = GroupElement(
            'particles',
            element=ColoredNoiseParticleElement,
            size=self.n_particles,
            values={'xi': noise, 'gamma': self.gamma, 'D': self.D, 'tau': self.tau},
        )
        drag = RateInteraction(
            'drag',
            target='particles',
            state='v',
            rate=lambda s: -s.target.gamma * s.target.v + s.target.xi,
        )
        noise = RelaxationTimeInteraction(
            'noise',
            target='particles',
            state='xi',
            time=lambda s: s.target.tau,
            noise=lambda s: sqrt(2 * s.target.D) / s.target.tau,
        )
        return CompositeSystem(self.label, elements=[particles], interactions=[drag, noise])
