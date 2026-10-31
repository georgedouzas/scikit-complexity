"""Colored noise: a linear Langevin equation driven by Ornstein-Uhlenbeck noise of finite correlation time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo, sqrt
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import RateInteraction, RelaxationInteraction
from ....core.system import System
from ....engine.solvers import EulerMaruyamaSolver
from ....specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_model


class ColoredNoiseParticleElement(Element):
    """Particle with the velocity `v`, driven by the noise `xi` of correlation time `tau`.

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
        >>> from skcomplex.models.physics.soft_matter import ColoredNoiseParticleElement
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


@register_model(
    field='physics/soft_matter',
    tags=('noise', 'sde', 'colored noise', 'langevin equation'),
    references=('Bertin 2021', 'Hänggi and Jung 1995'),
    defaults={'until': 50.0, 'dt': 0.1, 'solver': EulerMaruyamaSolver(dt=0.01)},
)
def colored_noise_langevin(
    n_particles: int = 500,
    *,
    gamma: float = 1.0,
    D: float = 1.0,
    tau: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return particles whose velocity `dv/dt = -gamma v + xi` is driven by Ornstein-Uhlenbeck noise `xi`.

    The noise follows `tau dxi = -xi dt + sqrt(2 D) dW`, so its correlation is `<xi(t) xi(t')> = D / tau
    exp(-|t - t'| / tau)`, and it tends to the white noise of intensity `2 D` as `tau` goes to 0. The stationary
    velocity variance is `<v^2> = D / (gamma (1 + gamma tau))`, which is `D / gamma` in the white noise limit. The
    velocities start at 0 and the noise starts in its stationary state. Solver steps must be short against `tau`.

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

    Returns:
        system:
            The model, with the velocities `particles.v` and the noise `particles.xi`.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import colored_noise_langevin
        >>> colored_noise_langevin(10, tau=0.5).elements[0].values['tau']
        0.5
    """
    noise = np.random.default_rng(random_state).normal(0.0, math.sqrt(D / tau), n_particles)
    particles = GroupElement(
        'particles',
        element=ColoredNoiseParticleElement,
        size=n_particles,
        values={'xi': noise, 'gamma': gamma, 'D': D, 'tau': tau},
    )
    drag = RateInteraction(
        'drag',
        target='particles',
        state='v',
        rate=lambda s: -s.target.gamma * s.target.v + s.target.xi,
    )
    noise = RelaxationInteraction(
        'noise',
        target='particles',
        state='xi',
        time=lambda s: s.target.tau,
        noise=lambda s: sqrt(2 * s.target.D) / s.target.tau,
    )
    return System(elements=[particles], interactions=[drag, noise])
