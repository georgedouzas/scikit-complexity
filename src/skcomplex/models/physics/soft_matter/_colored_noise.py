"""Colored noise: a linear Langevin equation driven by Ornstein-Uhlenbeck noise of finite correlation time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np
from sage.all__sagemath_symbolics import sqrt

from ....core.base import Domain, parameter
from ....core.model import Element, Noise, Rate, System
from ....core.runtime import register_model
from ....core.solvers import EulerMaruyama

ColoredNoiseParticle = Element.define(
    'ColoredNoiseParticle',
    states={'v': 0.0, 'xi': 0.0},
    parameters={
        'gamma': parameter(default=1.0, domain=Domain.POSITIVE),
        'D': parameter(default=1.0, domain=Domain.NONNEGATIVE),
        'tau': parameter(default=1.0, domain=Domain.POSITIVE),
    },
    laws=lambda s: [
        Rate(s.v, -s.gamma * s.v + s.xi),
        Rate(s.xi, -s.xi / s.tau),
        Noise(s.xi, sqrt(2 * s.D) / s.tau),
    ],
)


@register_model(
    field='physics/soft_matter',
    tags=('noise', 'sde', 'colored noise', 'langevin equation'),
    references=('Bertin 2021', 'Hänggi and Jung 1995'),
    defaults={'until': 50.0, 'dt': 0.1, 'solver': EulerMaruyama(dt=0.01)},
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
    particles = ColoredNoiseParticle.group('particles', n_particles, xi=noise, gamma=gamma, D=D, tau=tau)
    return System(components=[particles])
