"""Colored noise: a linear Langevin equation driven by Ornstein-Uhlenbeck noise of finite correlation time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

import numpy as np
from sage.all__sagemath_symbolics import sqrt

from skcomplex.core.declarations import state

from ....core.base import NumericDomain, parameter
from ....core.components import Element
from ....core.laws import NoiseLaw, RateLaw
from ....core.system import System
from ....engine.runtime import register_model
from ....engine.solvers import EulerMaruyamaSolver
from ...core.base import BaseLaw
from ...core.elements import GroupElement
from ...core.laws import Symbols


class ColoredNoiseParticleElement(Element):
    """Colored noise particle element."""

    v = state(0.0)
    xi = state(0.0)
    gamma = parameter(default=1.0, domain=NumericDomain.POSITIVE)
    D = parameter(default=1.0, domain=NumericDomain.NONNEGATIVE)
    tau = parameter(default=1.0, domain=NumericDomain.POSITIVE)

    def laws(self, s: Symbols) -> list[BaseLaw]:
        """Return the laws of the colored noise particle element."""
        return [
            RateLaw(s.v, -s.gamma * s.v + s.xi),
            RateLaw(s.xi, -s.xi / s.tau),
            NoiseLaw(s.xi, sqrt(2 * s.D) / s.tau),
        ]


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
    return System(elements=[particles])
