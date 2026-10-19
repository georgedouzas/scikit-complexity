"""Soft matter models: Brownian particles and noise-driven transitions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math

from sage.all__sagemath_symbolics import cos, sqrt

from ....core.base import Domain, parameter, state
from ....core.model import Boundary, Element, Noise, Rate, System
from ....core.runtime import register_model

DoubleWell = Element.define(
    'DoubleWell',
    states={'x': -1.0},
    parameters={
        'amplitude': 0.1,
        'frequency': 0.01,
        'D': parameter(default=0.12, domain=Domain.NONNEGATIVE),
    },
    laws=lambda s: [
        Rate(s.x, s.x - s.x**3 + s.amplitude * cos(2 * math.pi * s.frequency * s.t)),
        Noise(s.x, sqrt(2 * s.D)),
    ],
)


@register_model(
    field='physics/soft_matter',
    tags=('noise', 'sde', 'forcing', 'resonance'),
    references=('Benzi, Sutera, and Vulpiani 1981', 'Gammaitoni et al. 1998'),
    defaults={'until': 500.0, 'dt': 0.5},
)
def stochastic_resonance(
    x: float = -1.0,
    *,
    amplitude: float = 0.1,
    frequency: float = 0.01,
    D: float = 0.12,
) -> System:
    """Return a particle in a double well, `dx = (x - x^3 + A cos(2 pi f t)) dt + sqrt(2 D) dW`.

    The forcing alone is too weak to move the particle between the wells, but noise of the right intensity makes the
    hops follow the forcing.

    Args:
        x:
            The initial position.

        amplitude:
            The amplitude of the periodic forcing.

        frequency:
            The frequency of the forcing.

        D:
            The noise intensity.

    Returns:
        system:
            The model, with the position `particle.x`.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import stochastic_resonance
        >>> stochastic_resonance(D=0.2).elements[0].D
        0.2
    """
    return System(elements=[DoubleWell('particle', x=x, amplitude=amplitude, frequency=frequency, D=D)])


WallParticle = Element.define(
    'WallParticle',
    states={'z': state(default=1.0, domain=Domain.NONNEGATIVE)},
    parameters={
        'D0': parameter(default=1.0, domain=Domain.POSITIVE),
        'a': parameter(default=1.0, domain=Domain.POSITIVE),
        'drift': 0.2,
    },
    laws=lambda s: [
        Rate(s.z, -s.drift * s.z * abs(s.z) / (s.z**2 + s.a**2)),
        Noise(s.z, sqrt(2 * s.D0) * s.z / sqrt(s.z**2 + s.a**2)),
        Boundary(s.z, abs(s.z)),
    ],
)


@register_model(
    field='physics/soft_matter',
    tags=('brownian motion', 'sde', 'multiplicative noise', 'hydrodynamics'),
    references=('Brenner 1961', 'Volpe and Wehr 2016'),
    defaults={'until': 10.0, 'dt': 0.1},
)
def brownian_near_wall(
    z: float = 1.0,
    *,
    D0: float = 1.0,
    a: float = 1.0,
    drift: float = 0.2,
    noise: str = 'anti-ito',
) -> System:
    """Return a Brownian particle whose diffusion slows near a wall, `D(z) = D0 z² / (z² + a²)`, as it settles on it.

    The diffusion is a smooth form of the hydrodynamic hindrance that vanishes at the wall, and the drift is the
    settling velocity scaled by the same mobility. Diffusion that depends on the position makes the noise
    multiplicative, and the anti-Itô convention gives the equilibrium of physical particles. The laws are symmetric
    about the wall at `z = 0`, which reflects the particle at every frame.

    Args:
        z:
            The initial distance from the wall.

        D0:
            The diffusion coefficient far from the wall.

        a:
            The distance over which the wall slows diffusion, such as the particle radius.

        drift:
            The settling velocity far from the wall.

        noise:
            The convention of the noise, `'ito'`, `'stratonovich'`, or `'anti-ito'`.

    Returns:
        system:
            The model, with the distance `particle.z`.

    Examples:
        >>> from skcomplex.models.physics.soft_matter import brownian_near_wall
        >>> brownian_near_wall(noise='ito').noise
        'ito'
    """
    return System(elements=[WallParticle('particle', z=z, D0=D0, a=a, drift=drift)], noise=noise)
