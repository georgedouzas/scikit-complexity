"""Celestial mechanics models: orbits of bodies under gravity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import itertools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
from sage.all__sagemath_symbolics import cos

from ....core.base import Domain, ValidationError, parameter, state
from ....core.model import Element, EuclideanSpace, Gravity, Particle, Rate, System, derivative
from ....core.runtime import register_model

FIGURE_EIGHT = {
    'G': 1.0,
    'masses': [1.0, 1.0, 1.0],
    'positions': [[-0.97000436, 0.24308753], [0.97000436, -0.24308753], [0.0, 0.0]],
    'velocities': [[0.466203685, 0.43236573], [0.466203685, 0.43236573], [-0.93240737, -0.86473146]],
}
INNER_SOLAR_SYSTEM = {
    'G': 4 * math.pi**2,
    'masses': [1.0, 1.66e-7, 2.45e-6, 3.0e-6, 3.23e-7],
    'positions': [[0.0, 0.0], [0.387, 0.0], [0.723, 0.0], [1.0, 0.0], [1.524, 0.0]],
    'velocities': [
        [0.0, 0.0],
        [0.0, 2 * math.pi / math.sqrt(0.387)],
        [0.0, 2 * math.pi / math.sqrt(0.723)],
        [0.0, 2 * math.pi],
        [0.0, 2 * math.pi / math.sqrt(1.524)],
    ],
}
PRESETS = {'figure_eight': FIGURE_EIGHT, 'inner_solar_system': INNER_SOLAR_SYSTEM}

RotatingFrameBody = Element.define(
    'RotatingFrameBody',
    states={'x': state(default=0.5, order=2), 'y': state(default=0.0, order=2)},
    parameters={'mu': parameter(default=0.01215, domain=Domain.POSITIVE)},
    laws=lambda s: [
        Rate(
            s.x,
            2 * derivative(s.y, 1)
            + s.x
            - (1 - s.mu) * (s.x + s.mu) / ((s.x + s.mu) ** 2 + s.y**2) ** (3 / 2)
            - s.mu * (s.x - 1 + s.mu) / ((s.x - 1 + s.mu) ** 2 + s.y**2) ** (3 / 2),
        ),
        Rate(
            s.y,
            -2 * derivative(s.x, 1)
            + s.y
            - (1 - s.mu) * s.y / ((s.x + s.mu) ** 2 + s.y**2) ** (3 / 2)
            - s.mu * s.y / ((s.x - 1 + s.mu) ** 2 + s.y**2) ** (3 / 2),
        ),
    ],
)


def _preset(name: str) -> dict[str, Any]:
    """Return the masses, positions, velocities, and gravitational constant of a named configuration."""
    if name not in PRESETS:
        error_msg = f'Unknown preset {name!r}. Use one of {", ".join(sorted(PRESETS))}.'
        raise ValidationError(error_msg)
    return PRESETS[name]


@register_model(
    field='physics/celestial_mechanics',
    tags=('mechanics', 'ode', 'gravity'),
    references=('Kepler 1609', 'Newton 1687'),
    defaults={'until': 6.0, 'dt': 0.01, 'solver': 'rk4'},
)
def two_body_orbit(
    eccentricity: float = 0.0,
    *,
    G: float = 1.0,
    mass: float = 1000.0,
    radius: float = 10.0,
) -> System:
    """Return a light planet orbiting a heavy star under Newtonian gravity.

    Args:
        eccentricity:
            The eccentricity of the orbit, between 0 for a circle and 1 for an escape.

        G:
            The gravitational constant.

        mass:
            The mass of the star.

        radius:
            The distance of the planet from the star at its closest approach.

    Returns:
        system:
            The model, with the particles `star` and `planet` in two dimensions.

    Examples:
        >>> from skcomplex.models.physics.celestial_mechanics import two_body_orbit
        >>> [element.label for element in two_body_orbit().elements]
        ['star', 'planet']
    """
    speed = float(np.sqrt(G * mass * (1 + eccentricity) / radius))
    return System(
        elements=[Particle('star', m=mass), Particle('planet', x=[radius, 0.0], v=[0.0, speed], m=1e-6)],
        interactions=[Gravity('gravity', first='star', second='planet', G=G)],
        space=EuclideanSpace(2),
    )


@register_model(
    field='physics/celestial_mechanics',
    tags=('mechanics', 'ode', 'gravity', 'chaos'),
    references=('Argun et al. 2021', 'Dilão 2023', 'Chenciner and Montgomery 2000'),
    defaults={'until': 6.33, 'dt': 0.01},
)
def n_body(
    preset: str = 'figure_eight',
    *,
    masses: Sequence[float] | None = None,
    positions: Sequence[Sequence[float]] | None = None,
    velocities: Sequence[Sequence[float]] | None = None,
    G: float | None = None,
) -> System:
    """Return bodies in the plane that attract each other pairwise by Newtonian gravity.

    The preset `'figure_eight'` is the periodic orbit of three equal masses found by Chenciner and Montgomery, with
    period 6.3259 for `G = 1`. The preset `'inner_solar_system'` is the Sun with Mercury, Venus, the Earth, and Mars on
    circular orbits, in astronomical units and years, with `G = 4 π²`. Masses, positions, velocities, or `G` given
    explicitly replace those of the preset.

    Args:
        preset:
            `'figure_eight'` or `'inner_solar_system'`.

        masses:
            The masses of the bodies, or `None` for those of the preset.

        positions:
            The initial positions of the bodies, or `None` for those of the preset.

        velocities:
            The initial velocities of the bodies, or `None` for those of the preset.

        G:
            The gravitational constant, or `None` for that of the preset.

    Returns:
        system:
            The model, with the particles `body_<i>` in two dimensions.

    Raises:
        ValidationError:
            The preset is unknown.

    Examples:
        >>> from skcomplex.models.physics.celestial_mechanics import n_body
        >>> len(n_body('inner_solar_system').interactions)
        10
    """
    configuration = _preset(preset)
    masses = configuration['masses'] if masses is None else masses
    positions = configuration['positions'] if positions is None else positions
    velocities = configuration['velocities'] if velocities is None else velocities
    G = configuration['G'] if G is None else G
    bodies = [
        Particle(f'body_{index}', x=list(position), v=list(velocity), m=mass)
        for index, (mass, position, velocity) in enumerate(zip(masses, positions, velocities, strict=True))
    ]
    attractions = [
        Gravity(f'gravity_{first}_{second}', first=f'body_{first}', second=f'body_{second}', G=G)
        for first, second in itertools.combinations(range(len(bodies)), 2)
    ]
    return System(elements=bodies, interactions=attractions, space=EuclideanSpace(2))


@register_model(
    field='physics/celestial_mechanics',
    tags=('mechanics', 'ode', 'second order', 'gravity', 'chaos', 'rotating frame'),
    references=('Dilão 2023', 'Szebehely 1967'),
    defaults={'until': 20.0, 'dt': 0.02},
)
def restricted_three_body(
    x: float = 0.5,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.8,
    *,
    mu: float = 0.01215,
) -> System:
    """Return a light body moving in the plane of two primaries on circular orbits, in the frame that rotates with them.

    The primaries of masses `1 - μ` and `μ` sit at `(-μ, 0)` and `(1 - μ, 0)`, and the light body follows
    `x'' = 2 y' + x - (1 - μ)(x + μ) / r1³ - μ (x - 1 + μ) / r2³` and `y'' = -2 x' + y - (1 - μ) y / r1³ - μ y / r2³`,
    with the Coriolis and centrifugal forces of the rotating frame. The Jacobi constant
    `x² + y² + 2 (1 - μ) / r1 + 2 μ / r2 - x'² - y'²` is conserved. The default `μ` is that of the Earth and the Moon.

    Args:
        x:
            The initial first coordinate.

        y:
            The initial second coordinate.

        vx:
            The initial first velocity.

        vy:
            The initial second velocity.

        mu:
            The mass of the lighter primary, as a fraction of the total.

    Returns:
        system:
            The model, with the coordinates `body.x` and `body.y` of order 2.

    Examples:
        >>> from skcomplex.models.physics.celestial_mechanics import restricted_three_body
        >>> restricted_three_body(mu=0.1).elements[0].mu
        0.1
    """
    return System(elements=[RotatingFrameBody('body', x=x, y=y, x_d1=vx, y_d1=vy, mu=mu)])


SitnikovBody = Element.define(
    'SitnikovBody',
    states={'z': state(default=0.5, order=2)},
    parameters={'e': parameter(default=0.0, domain=Domain.NONNEGATIVE)},
    laws=lambda s: [Rate(s.z, -s.z / (s.z**2 + ((1 - s.e * cos(s.t)) / 2) ** 2) ** (3 / 2))],
)


@register_model(
    field='physics/celestial_mechanics',
    tags=('mechanics', 'ode', 'second order', 'gravity', 'chaos'),
    references=('Dilão 2023', 'Sitnikov 1960'),
    defaults={'until': 50.0, 'dt': 0.05},
)
def sitnikov(z: float = 0.5, vz: float = 0.0, *, e: float = 0.0) -> System:
    """Return a light body moving on the axis through the centre of mass of two equal primaries, normal to their orbit.

    The primaries of mass 1/2 orbit their centre of mass with unit mean separation and period `2π`, at distance
    `r(t) = (1 - e cos t) / 2` from it to first order in the eccentricity `e`, and the light body follows
    `z'' = -z / (z² + r(t)²)^(3/2)`. For circular primaries the energy `z'²/2 - 1 / √(z² + 1/4)` is conserved and the
    motion is integrable, and eccentric primaries make bounded orbits chaotic.

    Args:
        z:
            The initial height above the plane of the primaries.

        vz:
            The initial vertical velocity.

        e:
            The eccentricity of the primaries' orbits.

    Returns:
        system:
            The model, with the height `body.z` of order 2.

    Examples:
        >>> from skcomplex.models.physics.celestial_mechanics import sitnikov
        >>> sitnikov(e=0.1).elements[0].e
        0.1
    """
    return System(elements=[SitnikovBody('body', z=z, z_d1=vz, e=e)])
