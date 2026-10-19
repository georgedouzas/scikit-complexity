"""Celestial mechanics models: orbits of bodies under gravity."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np

from ....core.model import EuclideanSpace, Gravity, Particle, System
from ....core.runtime import register_model


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
