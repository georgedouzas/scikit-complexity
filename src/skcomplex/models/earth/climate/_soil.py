"""Coupled maps of the temperatures of the soil surface and of the deeper soil."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from ....core.model import Element, System, Update
from ....core.runtime import register_model

SoilTemperature = Element.define(
    'SoilTemperature',
    states={'x': 0.3, 'y': 0.1},
    parameters={'r': 3.8, 'epsilon': 0.1},
    laws=lambda s: [
        Update(s.x, s.r * s.x * (1 - s.x) + s.epsilon * s.y),
        Update(s.y, s.epsilon * (s.x + s.y)),
    ],
)


@register_model(
    field='earth/climate',
    tags=('discrete map', 'coupled maps', 'chaos', 'land surface'),
    references=('Mimić, Mihailović, and Budinčević 2013', 'Mihailović et al. 2023'),
    defaults={'until': 200, 'dt': 1},
)
def soil_temperature(x: float = 0.3, y: float = 0.1, *, r: float = 3.8, epsilon: float = 0.1) -> System:
    """Return the coupled maps of the soil surface temperature and the deeper soil temperature.

    The dimensionless surface temperature `x` follows the energy balance of the ground, reduced to a logistic map,
    and the deeper soil temperature `y` relaxes towards it: `x_{n+1} = r x_n (1 - x_n) + epsilon y_n` and
    `y_{n+1} = epsilon (x_n + y_n)`. Apart from the origin, the fixed point is
    `x* = (r + epsilon^2 / (1 - epsilon) - 1) / r` and `y* = epsilon x* / (1 - epsilon)`.

    Args:
        x:
            The initial dimensionless surface temperature.

        y:
            The initial dimensionless deeper soil temperature.

        r:
            The logistic parameter of the surface energy balance.

        epsilon:
            The coupling between the surface and the deeper soil.

    Returns:
        system:
            The model, with the temperatures `soil.x` and `soil.y`.

    Examples:
        >>> from skcomplex.models.earth.climate import soil_temperature
        >>> soil_temperature(r=2.5).elements[0].r
        2.5
    """
    return System(components=[SoilTemperature('soil', x=x, y=y, r=r, epsilon=epsilon)])
