"""Coupled maps of the temperatures of the soil surface and of the deeper soil."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from typing import Annotated

from ....core.components.elements import Element
from ....core.components.interactions import MapInteraction
from ....core.components.system import System
from ....core.specs import ParameterSpec, StateSpec, VisualSpec
from ..._catalog import register_system


class SoilTemperatureElement(Element):
    """Soil of the coupled temperature maps, its dimensionless surface temperature and deeper soil temperature.

    Two maps of `soil_temperature` iterate the states. The surface temperature follows the energy balance of the
    ground, reduced to a logistic map, and the deeper soil temperature relaxes towards it:
    `x_{n+1} = r x_n (1 - x_n) + epsilon y_n` and `y_{n+1} = epsilon (x_n + y_n)`.

    Args:
        label:
            The label of the soil.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial dimensionless surface temperature.

        y:
            The initial dimensionless deeper soil temperature.

        r:
            The logistic parameter of the surface energy balance.

        epsilon:
            The coupling between the surface and the deeper soil.

    Examples:
        >>> from skcomplex.systems.physics.climate import SoilTemperatureElement
        >>> SoilTemperatureElement('soil').r
        3.8
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.3,
        y: Annotated[float, StateSpec()] = 0.1,
        r: Annotated[float, ParameterSpec()] = 3.8,
        epsilon: Annotated[float, ParameterSpec()] = 0.1,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.r = r
        self.epsilon = epsilon


@register_system(
    field='physics/climate',
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
            The system, with the temperatures `soil.x` and `soil.y`.

    Examples:
        >>> from skcomplex.systems.physics.climate import soil_temperature
        >>> soil_temperature(r=2.5).elements[0].r
        2.5
    """
    surface = MapInteraction(
        'surface',
        target='soil',
        state='x',
        function=lambda s: s.target.r * s.target.x * (1 - s.target.x) + s.target.epsilon * s.target.y,
    )
    depth = MapInteraction(
        'depth',
        target='soil',
        state='y',
        function=lambda s: s.target.epsilon * (s.target.x + s.target.y),
    )
    return System(
        elements=[SoilTemperatureElement('soil', x=x, y=y, r=r, epsilon=epsilon)],
        interactions=[surface, depth],
    )
