"""Coupled maps of the temperatures of the soil surface and of the deeper soil."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


class SoilTemperatureSystem(DynamicalSystem):
    """The coupled maps of the surface and deeper temperatures of the soil.

    The dimensionless surface temperature `x` follows the energy balance of the ground, reduced to a logistic map, and
    the deeper soil temperature `y` relaxes towards it: `x_{n+1} = r x_n (1 - x_n) + epsilon y_n` and `y_{n+1} = epsilon
    (x_n + y_n)`. Apart from the origin, the fixed point is `x* = (r + epsilon^2 / (1 - epsilon) - 1) / r` and `y* =
    epsilon x* / (1 - epsilon)`. The system has the temperatures `soil.x` and `soil.y`.

    Args:
        x:
            The initial dimensionless surface temperature.

        y:
            The initial dimensionless deeper soil temperature.

        r:
            The logistic parameter of the surface energy balance.

        epsilon:
            The coupling between the surface and the deeper soil.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.physics.climate import SoilTemperatureSystem
        >>> SoilTemperatureSystem(r=2.5).prepare().system_.elements[0].get_value('r')
        2.5
    """

    field = 'physics/climate'
    topics = ('discrete map', 'coupled maps', 'chaos', 'land surface')
    references = ('Mimić, Mihailović, and Budinčević 2013', 'Mihailović et al. 2023')
    until = 200
    dt = 1

    def __init__(
        self,
        x: float = 0.3,
        y: float = 0.1,
        *,
        r: float = 3.8,
        epsilon: float = 0.1,
        label: str = 'system',
    ) -> None:
        super().__init__(label=label)
        self.x = x
        self.y = y
        self.r = r
        self.epsilon = epsilon

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'soil'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state."""
        return {
            'x': self.x,
            'y': self.y,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the parameters of the equations."""
        return {
            'r': self.r,
            'epsilon': self.epsilon,
        }

    def _get_equations(self) -> Equations:
        """Return the coupled maps of the surface and deeper temperatures of the soil."""
        return {
            'x': lambda s: s.target.r * s.target.x * (1 - s.target.x) + s.target.epsilon * s.target.y,
            'y': lambda s: s.target.epsilon * (s.target.x + s.target.y),
        }
