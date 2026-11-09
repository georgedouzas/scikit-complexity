"""Define the Langevin thermostat.

The thermostat relaxes the velocities of particles to rest under thermal noise.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, sqrt
from sage.sets.real_set import RealSet

from .....core.specs import ParameterSpec, RoleSpec
from ._relaxation_time import RelaxationTimeInteraction
from ._utils import velocity_of


class LangevinThermostatInteraction(RelaxationTimeInteraction):
    """Langevin thermostat.

    The thermostat adds a drag `-gamma * v` and a random force that hold particles at a temperature.

    Every velocity component follows `dv = -gamma * v / m * dt + sqrt(2 * gamma * temperature) / m * dW`, with the
    temperature in energy units, a relaxation of the velocity towards rest over the time `m / gamma` with additive
    noise. The velocity is the first derivative of the position, a state of order 2, and the mass is the divisor. A
    system with a thermostat is stochastic, and its velocities relax to the Maxwell-Boltzmann distribution at that
    temperature.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle or particle group.

        temperature:
            The temperature in energy units.

        gamma:
            The friction coefficient.

        state:
            The name of the vector state of order 2 that holds the positions of the particles.

        divisor:
            The name of the state or parameter that holds the masses of the particles.

    Examples:
        >>> from skcomplex.systems.physics.common.particles import LangevinThermostatInteraction
        >>> LangevinThermostatInteraction('bath', target='gas', temperature=2.0).temperature
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        temperature: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        state: str = 'x',
        divisor: str = 'm',
    ) -> None:
        super().__init__(label, target=target, state=state)
        self.temperature = temperature
        self.gamma = gamma
        self.divisor = divisor

    def _get_time(self, namespace: SimpleNamespace) -> object:
        """Return the relaxation time `1 / gamma` of the velocity.

        The time is taken before the division by the mass.

        Args:
            namespace:
                The namespace of the symbols the time can use.

        Returns:
            time:
                The relaxation time.
        """
        return 1 / namespace.gamma

    def _get_noise(self, namespace: SimpleNamespace) -> object:
        """Return the amplitude `sqrt(2 * gamma * temperature)` of the noise.

        The amplitude is taken before the division by the mass.

        Args:
            namespace:
                The namespace of the symbols the amplitude can use.

        Returns:
            amplitude:
                The noise amplitude.
        """
        return sqrt(2 * namespace.gamma * namespace.temperature)

    def _get_rate(self, namespace: SimpleNamespace) -> object:
        """Return the relaxation `-v / time` of the velocity `v`.

        The velocity is the first derivative of the position.

        Args:
            namespace:
                The namespace of the symbols the rate can use.

        Returns:
            rate:
                The relaxation of the velocity towards rest, before the division by the mass.
        """
        return -velocity_of(getattr(namespace.target, self.state)) / self._get_time(namespace)
