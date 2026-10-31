"""Define the Langevin thermostat, a relaxation of the velocities of particles to rest under thermal noise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, sqrt
from sage.sets.real_set import RealSet

from .....core.interactions import RelaxationInteraction
from .....specs import ParameterSpec, RoleSpec


class LangevinThermostatInteraction(RelaxationInteraction):
    """Langevin thermostat, which adds a drag `-gamma * v` and a random force that hold particles at a temperature.

    Every velocity component follows `dv = -gamma * v / m * dt + sqrt(2 * gamma * temperature) / m * dW`, with the
    temperature in energy units, a relaxation of the velocity towards rest over the time `m / gamma` with additive
    noise. A system with a thermostat is stochastic, and its velocities relax to the Maxwell-Boltzmann distribution at
    that temperature.

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
            The name of the vector state that holds the velocities of the particles.

        mass:
            The name of the state or parameter that holds the masses of the particles.

    Examples:
        >>> from skcomplex.models.physics.common.particles import LangevinThermostatInteraction
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
        state: str = 'v',
        mass: str = 'm',
    ) -> None:
        super().__init__(label, target=target, state=state)
        self.temperature = temperature
        self.gamma = gamma
        self.mass = mass

    def _get_time(self, namespace: SimpleNamespace) -> object:
        """Return the relaxation time `m / gamma` of the velocity.

        Args:
            namespace:
                The namespace of the symbols the time can use.

        Returns:
            time:
                The relaxation time.
        """
        return getattr(namespace.target, self.mass) / namespace.gamma

    def _get_noise(self, namespace: SimpleNamespace) -> object:
        """Return the amplitude `sqrt(2 * gamma * temperature) / m` of the noise of the velocity.

        Args:
            namespace:
                The namespace of the symbols the amplitude can use.

        Returns:
            amplitude:
                The noise amplitude.
        """
        return sqrt(2 * namespace.gamma * namespace.temperature) / getattr(namespace.target, self.mass)
