"""Define the Langevin thermostat.

The thermostat relaxes the velocities of particles to rest under thermal noise.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, sqrt
from sage.sets.real_set import RealSet

from .....core.components.interactions import Interaction
from .....core.components.laws import NoiseLaw, RateLaw
from .....core.components.spaces import BaseSpace
from .....core.types import Role


class LangevinThermostatInteraction(Interaction):
    """Langevin thermostat, a drag `-gamma x'` and a random force that hold particles at a temperature.

    The velocity `x'` of every particle of mass `m` follows `dx' = -gamma x' / m dt + sqrt(2 gamma temperature) / m
    dW`, with the temperature in energy units, so the velocities relax over the time `m / gamma` to the
    Maxwell-Boltzmann distribution at that temperature. It is the solvent around a colloid, the heat bath of a
    molecular simulation, or the gas around a dust grain. The LAMMPS solver runs it as `fix langevin`.

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
            The name of the position of the particles.


    Examples:
        >>> from skcomplex.systems.physics.common.particles import LangevinThermostatInteraction
        >>> LangevinThermostatInteraction('bath', target='gas', temperature=2.0).temperature
        2.0
    """

    law_classes = (RateLaw, NoiseLaw)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        temperature: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
        gamma: Annotated[float, RealSet.open(0, oo)] = 1.0,
        state: str = 'x',
    ) -> None:
        super().__init__(label)
        self.target = target
        self.temperature = temperature
        self.gamma = gamma
        self.state = state

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the drag and the random force on the velocity of the particles.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            s, t = context, context.t
            x, m = getattr(s.target, self.state), s.target.m
            self.laws_ = [
                RateLaw(x.diff(t, 2), -x.diff(t) / (1 / s.gamma) / m),
                NoiseLaw(x.diff(t, 2), sqrt(2 * s.gamma * s.temperature) / m),
            ]
