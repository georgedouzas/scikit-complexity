"""Define thermal sampling.

The Boltzmann weight of an energy at a temperature is sampled by the moves of a Metropolis solver.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, oo
from sage.sets.real_set import RealSet

from .....core.components.base import BaseComponent
from .....core.components.interactions import Interaction
from .....core.components.laws import ObjectiveLaw
from .....core.types import Role


class ThermalSamplingInteraction(Interaction):
    """Boltzmann weight `exp(-E / T)` of a state of the members of a group, at a temperature.

    The objective of every member is its local energy over the temperature, `E / T`, such as
    `-s sum_j s_j / T` for an Ising spin, and the `MetropolisSolver` of the system moves the state, such as
    `MetropolisSolver(moves={'spins.s': 'flip'})`. The temperature is a value of the system, so a sweep or a schedule
    changes it like any value. It samples the spins of a magnet, the angles of a lattice gauge field, or the opinions
    of a population with a social energy.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group on the sites of a lattice or the nodes of a network.

        state:
            The name of the state the moves change.

        energy:
            The function `energy(namespace)` of the local energy of a member, which reads the member as
            `namespace.target` and a neighbour as `namespace.target.other`.

        temperature:
            The temperature.

    Examples:
        >>> from skcomplex.core.components.laws.utils import neighbors
        >>> from skcomplex.systems.physics.common.spins import ThermalSamplingInteraction
        >>> energy = lambda n: -n.target.s * neighbors(n.target.other.s)
        >>> ThermalSamplingInteraction('ising', target='spins', state='s', energy=energy, temperature=2.0).temperature
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str,
        energy: Callable[[SimpleNamespace], object],
        temperature: Annotated[float, RealSet.open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.energy = energy
        self.temperature = temperature

    def _get_temperature(self, namespace: SimpleNamespace) -> object:
        """Return the temperature the energy is divided by, the parameter `temperature`."""
        return namespace.temperature

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the objective of every member, its local energy over the temperature.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        state = getattr(self._get_member(context, 'target'), self.state)
        objective = SR(self.energy(context)) / SR(self._get_temperature(context))
        self.laws_: list[BaseComponent] = [ObjectiveLaw(objective, state)]
