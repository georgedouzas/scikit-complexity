"""Define spin sampling.

The Boltzmann weight of the energy of spins is sampled at the temperature the spins hold.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace

from .....core.components.base import BaseComponent
from .....core.components.interactions import Interaction
from .....core.components.laws import ObjectiveLaw
from .....core.types import Role


class SpinSamplingInteraction(Interaction):
    """Boltzmann weight of the spins of a lattice or network group, at the temperature of the spins.

    The objective of every spin is its local energy over the value `temperature` of the spins, `E / T`, so the
    temperature lives with the couplings and fields of the spins, and a schedule of `<group>.temperature` changes it
    during a run. The `MetropolisSolver` of the system moves the spins, such as
    `MetropolisSolver(moves={'spins.spin': 'flip'})`. The local energy reads the spin as `namespace.target` and a
    neighbour as `namespace.target.other`.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group of spins.

        state:
            The name of the state the moves change.

        energy:
            The function `energy(namespace)` of the local energy of a spin.

    Examples:
        >>> from skcomplex.core.components.laws.utils import neighbors
        >>> from skcomplex.systems.physics.common.spins import SpinSamplingInteraction
        >>> energy = lambda n: -n.target.spin * neighbors(n.target.other.spin)
        >>> SpinSamplingInteraction('ising', target='spins', state='spin', energy=energy).state
        'spin'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        state: str,
        energy: Callable[[SimpleNamespace], object],
    ) -> None:
        super().__init__(label)
        self.target = target
        self.state = state
        self.energy = energy

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the objective of every spin, its local energy over its temperature.

        Args:
            context:
                The namespace of the target, the time `t`, and the parameters.
        """
        super()._prepare(context)
        spins = self._get_member(context, 'target')
        objective = self.energy(context) / spins.temperature
        self.laws_: list[BaseComponent] = [ObjectiveLaw(objective, getattr(spins, self.state))]
