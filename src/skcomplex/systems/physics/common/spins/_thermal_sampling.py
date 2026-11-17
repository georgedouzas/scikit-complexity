"""Define thermal sampling.

Monte Carlo sweeps read an energy at a temperature.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import SR, Expression, oo
from sage.sets.real_set import RealSet

from .....core.components.interactions import SamplingInteraction
from .....core.components.spaces import BaseNeighborhood
from .....core.specs import ParameterSpec, RoleSpec


class ThermalSamplingInteraction(SamplingInteraction):
    """Monte Carlo sweeps of a state in an energy at a temperature.

    The intensity is one over the temperature.

    Args:
        label:
            The label of the interaction, unique among the components of its system.

        target:
            The path of the group on the sites of a lattice or the nodes of a network.

        state:
            The name of the state the moves change.

        energy:
            The function `energy(namespace)` of the local energy of a member.

        move:
            `'flip'`, `'random-value'`, `'rotate'`, `'gaussian'`, `'cluster'`, or a function that proposes values.

        temperature:
            The temperature, where 0 accepts only moves that do not raise the energy.

        rule:
            The acceptance, `'metropolis'`, `'glauber'`, or `'heat-bath'`, which draws among all values of a discrete
            move by their Boltzmann weights.

        schedule:
            `'checkerboard'`, `'colored'`, or `'random-sequential'`.

        neighborhood:
            The neighbourhood the energy reduces over, or `None` for the von Neumann neighbourhood.

        values:
            The values a random-value move draws from.

        width:
            The largest change of a rotation, or the standard deviation of a Gaussian move.

        cooling:
            The function of the time symbol that multiplies the temperature, or `None` for a constant temperature.

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
        target: Annotated[str, RoleSpec()],
        state: str,
        energy: Callable[[SimpleNamespace], object],
        move: str | Callable[..., object] = 'flip',
        temperature: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        rule: str = 'metropolis',
        schedule: str = 'checkerboard',
        neighborhood: BaseNeighborhood | None = None,
        values: Sequence[float] = (),
        width: float = 1.0,
        cooling: Callable[[Expression], object] | None = None,
    ) -> None:
        super().__init__(
            label,
            target=target,
            state=state,
            objective=energy,
            move=move,
            rule=rule,
            schedule=schedule,
            neighborhood=neighborhood,
            values=values,
            width=width,
            cooling=cooling,
        )
        self.energy = energy
        self.temperature = temperature

    def _get_parameter(self, name: str) -> str:
        """Return `energy` for the objective of the general interaction.

        Any other name is returned unchanged.
        """
        return 'energy' if name == 'objective' else name

    def _get_intensity(self, namespace: SimpleNamespace) -> object:
        """Return one over the temperature.

        It is infinite at a temperature of zero, which a row score reads as a value.
        """
        temperature = SR(self._get_parameter_symbol('temperature', namespace))
        return float('inf') if temperature.is_zero() else 1 / temperature
