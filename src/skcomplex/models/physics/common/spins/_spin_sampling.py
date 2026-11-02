"""Define spin sampling, Monte Carlo sweeps of spins at the temperature the spins hold."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Annotated

from .....core.laws import BaseLaw
from .....core.spaces import BaseNeighborhood
from .....specs import RoleSpec
from ._thermal_sampling import ThermalSamplingInteraction


class SpinSamplingInteraction(ThermalSamplingInteraction):
    """Monte Carlo sweeps of the spins of a lattice or network group at the temperature of the spins.

    The temperature is the parameter `temperature` of the spins, so it lives with the couplings and fields of the
    spins, and a schedule of `<group>.temperature` changes it during a run. The local energy reads the states and
    parameters of the spin as `namespace.target`.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of spins.

        state:
            The name of the state the moves change.

        energy:
            The function `energy(namespace)` of the local energy of a spin.

        move:
            `'flip'`, `'random-value'`, `'rotate'`, `'gaussian'`, or a function that proposes values.

        rule:
            The acceptance, `'metropolis'`, `'glauber'`, or `'heat-bath'`.

        schedule:
            `'checkerboard'`, `'colored'`, or `'random-sequential'`.

        neighborhood:
            The neighbourhood the energy reduces over, or `None` for the von Neumann neighbourhood.

        values:
            The values a random-value move draws from.

        width:
            The largest change of a rotation, or the standard deviation of a Gaussian move.

    Examples:
        >>> from skcomplex.core.laws.utils import neighbors
        >>> from skcomplex.models.physics.common.spins import SpinSamplingInteraction
        >>> energy = lambda n: -n.target.spin * neighbors(n.target.other.spin)
        >>> SpinSamplingInteraction('ising', target='spins', state='spin', energy=energy).rule
        'metropolis'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str,
        energy: Callable[[SimpleNamespace], object],
        move: str | Callable[..., object] = 'flip',
        rule: str = 'metropolis',
        schedule: str = 'checkerboard',
        neighborhood: BaseNeighborhood | None = None,
        values: Sequence[float] = (),
        width: float = 1.0,
    ) -> None:
        super().__init__(
            label,
            target=target,
            state=state,
            energy=energy,
            move=move,
            rule=rule,
            schedule=schedule,
            neighborhood=neighborhood,
            values=values,
            width=width,
        )

    def _get_laws(self, namespace: SimpleNamespace) -> list[BaseLaw]:
        """Return the sample law of the state at the temperature of the spins.

        Args:
            namespace:
                The namespace of the symbols the laws can use, with `namespace.target`.

        Returns:
            laws:
                The sample law.
        """
        return super()._get_laws(SimpleNamespace(**vars(namespace), temperature=namespace.target.temperature))
