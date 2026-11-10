"""Define Ising sampling.

Monte Carlo sweeps flip up and down spins coupled to their neighbours and a field.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from .....core.components.laws.utils import neighbors
from .....core.components.spaces import BaseNeighborhood
from .....core.specs import RoleSpec
from ._spin_sampling import SpinSamplingInteraction


class IsingSamplingInteraction(SpinSamplingInteraction):
    """Monte Carlo sweeps of Ising spins `s = ±1` by flips.

    The local energy of a spin is `-J s sum_j s_j - h s`.

    The coupling `J`, the field `h`, and the temperature are parameters of the spins.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of spins.

        state:
            The name of the state that holds the spin.

        rule:
            The acceptance, `'metropolis'`, `'glauber'`, or `'heat-bath'`.

        schedule:
            `'checkerboard'` on a square lattice, `'colored'` on a network, or `'random-sequential'`.

        neighborhood:
            The neighbours of a spin, or `None` for the von Neumann neighbourhood. A network takes
            `AdjacencyNeighborhood()`.

    Examples:
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.systems.physics.common.spins import IsingSamplingInteraction
        >>> ising = IsingSamplingInteraction('ising', target='spins')
        >>> other = SimpleNamespace(spin=var('t'))
        >>> ising.energy(SimpleNamespace(target=SimpleNamespace(spin=var('s'), J=1, h=var('h'), other=other)))
        -h*s - s*neighbors(t, reduce='sum')
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        state: str = 'spin',
        rule: str = 'metropolis',
        schedule: str = 'checkerboard',
        neighborhood: BaseNeighborhood | None = None,
    ) -> None:
        super().__init__(
            label,
            target=target,
            state=state,
            energy=self._get_energy,
            rule=rule,
            schedule=schedule,
            neighborhood=neighborhood,
        )

    def _get_energy(self, namespace: SimpleNamespace) -> Expression:
        """Return the local energy `-J s sum_j s_j - h s` of a spin."""
        spin, other = getattr(namespace.target, self.state), getattr(namespace.target.other, self.state)
        return -namespace.target.J * spin * neighbors(other, over=self.neighborhood) - namespace.target.h * spin
