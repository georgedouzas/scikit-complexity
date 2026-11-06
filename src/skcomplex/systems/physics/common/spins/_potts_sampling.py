"""Define Potts sampling, Monte Carlo sweeps of spins of `q` states that favour neighbours in the same state."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from .....core.components.laws.utils import neighbors, where
from .....core.components.spaces import BaseNeighborhood
from .....core.specs import RoleSpec
from ._spin_sampling import SpinSamplingInteraction


class PottsSamplingInteraction(SpinSamplingInteraction):
    """Monte Carlo sweeps of Potts spins in states `0` to `q - 1`, of local energy `-J` times the like neighbours.

    Every move proposes a state drawn from the `q` states. The coupling `J` and the temperature are parameters of the
    spins.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of spins.

        q:
            The number of states.

        state:
            The name of the state that holds the spin.

        rule:
            The acceptance, `'metropolis'`, `'glauber'`, or `'heat-bath'`.

        neighborhood:
            The neighbours of a spin, or `None` for the von Neumann neighbourhood.

    Examples:
        >>> from skcomplex.systems.physics.common.spins import PottsSamplingInteraction
        >>> PottsSamplingInteraction('potts', target='spins', q=4).values
        (0.0, 1.0, 2.0, 3.0)
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        q: int = 3,
        state: str = 'color',
        rule: str = 'heat-bath',
        neighborhood: BaseNeighborhood | None = None,
    ) -> None:
        super().__init__(
            label,
            target=target,
            state=state,
            energy=self._energy,
            move='random-value',
            rule=rule,
            neighborhood=neighborhood,
            values=tuple(float(value) for value in range(q)),
        )
        self.q = q

    def _energy(self, namespace: SimpleNamespace) -> Expression:
        """Return the local energy `-J` times the number of neighbours in the state of the spin."""
        color, other = getattr(namespace.target, self.state), getattr(namespace.target.other, self.state)
        return -namespace.target.J * neighbors(where(other == color, 1, 0), over=self.neighborhood)
