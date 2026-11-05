"""Define XY sampling, Monte Carlo sweeps of planar spins that favour neighbours at the same angle."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, cos

from .....core.components.laws.utils import neighbors
from .....core.components.spaces import BaseNeighborhood
from .....core.specs import RoleSpec
from ._spin_sampling import SpinSamplingInteraction


class XYSamplingInteraction(SpinSamplingInteraction):
    """Monte Carlo sweeps of planar spins at angles `theta`, of local energy `-J sum_j cos(theta - theta_j)`.

    Every move rotates the angle by a uniform number in `[-width, width]`, accepted by the Metropolis rule. The
    coupling `J` and the temperature are parameters of the spins.

    Args:
        label:
            The label of the interaction, unique among the parts of its system.

        target:
            The path of the group of spins.

        width:
            The largest change of a proposed rotation.

        state:
            The name of the state that holds the angle.

        neighborhood:
            The neighbours of a spin, or `None` for the von Neumann neighbourhood.

    Examples:
        >>> from skcomplex.systems.physics.common.spins import XYSamplingInteraction
        >>> XYSamplingInteraction('xy', target='spins', width=0.5).move
        'rotate'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        width: float = 1.0,
        state: str = 'theta',
        neighborhood: BaseNeighborhood | None = None,
    ) -> None:
        super().__init__(
            label,
            target=target,
            state=state,
            energy=self._energy,
            move='rotate',
            neighborhood=neighborhood,
            width=width,
        )

    def _energy(self, namespace: SimpleNamespace) -> Expression:
        """Return the local energy `-J sum_j cos(theta - theta_j)` of a spin."""
        angle, other = getattr(namespace.target, self.state), getattr(namespace.target.other, self.state)
        return -namespace.target.J * neighbors(cos(angle - other), over=self.neighborhood)
