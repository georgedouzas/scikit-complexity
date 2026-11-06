"""Define the drive of a global inhibitor by the most active oscillator of a lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.components.interactions import CouplingInteraction
from ....core.components.laws.utils import logistic
from ....core.components.types import Target
from ....core.specs import RoleSpec


class InhibitorDriveInteraction(CouplingInteraction):
    """Drive of a global inhibitor by the largest activity `m H(x - theta_x)` of the oscillators of a lattice.

    The inhibitor gets `phi max_j m_j H(x_j - theta_x)`, with the rate `phi` of the inhibitor and a smooth step of
    width `kappa` of every oscillator, so it is driven when at least one stimulated oscillator is active. It is the
    coupling of a hub element to a group by the largest value over the group.

    Args:
        label:
            The label of the interaction.

        inhibitor:
            The path of the inhibitor.

        oscillators:
            The path of the lattice of oscillators.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import InhibitorDriveInteraction
        >>> InhibitorDriveInteraction('drive', inhibitor='z', oscillators='grid').reduce
        'max'
    """

    def __init__(
        self,
        label: str,
        *,
        inhibitor: Annotated[str, RoleSpec()],
        oscillators: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(
            label,
            target=inhibitor,
            source=oscillators,
            write='z',
            function=self._get_drive,
            reduce='max',
            mode='group',
        )
        self.inhibitor = inhibitor
        self.oscillators = oscillators

    @staticmethod
    def _get_drive(namespace: SimpleNamespace) -> Target:
        """Return the drive `phi m H(x - theta_x)` of the inhibitor by one oscillator."""
        other = namespace.other
        return namespace.member.phi * other.m * logistic(other.x - other.theta_x, 0, other.kappa)
