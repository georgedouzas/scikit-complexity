"""Inhibition of every oscillator of a lattice by a global inhibitor, once the inhibitor exceeds its threshold."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import CouplingInteraction
from ....core.components.laws.utils import logistic
from ....core.components.types import Target
from ....core.specs import ParameterSpec, RoleSpec


class GlobalInhibitionInteraction(CouplingInteraction):
    """Inhibition of every oscillator of a lattice by a global inhibitor, once the inhibitor exceeds its threshold.

    Every oscillator gets `-W_z H(z - theta_z)` from the inhibitor, a smooth step of width `kappa` of the oscillator.
    It is the coupling of the members of a group to a hub element, and `InhibitorDriveInteraction` drives the hub.

    Args:
        label:
            The label of the interaction.

        oscillators:
            The path of the lattice of oscillators.

        inhibitor:
            The path of the inhibitor.

        W_z:
            The strength of the inhibition.

        theta_z:
            The threshold of the inhibitor.

    Examples:
        >>> from skcomplex.systems.biology.neuroscience import GlobalInhibitionInteraction
        >>> GlobalInhibitionInteraction('inhibition', inhibitor='z', oscillators='grid').W_z
        1.5
    """

    def __init__(
        self,
        label: str,
        *,
        oscillators: Annotated[str, RoleSpec()],
        inhibitor: Annotated[str, RoleSpec()],
        W_z: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.5,
        theta_z: Annotated[float, ParameterSpec()] = 0.1,
    ) -> None:
        super().__init__(
            label,
            target=oscillators,
            source=inhibitor,
            write='x',
            function=self._get_inhibition,
            mode='group',
        )
        self.oscillators = oscillators
        self.inhibitor = inhibitor
        self.W_z = W_z
        self.theta_z = theta_z

    @staticmethod
    def _get_inhibition(namespace: SimpleNamespace) -> Target:
        """Return the inhibition `-W_z H(z - theta_z)` of an oscillator."""
        return -namespace.W_z * logistic(namespace.other.z - namespace.theta_z, 0, namespace.member.kappa)
