"""Transmission line of a capacity that couples the phases of two nodes by the sine of their difference."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, sin
from sage.sets.real_set import RealSet

from ....core.interactions import CouplingInteraction
from ....core.types import Target
from ....specs import ParameterSpec, RoleSpec


class TransmissionLineInteraction(CouplingInteraction):
    """Transmission line of a capacity that couples the phases of two nodes by the sine of their difference.

    It is the coupling of the two nodes as separate elements, each getting `K sin(phi_j - phi_i)` from the other.

    Args:
        label:
            The label of the interaction.

        ends:
            The paths of the two nodes the line joins.

        K:
            The capacity of the line.
    """

    def __init__(
        self,
        label: str,
        *,
        ends: Annotated[Sequence[str], RoleSpec()],
        K: Annotated[object, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, target=ends, read='phi', write='phi', function=self._get_flow, mode='group')
        self.ends = ends
        self.K = K

    @staticmethod
    def _get_flow(namespace: SimpleNamespace) -> Target:
        """Return the power flow from the other end of the line, `K sin(phi_j - phi_i)`."""
        return namespace.K * sin(namespace.other.phi - namespace.member.phi)
