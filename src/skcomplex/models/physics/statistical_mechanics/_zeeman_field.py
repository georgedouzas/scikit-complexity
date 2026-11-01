"""Uniform field `-h Σ S^z_i` on the spins of a chain."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.interactions import Interaction
from ....core.laws import BaseLaw, GeneratorLaw
from ....core.laws.utils import local_operator
from ....specs import ParameterSpec, RoleSpec


class ZeemanFieldInteraction(Interaction):
    """Uniform field `-h Σ S^z_i` on the spins of a chain.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the spin chain.

        h:
            The field.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import ZeemanFieldInteraction
        >>> ZeemanFieldInteraction('field', target='chain', h=0.2).h
        0.2
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        h: Annotated[float, ParameterSpec()] = 0.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.h = h

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the Hamiltonian of the field.

        Args:
            s:
                The namespace of the interaction.

        Returns:
            laws:
                The field Hamiltonian.
        """
        return [GeneratorLaw(-s.h * sum(local_operator(member.psi, 'z') for member in s.target))]
