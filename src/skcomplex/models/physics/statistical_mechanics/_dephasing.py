"""Dephasing of every spin of a chain, a dissipator with the collapse operator `S^z_i` at the rate `gamma`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.interactions import Interaction
from ....core.laws import BaseLaw, DissipatorLaw
from ....core.laws.utils import local_operator
from ....specs import ParameterSpec, RoleSpec


class DephasingInteraction(Interaction):
    """Dephasing of every spin of a chain, a dissipator with the collapse operator `S^z_i` at the rate `gamma`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the spin chain.

        gamma:
            The dephasing rate.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import DephasingInteraction
        >>> DephasingInteraction('dephasing', target='chain', gamma=0.1).gamma
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.gamma = gamma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return one dissipator per spin.

        Args:
            s:
                The namespace of the interaction.

        Returns:
            laws:
                The dissipators.
        """
        return [DissipatorLaw(local_operator(member.psi, 'z'), s.gamma) for member in s.target]
