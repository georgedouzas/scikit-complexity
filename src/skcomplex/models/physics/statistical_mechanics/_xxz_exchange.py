"""Exchange `J (S^x_i S^x_j + S^y_i S^y_j + Δ S^z_i S^z_j)` between neighbouring spins."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from ....core.interactions import NeighborInteraction
from ....core.laws import BaseLaw, GeneratorLaw
from ....core.laws.utils import local_operator
from ....specs import ParameterSpec, RoleSpec


class XXZExchangeInteraction(NeighborInteraction):
    """Exchange `J (S^x_i S^x_j + S^y_i S^y_j + Δ S^z_i S^z_j)` between neighbouring spins.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the spin chain.

        J:
            The exchange coupling, antiferromagnetic when positive.

        delta:
            The anisotropy `Δ`, 1 for the isotropic chain and 0 for the XX chain.

    Examples:
        >>> from skcomplex.models.physics.statistical_mechanics import XXZExchangeInteraction
        >>> XXZExchangeInteraction('exchange', target='chain', delta=0.5).delta
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        J: Annotated[float, ParameterSpec()] = 1.0,
        delta: Annotated[float, ParameterSpec()] = 1.0,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.J = J
        self.delta = delta

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the exchange Hamiltonian of one bond.

        Args:
            s:
                The namespace of the bond, with `s.source` and `s.target`.

        Returns:
            laws:
                The Hamiltonian of the bond.
        """
        first, second = s.source.psi, s.target.psi
        exchange = (
            local_operator(first, 'x') * local_operator(second, 'x')
            + local_operator(first, 'y') * local_operator(second, 'y')
            + s.delta * local_operator(first, 'z') * local_operator(second, 'z')
        )
        return [GeneratorLaw(s.J * exchange)]
