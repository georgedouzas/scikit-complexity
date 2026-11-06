"""Hamiltonian `-D laplacian + V` of a particle on the sites of a lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, GeneratorLaw
from ....core.components.laws.utils import local_operator
from ....core.specs import ParameterSpec, RoleSpec


class LatticeHamiltonianInteraction(Interaction):
    """Hamiltonian `-D laplacian + V` of a particle on the sites of a lattice.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the particle, whose Hilbert space holds the operators `laplacian` and `potential`.

        D:
            The coefficient of the Laplacian, `ħ / (2 m)`.

    Examples:
        >>> from skcomplex.systems.physics.waves import LatticeHamiltonianInteraction
        >>> LatticeHamiltonianInteraction('kinetic', target='packet').D
        0.5
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        D: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 0.5,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.D = D

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the Hamiltonian of the particle.

        Args:
            s:
                The namespace of the interaction.

        Returns:
            laws:
                The Hamiltonian.
        """
        psi = s.target.psi
        return [GeneratorLaw(-s.D * local_operator(psi, 'laplacian') + local_operator(psi, 'potential'))]
