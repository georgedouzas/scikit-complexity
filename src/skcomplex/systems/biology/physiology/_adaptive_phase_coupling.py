"""Phase coupling of the cells of a layer through links that adapt to the phase differences they carry."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, sin
from sage.sets.real_set import RealSet

from ....core.components.interactions import NeighborInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, RoleSpec


class AdaptivePhaseCouplingInteraction(NeighborInteraction):
    """Phase coupling of the cells of a layer through links that adapt to the phase differences they carry.

    Along the link from cell `j` to cell `i`, `dphi_i/dt += -(a_ij + kappa_ij) sin(phi_i - phi_j + alpha) / N` and
    `dkappa_ij/dt = -epsilon (kappa_ij + sin(phi_i - phi_j - beta))`. The phase lag `alpha` sets the preferred phase
    difference of the coupling, and the age parameter `beta` the rule of the adaptation.

    Args:
        label:
            The label of the interaction.

        cells:
            The path of the group of cells on the nodes.

        links:
            The path of the group of links on the edges of the layer.

        N:
            The number of cells of the layer, which normalizes the coupling.

        alpha:
            The phase lag of the coupling.

        beta:
            The age parameter of the adaptation.

        epsilon:
            The adaptation rate.

    Examples:
        >>> from skcomplex.systems.biology.physiology import AdaptivePhaseCouplingInteraction
        >>> AdaptivePhaseCouplingInteraction('coupling', cells='cells', links='links', epsilon=0.3).epsilon
        0.3
    """

    def __init__(
        self,
        label: str,
        *,
        cells: Annotated[str, RoleSpec()],
        links: Annotated[str, RoleSpec()],
        N: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        alpha: Annotated[float, ParameterSpec()] = 0.0,
        beta: Annotated[float, ParameterSpec()] = 0.0,
        epsilon: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.01,
    ) -> None:
        super().__init__(label)
        self.cells = cells
        self.links = links
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the coupling of the receiving cell and the adaptation of the link.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        difference = s.target.phi - s.source.phi
        kappa = s.edge.kappa
        return [
            RateLaw(s.target.phi, -(s.edge.a + kappa) * sin(difference + s.alpha) / s.N),
            RateLaw(kappa, -s.epsilon * (kappa + sin(difference - s.beta))),
        ]
