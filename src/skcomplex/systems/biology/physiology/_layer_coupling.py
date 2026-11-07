"""Coupling of every cell to its partner in another layer, `dphi_i/dt += -sigma sin(phi_i - phi_j)`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, sin
from sage.sets.real_set import RealSet

from ....core.components.interactions import NeighborInteraction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.specs import ParameterSpec, RoleSpec


class LayerCouplingInteraction(NeighborInteraction):
    """Coupling of every cell to its partner in another layer, `dphi_i/dt += -sigma sin(phi_i - phi_j)`.

    It runs on the edges of the layer of the network that joins the partners, both ways.

    Args:
        label:
            The label of the interaction.

        cells:
            The path of the group of cells on the nodes.

        sigma:
            The strength of the coupling between the layers.

        layer:
            The layer of the network whose edges join the partners.

    Examples:
        >>> from skcomplex.systems.biology.physiology import LayerCouplingInteraction
        >>> LayerCouplingInteraction('interlayer', cells='cells', sigma=0.5, layer='partners').layer
        'partners'
    """

    def __init__(
        self,
        label: str,
        *,
        cells: Annotated[str, RoleSpec()],
        sigma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        layer: str | None = None,
    ) -> None:
        super().__init__(label, layer=layer)
        self.cells = cells
        self.sigma = sigma

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the pull of the receiving cell towards its partner.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RateLaw(s.target.phi, -s.sigma * sin(s.target.phi - s.source.phi))]
