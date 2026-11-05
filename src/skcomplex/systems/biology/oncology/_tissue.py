"""Tissue in which oxygen and the vascular endothelial growth factor VEGF diffuse and decay."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class TissueElement(Element):
    """Tissue in which oxygen and the vascular endothelial growth factor VEGF diffuse and decay.

    Both follow `u' = D Δu - lambda u`, and cells add or remove amounts at their sites: vessels supply oxygen, tumour
    cells consume it, and hypoxic tumour cells secrete VEGF. The system adds the diffusion and the decay as
    interactions.

    Args:
        label:
            The label of the tissue or the lattice field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        oxygen:
            The initial oxygen.

        vegf:
            The initial VEGF.

        D_oxygen:
            The diffusion of oxygen.

        decay_oxygen:
            The decay rate of oxygen.

        D_vegf:
            The diffusion of VEGF.

        decay_vegf:
            The decay rate of VEGF.

    Examples:
        >>> from skcomplex.systems.biology.oncology import TissueElement
        >>> TissueElement('tissue').D_vegf
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        oxygen: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        vegf: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        D_oxygen: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        decay_oxygen: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        D_vegf: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        decay_vegf: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.005,
    ) -> None:
        super().__init__(label, visual=visual)
        self.oxygen = oxygen
        self.vegf = vegf
        self.D_oxygen = D_oxygen
        self.decay_oxygen = decay_oxygen
        self.D_vegf = D_vegf
        self.decay_vegf = decay_vegf
