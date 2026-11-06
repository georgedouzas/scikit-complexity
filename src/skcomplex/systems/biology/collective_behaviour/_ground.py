"""Ground whose structure `G` is raised by footprints and recovers, with a trail potential `V` that smooths it."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class GroundElement(Element):
    """Ground whose structure `G` is raised by footprints and recovers, with a trail potential `V` that smooths it.

    The structure relaxes back to bare ground as `G' = -G / T`, and walkers add footprints. The trail potential
    follows `V' = D ΔV + (G - V) / tau`, so it settles to the structure smoothed over a length `sqrt(D tau)`, the
    distance from which walkers notice a trail. The element holds the states and their parameters, and the system adds
    the fading, the diffusion, and the relaxation as interactions.

    Args:
        label:
            The label of the ground or the lattice field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        G:
            The initial ground structure.

        V:
            The initial trail potential.

        T:
            The durability of trails, the time over which they fade.

        D:
            The diffusion of the trail potential.

        tau:
            The relaxation time of the trail potential.

    Examples:
        >>> from skcomplex.systems.biology.collective_behaviour import GroundElement
        >>> GroundElement('ground').T
        200.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        G: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        V: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        T: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 200.0,
        D: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        tau: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.G = G
        self.V = V
        self.T = T
        self.D = D
        self.tau = tau
