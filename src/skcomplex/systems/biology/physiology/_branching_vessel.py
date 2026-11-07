"""Vessel of a space-filling branching network, sized by its level so that the network minimizes transport energy."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo, pi
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, EquationLaw
from ....core.components.laws.utils import where
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class BranchingVesselElement(Element):
    """Vessel of a space-filling branching network, sized by its level so that the network minimizes transport energy.

    Every vessel branches into `n` daughters. Area-preserving branching gives the radii the ratio `n^(-1/2)` between
    levels and space filling the lengths the ratio `n^(-1/3)`, and the capillaries at the last level `depth` have the
    same radius `r_c` and length `l_c` in every organism, so `r = r_c n^((depth - level) / 2)`, `l = l_c n^((depth -
    level) / 3)`, and the volume of the vessel is `pi r^2 l`.

    Args:
        label:
            The label of the vessel or of the group on the edges.

        visual:
            How the vessel is drawn, or `None` for `default_visual_`.

        r:
            The radius.

        l:
            The length.

        volume:
            The volume.

        terminal:
            Whether the vessel is a capillary.

        level:
            The level of the vessel, 1 for the aorta's daughters and `depth` for the capillaries.

        depth:
            The level of the capillaries.

        n:
            The branching ratio.

        r_c:
            The radius of a capillary.

        l_c:
            The length of a capillary.

    Examples:
        >>> from skcomplex.systems.biology.physiology import BranchingVesselElement
        >>> BranchingVesselElement('vessel', level=3.0).level
        3.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        r: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        l: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        volume: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        terminal: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.0,
        level: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        depth: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        n: Annotated[float, ParameterSpec(domain=RealSet.open(1, oo))] = 2.0,
        r_c: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        l_c: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.r = r
        self.l = l
        self.volume = volume
        self.terminal = terminal
        self.level = level
        self.depth = depth
        self.n = n
        self.r_c = r_c
        self.l_c = l_c

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the radius, the length, the volume, and the capillary indicator of the vessel.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        height = s.depth - s.level
        return [
            EquationLaw(s.r == s.r_c * s.n ** (height / 2), target=s.r),
            EquationLaw(s.l == s.l_c * s.n ** (height / 3), target=s.l),
            EquationLaw(s.volume == pi * s.r**2 * s.l, target=s.volume),
            EquationLaw(s.terminal == where(s.level > s.depth - 1 / 2, 1, 0), target=s.terminal),
        ]
