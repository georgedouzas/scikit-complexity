"""Site of the Gray-Scott model, the substrate `u` and the activator `v`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class GrayScottElement(Element):
    """Site of the Gray-Scott model, the substrate `u` and the activator `v`.

    The reactions of `gray_scott` at every site, the cubic autocatalysis of the activator on the substrate, the feed
    `F` of the substrate, and the removal `F + k` of the activator, read `F` and `k`, and its diffusions read `Du` and
    `Dv`.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        u:
            The initial substrate.

        v:
            The initial activator.

        Du:
            The diffusion of the substrate.

        Dv:
            The diffusion of the activator.

        F:
            The feed rate.

        k:
            The kill rate.

    Examples:
        >>> from skcomplex.models.chemistry.pattern_formation import GrayScottElement
        >>> GrayScottElement('cell').k
        0.065
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[object, StateSpec()] = 1.0,
        v: Annotated[object, StateSpec()] = 0.0,
        Du: Annotated[object, ParameterSpec()] = 0.16,
        Dv: Annotated[object, ParameterSpec()] = 0.08,
        F: Annotated[object, ParameterSpec()] = 0.035,
        k: Annotated[object, ParameterSpec()] = 0.065,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.v = v
        self.Du = Du
        self.Dv = Dv
        self.F = F
        self.k = k
