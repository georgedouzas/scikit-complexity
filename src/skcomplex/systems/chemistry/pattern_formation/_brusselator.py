"""Site of the Brusselator, the reactor of the Brusselator with the diffusion coefficients of its intermediates."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.specs import ParameterSpec, VisualSpec
from ..common.kinetics import BrusselatorReactionElement


class BrusselatorElement(BrusselatorReactionElement):
    """Site of the Brusselator, the reactor of the Brusselator with the diffusion coefficients of its intermediates.

    The four reactions of `brusselator` run at every site, and its diffusions read `Du` and `Dv`.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        u:
            The initial concentration of `u`.

        v:
            The initial concentration of `v`.

        a:
            The feed of `u`.

        b:
            The conversion of `u` into `v`.

        Du:
            The diffusion of `u`.

        Dv:
            The diffusion of `v`.

    Examples:
        >>> from skcomplex.systems.chemistry.pattern_formation import BrusselatorElement
        >>> BrusselatorElement('cell').Dv
        8.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: float = 1.0,
        v: float = 3.0,
        a: float = 1.0,
        b: float = 3.0,
        Du: Annotated[object, ParameterSpec()] = 1.0,
        Dv: Annotated[object, ParameterSpec()] = 8.0,
    ) -> None:
        super().__init__(label, visual=visual, u=u, v=v, a=a, b=b)
        self.Du = Du
        self.Dv = Dv
