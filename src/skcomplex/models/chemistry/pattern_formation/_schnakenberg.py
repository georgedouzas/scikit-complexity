"""Site of the Schnakenberg model, the activator `u` and the inhibitor `v`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from ....core.elements import Element
from ....specs import ParameterSpec, StateSpec, VisualSpec


class SchnakenbergElement(Element):
    """Site of the Schnakenberg model, the activator `u` and the inhibitor `v`.

    The reactions of `schnakenberg` at every site, the productions `a` of `u` and `b` of `v`, the decay of `u`, and
    the autocatalysis `2U + V -> 3U`, read `a` and `b`, and its diffusions read `Du` and `Dv`.

    Args:
        label:
            The label of the site or the field.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        u:
            The initial activator.

        v:
            The initial inhibitor.

        a:
            The production of `u`.

        b:
            The production of `v`.

        Du:
            The diffusion of the activator.

        Dv:
            The diffusion of the inhibitor.

    Examples:
        >>> from skcomplex.models.chemistry.pattern_formation import SchnakenbergElement
        >>> SchnakenbergElement('cell').Dv
        40.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        u: Annotated[object, StateSpec()] = 1.0,
        v: Annotated[object, StateSpec()] = 0.9,
        a: Annotated[object, ParameterSpec()] = 0.1,
        b: Annotated[object, ParameterSpec()] = 0.9,
        Du: Annotated[object, ParameterSpec()] = 1.0,
        Dv: Annotated[object, ParameterSpec()] = 40.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.u = u
        self.v = v
        self.a = a
        self.b = b
        self.Du = Du
        self.Dv = Dv
