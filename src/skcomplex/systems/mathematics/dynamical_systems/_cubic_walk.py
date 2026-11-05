"""Walker driven by Bertin's cubic chaotic map `x' = lam x (x - 1/2)(x - 1) + x`, which steps by `2 x - 1`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class CubicWalkElement(Element):
    """Walker driven by Bertin's cubic chaotic map `x' = lam x (x - 1/2)(x - 1) + x`, which steps by `2 x - 1`.

    Args:
        label:
            The label of the element.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value of the map, between 0 and 1.

        y:
            The initial position of the walker.

        lam:
            The nonlinearity of the map.

    Examples:
        >>> from skcomplex.systems.mathematics.dynamical_systems import CubicWalkElement
        >>> CubicWalkElement('walker', lam=8.0).lam
        8.0
    """

    visual = VisualSpec('points', {'position': 'y', 'states': ()})

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[object, StateSpec()] = 0.3,
        y: Annotated[object, StateSpec(role='position')] = 0.0,
        lam: Annotated[object, ParameterSpec(domain=RealSet.open(0, oo))] = 15.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.y = y
        self.lam = lam

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the next value of the map and the next position.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        return [
            UpdateLaw(s.x, s.lam * s.x * (s.x - 1 / 2) * (s.x - 1) + s.x),
            UpdateLaw(s.y, s.y + 2 * s.x - 1),
        ]
