"""Define logistic maps, the quadratic maps of the unit interval that reach chaos by period doubling."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression

from ...specs import ParameterSpec, StateSpec, VisualSpec
from ..base import BasePart
from ..laws import UpdateLaw
from ._element import Element


class LogisticMapElement(Element):
    """Logistic map `x_{t+1} = r x_t (1 - x_t)`, which reaches chaos through period doubling as `r` grows.

    For `1 < r < 3` the map converges to the fixed point `1 - 1 / r`, at `r = 3` the fixed point gives way to a cycle of
    period 2, and from `r` close to 3.57 the map is chaotic. A subclass that couples maps, such as a lattice of coupled
    maps, overrides `_get_laws` and applies `_get_map` to the mixture of the states it couples.

    Args:
        label:
            The label of the map, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value, in the unit interval.

        r:
            The growth parameter.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

    Examples:
        >>> from skcomplex.core.elements import LogisticMapElement
        >>> from skcomplex.core.system import System
        >>> from skcomplex.engine.runtime import Simulation
        >>> system = System(elements=[LogisticMapElement('map', x=0.2, r=2.0)])
        >>> Simulation(system).check().prepare().run(until=3).trajectory_.values('map.x').round(4).tolist()
        [0.2, 0.32, 0.4352, 0.4916]
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec()] = 0.2,
        r: Annotated[float, ParameterSpec()] = 3.7,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.r = r

    def _get_map(self, r: Expression, x: Expression) -> Expression:
        """Return the image `r x (1 - x)` of a value under the logistic map with growth parameter `r`."""
        return r * x * (1 - x)

    def _get_laws(self, namespace: SimpleNamespace) -> Sequence[BasePart]:
        """Return the next value of the map.

        Args:
            namespace:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The update law of `x`.
        """
        return [UpdateLaw(namespace.x, self._get_map(namespace.r, namespace.x))]
