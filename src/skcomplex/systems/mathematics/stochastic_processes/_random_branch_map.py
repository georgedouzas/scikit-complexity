"""Value of a random interval map, which applies `(1 + x) / 2` with probability `p1` and `mu x²` otherwise."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, UpdateLaw
from ....core.components.laws.utils import bernoulli
from ....core.specs import ParameterSpec, StateSpec, VisualSpec


class RandomBranchMapElement(Element):
    """Value of a random interval map, which applies `(1 + x) / 2` with probability `p1` and `mu x²` otherwise.

    Args:
        label:
            The label of the element or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        x:
            The initial value, between 0 and 1.

        p1:
            The probability of the branch `(1 + x) / 2`.

        mu:
            The coefficient of the branch `mu x²`, between 0 and 1.

    Examples:
        >>> from skcomplex.systems.mathematics.stochastic_processes import RandomBranchMapElement
        >>> RandomBranchMapElement('map', p1=0.3).p1
        0.3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        x: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 0.5,
        p1: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.5,
        mu: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.x = x
        self.p1 = p1
        self.mu = mu

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the update of the value by the branch drawn in the step.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the element.
        """
        first = bernoulli('branch', s.p1)
        return [UpdateLaw(s.x, first * (1 + s.x) / 2 + (1 - first) * s.mu * s.x**2)]
