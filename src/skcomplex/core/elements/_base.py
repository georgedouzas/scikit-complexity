"""Define the base of elements, the parts of a system that hold states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Self

from ...specs import VisualSpec
from ..base import BasePart


class BaseElement(BasePart):
    """Base of every element, a part of a system that holds states.

    A subclass takes its states and parameters as arguments of its own `__init__`, annotated with `Annotated` and the
    spec `StateSpec` or `ParameterSpec`, and stores each under its own name, as in scikit-learn. A subclass
    drawn in its own way gives `visual` a default in its signature. It returns its laws from `_get_laws`, which every
    subclass implements, returning an empty list when the element has none.

    Args:
        label:
            The label of the element, unique among the parts of its system.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.elements._base import BaseElement
        >>> from skcomplex.specs import ParameterSpec, StateSpec
        >>> from skcomplex.specs import VisualSpec
        >>> class PopulationElement(BaseElement):
        ...     def __init__(
        ...         self,
        ...         label,
        ...         n: Annotated[float, StateSpec()] = 1.0,
        ...         r: Annotated[float, ParameterSpec()] = 0.5,
        ...         visual=VisualSpec('line'),
        ...     ):
        ...         super().__init__(label, visual=visual)
        ...         self.n = n
        ...         self.r = r
        ...
        ...     def _get_laws(self, namespace):
        ...         return []
        >>> PopulationElement('rabbits', n=40).get_params()
        {'label': 'rabbits', 'n': 40, 'r': 0.5, 'visual': VisualSpec(kind='line', options={})}
    """

    def __init__(self, label: str, visual: VisualSpec | None = None) -> None:
        super().__init__(label)
        self.visual = visual

    def _check(self) -> Self:
        """Return the element, which has nothing to check besides its label.

        Returns:
            element:
                The element itself.
        """
        return self

    def _prepare(self, context: object = None) -> Self:
        """Set `default_visual_`, the plain visual of an element.

        Args:
            context:
                The space the element inherits from its system, which an element does not read.

        Returns:
            element:
                The element itself.
        """
        self.default_visual_ = VisualSpec()
        return self
