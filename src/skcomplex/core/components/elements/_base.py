"""Define the base of elements, the parts of a system that hold states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace

from ...specs import StateSpec, VisualSpec
from ..base import BasePart
from ..spaces import BaseSpace


class BaseElement(BasePart[BaseSpace, SimpleNamespace]):
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

        laws_:
            The laws the element contributes for the namespace of its last run.

    Examples:
        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements._base import BaseElement
        >>> from skcomplex.core.specs import ParameterSpec, StateSpec
        >>> from skcomplex.core.specs import VisualSpec
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
        >>> PopulationElement('rabbits', n=40).get_params(deep=False)
        {'label': 'rabbits', 'n': 40, 'r': 0.5, 'visual': VisualSpec(kind='line', options={})}
    """

    laws_: Sequence[BasePart]

    def _run(self, inputs: SimpleNamespace | None) -> None:
        """Set `laws_`, the laws of the element for a namespace of symbols.

        Args:
            inputs:
                The namespace of the symbols the laws can use.
        """
        self.laws_ = self._laws_of(inputs)

    def __init__(self, label: str, visual: VisualSpec | None = None) -> None:
        super().__init__(label)
        self.visual = visual

    def _check(self) -> None:
        """Accept the element, which has nothing to check besides its label."""

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Set `default_visual_`, the plain visual of an element.

        Args:
            context:
                The space the element inherits from its system, which an element does not read.
        """
        self.default_visual_ = VisualSpec()

    def get_states(self) -> dict[str, StateSpec]:
        """Return the states the element carries with their specs, those of its class and those it declares from data.

        Returns:
            states:
                A mapping from the name of every state to its spec.

        Examples:
            >>> from typing import Annotated
            >>> from skcomplex.core.components.elements import Element
            >>> from skcomplex.core.specs import StateSpec
            >>> class TankElement(Element):
            ...     def __init__(self, label, level: Annotated[float, StateSpec()] = 1.0):
            ...         super().__init__(label)
            ...         self.level = level
            >>> list(TankElement('tank').get_states())
            ['level']
        """
        return {name: spec for name, spec in self._get_part_specs().items() if isinstance(spec, StateSpec)}
