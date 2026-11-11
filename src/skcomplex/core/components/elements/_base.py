"""Define the abstract element, the part of a system that holds its states."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from dataclasses import replace
from types import SimpleNamespace

from ...specs import StateSpec, Tags, VisualSpec
from ..base import BaseComponent
from ..spaces import BaseSpace


class BaseElement(BaseComponent[BaseSpace, SimpleNamespace]):
    """An abstract element, the part of a system that holds its states.

    A state is a value that changes as the system runs, such as the size of a population or the position of a particle.
    An element names its states and parameters, may give laws of its own, and is drawn with a visual. A subclass takes
    each state and parameter as an argument of its `__init__`, annotated with `StateSpec` or `ParameterSpec` inside
    `Annotated`, and stores it under the same name, as a scikit-learn estimator does. It returns its laws from
    `_get_laws`, or an empty list when it has none.

    Unlike `BaseComponent`, it holds states, and the interactions and laws of a system read and change them through it.
    Choose it when a new kind of element stores its members in a way of its own, as `GroupElement` does. Choose
    `Element` for one entity with named states.

    It appears in every field through its subclasses, such as a population in ecology, a chemical species in a reactor,
    a bank account in economics, and a pendulum in mechanics.

    Args:
        label:
            The name of the element, unique among the parts of its system, such as `'rabbits'`.

        visual:
            How the element is drawn, such as `VisualSpec('line')`, or `None` for `default_visual_`.

    Attributes:
        default_visual_:
            The visual of the element when it sets none.

        laws_:
            The laws of the element for the namespace of its last run.

    Examples:
        A population with one state and one parameter. The constructor stores its arguments as given.

        >>> from typing import Annotated
        >>> from skcomplex.core.components.elements import BaseElement
        >>> from skcomplex.core.specs import ParameterSpec, StateSpec, VisualSpec
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

    laws_: Sequence[BaseComponent]

    def _run(self, inputs: SimpleNamespace | None) -> None:
        """Set `laws_`, the laws of the element for a namespace of symbols.

        Args:
            inputs:
                The namespace of the symbols the laws can use.
        """
        self.laws_ = self._get_laws_of(inputs)

    def __init__(self, label: str, visual: VisualSpec | None = None) -> None:
        super().__init__(label)
        self.visual = visual

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Set `default_visual_`, the plain visual of an element.

        Args:
            context:
                The space the element inherits from its system, which an element does not read.
        """
        super()._prepare(context)
        self.default_visual_ = VisualSpec()

    def _get_domain_tags(self) -> frozenset[str]:
        """Return the domain tags of the states of the element."""
        domains = {spec._get_domain_tag() for spec in self.get_states().values()}
        return frozenset(domain for domain in domains if domain is not None)

    def _get_tags(self) -> Tags:
        """Return the tags of the element.

        The tags are the domains of its states and the laws its class lists.
        """
        return replace(self._get_law_tags(self._get_declared_laws()), domains=self._get_domain_tags())

    def get_states(self) -> dict[str, StateSpec]:
        """Return the states of the element with their specs.

        The states are those its class declares and those it takes as data.

        Returns:
            states:
                The spec of every state, by name.

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
