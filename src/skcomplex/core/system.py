"""Define systems, which compose elements, interactions, and nested systems on a space, in a time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Self

from .base import BasePart
from .elements import BaseElement
from .interactions import BaseInteraction
from .spaces import BaseSpace, BaseTime


class System(BasePart):
    """Composition of elements and the interactions between them, with nested systems, on a space, in a time.

    A system is defined by its parts and by nothing about how it runs. An interaction refers to an element by its path
    relative to the system that holds the interaction, so `forest/foxes` names the element `foxes` of the nested system
    `forest`. Groups and nested systems without a space of their own live on the space of the system.

    Args:
        label:
            The label of the system, which names it when it is nested in another one.

        elements:
            The elements and groups.

        interactions:
            The interactions between the elements.

        systems:
            The nested systems.

        space:
            The space the elements live on, which gives vector states their dimension, or `None` for none.

        time:
            The time the system evolves in, `ContinuousTime`, `DiscreteTime` with its schedule, or `StaticTime`, or
            `None` to infer it from the laws.

    Examples:
        >>> from typing import Annotated
        >>> from types import SimpleNamespace
        >>> from sage.all__sagemath_symbolics import var
        >>> from skcomplex.core.elements import Element
        >>> from skcomplex.core.laws import RateLaw
        >>> from skcomplex.core.specs import StateSpec
        >>> from skcomplex.core.system import System
        >>> class DecayElement(Element):
        ...     def __init__(self, label, x: Annotated[float, StateSpec()] = 1.0):
        ...         super().__init__(label)
        ...         self.x = x
        ...
        ...     def _get_laws(self, namespace):
        ...         return [RateLaw(namespace.x, -namespace.x)]
        >>> system = System(elements=[DecayElement('decay')]).check().prepare()
        >>> namespace = SimpleNamespace(decay=SimpleNamespace(x=var('x')))
        >>> [type(law).__name__ for law in system.run(namespace)]
        ['RateLaw']
    """

    def __init__(
        self,
        label: str = 'system',
        *,
        elements: Sequence[BaseElement] = (),
        interactions: Sequence[BaseInteraction] = (),
        systems: Sequence['System'] = (),
        space: BaseSpace | None = None,
        time: BaseTime | None = None,
    ) -> None:
        """Store the label and the parts of the system."""
        super().__init__(label)
        self.elements = elements
        self.interactions = interactions
        self.systems = systems
        self.space = space
        self.time = time

    def _get_parts(self) -> list[BasePart]:
        """Return the parts of the system: its space, its time, its elements, its interactions, and its systems."""
        around = [part for part in (self.space, self.time) if part is not None]
        return [*around, *self.elements, *self.interactions, *self.systems]

    def _check(self) -> Self:
        """Check every part of the system.

        Returns:
            system:
                The system itself.
        """
        for part in self._get_parts():
            part.check()
        return self

    def _prepare(self, context: object = None) -> Self:
        """Prepare the space and the time, then the elements and the nested systems on the space, then the interactions.

        Args:
            context:
                The space a nested system inherits from the system that holds it, or `None`.

        Returns:
            system:
                The system itself.
        """
        space = self.space if self.space is not None else context
        for part in (self.space, self.time):
            if part is not None:
                part.prepare()
        for part in (*self.elements, *self.systems):
            part.prepare(space)
        for part in self.interactions:
            part.prepare()
        return self

    def _get_laws(self, namespace: object) -> list[object]:
        """Return the laws of every part, each for the namespace of its label.

        Args:
            namespace:
                The namespace of the system, which holds the namespace of every part under its label.

        Returns:
            laws:
                The laws of the space, the elements, the interactions, and the nested systems, in that order.
        """
        return [law for part in self._get_parts() for law in part._get_laws(getattr(namespace, part.label, None))]
