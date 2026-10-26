"""Define systems, which compose components and interactions on a space, in a time."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import ClassVar

from sklearn.base import BaseEstimator

from ..base import BaseInteraction, ElementKind
from ..components import AggregateQuantity, Element, GroupElement
from ..space import BaseSpace, BaseTime


class System(BaseEstimator):
    """Composition of components and the interactions between them, on a space, in a time.

    A system is defined by its components, its interactions, its space, and its time, and by nothing about how it
    runs. Components are elements, groups, and nested systems, so systems nest to any depth. An interaction
    refers to a component by its path relative to the system that holds the interaction, so `forest/foxes` names the
    element `foxes` of the nested system `forest`.

    Args:
        label:
            The label of the system. It names the system when it is nested in another one.

        components:
            The elements, groups, and nested systems.

        interactions:
            The interactions between the components.

        aggregates:
            The quantities that reduce a group to one value, such as its size, which roles bind like elements.

        space:
            The substrate the components live on, which gives vector states their dimension. Nested systems without a
            space use it too.

        time:
            The time the system evolves in, `ContinuousTime`, `DiscreteTime` with its schedule, or `StaticTime`, or
            `None` to infer it from the laws.


    Examples:
        >>> from skcomplex.core.system import System
        >>> from skcomplex.models.biology.ecology import LogisticGrowthInteraction, PopulationElement
        >>> system = System(
        ...     components=[PopulationElement('rabbits', n=10)],
        ...     interactions=[LogisticGrowthInteraction('growth', target='rabbits', r=1.0, K=100)],
        ... )
        >>> [element.label for element in system.elements], [item.label for item in system.interactions]
        (['rabbits'], ['growth'])
    """

    component_kind: ClassVar[ElementKind] = ElementKind.SYSTEM

    def __init__(
        self,
        label: str = 'system',
        *,
        components: Sequence['Element | GroupElement | System'] = (),
        interactions: Sequence[BaseInteraction] = (),
        aggregates: Sequence[AggregateQuantity] = (),
        space: BaseSpace | None = None,
        time: BaseTime | None = None,
    ) -> None:
        self.label = label
        self.components = components
        self.interactions = interactions
        self.aggregates = aggregates
        self.space = space
        self.time = time

    @property
    def elements(self) -> tuple['Element | GroupElement', ...]:
        """Return the elements and groups among the components, in order."""
        return tuple(component for component in self.components if not isinstance(component, System))

    @property
    def systems(self) -> tuple['System', ...]:
        """Return the nested systems among the components, in order."""
        return tuple(component for component in self.components if isinstance(component, System))
