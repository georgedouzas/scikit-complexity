"""Define systems, which compose elements, interactions, and subsystems."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

from sklearn.base import BaseEstimator

from ._components import Element, Interaction
from ._spaces import Space


class System(BaseEstimator):
    """Composition of elements, interactions, and child systems.

    An interaction refers to elements by their path relative to the system that holds the interaction, so
    `forest/foxes` names the element `foxes` of the child system `forest`.

    Args:
        label:
            The label of the system. It names the system when it is the child of another one.

        elements:
            The elements of the system.

        interactions:
            The interactions of the system.

        systems:
            The child systems.

        space:
            The space that gives vector states their dimension. Child systems without a space use it too.

    Examples:
        >>> from skcomplex.model import System
        >>> from skcomplex.models.ecology import LogisticGrowth, Population
        >>> system = System(
        ...     elements=[Population('rabbits', n=10)],
        ...     interactions=[LogisticGrowth('growth', target='rabbits', r=1.0, K=100)],
        ... )
        >>> system
        System(elements=1, interactions=1, systems=0)
    """

    def __init__(
        self,
        label: str = 'system',
        *,
        elements: Sequence[Element] = (),
        interactions: Sequence[Interaction] = (),
        systems: Sequence['System'] = (),
        space: Space | None = None,
    ) -> None:
        self.label = label
        self.elements = elements
        self.interactions = interactions
        self.systems = systems
        self.space = space

    def __repr__(self, N_CHAR_MAX: int = 700) -> str:
        """Return the label and the number of children of each kind."""
        label = '' if self.label == 'system' else f'{self.label!r}, '
        return (
            f'System({label}elements={len(self.elements)}, interactions={len(self.interactions)}, '
            f'systems={len(self.systems)})'
        )
