"""Employment systems: employers and workers on a checkerboard.

They follow each other's colours.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

import numpy as np
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components.elements import Element, LatticeElement
from ....core.components.interactions import LocalRuleInteraction
from ....core.components.spaces import LatticeNeighborhood, LatticeSpace
from ....core.components.system import BaseSystem, System
from ....core.specs import StateSpec, VisualSpec
from ....core.types import Array

TAKEOVER = 3
KEEP = 2
NEIGHBORS = LatticeNeighborhood(((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)))


class WorkplaceElement(Element):
    """Cell of the employment takeover system.

    It is an employer, white 0 or black 1, or a worker, red 0 or blue 1. Employers and workers alternate like the
    squares of a checkerboard. Every step each worker first takes the colour of the employer to its right, at `x + 1`.
    Then each employer turns black when at least three of its four workers are blue, stays as it is when two are, and
    turns white otherwise. The two phases are local rule interactions of the lattice, which `EmploymentTakeoverSystem`
    adds.

    Args:
        label:
            The label of the cell or the lattice group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        colour:
            The initial colour, 0 or 1.

        employer:
            Whether the cell is an employer, 1, or a worker, 0.

    Examples:
        >>> from skcomplex.systems.social.organizations import WorkplaceElement
        >>> WorkplaceElement('cell', employer=1).employer
        1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        colour: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
        employer: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.colour = colour
        self.employer = employer


class EmploymentTakeoverSystem(BaseSystem):
    """The employment takeover system on a periodic checkerboard.

    Half of each kind is coloured at random. The system has the colours `cells.colour` and the employers
    `cells.employer`.

    Args:
        size:
            The even number of rows and columns.

        random_state:
            The seed or generator of the cells coloured 1.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.organizations import EmploymentTakeoverSystem
        >>> EmploymentTakeoverSystem(4).prepare().system_.elements[0].prepare().size_
        16
    """

    field = 'social/organizations'
    topics = ('lattice', 'employment', 'takeover', 'deterministic')
    references = ('Chesney 2021',)
    until = 50
    dt = 1

    def __init__(
        self,
        size: int = 32,
        *,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.size = size
        self.random_state = random_state

    @staticmethod
    def _follow_employer(*values: Array) -> Array:
        """Return the colours after every worker follows its employer.

        Every worker takes the colour of the employer at `x + 1`.
        """
        colours, employers = values[:5], values[5:]
        return np.where(employers[0] > 0, colours[0], colours[1])

    @staticmethod
    def _follow_workers(*values: Array) -> Array:
        """Return the colours after every employer follows its workers.

        Every employer counts its workers of colour 1 and follows a majority of three.
        """
        colours, employers = values[:5], values[5:]
        count = sum(colours[1:])
        following = np.where(count >= TAKEOVER, 1.0, np.where(count == KEEP, colours[0], 0.0))
        return np.where(employers[0] > 0, following, colours[0])

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        rows, columns = np.indices((self.size, self.size))
        employers = ((rows + columns) % 2 == 0).astype(float)
        colours = np.zeros((self.size, self.size))
        for kind in (1.0, 0.0):
            cells = np.flatnonzero(employers.ravel() == kind)
            colours.ravel()[rng.choice(cells, len(cells) // 2, replace=False)] = 1.0
        cells = LatticeElement(
            'cells',
            element=WorkplaceElement,
            space=LatticeSpace((self.size, self.size)),
            values={'colour': colours, 'employer': employers},
        )
        phases = [
            LocalRuleInteraction(
                label,
                target='cells',
                state='colour',
                rule=rule,
                neighborhood=NEIGHBORS,
                reads=('colour', 'employer'),
            )
            for label, rule in (('workers', self._follow_employer), ('employers', self._follow_workers))
        ]
        return System(self.label, elements=[cells], interactions=phases)
