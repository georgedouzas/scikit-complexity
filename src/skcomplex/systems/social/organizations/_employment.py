"""Employment systems: employers and workers on a checkerboard.

They follow each other's colours.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet

from ....core.components import Observable
from ....core.components.base import BaseComponent
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace, LatticeNeighborhood, LatticeSpace
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role

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

        colour:
            The initial colour, 0 or 1.

        employer:
            Whether the cell is an employer, 1, or a worker, 0.

        space:
            The space with one member per site, or `None` for one member.

    Examples:
        >>> from skcomplex.systems.social.organizations import WorkplaceElement
        >>> WorkplaceElement('cell', employer=1).employer
        1
    """

    def __init__(
        self,
        label: str,
        *,
        colour: Annotated[object, FiniteEnumeratedSet([0, 1])] = 0,
        employer: Annotated[object, FiniteEnumeratedSet([0, 1])] = 0,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, space=space)
        self.colour = colour
        self.employer = employer


class EmploymentTakeoverInteraction(Interaction):
    """Colours of workers and employers that follow each other on a checkerboard.

    Every step each worker first takes the colour of the employer to its right, and then each employer takes the colour
    of three or four of its four workers, keeping its own colour on a tie of two.

    Args:
        label:
            The label of the interaction.

        target:
            The cells whose state `colour` the rules update, by their states `colour` and `employer`.

    Examples:
        >>> from skcomplex.systems.social.organizations import EmploymentTakeoverInteraction
        >>> EmploymentTakeoverInteraction('takeover', target='cells').target
        'cells'
    """

    def __init__(self, label: str, *, target: Role = 'cells') -> None:
        super().__init__(label)
        self.target = target

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

    def _prepare(self, context: BaseSpace | SimpleNamespace | None = None) -> None:
        """Set the workers following their employers, then the employers following their workers, as the laws."""
        super()._prepare(context)
        if isinstance(context, SimpleNamespace):
            colour, employer = context.target.colour, context.target.employer
            reads = (colour, employer)
            self.laws_: list[BaseComponent] = [
                UpdateLaw(colour, self._follow_employer, neighborhood=NEIGHBORS, reads=reads, label='workers'),
                UpdateLaw(colour, self._follow_workers, neighborhood=NEIGHBORS, reads=reads, label='employers'),
            ]


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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.organizations import EmploymentTakeoverSystem
        >>> EmploymentTakeoverSystem(4).prepare().system_.elements[0].colour.size
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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.size = size
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        rows, columns = np.indices((self.size, self.size))
        employers = ((rows + columns) % 2 == 0).astype(float)
        colours = np.zeros((self.size, self.size))
        for kind in (1.0, 0.0):
            cells = np.flatnonzero(employers.ravel() == kind)
            colours.ravel()[rng.choice(cells, len(cells) // 2, replace=False)] = 1.0
        cells = WorkplaceElement(
            'cells',
            colour=colours,
            employer=employers,
            space=LatticeSpace((self.size, self.size)),
        )
        takeover = EmploymentTakeoverInteraction('takeover', target='cells')
        self.system_ = System(self.label, elements=[cells], interactions=[takeover])
