"""Employment models: employers and workers on a checkerboard who follow each other's colours."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, state
from ....core.components import Element
from ....core.laws import Law, LocalRule, Symbols
from ....core.runtime import register_model
from ....core.space import LatticeSpace, Neighborhood
from ....core.system import System

type Array = npt.NDArray[np.float64]

TAKEOVER = 3
KEEP = 2
NEIGHBORS = Neighborhood(((0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)))


def _follow_employer(*values: Array) -> Array:
    """Return the colours after every worker takes the colour of the employer at `x + 1`."""
    colours, employers = values[:5], values[5:]
    return np.where(employers[0] > 0, colours[0], colours[1])


def _follow_workers(*values: Array) -> Array:
    """Return the colours after every employer counts its workers of colour 1 and follows a majority of three."""
    colours, employers = values[:5], values[5:]
    count = sum(colours[1:])
    following = np.where(count >= TAKEOVER, 1.0, np.where(count == KEEP, colours[0], 0.0))
    return np.where(employers[0] > 0, following, colours[0])


class Workplace(Element):
    """Cell of the employment takeover model: an employer, white at 0 or black at 1, or a worker, red at 0 or blue at 1.

    Employers and workers alternate like the squares of a checkerboard. Every step each worker first takes the colour
    of the employer to its right, at `x + 1`. Then each employer turns black when at least three of its four workers
    are blue, stays as it is when two are, and turns white otherwise.

    Args:
        label:
            The label of the cell or the lattice group.

        colour:
            The initial colour, 0 or 1.

        employer:
            Whether the cell is an employer, 1, or a worker, 0.

    Examples:
        >>> from skcomplex.models.social.organizations import Workplace
        >>> Workplace('cell', employer=1).employer
        1
    """

    colour: float = state(default=0, domain=Domain.BOOLEAN)
    employer: float = state(default=0, domain=Domain.BOOLEAN)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the worker phase and the employer phase.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        reads = (s.colour, s.employer)
        return [
            LocalRule(s.colour, NEIGHBORS, _follow_employer, reads=reads),
            LocalRule(s.colour, NEIGHBORS, _follow_workers, reads=reads),
        ]


@register_model(
    field='social/organizations',
    tags=('lattice', 'employment', 'takeover', 'deterministic'),
    references=('Chesney 2021',),
    defaults={'until': 50, 'dt': 1},
)
def employment_takeover(size: int = 32, *, random_state: int | np.random.Generator | None = 0) -> System:
    """Return the employment takeover model on a periodic checkerboard, with half of each kind coloured at random.

    Args:
        size:
            The even number of rows and columns.

        random_state:
            The seed or generator of the cells coloured 1.

    Returns:
        system:
            The model, with the colours `cells.colour` and the employers `cells.employer`.

    Examples:
        >>> from skcomplex.models.social.organizations import employment_takeover
        >>> employment_takeover(4).elements[0].size
        16
    """
    rng = np.random.default_rng(random_state)
    rows, columns = np.indices((size, size))
    employers = ((rows + columns) % 2 == 0).astype(float)
    colours = np.zeros((size, size))
    for kind in (1.0, 0.0):
        cells = np.flatnonzero(employers.ravel() == kind)
        colours.ravel()[rng.choice(cells, len(cells) // 2, replace=False)] = 1.0
    cells = Workplace.lattice('cells', LatticeSpace((size, size)), colour=colours, employer=employers)
    return System(components=[cells])
