"""Axelrod's system of the dissemination of culture on a lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from functools import partial
from typing import Annotated, Protocol, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet
from scipy import ndimage

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import AnyArray, IntArray, Role
from ._random_pairs import NeighborhoodsInteraction


class VillageView(AgentViewProtocol, Protocol):
    """View of a village of the Axelrod culture system.

    Its states and parameters are attributes.
    """

    culture: float
    features: float
    traits: float


class VillageElement(Element):
    """Village of Axelrod's system, with a culture of `F` features of `q` traits each.

    The culture is stored as one integer.

    Args:
        label:
            The label of the agent or group.

        culture:
            The culture code, the traits of the features as the digits of a number in base `q`.

        features:
            The number of features `F`.

        traits:
            The number of traits `q` of every feature.

        x:
            The column of the cell of the village on a lattice, one per village or shared.

        y:
            The row of the cell of the village on a lattice, one per village or shared.

        size:
            The number of villages, or `None` for one.

        space:
            The lattice the villages live on, whose axes `x` and `y` make `x` and `y` their cell, or `None`.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import VillageElement
        >>> VillageElement('villages', culture=0, features=3, traits=5, size=4).size
        4
    """

    visual = VisualSpec('agents', {'color_by': 'culture'})

    def __init__(
        self,
        label: str,
        *,
        culture: Annotated[object, NN] = 0,
        features: Annotated[float, RealSet.open(0, oo)] = 5.0,
        traits: Annotated[float, RealSet.open(0, oo)] = 10.0,
        x: object = 0,
        y: object = 0,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, size=size, space=space)
        self.culture = culture
        self.features = features
        self.traits = traits
        self.x = x
        self.y = y


def cultural_traits(culture: float, features: int, traits: int) -> IntArray:
    """Return the traits of a culture vector stored as one integer.

    They are its digits in base `q`.

    Args:
        culture:
            The integer code of the culture vector.

        features:
            The number of features `F`.

        traits:
            The number of traits `q` of every feature.

    Returns:
        traits:
            The trait of every feature, the first feature first.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import cultural_traits
        >>> cultural_traits(7, 3, 2).tolist()
        [1, 1, 1]
    """
    return (int(culture) // traits ** np.arange(features, dtype=np.int64)) % traits


class CulturalInfluenceInteraction(NeighborhoodsInteraction):
    """Cultural influence of Axelrod's system.

    Every step every village, in random order, picks a random neighbour and, with probability equal to the fraction of
    features they share, copies the trait of one feature on which they differ.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the group of villages.

        phase:
            The phase of the step the influence runs in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import CulturalInfluenceInteraction
        >>> CulturalInfluenceInteraction('influence', target='villages').target
        'villages'
    """

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        phase: str | None = None,
    ) -> None:
        super().__init__(label, target=target, phase=phase)

    def act_on(self, members: list[AgentViewProtocol], system: SystemView) -> None:
        """Copy one differing trait of a random neighbour.

        The probability is equal to the cultural overlap.

        Args:
            members:
                The village, followed by its neighbours.

            system:
                The system view of the rule.
        """
        agent, *neighbors = cast(list[VillageView], members)
        if not neighbors:
            return
        neighbor = neighbors[int(system.rng.integers(len(neighbors)))]
        features, traits = int(agent.features), int(agent.traits)
        own = cultural_traits(agent.culture, features, traits)
        other = cultural_traits(neighbor.culture, features, traits)
        different = np.flatnonzero(own != other)
        if len(different) and system.rng.random() < 1 - len(different) / features:
            feature = int(different[int(system.rng.integers(len(different)))])
            agent.culture += float((int(other[feature]) - int(own[feature])) * traits**feature)


def largest_region(locations: npt.ArrayLike, cultures: npt.ArrayLike, width: int, height: int) -> float:
    """Return the fraction of the cells in the largest connected region of one culture.

    Args:
        locations:
            The cell of every agent, one `(x, y)` row per agent.

        cultures:
            The culture code of every agent.

        width:
            The number of columns of the grid.

        height:
            The number of rows of the grid.

    Returns:
        fraction:
            The size of the largest region of adjacent cells with the same culture over the number of cells.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import largest_region
        >>> largest_region([(0, 0), (1, 0), (0, 1), (1, 1)], [1, 1, 2, 3], 2, 2)
        0.5
    """
    cells = np.asarray(locations, dtype=np.int64).reshape(-1, 2)
    grid = np.full((width, height), -1.0)
    grid[cells[:, 0], cells[:, 1]] = np.asarray(cultures, dtype=float)
    largest = 0
    for culture in np.unique(grid[grid >= 0]):
        labels, count = ndimage.label(grid == culture)
        if count:
            largest = max(largest, int(np.bincount(labels.ravel())[1:].max()))
    return largest / (width * height)


def _count_cultures(cultures: AnyArray) -> int:
    """Return the number of distinct cultures."""
    return len(np.unique(cultures))


def culture_observables(width: int, height: int) -> list[Observable]:
    """Return the share of the largest cultural region and the number of cultures.

    They are read from a full grid of villages.

    Args:
        width:
            The number of columns of the grid.

        height:
            The number of rows of the grid.

    Returns:
        observables:
            The observables `largest_region` and `cultures`, each reducing the cultures of the group `villages`.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import culture_observables
        >>> [item.label for item in culture_observables(5, 4)]
        ['largest_region', 'cultures']
    """
    locations = [(x, y) for x in range(width) for y in range(height)]
    region = partial(largest_region, locations, width=width, height=height)
    return [
        Observable('largest_region', group='villages', variable='culture', reduce=region),
        Observable('cultures', group='villages', variable='culture', reduce=_count_cultures),
    ]


class AxelrodSystem(BaseSystem):
    """Axelrod's culture dissemination system on a full grid.

    The grid has von Neumann neighbourhoods. Every village starts with random traits. For few traits per feature the
    villages converge to one culture, and for many traits they freeze into many small cultural regions, an
    order-disorder transition at a critical `q`. The system has the group `villages` and the influence `influence`.
    `culture_observables` gives the share of the largest cultural region and the number of cultures.

    Args:
        width:
            The number of columns.

        height:
            The number of rows.

        features:
            The number of cultural features `F`.

        traits:
            The number of traits `q` of every feature.

        random_state:
            The seed or generator of the initial cultures.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.social.opinion_dynamics import AxelrodSystem
        >>> AxelrodSystem(5, 4).prepare().system_.elements[0].size
        20
    """

    field = 'social/opinion_dynamics'
    topics = ('culture', 'agents', 'grid', 'social influence', 'phase transition')
    references = ('Axelrod 1997', 'Castellano, Fortunato, and Loreto 2009')
    until = 200

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        *,
        features: int = 5,
        traits: int = 10,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.width = width
        self.height = height
        self.features = features
        self.traits = traits
        self.random_state = random_state

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set `system_` to the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        size = self.width * self.height
        digits = rng.integers(self.traits, size=(size, self.features))
        cultures = (digits * self.traits ** np.arange(self.features)).sum(axis=1).astype(float)
        columns, rows = np.divmod(np.arange(size), self.height)
        villages = VillageElement(
            'villages',
            culture=cultures,
            features=self.features,
            traits=self.traits,
            x=columns,
            y=rows,
            size=size,
            space=LatticeSpace((self.width, self.height), boundary='fixed', metric='manhattan', capacity=1),
        )
        self.system_ = System(
            self.label,
            elements=[villages],
            interactions=[CulturalInfluenceInteraction('influence', target='villages')],
        )
        if not self.observables_:
            self.observables_ = tuple(culture_observables(self.width, self.height))
