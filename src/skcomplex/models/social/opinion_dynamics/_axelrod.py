"""Axelrod's model of the dissemination of culture on a lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from ....core.base import Domain, VisualSpec, parameter, role, state
from ....core.model import AgentType, GridSpace, Institution, Law, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

type Integers = npt.NDArray[np.int64]


def cultural_traits(culture: float, features: int, traits: int) -> Integers:
    """Return the traits of a culture vector stored as one integer, its digits in base `q`.

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
        >>> from skcomplex.models.social.opinion_dynamics import cultural_traits
        >>> cultural_traits(7, 3, 2).tolist()
        [1, 1, 1]
    """
    return (int(culture) // traits ** np.arange(features, dtype=np.int64)) % traits


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
        >>> from skcomplex.models.social.opinion_dynamics import largest_region
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


class Village(AgentType):
    """Village of Axelrod's model, with a culture of `F` features of `q` traits each, stored as one integer.

    Every step the village picks a random neighbour and, with probability equal to the fraction of features they
    share, copies the trait of one feature on which they differ.

    Args:
        label:
            The label of the agent type or group.

        location:
            The cell of the village.

        culture:
            The culture code, the traits of the features as the digits of a number in base `q`.

        features:
            The number of features `F`.

        traits:
            The number of traits `q` of every feature.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import Village
        >>> Village.group('villages', 4, culture=0, features=3, traits=5).size
        4
    """

    visual: ClassVar[VisualSpec] = VisualSpec('agents', {'color_by': 'culture'})
    culture: float = state(default=0, domain=Domain.NONNEGATIVE_INTEGER)
    features: float = parameter(default=5.0, domain=Domain.POSITIVE)
    traits: float = parameter(default=10.0, domain=Domain.POSITIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the cultural interaction rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.interact)]

    def interact(self, agent: AgentView, model: ModelView) -> None:
        """Copy one differing trait of a random neighbour with probability equal to the cultural overlap.

        Args:
            agent:
                The village.

            model:
                The model.
        """
        neighbors = agent.neighbors()
        if not neighbors:
            return
        neighbor = neighbors[int(model.rng.integers(len(neighbors)))]
        features, traits = int(agent.features), int(agent.traits)
        own = cultural_traits(agent.culture, features, traits)
        other = cultural_traits(neighbor.culture, features, traits)
        different = np.flatnonzero(own != other)
        if len(different) and model.rng.random() < 1 - len(different) / features:
            feature = int(different[int(model.rng.integers(len(different)))])
            agent.culture += float((int(other[feature]) - int(own[feature])) * traits**feature)


class CultureCensus(Institution):
    """Census of the cultures of the villages, which reports the share of the largest cultural region.

    Args:
        label:
            The label of the census.

        villages:
            The path of the group of villages.

        width:
            The number of columns of the grid.

        height:
            The number of rows of the grid.

        phase:
            The phase in which the census counts.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import CultureCensus
        >>> CultureCensus('census', villages='villages', width=5, height=5).width
        5
    """

    reports: ClassVar[tuple[str, ...]] = ('largest_region', 'cultures')
    villages: str = role()
    width: int = 10
    height: int = 10

    def act(self, model: ModelView) -> None:
        """Report the share of the largest region and the number of distinct cultures.

        Args:
            model:
                The model view of the census.
        """
        agents = model.agents('villages')
        cultures = [agent.culture for agent in agents]
        locations = np.array([agent.location for agent in agents], dtype=np.int64)
        model.report('largest_region', largest_region(locations, cultures, self.width, self.height))
        model.report('cultures', len(set(cultures)))


@register_model(
    field='social/opinion_dynamics',
    tags=('culture', 'agents', 'grid', 'social influence', 'phase transition'),
    references=('Axelrod 1997', 'Castellano, Fortunato, and Loreto 2009'),
    defaults={'until': 200},
)
def axelrod(
    width: int = 20,
    height: int = 20,
    *,
    features: int = 5,
    traits: int = 10,
    random_state: int | np.random.Generator | None = None,
) -> System:
    """Return Axelrod's culture dissemination model on a full grid with von Neumann neighbourhoods.

    Every village starts with random traits. For few traits per feature the villages converge to one culture, and
    for many traits they freeze into many small cultural regions, an order-disorder transition at a critical `q`.

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

    Returns:
        system:
            The model, with the group `villages` and the census `census`, which reports the share of the largest
            cultural region `census.largest_region` and the number of cultures `census.cultures`.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import axelrod
        >>> axelrod(5, 4).elements[0].size
        20
    """
    rng = np.random.default_rng(random_state)
    size = width * height
    digits = rng.integers(traits, size=(size, features))
    cultures = (digits * traits ** np.arange(features)).sum(axis=1).astype(float)
    locations = [(x, y) for x in range(width) for y in range(height)]
    villages = Village.group('villages', size, location=locations, culture=cultures, features=features, traits=traits)
    return System(
        elements=[villages],
        interactions=[CultureCensus('census', villages='villages', width=width, height=height)],
        space=GridSpace(width, height, neighborhood='von_neumann', capacity=1),
    )
