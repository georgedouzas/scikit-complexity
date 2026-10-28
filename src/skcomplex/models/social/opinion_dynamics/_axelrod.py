"""Axelrod's model of the dissemination of culture on a lattice."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet
from scipy import ndimage

from ....core.elements import Element, GroupElement
from ....core.interactions import Interaction
from ....core.laws import BaseLaw, RuleLaw
from ....core.spaces import LatticeSpace
from ....core.system import System
from ....engine.runtime import register_model
from ....engine.solvers import AgentView, ModelView
from ....specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....types import IntArray


def cultural_traits(culture: float, features: int, traits: int) -> IntArray:
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


class VillageElement(Element):
    """Village of Axelrod's model, with a culture of `F` features of `q` traits each, stored as one integer.

    Every step the village picks a random neighbour and, with probability equal to the fraction of features they
    share, copies the trait of one feature on which they differ.

    Args:
        label:
            The label of the agent or group.

        culture:
            The culture code, the traits of the features as the digits of a number in base `q`.

        features:
            The number of features `F`.

        traits:
            The number of traits `q` of every feature.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import VillageElement
        >>> GroupElement(
        ...     'villages', element=VillageElement, size=4, values={'culture': 0, 'features': 3, 'traits': 5}
        ... ).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'culture'}),
        *,
        culture: Annotated[float, StateSpec(domain=NN)] = 0,
        features: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 5.0,
        traits: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 10.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.culture = culture
        self.features = features
        self.traits = traits

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the cultural interaction rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.interact)]

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


class CultureCensusInteraction(Interaction):
    """Census of the cultures of the villages, which reports the share of the largest cultural region.

    Args:
        label:
            The label of the census.

        width:
            The number of columns of the grid.

        height:
            The number of rows of the grid.

        phase:
            The phase in which the census counts.

        **values:
            The values of the declared variables, by name:

            villages:
                The path of the group of villages.

    Examples:
        >>> from skcomplex.models.social.opinion_dynamics import CultureCensusInteraction
        >>> CultureCensusInteraction('census', villages='villages', width=5, height=5).width
        5
    """

    reports: ClassVar[tuple[str, ...]] = ('largest_region', 'cultures')

    def __init__(
        self,
        label: str,
        *,
        villages: Annotated[str, RoleSpec()],
        width: int = 10,
        height: int = 10,
        phase: str | None = None,
    ) -> None:
        super().__init__(label)
        self.villages = villages
        self.width = width
        self.height = height
        self.phase = phase

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of the census in its phase.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.act, phase=self.phase)]

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
    random_state: int | np.random.Generator | None = 0,
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
        >>> axelrod(5, 4).elements[0].check().prepare().size_
        20
    """
    rng = np.random.default_rng(random_state)
    size = width * height
    digits = rng.integers(traits, size=(size, features))
    cultures = (digits * traits ** np.arange(features)).sum(axis=1).astype(float)
    locations = [(x, y) for x in range(width) for y in range(height)]
    villages = GroupElement(
        'villages',
        element=VillageElement,
        size=size,
        values={'culture': cultures, 'features': features, 'traits': traits},
        location=locations,
    )
    return System(
        elements=[villages],
        interactions=[CultureCensusInteraction('census', villages='villages', width=width, height=height)],
        space=LatticeSpace((width, height), boundary='fixed', neighborhood='von_neumann', capacity=1),
    )
