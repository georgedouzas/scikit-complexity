"""Markets models: Epstein and Axtell's Sugarscape of foragers on a landscape of regrowing sugar."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated, Any, ClassVar

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.laws import BaseLaw, RuleLaw
from ....core.spaces import DiscreteTime, LatticeSpace
from ....core.specs import ParameterSpec, PhaseSpec, PhasesScheduleSpec, RoleSpec, StateSpec, VisualSpec
from ....core.system import System
from ....engine.runtime import register_model
from ....engine.solvers import AgentView, ModelView
from ..common.institutions import InstitutionInteraction

type Array = npt.NDArray[np.float64]

PEAKS = ((0.3, 0.3), (0.7, 0.7))
BOUNDS = (21, 16, 11, 6)
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def landscape(width: int, height: int) -> Array:
    """Return the sugar capacity of every cell, from 0 to 4 in rings around two peaks.

    The peaks sit at 30 and 70 percent of the extent of the grid, and the rings of Downey (2018 ch. 9) have radii of
    21, 16, 11, and 6 cells on a grid of 50 cells, scaled with the grid.

    Args:
        width:
            The number of columns.

        height:
            The number of rows.

    Returns:
        capacity:
            The capacity of the cell `(x, y)` at `[x, y]`.

    Examples:
        >>> from skcomplex.models.economics.markets._sugarscape import landscape
        >>> float(landscape(50, 50).max()), float(landscape(50, 50)[0, 49])
        (4.0, 0.0)
    """
    x, y = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
    scale = min(width, height) / 50
    distance = np.min([np.hypot(x - px * width, y - py * height) for px, py in PEAKS], axis=0)
    return np.sum([distance < bound * scale for bound in BOUNDS], axis=0).astype(float)


class PatchElement(Element):
    """Cell of the landscape whose sugar grows back by a fixed amount a step up to its capacity.

    Args:
        label:
            The label of the agent or group.

        location:
            The cell of the patch.

        sugar:
            The sugar on the patch.

        capacity:
            The most sugar the patch holds.

        growback:
            The sugar that grows back in a step.

    Examples:
        >>> from skcomplex.models.economics.markets import PatchElement
        >>> GroupElement('patches', element=PatchElement, size=4, values={'capacity': 2.0}).size
        4
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'sugar', 'color_range': (0.0, 4.0)}),
        *,
        sugar: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        capacity: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        growback: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.sugar = sugar
        self.capacity = capacity
        self.growback = growback

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the growback rule in the phase of growth.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.grow, phase='growth')]

    def grow(self, agent: AgentView, model: ModelView) -> None:
        """Grow sugar back up to the capacity.

        Args:
            agent:
                The patch.

            model:
                The model.
        """
        agent.sugar = min(agent.capacity, agent.sugar + agent.growback)


class ForagerElement(Element):
    """Agent that moves to the richest free cell in sight, harvests its sugar, burns its metabolism, and may starve.

    A forager sees `vision` cells along each of the four lattice directions and moves to the free cell with the most
    sugar, the nearest among equals, or stays when no cell in sight beats its own. It dies when its wealth falls
    below zero.

    Args:
        label:
            The label of the agent or group.

        patches:
            The path of the group of patches.

        sugar:
            The wealth of the forager.

        vision:
            The number of cells the forager sees in every direction.

        metabolism:
            The sugar the forager burns in a step.

    Examples:
        >>> from skcomplex.models.economics.markets import ForagerElement
        >>> GroupElement('foragers', element=ForagerElement, size=3, values={'vision': 2.0}).size
        3
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'sugar'}),
        *,
        sugar: Annotated[float, StateSpec()] = 10.0,
        vision: Annotated[float, StateSpec(domain=NN)] = 1.0,
        metabolism: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.sugar = sugar
        self.vision = vision
        self.metabolism = metabolism

    patches: ClassVar[str] = 'patches'

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the foraging rule in the phase of foraging.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.forage, phase='foraging')]

    def _patch_map(self, model: ModelView) -> dict[tuple[int, int], AgentView]:
        """Return the patches by cell, built once per run in the storage of the foragers."""
        if 'patches' not in model.storage:
            model.storage['patches'] = {
                tuple(int(item) for item in patch.location): patch  # type: ignore[attr-defined]
                for patch in model.agents(self.patches)
            }
        return model.storage['patches']

    def forage(self, agent: AgentView, model: ModelView) -> None:
        """Move to the best free cell in sight, harvest it, and starve when the wealth falls below zero.

        Args:
            agent:
                The forager.

            model:
                The model.
        """
        patches = self._patch_map(model)
        x, y = (int(item) for item in agent.location)  # type: ignore[attr-defined]
        best, best_sugar, best_distance = (x, y), patches[x, y].sugar, 0
        for dx, dy in DIRECTIONS:
            for distance in range(1, int(agent.vision) + 1):
                cell = (x + distance * dx, y + distance * dy)
                patch = patches.get(cell)
                if patch is None:
                    break
                better = patch.sugar > best_sugar or (patch.sugar == best_sugar and distance < best_distance)
                if better and not patch.agents_here(agent.group):
                    best, best_sugar, best_distance = cell, patch.sugar, distance
        if best != (x, y):
            agent.move_to(best)
        patch = patches[best]
        agent.sugar += patch.sugar - agent.metabolism
        patch.sugar = 0.0
        if agent.sugar < 0:
            agent.remove()


def gini(values: npt.ArrayLike) -> float:
    """Return the Gini coefficient of nonnegative values, zero for equal values and near one for one holder.

    Args:
        values:
            The values, such as the wealth of every agent.

    Returns:
        gini:
            The Gini coefficient, zero without values.

    Examples:
        >>> from skcomplex.models.economics.markets._sugarscape import gini
        >>> gini([1.0, 1.0]), gini([0.0, 0.0, 0.0, 3.0])
        (0.0, 0.75)
    """
    array = np.sort(np.asarray(values, dtype=float))
    if not len(array) or not array.sum():
        return 0.0
    ranks = np.arange(1, len(array) + 1)
    return float(np.sum((2 * ranks - len(array) - 1) * array) / (len(array) * array.sum()))


class CensusInteraction(InstitutionInteraction):
    """Census that reports the population of foragers and the Gini coefficient of their wealth.

    Args:
        label:
            The label of the institution.

        foragers:
            The path of the group of foragers.

        rule:
            Unused, since the census overrides `act`.

        phase:
            The phase the census counts in.

    Examples:
        >>> from skcomplex.models.economics.markets import CensusInteraction
        >>> CensusInteraction('census', foragers='foragers').reports
        ('population', 'gini')
    """

    reports: ClassVar[tuple[str, ...]] = ('population', 'gini')

    def __init__(
        self,
        label: str,
        *,
        rule: Callable[[object], None] | None = None,
        phase: str | None = None,
        foragers: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.foragers = foragers

    def act(self, model: Any) -> None:  # noqa: ANN401
        """Report the population and the Gini coefficient of wealth.

        Args:
            model:
                The model view of the census.
        """
        wealth = [forager.sugar for forager in model.agents('foragers')]
        model.report('population', len(wealth))
        model.report('gini', gini(wealth))


@register_model(
    field='economics/markets',
    tags=('sugarscape', 'wealth', 'agents', 'grid'),
    references=('Argun et al. 2021 sec. 16.2', 'Downey 2018 ch. 9'),
    defaults={'until': 100},
)
def sugarscape(
    width: int = 50,
    height: int = 50,
    *,
    n_foragers: int = 400,
    max_vision: int = 6,
    max_metabolism: float = 4.0,
    low_sugar: float = 5.0,
    high_sugar: float = 25.0,
    growback: float = 1.0,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the Sugarscape of Epstein and Axtell, foragers of random vision and metabolism on two sugar peaks.

    Every step the foragers, in random order, move to the richest free cell in sight, harvest it, and burn their
    metabolism, and those whose wealth falls below zero die. Then the sugar grows back. Foragers of poor vision or
    high metabolism starve, so the population falls to the carrying capacity of the landscape, and the wealth of the
    survivors becomes skewed, with a Gini coefficient that rises from its initial value.

    Args:
        width:
            The number of columns.

        height:
            The number of rows.

        n_foragers:
            The initial number of foragers.

        max_vision:
            The largest vision, every forager seeing from 1 to `max_vision` cells.

        max_metabolism:
            The largest metabolism, every forager burning from 1 to `max_metabolism` sugar a step.

        low_sugar:
            The lowest initial wealth.

        high_sugar:
            The highest initial wealth.

        growback:
            The sugar that grows back on every patch in a step.

        random_state:
            The seed or generator of the foragers' cells, visions, metabolisms, and wealth.

    Returns:
        system:
            The model, with the groups `patches` and `foragers` and the census `census`, whose reporters are the
            population `census.population` and the Gini coefficient of wealth `census.gini`.

    Examples:
        >>> from skcomplex.models.economics.markets import sugarscape
        >>> [group.size for group in sugarscape(10, 10, n_foragers=20).elements]
        [100, 20]
    """
    rng = np.random.default_rng(random_state)
    capacity = landscape(width, height).ravel()
    cells = [(x, y) for x in range(width) for y in range(height)]
    patches = GroupElement(
        'patches',
        element=PatchElement,
        size=len(cells),
        values={'sugar': capacity, 'capacity': capacity, 'growback': growback},
        location=cells,
    )
    occupied = rng.choice(len(cells), size=min(n_foragers, len(cells)), replace=False)
    foragers = GroupElement(
        'foragers',
        element=ForagerElement,
        size=len(occupied),
        values={
            'sugar': rng.uniform(low_sugar, high_sugar, len(occupied)),
            'vision': rng.integers(1, max_vision + 1, len(occupied)).astype(float),
            'metabolism': rng.uniform(1.0, max_metabolism, len(occupied)),
        },
        variable=True,
        location=[cells[index] for index in occupied],
    )
    return System(
        elements=[patches, foragers],
        interactions=[CensusInteraction('census', foragers='foragers', phase='growth')],
        space=LatticeSpace((width, height), boundary='fixed', neighborhood='von_neumann'),
        time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('foraging'), PhaseSpec('growth')])),
    )
