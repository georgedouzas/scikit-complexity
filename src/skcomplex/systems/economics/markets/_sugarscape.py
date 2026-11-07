"""Markets systems: Epstein and Axtell's Sugarscape of foragers on a landscape of regrowing sugar."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.analysis import distribution
from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import LifeCycleInteraction, MovementInteraction, TransferInteraction
from ....core.components.laws import Sink
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.system import BaseSystem, System
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import AgentView
from ....core.specs import BaseSpec, ObservableSpec, ParameterSpec, PhaseSpec, PhasesScheduleSpec, StateSpec, VisualSpec
from ....core.types import Array
from ..common.resources import RegrowthInteraction


class ForagerView(AgentView):
    """Forager as the rules see it: its wealth `sugar`, its `vision`, and its `metabolism`."""

    sugar: float
    vision: float
    metabolism: float


class ForagerElement(Element):
    """Agent that moves to the richest free cell in sight, harvests its sugar, burns its metabolism, and may starve.

    In its own activation, a forager sees `vision` cells along each of the four lattice directions and moves to the
    free cell with the most sugar, the nearest among equals, or stays when no cell in sight beats its own, by a
    `MovementInteraction` of the best step with a vision. Transfers then harvest the sugar of its cell and burn its
    metabolism, and a `LifeCycleInteraction` removes it when its wealth falls below zero.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        sugar:
            The wealth of the forager.

        vision:
            The number of cells the forager sees in every direction.

        metabolism:
            The sugar the forager burns in a step.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import ForagerElement
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

    @classmethod
    def _get_view(cls) -> type[ForagerView]:
        """Return the view of a forager, with its sugar, vision, and metabolism."""
        return ForagerView


class SugarPatchView(AgentView):
    """Patch as the rules see it: its `sugar`, its `capacity`, and its `growback`."""

    sugar: float
    capacity: float
    growback: float


@dataclass(frozen=True)
class _PatchMapSpec(BaseSpec):
    """Patches of a run by cell, which the foragers build once and keep in the storage of the system.

    Args:
        patches:
            The patch of every cell.
    """

    patches: Mapping[tuple[int, int], SugarPatchView]


class SugarPatchElement(Element):
    """Cell of the landscape whose sugar grows back by a fixed amount a step up to its capacity.

    The growback is a `RegrowthInteraction` of the group of patches.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        sugar:
            The sugar on the patch.

        capacity:
            The most sugar the patch holds.

        growback:
            The sugar that grows back in a step.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.markets import SugarPatchElement
        >>> GroupElement('patches', element=SugarPatchElement, size=4, values={'capacity': 2.0}).size
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

    @classmethod
    def _get_view(cls) -> type[SugarPatchView]:
        """Return the view of a patch, with its sugar, capacity, and growback."""
        return SugarPatchView


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
        >>> from skcomplex.systems.economics.markets._sugarscape import landscape
        >>> float(landscape(50, 50).max()), float(landscape(50, 50)[0, 49])
        (4.0, 0.0)
    """
    x, y = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
    scale = min(width, height) / 50
    distance = np.min([np.hypot(x - px * width, y - py * height) for px, py in PEAKS], axis=0)
    return np.sum([distance < bound * scale for bound in BOUNDS], axis=0).astype(float)


def _cell_of(location: object) -> tuple[int, int]:
    """Return the cell of a location on the lattice."""
    x, y = (int(item) for item in np.asarray(location, dtype=float))
    return x, y


def _get_patches(system: SystemViewProtocol) -> Mapping[tuple[int, int], SugarPatchView]:
    """Return the patches by cell, built once per run in the storage of the movement."""
    cached = system.storage.get('patches')
    if not isinstance(cached, _PatchMapSpec):
        patches = {
            _cell_of(patch.location): patch for patch in system.agents('patches') if isinstance(patch, SugarPatchView)
        }
        cached = _PatchMapSpec(patches)
        system.storage['patches'] = cached
    return cached.patches


def _get_sugar(agent: AgentViewProtocol, cell: object, system: SystemViewProtocol) -> float:
    """Return the sugar of the patch of a cell, the score a forager maximizes."""
    return float(_get_patches(system)[_cell_of(cell)].sugar)


def _get_vision(agent: AgentViewProtocol, system: SystemViewProtocol) -> float:
    """Return the number of cells a forager sees along every direction."""
    return float(agent.vision) if isinstance(agent, ForagerView) else 0.0


def _get_harvest(patch: AgentViewProtocol, forager: AgentViewProtocol, system: SystemViewProtocol) -> float:
    """Return the whole sugar of the patch of a forager."""
    return float(patch.sugar) if isinstance(patch, SugarPatchView) else 0.0


def _get_metabolism(forager: AgentViewProtocol, other: AgentViewProtocol, system: SystemViewProtocol) -> float:
    """Return the sugar a forager burns in a step."""
    return float(forager.metabolism) if isinstance(forager, ForagerView) else 0.0


def _is_starved(forager: AgentViewProtocol, system: SystemViewProtocol) -> bool:
    """Return whether the wealth of a forager has fallen below zero."""
    return isinstance(forager, ForagerView) and forager.sugar < 0


def _gini(wealth: object) -> float:
    """Return the Gini coefficient of the wealth of the foragers."""
    gini = distribution(np.asarray(wealth, dtype=float), kind='gini').gini
    return 0.0 if gini is None else gini


class SugarscapeSystem(BaseSystem):
    """The Sugarscape of Epstein and Axtell, foragers of random vision and metabolism on two sugar peaks.

    Every step the foragers, in random order, move to the richest free cell in sight, harvest it, and burn their
    metabolism, and those whose wealth falls below zero die. Then the sugar grows back. Foragers of poor vision or high
    metabolism starve, so the population falls to the carrying capacity of the landscape, and the wealth of the
    survivors becomes skewed, with a Gini coefficient that rises from its initial value. The system has the groups
    `patches` and `foragers`, whose catalog observables are the population `population` and the Gini coefficient of
    wealth `gini`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.markets import SugarscapeSystem
        >>> [group.size for group in SugarscapeSystem(10, 10, n_foragers=20).prepare().system_.elements]
        [100, 20]
    """

    field = 'economics/markets'
    topics = ('sugarscape', 'wealth', 'agents', 'grid')
    references = ('Argun et al. 2021 sec. 16.2', 'Downey 2018 ch. 9')
    until = 100
    observables = (
        ObservableSpec('population', group='foragers', variable='sugar', reduce='count'),
        ObservableSpec('gini', group='foragers', variable='sugar', reduce=_gini),
    )

    def __init__(
        self,
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
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.width = width
        self.height = height
        self.n_foragers = n_foragers
        self.max_vision = max_vision
        self.max_metabolism = max_metabolism
        self.low_sugar = low_sugar
        self.high_sugar = high_sugar
        self.growback = growback
        self.random_state = random_state

    @staticmethod
    def _get_foraging() -> list[MovementInteraction | TransferInteraction | LifeCycleInteraction]:
        """Return the member interactions of a forager: move to the best cell in sight, harvest, burn, and starve."""
        return [
            MovementInteraction(
                'search',
                target='foragers',
                step='best',
                score=_get_sugar,
                vision=_get_vision,
                exclusive=True,
                schedule='member',
                phase='foraging',
            ),
            TransferInteraction(
                'harvest',
                source='patches',
                target='foragers',
                state='sugar',
                amount=_get_harvest,
                pairing='cell',
                form='agent',
                actor='target',
                schedule='member',
                phase='foraging',
            ),
            TransferInteraction(
                'metabolism',
                source='foragers',
                target='foragers',
                state='sugar',
                into=Sink(),
                amount=_get_metabolism,
                form='agent',
                schedule='member',
                phase='foraging',
            ),
            LifeCycleInteraction(
                'starvation',
                target='foragers',
                death=_is_starved,
                schedule='member',
                phase='foraging',
            ),
        ]

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        capacity = landscape(self.width, self.height).ravel()
        cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        patches = GroupElement(
            'patches',
            element=SugarPatchElement,
            size=len(cells),
            values={'sugar': capacity, 'capacity': capacity, 'growback': self.growback},
            location=cells,
        )
        occupied = rng.choice(len(cells), size=min(self.n_foragers, len(cells)), replace=False)
        foragers = GroupElement(
            'foragers',
            element=ForagerElement,
            size=len(occupied),
            values={
                'sugar': rng.uniform(self.low_sugar, self.high_sugar, len(occupied)),
                'vision': rng.integers(1, self.max_vision + 1, len(occupied)).astype(float),
                'metabolism': rng.uniform(1.0, self.max_metabolism, len(occupied)),
            },
            variable=True,
            location=[cells[index] for index in occupied],
        )
        return System(
            self.label,
            elements=[patches, foragers],
            interactions=[
                *self._get_foraging(),
                RegrowthInteraction('growback', target='patches', state='sugar', growth='growback', phase='growth'),
            ],
            space=LatticeSpace((self.width, self.height), boundary='fixed', neighborhood='von_neumann'),
            time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec('foraging'), PhaseSpec('growth')])),
        )
