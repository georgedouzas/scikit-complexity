"""Markets systems: Epstein and Axtell's Sugarscape.

Foragers live on a landscape of regrowing sugar.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Protocol, runtime_checkable

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.analysis import distribution
from ....core.base import BaseSpec
from ....core.components import Observable
from ....core.components.elements import Element, VisualSpec
from ....core.components.interactions import Interaction
from ....core.components.laws import UpdateLaw
from ....core.components.spaces import BaseSpace, LatticeSpace
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.compile import PhaseSpec
from ....core.engine.solvers import BaseSolver, MesaSolver
from ....core.systems import BaseSystem, System
from ....core.types import Array
from ...general.agents import BirthDeathInteraction, MovementInteraction
from ..common.resources import RegrowthInteraction

PEAKS = ((0.3, 0.3), (0.7, 0.7))
BOUNDS = (21, 16, 11, 6)
DIRECTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))


@runtime_checkable
class ForagerView(AgentViewProtocol, Protocol):
    """Forager as the rules see it.

    It holds its wealth `sugar`, its `vision`, and its `metabolism`.
    """

    sugar: float
    vision: float
    metabolism: float


class ForagerElement(Element):
    """Agent that moves to the richest free cell in sight and harvests its sugar.

    It burns its metabolism, and may starve. In its own activation, a forager sees `vision` cells along each of the four
    lattice directions and moves to the free cell with the most sugar, the nearest among equals, or stays when no cell
    in sight beats its own, by a `MovementInteraction` of the best step with a vision. Transfers then harvest the sugar
    of its cell and burn its metabolism, and a `BirthDeathInteraction` removes it when its wealth falls below zero.

    Args:
        label:
            The label of the agent or group.

        sugar:
            The wealth of the forager.

        vision:
            The number of cells the forager sees in every direction.

        metabolism:
            The sugar the forager burns in a step.

        values:
            Further values by name, such as the cell `x` and `y` of every member.

        size:
            The number of members, or `None` for one.

        space:
            The grid whose axes `x` and `y` place the members, or `None` for none.

    Examples:
        >>> from skcomplex.systems.economics.markets import ForagerElement
        >>> ForagerElement('foragers', vision=2.0, size=3).size
        3
    """

    visual = VisualSpec('agents', {'color_by': 'sugar'})

    def __init__(
        self,
        label: str,
        *,
        sugar: object = 10.0,
        vision: Annotated[object, NN] = 1.0,
        metabolism: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.sugar = sugar
        self.vision = vision
        self.metabolism = metabolism


@runtime_checkable
class SugarPatchView(AgentViewProtocol, Protocol):
    """Patch as the rules see it: its `sugar`, its `capacity`, and its `growback`."""

    sugar: float
    capacity: float
    growback: float


@dataclass(frozen=True)
class _PatchMapSpec(BaseSpec):
    """Patches of a run by cell.

    The foragers build them once and keep them in the storage of the system.

    Args:
        patches:
            The patch of every cell.
    """

    patches: Mapping[tuple[int, int], SugarPatchView]


class SugarPatchElement(Element):
    """Cell of the landscape whose sugar grows back up to its capacity.

    The sugar grows back by a fixed amount a step. The growback is a `RegrowthInteraction` of the group of patches.

    Args:
        label:
            The label of the agent or group.

        sugar:
            The sugar on the patch.

        capacity:
            The most sugar the patch holds.

        growback:
            The sugar that grows back in a step.

        values:
            Further values by name, such as the cell `x` and `y` of every member.

        size:
            The number of members, or `None` for one.

        space:
            The grid whose axes `x` and `y` place the members, or `None` for none.

    Examples:
        >>> from skcomplex.systems.economics.markets import SugarPatchElement
        >>> SugarPatchElement('patches', capacity=2.0, size=4).size
        4
    """

    visual = VisualSpec('agents', {'color_by': 'sugar', 'color_range': (0.0, 4.0)})

    def __init__(
        self,
        label: str,
        *,
        sugar: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        capacity: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        growback: Annotated[object, RealSet.closed_open(0, oo)] = 1.0,
        values: Mapping[str, object] | None = None,
        size: int | None = None,
        space: BaseSpace | None = None,
    ) -> None:
        super().__init__(label, values=values, size=size, space=space)
        self.sugar = sugar
        self.capacity = capacity
        self.growback = growback


def _get_gini(wealth: object) -> float:
    """Return the Gini coefficient of the wealth of the foragers."""
    gini = distribution(np.asarray(wealth, dtype=float), kind='gini').gini
    return 0.0 if gini is None else gini


class SugarscapeSystem(BaseSystem):
    """The Sugarscape of Epstein and Axtell.

    Foragers of random vision and metabolism live on two sugar peaks. Every step the foragers, in random order, move to
    the richest free cell in sight, harvest it, and burn their metabolism, and those whose wealth falls below zero die.
    Then the sugar grows back. Foragers of poor vision or high metabolism starve, so the population falls to the
    carrying capacity of the landscape, and the wealth of the survivors becomes skewed, with a Gini coefficient that
    rises from its initial value. The system has the groups `patches` and `foragers`, whose catalog observables are the
    population `population` and the Gini coefficient of wealth `gini`.

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

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

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
        Observable('population', group='foragers', variable='sugar', reduce='count'),
        Observable('gini', group='foragers', variable='sugar', reduce=_get_gini),
    )

    @staticmethod
    def _get_cell_of(location: object) -> tuple[int, int]:
        """Return the cell of a location on the lattice."""
        x, y = (int(item) for item in np.asarray(location, dtype=float))
        return x, y

    @staticmethod
    def _get_patches(system: SystemViewProtocol) -> Mapping[tuple[int, int], SugarPatchView]:
        """Return the patches by cell, built once per run in the storage of the movement."""
        cached = system.storage.get('patches')
        if not isinstance(cached, _PatchMapSpec):
            patches = {
                SugarscapeSystem._get_cell_of(patch.location): patch
                for patch in system.get_agents('patches')
                if isinstance(patch, SugarPatchView)
            }
            cached = _PatchMapSpec(patches)
            system.storage['patches'] = cached
        return cached.patches

    @staticmethod
    def _get_sugar(agent: AgentViewProtocol, cell: object, system: SystemViewProtocol) -> float:
        """Return the sugar of the patch of a cell, the score a forager maximizes."""
        return float(SugarscapeSystem._get_patches(system)[SugarscapeSystem._get_cell_of(cell)].sugar)

    @staticmethod
    def _get_vision(agent: AgentViewProtocol, system: SystemViewProtocol) -> float:
        """Return the number of cells a forager sees along every direction."""
        return float(agent.vision) if isinstance(agent, ForagerView) else 0.0

    @staticmethod
    def _harvest(forager: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Move the whole sugar of the patch of a forager to the forager."""
        groups = set(system.get_groups('patches'))
        for patch in [other for other in forager.get_agents_here() if other.group in groups]:
            if forager.alive and patch.alive and isinstance(patch, SugarPatchView) and isinstance(forager, ForagerView):
                amount = float(patch.sugar)
                patch.sugar -= amount
                forager.sugar += amount

    @staticmethod
    def _burn(forager: AgentViewProtocol, system: SystemViewProtocol) -> None:
        """Burn the sugar a forager needs in a step."""
        del system
        if forager.alive and isinstance(forager, ForagerView):
            forager.sugar -= float(forager.metabolism)

    @staticmethod
    def _is_starved(forager: AgentViewProtocol, system: SystemViewProtocol) -> bool:
        """Return whether the wealth of a forager has fallen below zero."""
        return isinstance(forager, ForagerView) and forager.sugar < 0

    @staticmethod
    def _get_landscape(width: int, height: int) -> Array:
        """Return the sugar capacity of every cell, from 0 to 4 in rings around two peaks.

        The peaks sit at 30 and 70 percent of the extent of the grid, and the rings of Downey (2018 ch. 9) have radii
        of 21, 16, 11, and 6 cells on a grid of 50 cells, scaled with the grid. The capacity of the cell `(x, y)` is
        at `[x, y]`.
        """
        x, y = np.meshgrid(np.arange(width), np.arange(height), indexing='ij')
        scale = min(width, height) / 50
        distance = np.min([np.hypot(x - px * width, y - py * height) for px, py in PEAKS], axis=0)
        return np.sum([distance < bound * scale for bound in BOUNDS], axis=0).astype(float)

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
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
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
    def _get_foraging() -> list[MovementInteraction | Interaction | BirthDeathInteraction]:
        """Return the member interactions of a forager.

        They move to the best cell in sight, harvest, burn, and starve.
        """
        return [
            MovementInteraction(
                'search',
                target='foragers',
                step='best',
                score=SugarscapeSystem._get_sugar,
                vision=SugarscapeSystem._get_vision,
                exclusive=True,
                schedule='member',
                phase='foraging',
            ),
            Interaction(
                'harvest',
                roles={'target': 'foragers', 'patches': 'patches'},
                laws=lambda s: [UpdateLaw(s.target, SugarscapeSystem._harvest, phase='foraging')],
            ),
            Interaction(
                'metabolism',
                roles={'target': 'foragers'},
                laws=lambda s: [UpdateLaw(s.target, SugarscapeSystem._burn, phase='foraging')],
            ),
            BirthDeathInteraction(
                'starvation',
                target='foragers',
                death=SugarscapeSystem._is_starved,
                schedule='member',
                phase='foraging',
            ),
        ]

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        capacity = self._get_landscape(self.width, self.height).ravel()
        cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        space = LatticeSpace((self.width, self.height), boundary='fixed', metric='manhattan')
        xs, ys = np.array(cells).T
        patches = SugarPatchElement(
            'patches',
            sugar=capacity,
            capacity=capacity,
            growback=self.growback,
            values={'x': xs, 'y': ys},
            size=len(cells),
            space=space,
        )
        occupied = rng.choice(len(cells), size=min(self.n_foragers, len(cells)), replace=False)
        foragers = ForagerElement(
            'foragers',
            sugar=rng.uniform(self.low_sugar, self.high_sugar, len(occupied)),
            vision=rng.integers(1, self.max_vision + 1, len(occupied)).astype(float),
            metabolism=rng.uniform(1.0, self.max_metabolism, len(occupied)),
            values={'x': xs[occupied], 'y': ys[occupied]},
            size=len(occupied),
            space=space,
        )
        self.system_ = System(
            self.label,
            elements=[patches, foragers],
            interactions=[
                *self._get_foraging(),
                RegrowthInteraction('growback', target='patches', state='sugar', growth='growback', phase='growth'),
            ],
            space=space,
            solver=MesaSolver(phases=[PhaseSpec('foraging'), PhaseSpec('growth')]),
        )
