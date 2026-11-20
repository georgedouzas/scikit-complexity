"""Markets systems: Chesney's Fair Food Program.

Tomato buyers and farms join a programme of fair wages.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated, ClassVar, Protocol, cast

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.types import AgentViewProtocol
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver, SystemView
from ....core.specs import VisualSpec
from ....core.systems import BaseSystem, System
from ....core.types import Array, Role
from ..common.policy import PolicyRuleInteraction

OUTSIDE, COMMITTED, MEMBER = 0, 1, 2


class ProgramMemberView(AgentViewProtocol, Protocol):
    """Buyer or farm as the programme sees it.

    It holds its acres of tomatoes, its level, and what it traded in the period.
    """

    acres: float
    level: float
    traded: float


class ProgramMemberElement(Element):
    """Tomato buyer or farm with its acres of tomatoes.

    It holds its level in the Fair Food Program and what it traded. A buyer needs `acres` of tomatoes every period and
    has the level 0 outside the programme, 1 when it buys from member farms first, and 2 when it buys only from member
    farms. A farm grows `acres` of tomatoes and has the level 0 outside the programme and 1 inside. `traded` is what a
    buyer bought or a farm sold in the period.

    Args:
        label:
            The label of the agent or group.

        acres:
            The tomatoes a buyer needs or a farm grows every period, in acres.

        level:
            The level of involvement in the programme.

        traded:
            The tomatoes bought or sold in the period.

        size:
            The number of members, or `None` for one.

    Examples:
        >>> from skcomplex.systems.economics.markets import ProgramMemberElement
        >>> ProgramMemberElement('farms', acres=10.0, size=3).size
        3
    """

    visual = VisualSpec('agents', {'color_by': 'level'})

    def __init__(
        self,
        label: str,
        *,
        acres: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        level: Annotated[object, NN] = 0.0,
        traded: Annotated[object, RealSet.closed_open(0, oo)] = 0.0,
        size: int | None = None,
    ) -> None:
        super().__init__(label, size=size)
        self.acres = acres
        self.level = level
        self.traded = traded


class FairFoodMarketInteraction(PolicyRuleInteraction[SystemView]):
    """Tomato market and the Fair Food Program.

    Buyers and farms join the programme by the four rules of Chesney. Every period, in the order of the code of Chesney
    (2021, Section 7.6), every buyer outside the programme joins it at level 1 with the probability `pressure` (Rule 1).
    Every buyer at level 1, in random order, moves to level 2 when the member farms grow at least what the buyers at
    level 2 need plus its own need (Rule 2). The buyers then fill their needs in random order from the farms in random
    order, any farm at level 0, member farms first at level 1, and only member farms at level 2 (Rule 3). Finally every
    farm outside the programme joins it when it grows more than the buyers outside the programme need (Rule 4). Levels
    only rise.

    Args:
        label:
            The label of the market.

        rule:
            Unused, since the market overrides `act`.

        phase:
            The phase the market acts in, or `None` for the default phase.

        shops:
            The path of the group of buyers.

        farms:
            The path of the group of farms.

        pressure:
            The probability per period that a buyer outside the programme joins it at level 1.

    Examples:
        >>> from skcomplex.systems.economics.markets import FairFoodMarketInteraction
        >>> FairFoodMarketInteraction('market', shops='shops', farms='farms').reports[:2]
        ('committed_shops', 'member_shops')
    """

    reports: ClassVar[tuple[str, ...]] = ('committed_shops', 'member_shops', 'member_farms', 'unfulfilled', 'unsold')

    def __init__(
        self,
        label: str,
        *,
        rule: None = None,
        phase: str | None = None,
        shops: Role,
        farms: Role,
        pressure: Annotated[float, RealSet.closed(0, 1)] = 0.001,
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.shops = shops
        self.farms = farms
        self.pressure = pressure

    @staticmethod
    def _buy(need: float, order: npt.NDArray[np.int64], left: Array) -> float:
        """Take a need from the farms in order and return the tomatoes bought.

        It lowers what the farms have left.
        """
        taken = np.minimum(left[order], np.maximum(need - np.concatenate(([0.0], np.cumsum(left[order])[:-1])), 0.0))
        left[order] -= taken
        return float(taken.sum())

    def act(self, system: SystemView) -> None:
        """Run the four rules of one period.

        Args:
            system:
                The system view of the market.
        """
        rng = system.rng
        shops = sorted(cast(list[ProgramMemberView], system.get_agents('shops')), key=lambda shop: shop.id)
        farms = sorted(cast(list[ProgramMemberView], system.get_agents('farms')), key=lambda farm: farm.id)
        need = np.array([shop.acres for shop in shops])
        shop_level = np.array([shop.level for shop in shops])
        acres = np.array([farm.acres for farm in farms])
        farm_level = np.array([farm.level for farm in farms])
        joining = (shop_level == OUTSIDE) & (rng.random(len(shops)) < system.get_parameter('pressure'))
        shop_level = np.where(joining, float(COMMITTED), shop_level)
        for shop in rng.permutation(len(shops)):
            member_acres = acres[farm_level == 1].sum()
            if shop_level[shop] == COMMITTED and member_acres >= need[shop_level == MEMBER].sum() + need[shop]:
                shop_level[shop] = float(MEMBER)
        left, bought = acres.copy(), np.zeros(len(shops))
        for shop in rng.permutation(len(shops)):
            order = rng.permutation(len(farms))
            members, others = order[farm_level[order] == 1], order[farm_level[order] == 0]
            if shop_level[shop] == OUTSIDE:
                bought[shop] = self._buy(need[shop], order, left)
            elif shop_level[shop] == COMMITTED:
                bought[shop] = self._buy(need[shop], members, left)
                bought[shop] += self._buy(need[shop] - bought[shop], others, left)
            else:
                bought[shop] = self._buy(need[shop], members, left)
        for farm in range(len(farms)):
            if farm_level[farm] == 0 and acres[farm] > need[shop_level == OUTSIDE].sum():
                farm_level[farm] = 1.0
        for view, level, amount in zip(shops, shop_level, bought, strict=True):
            view.level, view.traded = float(level), float(amount)
        for grower, level, rest, grown in zip(farms, farm_level, left, acres, strict=True):
            grower.level, grower.traded = float(level), float(grown - rest)
        system.report('committed_shops', float(np.sum(shop_level == COMMITTED)))
        system.report('member_shops', float(np.sum(shop_level == MEMBER)))
        system.report('member_farms', float(np.sum(farm_level == 1)))
        system.report('unfulfilled', float((need - bought).sum()))
        system.report('unsold', float(left.sum()))


class FairFoodProgramSystem(BaseSystem):
    """Chesney's Fair Food Program.

    Pressure on tomato buyers draws farms into fair wages. The Coalition of Immokalee Workers presses tomato buyers to
    join the Fair Food Program, whose member farms pay pickers a penny more per pound. Buyers join at level 1, buying
    from member farms first, then move to level 2, buying only from member farms, once the member farms grow enough, and
    farms join when the buyers outside the programme need less than they grow. Buyers and farms have random sizes that
    share the acres of farmland, and the buyers need as much as the farms grow. This is the model of Chesney (2021,
    Sections 7.2 to 7.6), with the `floor` and `ceiling` of its NetLogo code, whose `sell_me` subtracts the last
    tomatoes of a farm from the buyer instead of adding them, which this system corrects. Farms hold out until most
    buyers have joined, then join in a cascade, a tipping point that comes later the more unequal the agents.

    The system has the groups `shops` and `farms` and the market `market`, whose reporters are the buyers at level 1
    `market.committed_shops` and level 2 `market.member_shops`, the member farms `market.member_farms`, the unfulfilled
    needs `market.unfulfilled`, and the unsold tomatoes `market.unsold`.

    Args:
        n_shops:
            The number of buyers.

        n_farms:
            The number of farms.

        acres:
            The acres of farmland, which the buyers need and the farms grow.

        pressure:
            The probability per period that a buyer outside the programme joins it at level 1.

        random_state:
            The seed or generator of the sizes of the buyers and farms.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.markets import FairFoodProgramSystem
        >>> [group.size for group in FairFoodProgramSystem(5, 3).prepare().system_.elements]
        [5, 3]
    """

    field = 'economics/markets'
    topics = ('fair trade', 'supply chain', 'agents', 'tipping point', 'labour standards', 'modern slavery')
    references = ('Chesney 2021',)
    until = 5000

    def __init__(
        self,
        n_shops: int = 30,
        n_farms: int = 20,
        *,
        acres: float = 1000.0,
        pressure: float = 0.001,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.n_shops = n_shops
        self.n_farms = n_farms
        self.acres = acres
        self.pressure = pressure
        self.random_state = random_state

    @staticmethod
    def _get_sizes(count: int, acres: float, rng: np.random.Generator) -> tuple[Array, Array]:
        """Return random weights of the agents and their shares of the acres.

        The shares are `farm_size / total_sizes * acres`.
        """
        weights = rng.random(count)
        return weights, weights / weights.sum() * acres

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        rng = np.random.default_rng(self.random_state)
        _, needs = self._get_sizes(self.n_shops, self.acres, rng)
        _, grown = self._get_sizes(self.n_farms, self.acres, rng)
        shops = ProgramMemberElement('shops', acres=np.floor(needs), size=self.n_shops)
        farms = ProgramMemberElement('farms', acres=np.ceil(grown), size=self.n_farms)
        market = FairFoodMarketInteraction('market', shops='shops', farms='farms', pressure=self.pressure)
        self.system_ = System(self.label, elements=[shops, farms], interactions=[market])
