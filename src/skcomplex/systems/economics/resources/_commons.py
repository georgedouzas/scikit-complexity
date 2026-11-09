"""Natural resources systems: the tragedy of the commons.

Herders' cows graze a shared meadow.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import MapInteraction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.laws.utils import transfer
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.components.types import AgentViewProtocol, SystemViewProtocol
from ....core.engine.solvers import AgentView, SystemView
from ....core.errors import SimulationError
from ....core.specs import (
    ObservableSpec,
    ParameterSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    RoleSpec,
    StateSpec,
    VisualSpec,
)
from ...general.agents import BirthDeathInteraction
from ...general.matching import MatchingInteraction, MatchSpec, PartnerMatchingSpec, RandomSearchSpec
from ...general.matching.types import MatchingRule
from ..common.policy import PolicyRuleInteraction


class HerderView(AgentView):
    """Herder as the rules see it, with its income, herd, and quota.

    It reads its `income` and `herd` of the step, the `upkeep` and the `newborn` energy of a cow, and its `quota`.
    """

    income: float
    herd: float
    upkeep: float
    newborn: float
    quota: float


UNLIMITED = 1e9


class HerderElement(Element):
    """Herder who adds a cow whenever a cow earned more than its upkeep.

    It sells one when the herd exceeds its quota. The life cycle `investment` of the system buys a cow, at a random
    cell, or sells the first cow the herder owns, and two maps clear the income and the herd the grazing tallies, all
    member interactions in the phase of herding.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        income:
            The milk income of the herder in the last step.

        herd:
            The number of cows of the herder in the last step.

        upkeep:
            The cost of keeping a cow for a step.

        newborn:
            The energy of a new cow.

        quota:
            The largest herd of a herder, an adaptive parameter that a regulator may change.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.resources import HerderElement
        >>> GroupElement('herders', element=HerderElement, size=5).size
        5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        income: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        herd: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        upkeep: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        newborn: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 4.0,
        quota: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo), adaptive=True)] = UNLIMITED,
    ) -> None:
        super().__init__(label, visual=visual)
        self.income = income
        self.herd = herd
        self.upkeep = upkeep
        self.newborn = newborn
        self.quota = quota

    @classmethod
    def _get_view(cls) -> type[HerderView]:
        """Return the view of the agents of the element."""
        return HerderView


class CowView(AgentView):
    """Cow as the grazing sees it: the number of its `owner` and its `energy`."""

    owner: float
    energy: float


class CowElement(Element):
    """Cow of a herder that gains the energy of the grass it eats.

    It burns its metabolism, and starves without food.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        owner:
            The id of the herder that owns the cow, given initially as its position in the group `herders`.

        energy:
            The energy of the cow.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.resources import CowElement
        >>> GroupElement('cows', element=CowElement, size=3, values={'energy': 4.0}).size
        3
    """

    law_classes = ()

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'energy'}),
        *,
        owner: Annotated[float, StateSpec(reference='herders')] = -1.0,
        energy: Annotated[float, StateSpec()] = 4.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.owner = owner
        self.energy = energy

    @classmethod
    def _get_view(cls) -> type[CowView]:
        """Return the view of the agents of the element."""
        return CowView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since the grazing moves the cows and their energy.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


class PastureView(AgentView):
    """Pasture as the rules see it.

    It holds its `grass`, its `capacity`, and the `regrowth` and `seed` of its grass.
    """

    grass: float
    capacity: float
    regrowth: float
    seed: float


class PastureElement(Element):
    """Cell of the meadow whose grass grows back logistically.

    When it is grazed bare, the grass grows from a seed bank.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        grass:
            The grass on the cell.

        capacity:
            The most grass the cell holds.

        regrowth:
            The logistic growth rate of the grass.

        seed:
            The share of the capacity that grows back as if it stood on a bare cell.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.resources import PastureElement
        >>> GroupElement('meadow', element=PastureElement, size=4, values={'capacity': 2.0}).size
        4
    """

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'grass', 'color_range': (0.0, 2.0)}),
        *,
        grass: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        capacity: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 2.0,
        regrowth: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        seed: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
    ) -> None:
        super().__init__(label, visual=visual)
        self.grass = grass
        self.capacity = capacity
        self.regrowth = regrowth
        self.seed = seed

    @classmethod
    def _get_view(cls) -> type[PastureView]:
        """Return the view of the agents of the element."""
        return PastureView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the regrowth rule of the phase of growth.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.grow, phase='growth')]

    def grow(self, agent: PastureView, system: SystemView) -> None:
        """Grow the grass logistically up to the capacity.

        Args:
            agent:
                The pasture.

            system:
                The system.
        """
        stand = max(agent.grass, system.get_parameter('seed') * agent.capacity)
        growth = system.get_parameter('regrowth') * stand * (1 - stand / agent.capacity)
        agent.grass = min(agent.capacity, agent.grass + growth)


class GrazingInteraction(MatchingInteraction[SystemView]):
    """Grazing of the meadow, a matching of the cows with the pastures.

    After the matching every cow eats and burns energy. The rule of the matching pairs the cows with the pastures, such
    as `PartnerMatchingSpec('local', 'score', score=GrazingInteraction.score, value='provider')`, which pairs every cow,
    in random order, with the free pasture of most grass in its cell or the cells around it, ties broken at random. A
    pasture takes one cow, its default quantity of one unit. Every matched cow moves to its pasture and eats up to its
    appetite as a transfer of grass to its energy, and its herder earns milk for what it ate. Then every cow burns its
    metabolism and counts in the herd of its herder, and a cow whose energy runs out dies.

    Args:
        label:
            The label of the grazing.

        requesters:
            The path of the group of cows.

        providers:
            The path of the group of pastures.

        rule:
            The spec of the algorithm that pairs the cows with the pastures.

        phase:
            The phase the grazing runs in.

        herders:
            The path of the group of herders.

        appetite:
            The most grass a cow eats in a step.

        metabolism:
            The energy a cow burns in a step.

        milk:
            The income of a herder per unit of grass its cows eat.

    Examples:
        >>> from skcomplex.systems.economics.resources import GrazingInteraction
        >>> GrazingInteraction('grazing', requesters='cows', providers='meadow', herders='herders').appetite
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        requesters: Annotated[str, RoleSpec()],
        providers: Annotated[str, RoleSpec()],
        rule: MatchingRule = RandomSearchSpec(),
        phase: str | None = 'grazing',
        herders: Annotated[str, RoleSpec()],
        appetite: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        metabolism: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        milk: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, requesters=requesters, providers=providers, rule=rule, phase=phase)
        self.herders = herders
        self.appetite = appetite
        self.metabolism = metabolism
        self.milk = milk

    @staticmethod
    def score(cow: object, pasture: object) -> float:
        """Return the score of a pasture for a cow, the grass on the pasture.

        Args:
            cow:
                The agent view of the cow.

            pasture:
                The agent view of the pasture.

        Returns:
            score:
                The grass on the pasture, or minus infinity for an agent that is not a pasture.
        """
        del cow
        return float(pasture.grass) if isinstance(pasture, PastureView) else -np.inf

    @staticmethod
    def _get_herder(cow: CowView) -> HerderView:
        """Return the herder that owns a cow, which its reference state `owner` names."""
        herder = cow.get_referenced('owner')
        if not isinstance(herder, HerderView):
            error_msg = f'Cow {cow.id} names no herder as its owner.'
            raise SimulationError(error_msg)
        return herder

    def _settle(self, match: MatchSpec, system: SystemView) -> None:
        """Move the cow of a match to its pasture and feed it.

        Its herder is paid for what it ate.

        Args:
            match:
                The match of a cow with a pasture.

            system:
                The system view of the grazing.
        """
        cow, pasture = match.requester, match.provider
        if not isinstance(cow, CowView) or not isinstance(pasture, PastureView):
            return
        cow.move_to(pasture.location)
        eaten = min(pasture.grass, system.get_parameter('appetite'))
        transfer(pasture, cow, eaten, 'grass', 'energy')
        self._get_herder(cow).income += system.get_parameter('milk') * eaten

    def match(self, system: SystemView) -> None:
        """Pair and feed the cows, then let every cow burn its metabolism.

        Every cow is counted in its herd, and it starves.

        Args:
            system:
                The system view of the grazing.
        """
        super().match(system)
        for cow in system.get_agents('requesters', CowView):
            cow.energy -= system.get_parameter('metabolism')
            self._get_herder(cow).herd += 1
            if cow.energy < 0:
                cow.remove()


class QuotaRegulatorInteraction(PolicyRuleInteraction[SystemView]):
    """Regulator of the quota of cows per herder.

    It halves the quota when the grass runs short and raises it when it recovers. The quota is the adaptive parameter
    `quota` of every herder. It halves, down to one cow, when the grass falls below the share `low` of the capacity of
    the meadow, and grows by one cow when the grass is above the share `high`. Without regulation it stays as it is.

    Args:
        label:
            The label of the regulator.

        rule:
            Unused, since the regulator overrides `act`.

        phase:
            The phase the regulator acts in.

        regulated:
            One when the regulator changes the quota, zero otherwise.

        low:
            The share of the capacity of the meadow below which the quota halves.

        high:
            The share of the capacity of the meadow above which the quota rises by one cow.

    Examples:
        >>> from skcomplex.systems.economics.resources import QuotaRegulatorInteraction
        >>> QuotaRegulatorInteraction('regulator', regulated=1.0).regulated
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        rule: Callable[[SystemView], None] | None = None,
        phase: str | None = None,
        regulated: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        low: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        high: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.regulated = regulated
        self.low = low
        self.high = high

    def act(self, system: SystemView) -> None:
        """Change the quota with the share of grass left on the meadow.

        Args:
            system:
                The system view of the regulator.
        """
        if not system.get_parameter('regulated'):
            return
        pastures = system.get_agents('meadow', PastureView)
        share = sum(pasture.grass for pasture in pastures) / sum(pasture.capacity for pasture in pastures)
        for herder in system.get_agents('herders', HerderView):
            if share < system.get_parameter('low'):
                herder.quota = max(1.0, float(np.floor(herder.quota / 2)))
            elif share > system.get_parameter('high'):
                herder.quota += 1


def _pays(herder: AgentViewProtocol, system: SystemViewProtocol) -> bool:
    """Return whether a cow of a herder earned more than its upkeep in the last step.

    The herd must also be below quota.
    """
    del system
    if not isinstance(herder, HerderView):
        return False
    earning = herder.income / herder.herd if herder.herd else herder.upkeep + 1
    return earning > herder.upkeep and herder.herd < herder.quota


def _exceeds(herder: AgentViewProtocol, system: SystemViewProtocol) -> bool:
    """Return whether the herd of a herder in the last step exceeds its quota.

    The herder then sells a cow.
    """
    del system
    return isinstance(herder, HerderView) and herder.herd > herder.quota


def _owns(herder: AgentViewProtocol, cow: AgentViewProtocol) -> bool:
    """Return whether a herder owns a cow.

    The reference state `owner` of the cow names the herder.
    """
    return cow.get_referenced('owner') is herder


def _get_calf(herder: AgentViewProtocol | None, system: SystemViewProtocol) -> dict[str, object]:
    """Return the owner and the energy of a new cow of a herder."""
    del system
    if not isinstance(herder, HerderView):
        return {}
    return {'owner': float(herder.id), 'energy': herder.newborn}


def _get_cell(herder: AgentViewProtocol | None, system: SystemViewProtocol) -> tuple[int, ...]:
    """Return a random cell of the meadow, where a new cow starts."""
    del herder
    return tuple(int(system.rng.integers(int(size))) for size in system.dimensions or ())


def _get_cleared(herder: AgentViewProtocol, system: SystemViewProtocol) -> float:
    """Return zero, the income and the herd a herder starts a step with."""
    del herder, system
    return 0.0


class TragedyOfTheCommonsSystem(BaseSystem):
    """Herders whose cows graze a shared meadow, until the grass collapses.

    Each herder adds cows while a cow pays. Every step every cow moves to the richest pasture it can reach, eats, and
    earns its herder milk, and the grass grows back logistically. Every herder adds a cow whenever a cow earned more
    than its upkeep, since the gain of a cow goes to its herder while the cost of the grass it eats falls on all. With
    more herders than the meadow carries, the herd grows past what the grass sustains, the grass collapses below a tenth
    of its capacity, and the cows starve, so the herd shrinks after it. A regulator with an adaptive quota of cows per
    herder, lowered when the grass runs short and raised when it recovers, keeps the meadow from collapse (Hamill and
    Gilbert 2016 sec. 10.2).

    The system has the groups `meadow`, `cows`, and `herders`, whose adaptive parameter `quota` the regulator
    `regulator` changes, and the grazing `grazing`.

    Args:
        n_herders:
            The number of herders.

        width:
            The number of columns of the meadow.

        height:
            The number of rows of the meadow.

        cows:
            The initial number of cows of every herder.

        quota:
            The initial quota of cows per herder of a regulator that adapts it, or `None` for no regulation.

        regrowth:
            The logistic growth rate of the grass.

        capacity:
            The most grass a cell holds.

        upkeep:
            The cost of keeping a cow for a step.

        random_state:
            The seed or generator of the cells of the cows.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.resources import TragedyOfTheCommonsSystem
        >>> [group.size for group in TragedyOfTheCommonsSystem(4, width=5, height=5).prepare().system_.elements]
        [25, 4, 4]
    """

    field = 'economics/resources'
    topics = ('commons', 'agents', 'overgrazing', 'regulation', 'grid')
    references = ('Hamill and Gilbert 2016 sec. 10.2', 'Hardin 1968')
    until = 100
    observables = (
        ObservableSpec('grass', group='meadow', variable='grass', reduce='sum'),
        ObservableSpec('herd', group='cows', variable='energy', reduce='count'),
        ObservableSpec('quota', group='herders', variable='quota', reduce='mean'),
    )

    def __init__(
        self,
        n_herders: int = 20,
        *,
        width: int = 20,
        height: int = 20,
        cows: int = 1,
        quota: float | None = None,
        regrowth: float = 0.2,
        capacity: float = 2.0,
        upkeep: float = 0.3,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_herders = n_herders
        self.width = width
        self.height = height
        self.cows = cows
        self.quota = quota
        self.regrowth = regrowth
        self.capacity = capacity
        self.upkeep = upkeep
        self.random_state = random_state

    @staticmethod
    def _get_investment() -> list[BirthDeathInteraction | MapInteraction]:
        """Return the member interactions of a herder.

        They buy or sell a cow, then clear its income and herd.
        """
        return [
            BirthDeathInteraction(
                'investment',
                target='herders',
                group='cows',
                birth=_pays,
                values=_get_calf,
                placement=_get_cell,
                cull=_exceeds,
                kin=_owns,
                schedule='member',
                phase='herding',
            ),
            *(
                MapInteraction(
                    f'{state}_clearing',
                    target='herders',
                    state=state,
                    function=_get_cleared,
                    form='agent',
                    schedule='member',
                    phase='herding',
                )
                for state in ('income', 'herd')
            ),
        ]

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        cells = [(x, y) for x in range(self.width) for y in range(self.height)]
        meadow = GroupElement(
            'meadow',
            element=PastureElement,
            size=len(cells),
            values={'grass': self.capacity, 'capacity': self.capacity, 'regrowth': self.regrowth},
            location=cells,
        )
        herders = GroupElement(
            'herders',
            element=HerderElement,
            size=self.n_herders,
            values={
                'upkeep': self.upkeep,
                'quota': UNLIMITED if self.quota is None else self.quota,
            },
            location=[cells[0]] * self.n_herders,
        )
        n_cows = self.n_herders * self.cows
        herd = GroupElement(
            'cows',
            element=CowElement,
            size=n_cows,
            values={'owner': np.repeat(np.arange(self.n_herders, dtype=float), self.cows)},
            variable=True,
            location=[cells[index] for index in rng.integers(len(cells), size=n_cows)],
        )
        regulator = QuotaRegulatorInteraction('regulator', phase='herding', regulated=float(self.quota is not None))
        grazing = GrazingInteraction(
            'grazing',
            requesters='cows',
            providers='meadow',
            rule=PartnerMatchingSpec('local', 'score', score=GrazingInteraction.score, value='provider'),
            herders='herders',
        )
        return CompositeSystem(
            self.label,
            elements=[meadow, herd, herders],
            interactions=[grazing, *self._get_investment(), regulator],
            space=LatticeSpace((self.width, self.height), boundary='periodic', metric='chebyshev'),
            time=DiscreteTime(
                schedule=PhasesScheduleSpec([PhaseSpec('grazing'), PhaseSpec('growth'), PhaseSpec('herding')]),
            ),
        )
