"""Natural resources models: herders whose cows graze a shared meadow, the tragedy of the commons."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.elements import Element, GroupElement
from ....core.interactions import FeedbackInteraction, MatchingInteraction
from ....core.laws import BaseLaw, RuleLaw
from ....core.laws.utils import transfer
from ....core.spaces import DiscreteTime, LatticeSpace
from ....core.system import System
from ....core.types import MatchingRule
from ....engine.solvers import ModelView
from ....errors import SimulationError
from ....specs import (
    GreedyMatchingSpec,
    MatchSpec,
    ObservableSpec,
    ParameterSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    RandomSearchSpec,
    RoleSpec,
    StateSpec,
    VisualSpec,
)
from ..._catalog import register_model
from ._cow_view import CowView
from ._herder_view import HerderView
from ._pasture_view import PastureView

UNLIMITED = 1e9


def _commons_observables(system: System) -> list[ObservableSpec]:
    """Return the grass on the meadow, the size of the herd, and the mean quota of the herders."""
    del system
    return [
        ObservableSpec('grass', group='meadow', variable='grass', reduce='sum'),
        ObservableSpec('herd', group='cows', variable='energy', reduce='count'),
        ObservableSpec('quota', group='herders', variable='quota', reduce='mean'),
    ]


class PastureElement(Element):
    """Cell of the meadow whose grass grows back logistically, from a seed bank when it is grazed bare.

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
        >>> from skcomplex.models.economics.resources import PastureElement
        >>> GroupElement('meadow', element=PastureElement, size=4, values={'capacity': 2.0}).size
        4
    """

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

    def grow(self, agent: PastureView, model: ModelView) -> None:
        """Grow the grass logistically up to the capacity.

        Args:
            agent:
                The pasture.

            model:
                The model.
        """
        stand = max(agent.grass, model.parameter('seed') * agent.capacity)
        growth = model.parameter('regrowth') * stand * (1 - stand / agent.capacity)
        agent.grass = min(agent.capacity, agent.grass + growth)


class CowElement(Element):
    """Cow of a herder that gains the energy of the grass it eats, burns its metabolism, and starves without food.

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
        >>> from skcomplex.models.economics.resources import CowElement
        >>> GroupElement('cows', element=CowElement, size=3, values={'energy': 4.0}).size
        3
    """

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


class HerderElement(Element):
    """Herder who adds a cow whenever a cow earned more than its upkeep, and sells one when the herd exceeds its quota.

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
        >>> from skcomplex.models.economics.resources import HerderElement
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

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule that grows the herd in the phase of herding.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.invest, phase='herding')]

    def invest(self, agent: HerderView, model: ModelView) -> None:
        """Add a cow while a cow earns more than its upkeep below the quota, and sell a cow above the quota.

        Args:
            agent:
                The herder.

            model:
                The model.
        """
        earning = agent.income / agent.herd if agent.herd else model.parameter('upkeep') + 1
        if earning > model.parameter('upkeep') and agent.herd < agent.quota:
            model.create('cows', owner=float(agent.id), energy=model.parameter('newborn'))
        elif agent.herd > agent.quota:
            next(cow for cow in model.agents('cows', CowView) if cow.referenced('owner') is agent).remove()
        agent.income = 0.0
        agent.herd = 0.0


class GrazingInteraction(MatchingInteraction[ModelView]):
    """Grazing of the meadow, a matching of the cows with the pastures, after which every cow eats and burns energy.

    The rule of the matching pairs the cows with the pastures, such as
    `GreedyMatchingSpec(score=GrazingInteraction.score, candidates='neighbors')`, which pairs every cow, in random
    order, with the free pasture of most grass in its cell or the cells around it, ties broken at random. A pasture
    takes one cow, its default quantity of one unit. Every matched cow moves to its pasture and eats up to its
    appetite as a transfer of grass to its energy, and its herder earns milk for what it ate. Then every cow burns
    its metabolism and counts in the herd of its herder, and a cow whose energy runs out dies.

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
        >>> from skcomplex.models.economics.resources import GrazingInteraction
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
        herder = cow.referenced('owner')
        if not isinstance(herder, HerderView):
            error_msg = f'Cow {cow.id} names no herder as its owner.'
            raise SimulationError(error_msg)
        return herder

    def _settle(self, match: MatchSpec, model: ModelView) -> None:
        """Move the cow of a match to its pasture, feed it, and pay its herder for what it ate.

        Args:
            match:
                The match of a cow with a pasture.

            model:
                The model view of the grazing.
        """
        cow, pasture = match.requester, match.provider
        if not isinstance(cow, CowView) or not isinstance(pasture, PastureView):
            return
        cow.move_to(pasture.location)
        eaten = min(pasture.grass, model.parameter('appetite'))
        transfer(pasture, cow, eaten, 'grass', 'energy')
        self._get_herder(cow).income += model.parameter('milk') * eaten

    def match(self, model: ModelView) -> None:
        """Pair and feed the cows, then let every cow burn its metabolism, count it in its herd, and starve.

        Args:
            model:
                The model view of the grazing.
        """
        super().match(model)
        for cow in model.agents('requesters', CowView):
            cow.energy -= model.parameter('metabolism')
            self._get_herder(cow).herd += 1
            if cow.energy < 0:
                cow.remove()


class QuotaRegulatorInteraction(FeedbackInteraction[ModelView]):
    """Regulator that halves the quota of cows per herder when the grass runs short and raises it when it recovers.

    The quota is the adaptive parameter `quota` of every herder. It halves, down to one cow, when the grass falls below
    the share `low` of the capacity of the meadow, and grows by one cow when the grass is above the share `high`.
    Without regulation it stays as it is.

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
        >>> from skcomplex.models.economics.resources import QuotaRegulatorInteraction
        >>> QuotaRegulatorInteraction('regulator', regulated=1.0).regulated
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        rule: Callable[[ModelView], None] | None = None,
        phase: str | None = None,
        regulated: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        low: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        high: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.8,
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.regulated = regulated
        self.low = low
        self.high = high

    def act(self, model: ModelView) -> None:
        """Change the quota with the share of grass left on the meadow.

        Args:
            model:
                The model view of the regulator.
        """
        if not model.parameter('regulated'):
            return
        pastures = model.agents('meadow', PastureView)
        share = sum(pasture.grass for pasture in pastures) / sum(pasture.capacity for pasture in pastures)
        for herder in model.agents('herders', HerderView):
            if share < model.parameter('low'):
                herder.quota = max(1.0, float(np.floor(herder.quota / 2)))
            elif share > model.parameter('high'):
                herder.quota += 1


@register_model(
    field='economics/resources',
    tags=('commons', 'agents', 'overgrazing', 'regulation', 'grid'),
    references=('Hamill and Gilbert 2016 sec. 10.2', 'Hardin 1968'),
    defaults={'until': 100},
    observables=_commons_observables,
)
def tragedy_of_the_commons(
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
) -> System:
    """Return herders whose cows graze a shared meadow, each adding cows while a cow pays, until the grass collapses.

    Every step every cow moves to the richest pasture it can reach, eats, and earns its herder milk, and the grass
    grows back logistically. Every herder adds a cow whenever a cow earned more than its upkeep, since the gain of a
    cow goes to its herder while the cost of the grass it eats falls on all. With more herders than the meadow
    carries, the herd grows past what the grass sustains, the grass collapses below a tenth of its capacity, and the
    cows starve, so the herd shrinks after it. A regulator with an adaptive quota of cows per herder, lowered when the
    grass runs short and raised when it recovers, keeps the meadow from collapse (Hamill and Gilbert 2016 sec. 10.2).

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

    Returns:
        system:
            The model, with the groups `meadow`, `cows`, and `herders`, whose adaptive parameter `quota` the regulator
            `regulator` changes, and the grazing `grazing`.

    Examples:
        >>> from skcomplex.models.economics.resources import tragedy_of_the_commons
        >>> [group.size for group in tragedy_of_the_commons(4, width=5, height=5).elements]
        [25, 4, 4]
    """
    rng = np.random.default_rng(random_state)
    cells = [(x, y) for x in range(width) for y in range(height)]
    meadow = GroupElement(
        'meadow',
        element=PastureElement,
        size=len(cells),
        values={'grass': capacity, 'capacity': capacity, 'regrowth': regrowth},
        location=cells,
    )
    herders = GroupElement(
        'herders',
        element=HerderElement,
        size=n_herders,
        values={
            'upkeep': upkeep,
            'quota': UNLIMITED if quota is None else quota,
        },
        location=[cells[0]] * n_herders,
    )
    n_cows = n_herders * cows
    herd = GroupElement(
        'cows',
        element=CowElement,
        size=n_cows,
        values={'owner': np.repeat(np.arange(n_herders, dtype=float), cows)},
        variable=True,
        location=[cells[index] for index in rng.integers(len(cells), size=n_cows)],
    )
    regulator = QuotaRegulatorInteraction('regulator', phase='herding', regulated=float(quota is not None))
    grazing = GrazingInteraction(
        'grazing',
        requesters='cows',
        providers='meadow',
        rule=GreedyMatchingSpec(score=GrazingInteraction.score, candidates='neighbors'),
        herders='herders',
    )
    return System(
        elements=[meadow, herd, herders],
        interactions=[grazing, regulator],
        space=LatticeSpace((width, height), boundary='periodic', neighborhood='moore'),
        time=DiscreteTime(
            schedule=PhasesScheduleSpec([PhaseSpec('grazing'), PhaseSpec('growth'), PhaseSpec('herding')]),
        ),
    )
