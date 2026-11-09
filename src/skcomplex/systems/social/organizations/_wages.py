"""Wage systems: Ajani's Choice.

In it, employers learn to pay migrant workers below the minimum wage.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.finite_enumerated_set import FiniteEnumeratedSet
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.systems import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec


class FarmerView(AgentView):
    """View of an employer of Ajani's Choice, with its states as attributes."""

    wage: float
    workforce: float
    employed: float
    area: float


class HarvesterView(AgentView):
    """View of a worker of Ajani's Choice, with its states as attributes."""

    slum: float
    working: float
    pay: float
    total_pay: float


class FarmerElement(Element):
    """Employer of Ajani's Choice.

    It offers a daily wage until the workforce it needs is filled.

    Args:
        label:
            The label of the employer or the group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        wage:
            The daily wage the employer offers, the minimum wage being 1.

        workforce:
            The number of workers the employer needs every day.

        employed:
            The number of workers the employer has hired today.

        area:
            The area of the farm, 0 near the first slum and 1 near the second.

    Examples:
        >>> from skcomplex.systems.social.organizations import FarmerElement
        >>> FarmerElement('farmer', wage=1.5).wage
        1.5
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'wage', 'color_range': (0.0, 2.0)}),
        *,
        wage: Annotated[float, StateSpec(domain=RealSet.closed(0, 2))] = 1.0,
        workforce: Annotated[float, StateSpec(domain=NN)] = 50.0,
        employed: Annotated[float, StateSpec(domain=NN)] = 0.0,
        area: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.wage = wage
        self.workforce = workforce
        self.employed = employed
        self.area = area

    @classmethod
    def _get_view(cls) -> type[FarmerView]:
        """Return the class of the views of the agents, whose attributes are the states."""
        return FarmerView


class HarvesterElement(Element):
    """Migrant worker of Ajani's Choice.

    It asks the employers near its slum for a day of work.

    Args:
        label:
            The label of the worker or the group.

        visual:
            How the agent is drawn, or `None` for `default_visual_`.

        slum:
            The slum the worker lives in, 0 or 1.

        working:
            Whether the worker works today.

        pay:
            The pay of the worker today.

        total_pay:
            The pay the worker has earned so far.

    Examples:
        >>> from skcomplex.systems.social.organizations import HarvesterElement
        >>> HarvesterElement('worker', slum=1.0).slum
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = VisualSpec('agents', {'color_by': 'pay', 'color_range': (0.0, 2.0)}),
        *,
        slum: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        working: Annotated[float, StateSpec(domain=FiniteEnumeratedSet([0, 1]))] = 0.0,
        pay: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        total_pay: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.slum = slum
        self.working = working
        self.pay = pay
        self.total_pay = total_pay

    @classmethod
    def _get_view(cls) -> type[HarvesterView]:
        """Return the class of the views of the agents, whose attributes are the states."""
        return HarvesterView


class HarvestInteraction(Interaction):
    """Working day of Ajani's Choice.

    Hiring happens in the morning, and wage adjustment and chat among employers in the evening. Every worker asks the
    employers of its local area, the area of its slum, and moves to the other slum when its area has none. In each of
    `asks` rounds every worker without work, in random order, asks a random employer it has not asked yet, and an
    employer whose workforce is not filled offers its wage. The worker accepts if the wage is at least the minimum wage
    1, if it did not work yesterday for at least the minimum wage, or if no employer is left to ask. In the evening
    every employer lowers its wage by `adjustment` if it filled its workforce and raises it otherwise, within
    `[0.1, 2]`, and then, in random order, sets its wage to the mean wage of `chat` random employers. The interaction
    reports the number of workers paid below the minimum wage, `slaves`, and of workers paid at least the minimum wage,
    `in_work`.

    Args:
        label:
            The label of the interaction.

        employers:
            The path of the group of employers.

        workers:
            The path of the group of workers.

        asks:
            The number of employers a worker asks for work in a day.

        chat:
            The number of employers whose wages an employer averages in the evening.

        adjustment:
            The change of the wage of an employer after a day.

    Examples:
        >>> from skcomplex.systems.social.organizations import HarvestInteraction
        >>> HarvestInteraction('day', employers='employers', workers='workers').chat
        3
    """

    draws = True

    law_classes = (RuleLaw,)

    reports: ClassVar[tuple[str, ...]] = ('slaves', 'in_work')

    def __init__(
        self,
        label: str,
        *,
        employers: Annotated[str, RoleSpec()],
        workers: Annotated[str, RoleSpec()],
        asks: Annotated[int, ParameterSpec(domain=NN)] = 1,
        chat: Annotated[int, ParameterSpec(domain=NN)] = 3,
        adjustment: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
    ) -> None:
        super().__init__(label)
        self.employers = employers
        self.workers = workers
        self.asks = asks
        self.chat = chat
        self.adjustment = adjustment

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of the working day.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.work)]

    def _hire(self, farmers: list[FarmerView], harvesters: list[HarvesterView], rng: np.random.Generator) -> None:
        """Let the workers ask the employers of their area for work.

        The workers accept or refuse their offers.
        """
        areas = {area: [farmer for farmer in farmers if farmer.area == area] for area in (0.0, 1.0)}
        asked: dict[int, list[FarmerView]] = {}
        desperate: dict[int, bool] = {}
        for harvester in harvesters:
            if not areas[harvester.slum]:
                harvester.slum = 1.0 - harvester.slum
            asked[harvester.id] = list(areas[harvester.slum])
            desperate[harvester.id] = not (harvester.working and harvester.pay >= 1)
            harvester.working, harvester.pay = 0.0, 0.0
        for _ in range(int(self.asks)):
            for position in rng.permutation(len(harvesters)).tolist():
                harvester = harvesters[position]
                left = asked[harvester.id]
                if harvester.working or not left:
                    continue
                farmer = left.pop(int(rng.integers(len(left))))
                if farmer.employed >= farmer.workforce:
                    continue
                if farmer.wage >= 1 or desperate[harvester.id] or not left:
                    harvester.working, harvester.pay = 1.0, farmer.wage
                    farmer.employed += 1

    def work(self, system: SystemView) -> None:
        """Run one working day.

        Args:
            system:
                The system.
        """
        farmers = system.agents(self.employers, FarmerView)
        harvesters = system.agents(self.workers, HarvesterView)
        rng = system.rng
        for farmer in farmers:
            farmer.employed = 0.0
        self._hire(farmers, harvesters, rng)
        for harvester in harvesters:
            harvester.total_pay += harvester.pay
        for farmer in farmers:
            step = -self.adjustment if farmer.employed >= farmer.workforce else self.adjustment
            farmer.wage = float(np.clip(farmer.wage + step, 0.1, 2.0))
        chat = min(int(self.chat), len(farmers))
        for position in rng.permutation(len(farmers)).tolist():
            partners = rng.choice(len(farmers), chat, replace=False).tolist()
            farmers[position].wage = float(np.mean([farmers[partner].wage for partner in partners]))
        system.report('slaves', float(sum(item.working and item.pay < 1 for item in harvesters)))
        system.report('in_work', float(sum(item.working and item.pay >= 1 for item in harvesters)))


class AjanisChoiceSystem(BaseSystem):
    """Chesney's Ajani's Choice, migrant farm workers and employers.

    Underpayment emerges among them. Employers need between 50 and 99 workers a day each and start with a random wage
    between 1 and 2, the minimum wage being 1. Workers live in two slums, `n_workers` in each, and ask the employers of
    their local area for work every morning, accepting an offer below the minimum wage only when desperate, after a day
    without work at the minimum wage, or when no employer is left to ask. Every evening employers lower their wage if
    they filled their workforce and raise it otherwise, by `adjustment`, within `[0.1, 2]`, and then average the wages
    of `chat` random employers (Chesney 2021 sections 3.3 and 3.8). A worker paid below the minimum wage is a slave, and
    a run lasts 60 days, two summer months. When workers outnumber the jobs, desperation drags the wages below the
    minimum, as with the default 10 employers and 1000 workers. The NetLogo listing of section 3.8 accepts a low offer
    when yesterday's pay was above the minimum, against rule 3 of the text, and the system follows the text.

    The system has the groups `employers` and `workers`, and the reporters `day.slaves` and `day.in_work`.

    Args:
        n_employers:
            The number of employers.

        n_workers:
            The number of workers in each slum.

        asks:
            The number of employers a worker asks for work in a day.

        chat:
            The number of employers whose wages an employer averages in the evening.

        adjustment:
            The change of the wage of an employer after a day.

        random_state:
            The seed or generator of the employers.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.social.organizations import AjanisChoiceSystem
        >>> AjanisChoiceSystem(5, n_workers=10).prepare().system_.elements[1].prepare().size_
        20
    """

    field = 'social/organizations'
    topics = ('agents', 'labour exploitation', 'wages', 'emergence')
    references = ('Chesney 2021', 'Chesney, Evans, Gold, and Trautrims 2018')
    until = 60

    def __init__(
        self,
        n_employers: int = 10,
        *,
        n_workers: int = 500,
        asks: int = 1,
        chat: int = 3,
        adjustment: float = 0.2,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_employers = n_employers
        self.n_workers = n_workers
        self.asks = asks
        self.chat = chat
        self.adjustment = adjustment
        self.random_state = random_state

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        employers = GroupElement(
            'employers',
            element=FarmerElement,
            size=self.n_employers,
            values={
                'wage': 1.0 + rng.random(self.n_employers),
                'workforce': rng.integers(50, 100, size=self.n_employers).astype(float),
                'area': rng.integers(2, size=self.n_employers).astype(float),
            },
        )
        workers = GroupElement(
            'workers',
            element=HarvesterElement,
            size=2 * self.n_workers,
            values={'slum': np.repeat([0.0, 1.0], self.n_workers)},
        )
        day = HarvestInteraction(
            'day',
            employers='employers',
            workers='workers',
            asks=self.asks,
            chat=self.chat,
            adjustment=self.adjustment,
        )
        return System(self.label, elements=[employers, workers], interactions=[day])
