"""Metaphoria: a mortal population of hunter-gatherers or of a monetary economy.

The monetary economy has firms and a bank.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.analysis import distribution
from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import BaseInteraction, Interaction, TransferInteraction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.laws.utils import transfer
from ....core.components.spaces import DiscreteTime, LatticeSpace
from ....core.components.systems import BaseSystem, CompositeSystem, RunSettings
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import (
    BaseSpec,
    ObservableSpec,
    ParameterSpec,
    PhaseSpec,
    PhasesScheduleSpec,
    RoleSpec,
    StateSpec,
    VisualSpec,
)
from ....core.types import Array
from ...general.matching import MatchingInteraction, MatchSpec, PartnerMatchingSpec
from ..common.resources import RegrowthInteraction

PROPENSITIES = {'fertility': 0.3, 'exploration': 1.0, 'enterprise': 0.3, 'restlessness': 0.1}
ALIVE, STARVED, AGED = 0.0, 1.0, 2.0
OLD_AGE_DEATH = 1 / 7
MODES = ('hunter_gatherer', 'monetary')
PHASES = ('growth', 'work', 'banking', 'hiring', 'market', 'living', 'census')
SIGHT = 5.0


class BusinessView(AgentView):
    """Firm as the rules see it.

    It holds its holdings, stock, wage, price, owner, workforce, sales, and policy parameters.
    """

    money: float
    loans: float
    stock: float
    wage: float
    price: float
    owner: float
    workers: float
    vacancies: float
    revenue: float
    interest: float
    repayment: float
    adjustment: float
    stock_limit: float


class CentralBankView(AgentView):
    """Bank as its rule sees it: its `money` and its `loans`."""

    money: float
    loans: float


class CentralBankElement(Element):
    """Bank that creates money as loans.

    The transfer `dividends` of the system pays the income to a random person, or to a random firm when nobody lives.

    The money it issued, a negative holding, and the loans it holds sum to zero after every step, so every unit of
    money in the economy is matched by a loan.

    Args:
        label:
            The label of the bank or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        money:
            The money of the bank, the negative of the money it issued plus its income of the step.

        loans:
            The loans the bank holds.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.macroeconomics import CentralBankElement
        >>> GroupElement('bank', element=CentralBankElement, size=1).size
        1
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        money: Annotated[float, StateSpec(quantity='money')] = 0.0,
        loans: Annotated[float, StateSpec(quantity='credit')] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.money = money
        self.loans = loans

    @classmethod
    def _get_view(cls) -> type[CentralBankView]:
        """Return the view of the bank, with its money and loans."""
        return CentralBankView

    @staticmethod
    def get_income(agent: CentralBankView, holder: AgentView, system: SystemView) -> float:
        """Return the income of the step that the bank pays as a dividend.

        It is its money plus its loans.

        Args:
            agent:
                The bank.

            holder:
                The person, or the firm when nobody lives, that receives the dividend.

            system:
                The system view of the dividends.

        Returns:
            income:
                The income.
        """
        return agent.money + agent.loans


class PlotView(AgentView):
    """Patch as the rules see it: its `food`, `capacity`, `resistance`, and `regrowth`."""

    food: float
    capacity: float
    resistance: float
    regrowth: float


class PlotElement(Element):
    """Patch of land whose food grows back by a share of its capacity every step.

    It is harvested against a resistance. The regrowth is a `RegrowthInteraction` of the group of patches, relative to
    the capacity.

    Args:
        label:
            The label of the patch or the site group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        food:
            The food on the patch.

        capacity:
            The most food the patch holds.

        resistance:
            The labour lost to the resistance of the patch before any food is harvested.

        regrowth:
            The share of the capacity that grows back in a step.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import PlotElement
        >>> PlotElement('patch').regrowth
        1.0
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        food: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        capacity: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 10.0,
        resistance: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        regrowth: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.food = food
        self.capacity = capacity
        self.resistance = resistance
        self.regrowth = regrowth

    @classmethod
    def _get_view(cls) -> type[PlotView]:
        """Return the view of a patch, with its food, capacity, resistance, and regrowth."""
        return PlotView


class SettlerView(AgentView):
    """Person as the rules see it.

    It holds its holdings, age, family, job, fate, propensities, and needs. Besides its states and parameters it reads
    two quantities of the markets: `hireable`, the highest wage limit of a job it accepts, which admits every job for an
    unemployed adult and none otherwise, and `shortfall`, the food it lacks for `reserve` steps, whose decrease by a
    purchase is the food the person receives.
    """

    food: float
    money: float
    loans: float
    age: float
    parent: float
    employer: float
    harvested: float
    born: float
    fate: float
    fertility: float
    exploration: float
    enterprise: float
    restlessness: float
    labour: float
    need: float
    maturity: float
    lifespan: float
    spacing: float
    endowment: float
    mutation: float
    sight: float
    reserve: float
    monetary: float

    @property
    def hireable(self) -> float:
        """Return the limit of the jobs the person accepts.

        It is unlimited for an unemployed adult and none otherwise.
        """
        return math.inf if self.employer < 0 and self.age >= self.maturity else -math.inf

    @property
    def shortfall(self) -> float:
        """Return the food the person lacks for `reserve` steps."""
        return self.reserve * self.need - self.food

    @shortfall.setter
    def shortfall(self, value: float) -> None:
        """Set the food so that the person lacks `value` for `reserve` steps."""
        self.food = self.reserve * self.need - value


@dataclass(frozen=True)
class _PeopleSpec(BaseSpec):
    """Living people by id at one time.

    The rules build them once per step and keep them in the storage of the system.

    Args:
        time:
            The time the people were read at.

        people:
            The living people by id.
    """

    time: float
    people: Mapping[int, SettlerView]


class SettlerElement(Element):
    """Mortal person who eats, ages, raises children, and moves.

    In the monetary economy, the person also works and founds firms. Every step a person eats `need` food and starves
    when its food runs out, and past `lifespan` it dies of old age with probability 1/7. A child takes `need` food from
    its parent and follows it. A fed adult has a child with the probability `fertility`, at most once every `spacing`
    steps, and gives it `endowment` food. The child inherits the four propensities, each moved by a normal mutation and
    kept in [0, 1]. A person that harvested too little and holds less than `reserve` steps of food moves, far with the
    probability `exploration`. In the monetary economy an unemployed adult on a patch with food but no firm founds one
    with the probability `enterprise`, and a worker quits with the probability `restlessness`.

    Args:
        label:
            The label of the person or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        food:
            The food the person holds.

        money:
            The money the person holds.

        loans:
            The loans the person holds, negative for a debt taken over from a firm.

        age:
            The age in steps.

        parent:
            The id of the parent, or -1.

        employer:
            The id of the firm that employs the person, or -1.

        harvested:
            The food the person harvested or earned in the step.

        born:
            The step of the person's last child.

        fate:
            0 while alive, 1 after starving, and 2 after dying of old age, until the census removes the person.

        fertility:
            The probability of a child in a step.

        exploration:
            The probability of moving far.

        enterprise:
            The probability of founding a firm.

        restlessness:
            The probability of quitting a job.

        labour:
            The labour of an adult in a step.

        need:
            The food a person eats in a step.

        maturity:
            The age of adulthood.

        lifespan:
            The age after which death by old age is possible.

        spacing:
            The fewest steps between two children.

        endowment:
            The food a parent gives a newborn child.

        mutation:
            The standard deviation of the mutation of a propensity.

        sight:
            The distance within which a person sees firms and explores.

        reserve:
            The steps of food a person buys up to, and below which a person that harvests too little moves.

        monetary:
            One in the monetary economy, zero among hunter-gatherers.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.macroeconomics import SettlerElement
        >>> GroupElement('people', element=SettlerElement, size=3, variable=True).size
        3
    """

    draws = True

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        food: Annotated[float, StateSpec()] = 10.0,
        money: Annotated[float, StateSpec(quantity='money')] = 0.0,
        loans: Annotated[float, StateSpec(quantity='credit')] = 0.0,
        age: Annotated[float, StateSpec(domain=NN)] = 20.0,
        parent: Annotated[float, StateSpec(reference='people')] = -1.0,
        employer: Annotated[float, StateSpec(reference='businesses')] = -1.0,
        harvested: Annotated[float, StateSpec()] = 0.0,
        born: Annotated[float, StateSpec()] = -100.0,
        fate: Annotated[float, StateSpec()] = ALIVE,
        fertility: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        exploration: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        enterprise: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        restlessness: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
        labour: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        need: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        maturity: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 15.0,
        lifespan: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 85.0,
        spacing: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        endowment: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        mutation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        sight: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        reserve: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 5.0,
        monetary: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.food = food
        self.money = money
        self.loans = loans
        self.age = age
        self.parent = parent
        self.employer = employer
        self.harvested = harvested
        self.born = born
        self.fate = fate
        self.fertility = fertility
        self.exploration = exploration
        self.enterprise = enterprise
        self.restlessness = restlessness
        self.labour = labour
        self.need = need
        self.maturity = maturity
        self.lifespan = lifespan
        self.spacing = spacing
        self.endowment = endowment
        self.mutation = mutation
        self.sight = sight
        self.reserve = reserve
        self.monetary = monetary

    @classmethod
    def _get_view(cls) -> type[SettlerView]:
        """Return the view of a person, with its states and parameters."""
        return SettlerView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of living in the phase of living.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.live, phase='living')]

    def _feed_child(self, agent: SettlerView, system: SystemView) -> None:
        """Take food from the living parent and follow it."""
        parent = agent.get_referenced('parent')
        if not isinstance(parent, SettlerView):
            return
        given = min(agent.need, max(parent.food - parent.need, 0.0))
        parent.food -= given
        agent.food += given
        agent.move_to(parent.location)

    def _breed(self, agent: SettlerView, system: SystemView) -> None:
        """Have a child with mutated propensities at the probability of fertility."""
        fed = agent.food >= agent.endowment + agent.need
        spaced = system.time - agent.born >= agent.spacing
        if not (fed and spaced and system.rng.random() < agent.fertility):
            return
        traits = {
            name: float(np.clip(getattr(agent, name) + system.rng.normal(0.0, agent.mutation), 0.0, 1.0))
            for name in PROPENSITIES
        }
        agent.food -= agent.endowment
        agent.born = system.time
        agent.create(None, None, food=agent.endowment, age=0.0, parent=float(agent.id), born=system.time, **traits)

    def _work_or_found(self, agent: SettlerView, system: SystemView) -> bool:
        """Quit a job at the restlessness, or found a firm on a free patch with food.

        It returns whether the person is employed.
        """
        if agent.employer >= 0:
            if system.rng.random() < agent.restlessness:
                agent.employer = -1.0
            return agent.employer >= 0
        if agent.get_agents_here('businesses') or SettlerElement._get_patch_here(agent).food <= 0:
            return False
        if system.rng.random() >= agent.enterprise:
            return False
        firm = agent.create('businesses', owner=float(agent.id))
        agent.employer = float(firm.id)
        return True

    def live(self, agent: SettlerView, system: SystemView) -> None:
        """Age, eat, die, have children, and move or work.

        Args:
            agent:
                The person.

            system:
                The system.
        """
        agent.age += 1
        adult = agent.age >= agent.maturity
        if not adult:
            self._feed_child(agent, system)
        agent.food -= agent.need
        if agent.food < 0:
            agent.fate = STARVED
            return
        if agent.age > agent.lifespan and system.rng.random() < OLD_AGE_DEATH:
            agent.fate = AGED
            return
        if not adult:
            return
        self._breed(agent, system)
        employed = agent.monetary > 0 and self._work_or_found(agent, system)
        if not employed and agent.harvested < agent.need and agent.food < agent.reserve * agent.need:
            SettlerElement._wander(agent, system, agent.sight)
        agent.harvested = 0.0

    @staticmethod
    def _get_patch_here(agent: AgentView) -> PlotView:
        """Return the patch of the cell of an agent."""
        return next(patch for patch in agent.get_agents_here('patches') if isinstance(patch, PlotView))

    @staticmethod
    def _get_cell(agent: AgentView) -> tuple[int, ...]:
        """Return the cell of an agent of the lattice."""
        location = agent.location
        if not isinstance(location, tuple):
            error_msg = f'The agent {agent.id} is not on a lattice cell.'
            raise TypeError(error_msg)
        return tuple(int(item) for item in location)

    @staticmethod
    def _wander(agent: SettlerView, system: SystemView, sight: float) -> None:
        """Move to a random cell within sight with the propensity to explore.

        Otherwise move to a neighbouring cell.
        """
        if system.rng.random() < agent.exploration:
            width, height = system.dimensions or (1, 1)
            reach = int(sight)
            dx, dy = system.rng.integers(-reach, reach + 1, size=2)
            x, y = SettlerElement._get_cell(agent)
            agent.move_to(((x + int(dx)) % int(width), (y + int(dy)) % int(height)))
            return
        cells = agent.get_neighbor_cells()
        agent.move_to(cells[int(system.rng.integers(len(cells)))])


class BusinessElement(Element):
    """Firm on a patch that harvests with its workers and sells food in its shop.

    It pays wages with money it borrows. Every step the workers pool their labour on the firm's patch, the firm pays
    each the wage, borrowing what it lacks from the bank, and pays the interest and a share of the principal of its
    loans. A firm that sold out raises its price, and one with a stock above `stock_limit` lowers it and lays a worker
    off. Otherwise a firm posts vacancies up to the workforce that harvests the whole capacity of its patch, raises the
    wage while a vacancy went unfilled, and lowers it while its sales fall short of its wage bill. Profits above twice
    the wage bill go to the owner, and a firm without a living owner passes to one of its workers.

    Args:
        label:
            The label of the firm or the group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        money:
            The money the firm holds.

        loans:
            The loans of the firm, negative for its debt.

        stock:
            The food in the shop.

        wage:
            The wage of a worker.

        price:
            The price of a unit of food.

        owner:
            The id of the owner.

        workers:
            The number of workers in the step.

        vacancies:
            The open vacancies.

        revenue:
            The sales of the step.

        interest:
            The interest rate of loans per step.

        repayment:
            The share of the principal repaid per step.

        adjustment:
            The relative step of a change of price or wage.

        stock_limit:
            The stock above which the firm lowers its price and lays off.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.macroeconomics import BusinessElement
        >>> GroupElement('businesses', element=BusinessElement, size=2, variable=True).size
        2
    """

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        money: Annotated[float, StateSpec(quantity='money')] = 0.0,
        loans: Annotated[float, StateSpec(quantity='credit')] = 0.0,
        stock: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        wage: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 4.0,
        price: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        owner: Annotated[float, StateSpec(reference='people')] = -1.0,
        workers: Annotated[float, StateSpec(domain=NN)] = 1.0,
        vacancies: Annotated[float, StateSpec(domain=NN)] = 1.0,
        revenue: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        interest: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.003,
        repayment: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.004,
        adjustment: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.05,
        stock_limit: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 30.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.money = money
        self.loans = loans
        self.stock = stock
        self.wage = wage
        self.price = price
        self.owner = owner
        self.workers = workers
        self.vacancies = vacancies
        self.revenue = revenue
        self.interest = interest
        self.repayment = repayment
        self.adjustment = adjustment
        self.stock_limit = stock_limit

    @classmethod
    def _get_view(cls) -> type[BusinessView]:
        """Return the view of a firm, with its states and parameters."""
        return BusinessView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the rule of production in the phase of work.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.produce, phase='work')]

    def _borrow(self, agent: BusinessView, bank: CentralBankView, amount: float) -> None:
        """Borrow an amount the bank creates as money and records as a loan."""
        if amount > 0:
            transfer(bank, agent, amount, 'money')
            transfer(agent, bank, amount, 'loans')

    def _pay_debt(self, agent: BusinessView, bank: CentralBankView) -> None:
        """Pay the interest and the repayment of the debt.

        The firm borrows anew what it lacks.
        """
        debt = -agent.loans
        interest, principal = agent.interest * debt, agent.repayment * debt
        self._borrow(agent, bank, interest + principal - agent.money)
        transfer(agent, bank, interest + principal, 'money')
        transfer(bank, agent, principal, 'loans')

    def _adjust(self, agent: BusinessView, workers: list[SettlerView], patch: PlotView, sold_out: bool) -> None:
        """Adjust the price, the wage, and the workforce.

        They follow the stock, the labour, and the sales.
        """
        step = agent.adjustment
        if sold_out:
            agent.price *= 1 + step
        if agent.stock > agent.stock_limit:
            agent.price *= 1 - step
            agent.vacancies = 0.0
            if len(workers) > 1:
                workers[-1].employer = -1.0
            return
        if agent.vacancies > 0:
            agent.wage *= 1 + step
        if agent.revenue < agent.wage * len(workers):
            agent.wage *= 1 - step
        wanted = math.ceil((patch.capacity + patch.resistance) / workers[0].labour) if workers else 1
        agent.vacancies = float(max(wanted - len(workers), 0))

    def produce(self, agent: BusinessView, system: SystemView) -> None:
        """Harvest with the workers, pay wages and the bank, and adjust.

        It then pays a dividend to the owner.

        Args:
            agent:
                The firm.

            system:
                The system.
        """
        people = BusinessElement._get_people_by_id(system)
        workers = [person for person in people.values() if person.employer == agent.id]
        bank = system.get_agents('bank', CentralBankView)[0]
        patch = SettlerElement._get_patch_here(agent)
        sold_out = agent.stock <= 0
        for worker in workers:
            worker.move_to(agent.location)
        labour = sum(worker.labour for worker in workers)
        harvest = min(patch.food, max(labour - patch.resistance, 0.0))
        patch.food -= harvest
        agent.stock += harvest
        bill = agent.wage * len(workers)
        self._borrow(agent, bank, bill - agent.money)
        for worker in workers:
            transfer(agent, worker, agent.wage, 'money')
            worker.harvested = worker.need
        self._pay_debt(agent, bank)
        self._adjust(agent, workers, patch, sold_out)
        agent.workers, agent.revenue = float(len(workers)), 0.0
        owner = agent.get_referenced('owner')
        if not isinstance(owner, SettlerView) and workers:
            owner = workers[0]
            agent.owner = float(owner.id)
        if isinstance(owner, SettlerView) and agent.money > 2 * bill:
            transfer(agent, owner, agent.money - 2 * bill, 'money')

    @staticmethod
    def _get_people_by_id(system: SystemView) -> dict[int, SettlerView]:
        """Return the living people by id.

        They are built once per step in the storage of the caller.
        """
        cached = system.storage.get('people')
        if not isinstance(cached, _PeopleSpec) or cached.time != system.time:
            people = {person.id: person for person in system.get_agents('people', SettlerView) if person.alive}
            cached = _PeopleSpec(system.time, people)
            system.storage['people'] = cached
        return dict(cached.people)


class CommonsInteraction(Interaction):
    """Harvest of the hunter-gatherers.

    The adults of a patch harvest `min(food, labour - resistance)`, and everyone on the patch, child or adult, gets an
    equal share.

    Args:
        label:
            The label of the interaction.

        people:
            The path of the group of people.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import CommonsInteraction
        >>> CommonsInteraction('commons', people='people').people
        'people'
    """

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        *,
        people: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.people = people

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the harvest rule in the phase of work.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.harvest, phase='work')]

    def harvest(self, system: SystemView) -> None:
        """Harvest every occupied patch and share the food among the people on it.

        Args:
            system:
                The system view of the interaction.
        """
        cells: dict[tuple[int, ...], list[SettlerView]] = {}
        for person in system.get_agents('people', SettlerView):
            cells.setdefault(SettlerElement._get_cell(person), []).append(person)
        for group in cells.values():
            patch = SettlerElement._get_patch_here(group[0])
            labour = sum(person.labour for person in group if person.age >= person.maturity)
            harvest = min(patch.food, max(labour - patch.resistance, 0.0))
            patch.food -= harvest
            for person in group:
                person.food += harvest / len(group)
                person.harvested = harvest / len(group)


class EstateInteraction(Interaction):
    """Settlement of the estates of the dead, with a census of the step.

    The dead leave their money and loans to an heir. The money and the loans of a dead person go to a random living
    person of its patch, or of anywhere. When nobody lives to inherit them, the dead person stays as an estate that
    holds them. The reporters `births`, `starvations`, and `old_age` count the events of a step.

    Args:
        label:
            The label of the interaction.

        people:
            The path of the group of people.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import EstateInteraction
        >>> EstateInteraction('census', people='people').reports
        ('births', 'starvations', 'old_age')
    """

    draws = True

    law_classes = (RuleLaw,)

    reports: ClassVar[tuple[str, ...]] = ('births', 'starvations', 'old_age')

    def __init__(
        self,
        label: str,
        *,
        people: Annotated[str, RoleSpec()],
    ) -> None:
        super().__init__(label)
        self.people = people

    def _settle(self, dead: SettlerView, living: list[SettlerView], system: SystemView) -> None:
        """Leave the money and the loans of a dead person to an heir.

        Then remove the person.
        """
        near = [person for person in living if person.location == dead.location] or living
        if near:
            heir = near[int(system.rng.integers(len(near)))]
            transfer(dead, heir, dead.money, 'money')
            transfer(dead, heir, dead.loans, 'loans')
        if dead.money == 0 and dead.loans == 0:
            dead.remove()

    def count(self, system: SystemView) -> None:
        """Count the births and deaths of the step and remove the dead.

        Args:
            system:
                The system view of the interaction.
        """
        people = system.get_agents('people', SettlerView)
        living = [person for person in people if person.fate == ALIVE]
        dead = [person for person in people if person.fate != ALIVE]
        system.report('births', float(sum(person.age == 0 for person in living)))
        system.report('starvations', float(sum(person.fate == STARVED for person in dead)))
        system.report('old_age', float(sum(person.fate == AGED for person in dead)))
        for person in dead:
            self._settle(person, living, system)

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the counting rule in the phase of the census.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.count, phase='census')]


class FoodMarketInteraction(MatchingInteraction[SystemView]):
    """Food market in which people short of food buy from the cheapest shop in sight.

    They buy in random order, from stocked shops. A matching of the people, who request their `shortfall` of food for
    `reserve` steps within the budget of their money, with the firms, which provide their stock at their price. Every
    purchase moves the money to the firm, which books it as revenue, and the food to the buyer. Build it with
    `PartnerMatchingSpec(sight, 'score', score=FoodMarketInteraction.score, value='provider')`, whose score prefers the
    lowest price. Besides the reporters of a matching, the reporter `gdp` sums the value of the sales of a step.

    Examples:
        >>> from skcomplex.core.specs import PartnerSpec
        >>> from skcomplex.systems.economics.macroeconomics import FoodMarketInteraction
        >>> rule = PartnerMatchingSpec(5.0, 'score', score=FoodMarketInteraction.score, value='provider')
        >>> FoodMarketInteraction('shops', requesters='people', providers='businesses', rule=rule).reports[-1]
        'gdp'
    """

    reports: ClassVar[tuple[str, ...]] = (*MatchingInteraction.reports, 'gdp')

    @staticmethod
    def score(person: object, firm: object) -> float:
        """Return the score of a firm for a person, higher for a lower price.

        Args:
            person:
                The person that buys.

            firm:
                The firm that sells.

        Returns:
            score:
                Minus the price of the firm.
        """
        del person
        return -float(firm.price) if isinstance(firm, BusinessView) else -math.inf

    @staticmethod
    def sell(match: MatchSpec, system: SystemView) -> None:
        """Book the value of a sale as revenue of the firm.

        Args:
            match:
                The match of a buyer with a firm.

            system:
                The system view of the market.
        """
        del system
        if isinstance(match.provider, BusinessView):
            match.provider.revenue += match.quantity * match.value

    def _report_matches(self, matches: list[MatchSpec], unmatched: float, system: SystemView) -> None:
        """Report the price, the volume, the unmatched quantity, and the sales value.

        The sales are those of the step.

        Args:
            matches:
                The sales of the step.

            unmatched:
                The quantity of both sides left after the sales.

            system:
                The system view of the market.
        """
        super()._report_matches(matches, unmatched, system)
        system.report('gdp', sum(match.quantity * match.value for match in matches))


class LabourMarketInteraction(MatchingInteraction[SystemView]):
    """Labour market in which unemployed adults take the best-paying vacancy in sight.

    They take it in random order. A matching of the people, who request a job, with the firms, which provide their
    vacancies: a person whose `hireable` limit admits a job takes the firm of the highest wage within its sight that
    still has a vacancy, and moves to it. Build it with
    `PartnerMatchingSpec(sight, 'score', score=LabourMarketInteraction.score, value='provider')`.

    Examples:
        >>> from skcomplex.core.specs import PartnerSpec
        >>> from skcomplex.systems.economics.macroeconomics import LabourMarketInteraction
        >>> LabourMarketInteraction(
        ...     'jobs',
        ...     requesters='people',
        ...     providers='businesses',
        ...     rule=PartnerMatchingSpec(5.0, 'score', score=LabourMarketInteraction.score, value='provider'),
        ...     reserve=('hireable', 'reserve'),
        ...     quantity=('quantity', 'vacancies'),
        ...     on_match=LabourMarketInteraction.hire,
        ... ).providers
        'businesses'
    """

    @staticmethod
    def score(person: object, firm: object) -> float:
        """Return the wage of a firm, the score by which a person prefers it.

        Args:
            person:
                The person who seeks a job.

            firm:
                The firm with a vacancy.

        Returns:
            score:
                The wage of the firm.
        """
        del person
        return float(firm.wage) if isinstance(firm, BusinessView) else -math.inf

    @staticmethod
    def hire(match: MatchSpec, system: SystemView) -> None:
        """Employ a matched person at its firm and move it there.

        Args:
            match:
                The match of a person with a firm.

            system:
                The system view of the market.
        """
        del system
        person, firm = match.requester, match.provider
        if isinstance(person, SettlerView) and isinstance(firm, BusinessView):
            person.employer = float(firm.id)
            person.move_to(firm.location)


class MetaphoriaSystem(BaseSystem):
    """Gooding's Metaphoria.

    It is a mortal population of hunter-gatherers or of a monetary economy. Patches of a periodic lattice hold food that
    grows back to a random capacity and resist harvesting with a random resistance. People need food every step, age,
    die of starvation or old age, and have children who inherit their propensities to breed, explore, found firms, and
    quit jobs, with mutations, so the propensities evolve under the test of survival. Hunter-gatherers pool the labour
    of the adults on a patch and share the harvest. In the monetary economy firms harvest with paid workers, sell the
    food in shops, and borrow the wages from a bank that creates money as loans, so the money of all holders sums to
    zero at every step. Gooding found the hunter-gatherers stable, and the monetary population smaller and less healthy,
    with money concentrating to a Gini coefficient near 0.5. On land where the hunter-gatherers thrive the monetary
    population dwindles towards extinction, and with abundant food it lives on with persistent starvation.

    The system has the group `patches`, one per cell, the variable group `people`, the reporters `census.births`,
    `census.starvations`, and `census.old_age`, and the observables `population` and `money_gini`. The monetary economy
    adds the variable group `businesses`, the group `bank`, and the reporter `shops.gdp`.

    Args:
        n_people:
            The initial number of adults.

        mode:
            `'hunter_gatherer'` or `'monetary'`.

        shape:
            The number of patches along each axis.

        max_food:
            The largest capacity of a patch, drawn uniformly from 0.

        max_resistance:
            The largest resistance of a patch, drawn uniformly from 0.

        regrowth:
            The share of the capacity that grows back in a step.

        need:
            The food a person eats in a step.

        labour:
            The labour of an adult in a step.

        interest:
            The interest rate of loans per step.

        repayment:
            The share of the principal repaid per step.

        n_firms:
            The number of firms of the monetary economy at the start, each without an owner until it has a worker.

        random_state:
            The seed or generator of the patches, the places, ages, and propensities of the people.

        label:
            The label of the system, which names it when it is nested in another one.

    Raises:
        ValueError:
            The mode is unknown.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import MetaphoriaSystem
        >>> [group.label for group in MetaphoriaSystem(10, mode='monetary', shape=(5, 5)).prepare().system_.elements]
        ['patches', 'people', 'businesses', 'bank']
    """

    field = 'economics/macroeconomics'
    topics = ('agents', 'mortality', 'evolution', 'money', 'hunter-gatherers', 'stock-flow consistent')
    references = ('Gooding 2019',)
    until = 300

    @staticmethod
    def _get_bank_balance(bank: SimpleNamespace) -> float:
        """Return the money the bank has issued plus the loans it holds.

        It is zero when every unit of money is a loan.
        """
        return float(bank.money + bank.loans)

    @staticmethod
    def _get_living(fates: object) -> float:
        """Return the number of living people from their fates."""
        return float(np.sum(np.asarray(fates) == ALIVE))

    @staticmethod
    def _get_money_gini(money: object) -> float:
        """Return the Gini coefficient of the money of people."""
        gini = distribution(np.asarray(money, dtype=float), kind='gini').gini
        return 0.0 if gini is None else gini

    @staticmethod
    def _get_metaphoria_observables(system: CompositeSystem) -> list[ObservableSpec]:
        """Return the population, the Gini coefficient of money, and the bank balance.

        The population is the living one, of a monetary economy.
        """
        observables = [
            ObservableSpec('population', group='people', variable='fate', reduce=MetaphoriaSystem._get_living),
            ObservableSpec('money_gini', group='people', variable='money', reduce=MetaphoriaSystem._get_money_gini),
        ]
        if any(element.label == 'bank' for element in system.elements):
            observables.append(
                ObservableSpec('bank_balance', MetaphoriaSystem._get_bank_balance, group='bank', reduce='sum'),
            )
        return observables

    def __init__(
        self,
        n_people: int = 100,
        *,
        mode: str = 'hunter_gatherer',
        shape: Sequence[int] = (20, 20),
        max_food: float = 15.0,
        max_resistance: float = 5.0,
        regrowth: float = 1.0,
        need: float = 2.0,
        labour: float = 5.0,
        interest: float = 0.003,
        repayment: float = 0.004,
        n_firms: int = 5,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_people = n_people
        self.mode = mode
        self.shape = shape
        self.max_food = max_food
        self.max_resistance = max_resistance
        self.regrowth = regrowth
        self.need = need
        self.labour = labour
        self.interest = interest
        self.repayment = repayment
        self.n_firms = n_firms
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        shape = self.shape
        if self.mode not in MODES:
            error_msg = f'Unknown mode {self.mode!r}. Use one of {MODES}.'
            raise ValueError(error_msg)
        rng = np.random.default_rng(self.random_state)
        shape = tuple(int(item) for item in shape)
        space = LatticeSpace(shape)
        sites = list(np.ndindex(*shape))
        capacity = rng.uniform(0.0, self.max_food, len(sites))
        patches = GroupElement(
            'patches',
            element=PlotElement,
            size=len(sites),
            values={
                'food': capacity,
                'capacity': capacity,
                'resistance': rng.uniform(0.0, self.max_resistance, len(sites)),
                'regrowth': self.regrowth,
            },
            location=sites,
        )
        cells = rng.integers(0, shape, size=(self.n_people, len(shape)))
        traits: dict[str, Array] = {
            name: rng.uniform(0.0, bound, self.n_people) for name, bound in PROPENSITIES.items()
        }
        people = GroupElement(
            'people',
            element=SettlerElement,
            size=self.n_people,
            values={
                'age': rng.integers(15, 40, self.n_people).astype(float),
                'need': self.need,
                'labour': self.labour,
                'monetary': float(self.mode == 'monetary'),
                'sight': SIGHT,
                **traits,
            },
            variable=True,
            location=[tuple(int(item) for item in cell) for cell in cells],
        )
        firms = rng.integers(0, shape, size=(self.n_firms, len(shape)))
        businesses = GroupElement(
            'businesses',
            element=BusinessElement,
            size=self.n_firms,
            values={'interest': self.interest, 'repayment': self.repayment},
            variable=True,
            location=[tuple(int(item) for item in cell) for cell in firms],
        )
        bank = GroupElement('bank', element=CentralBankElement, size=1, location=(0,) * len(shape))
        economy = [businesses, bank] if self.mode == 'monetary' else []
        jobs = LabourMarketInteraction(
            'jobs',
            requesters='people',
            providers='businesses',
            rule=PartnerMatchingSpec(SIGHT, 'score', score=LabourMarketInteraction.score, value='provider'),
            reserve=('hireable', 'reserve'),
            quantity=('quantity', 'vacancies'),
            phase='hiring',
            on_match=LabourMarketInteraction.hire,
        )
        shops = FoodMarketInteraction(
            'shops',
            requesters='people',
            providers='businesses',
            rule=PartnerMatchingSpec(SIGHT, 'score', score=FoodMarketInteraction.score, value='provider'),
            reserve=('reserve', 'price'),
            quantity=('shortfall', 'stock'),
            cap='money',
            transfers=(('money', -1),),
            phase='market',
            on_match=FoodMarketInteraction.sell,
        )
        dividends = TransferInteraction(
            'dividends',
            source='bank',
            target='people',
            state='money',
            amount=CentralBankElement.get_income,
            pairing='random',
            fallback='businesses',
            guard=lambda agent, holder, system: CentralBankElement.get_income(agent, holder, system) > 0,
            form='agent',
            phase='banking',
        )
        exchange: list[BaseInteraction] = (
            [CommonsInteraction('commons', people='people')]
            if self.mode == 'hunter_gatherer'
            else [jobs, dividends, shops]
        )
        return CompositeSystem(
            self.label,
            elements=[patches, people, *economy],
            interactions=[
                RegrowthInteraction(
                    'regrowth',
                    target='patches',
                    state='food',
                    growth='regrowth',
                    relative=True,
                    phase='growth',
                ),
                *exchange,
                EstateInteraction('census', people='people'),
            ],
            space=space,
            time=DiscreteTime(schedule=PhasesScheduleSpec([PhaseSpec(name) for name in PHASES])),
        )

    def get_run_settings(self) -> RunSettings:
        """Return the tuned run settings.

        The observables of the parts are built from the model parameters.
        """
        settings = super().get_run_settings()
        settings['observables'] = MetaphoriaSystem._get_metaphoria_observables(self._get_pass_parameters())
        return settings
