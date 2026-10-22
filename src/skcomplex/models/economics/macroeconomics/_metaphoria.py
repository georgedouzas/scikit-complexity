"""Metaphoria: a mortal population that lives as hunter-gatherers or in a monetary economy of firms and a bank."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

from ....core.analysis import gini
from ....core.backends import AgentView, ModelView
from ....core.base import Domain, parameter, role, state
from ....core.components import Element
from ....core.interactions import Interaction, Match, matches
from ....core.laws import Invariant, Law, Rule, Symbols, transfer
from ....core.runtime import register_model
from ....core.space import DiscreteTime, LatticeSpace, Phase, Phases
from ....core.system import Observable, System

MODES = ('hunter_gatherer', 'monetary')
PHASES = ('growth', 'work', 'banking', 'hiring', 'market', 'living', 'census')
PROPENSITIES = {'fertility': 0.3, 'exploration': 1.0, 'enterprise': 0.3, 'restlessness': 0.1}
ALIVE, STARVED, AGED = 0.0, 1.0, 2.0
OLD_AGE_DEATH = 1 / 7


def _bank_balance(bank: Any) -> float:  # noqa: ANN401
    """Return the money the bank has issued plus the loans it holds, zero when every unit of money is a loan."""
    return float(bank.money + bank.loans)


def _living(fates: npt.ArrayLike) -> float:
    """Return the number of living people from their fates."""
    return float(np.sum(np.asarray(fates) == ALIVE))


def _people_by_id(model: ModelView) -> dict[int, AgentView]:
    """Return the living people by id, built once per step in the storage of the caller."""
    cached = model.storage.get('people')
    if cached is None or cached[0] != model.time:
        cached = (model.time, {person.id: person for person in model.agents('people') if person.alive})
        model.storage['people'] = cached
    return cached[1]


def _patch_here(agent: AgentView) -> AgentView:
    """Return the patch of the cell of an agent."""
    return agent.agents_here('patches')[0]


def _wander(agent: AgentView, model: ModelView, sight: float) -> None:
    """Move to a random cell within sight with the propensity to explore, and otherwise to a neighbouring cell."""
    if model.rng.random() < agent.exploration:
        width, height = model.dimensions or (1, 1)
        reach = int(sight)
        dx, dy = model.rng.integers(-reach, reach + 1, size=2)
        x, y = (int(item) for item in agent.location)  # type: ignore[attr-defined]
        agent.move_to(((x + int(dx)) % int(width), (y + int(dy)) % int(height)))
        return
    cells = agent.neighbor_cells()
    agent.move_to(cells[int(model.rng.integers(len(cells)))])


def _offers(
    left: Sequence[Any],
    right: Sequence[Any],
    rng: np.random.Generator,
    capacity: int,
) -> list[tuple[Any, Any]]:
    """Return unemployed adults, in random order, paired with the best-paying firm with a vacancy within their sight."""
    open_firms = {firm: int(firm.vacancies) for firm in left if firm.vacancies > 0}
    pairs = []
    for index in rng.permutation(len(right)):
        person = right[int(index)]
        if person.employer >= 0 or person.age < person.maturity or not open_firms:
            continue
        near = [firm for firm in open_firms if person.distance_to(firm) <= person.sight]
        if near:
            firm = max(near, key=lambda item: item.wage)
            pairs.append((firm, person))
            open_firms[firm] -= 1
            if open_firms[firm] == 0:
                del open_firms[firm]
    return pairs


def _shops(left: Sequence[Any], right: Sequence[Any], rng: np.random.Generator, capacity: int) -> list[tuple[Any, Any]]:
    """Return people short of food, in random order, paired with the cheapest stocked firm within their sight."""
    stocked = [firm for firm in right if firm.stock > 0]
    pairs = []
    for index in rng.permutation(len(left)):
        person = left[int(index)]
        if person.money <= 0 or person.food >= person.reserve * person.need:
            continue
        near = [firm for firm in stocked if person.distance_to(firm) <= person.sight]
        if near:
            pairs.append((person, min(near, key=lambda item: item.price)))
    return pairs


class Plot(Element):
    """Patch of land whose food grows back by a share of its capacity every step, harvested against a resistance.

    Args:
        label:
            The label of the patch or the site group.

        food:
            The food on the patch.

        capacity:
            The most food the patch holds.

        resistance:
            The labour lost to the resistance of the patch before any food is harvested.

        regrowth:
            The share of the capacity that grows back in a step.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import Plot
        >>> Plot('patch').regrowth
        1.0
    """

    food: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    capacity: float = state(default=10.0, domain=Domain.NONNEGATIVE)
    resistance: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    regrowth: float = parameter(default=1.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the regrowth rule in the phase of growth.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.grow, phase='growth')]

    def grow(self, agent: AgentView, model: ModelView) -> None:
        """Grow food back up to the capacity.

        Args:
            agent:
                The patch.

            model:
                The model.
        """
        agent.food = min(agent.capacity, agent.food + agent.regrowth * agent.capacity)


class Settler(Element):
    """Mortal person who eats, ages, raises children, moves, and, in the monetary economy, works and founds firms.

    Every step a person eats `need` food and starves when its food runs out, and past `lifespan` it dies of old age
    with probability 1/7. A child takes `need` food from its parent and follows it. A fed adult has a child with the
    probability `fertility`, at most once every `spacing` steps, and gives it `endowment` food. The child inherits the
    four propensities, each moved by a normal mutation and kept in [0, 1]. A person that harvested too little and holds
    less than `reserve` steps of food moves, far with the probability `exploration`. In the monetary economy an
    unemployed adult on a patch with food but no firm founds one with the probability `enterprise`, and a worker quits
    with the probability `restlessness`.

    Args:
        label:
            The label of the person or the group.

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
        >>> from skcomplex.models.economics.macroeconomics import Settler
        >>> Settler.group('people', 3, variable=True).size
        3
    """

    food: float = state(default=10.0)
    money: float = state(default=0.0, quantity='money')
    loans: float = state(default=0.0, quantity='credit')
    age: float = state(default=20.0, domain=Domain.NONNEGATIVE_INTEGER)
    parent: float = state(default=-1.0)
    employer: float = state(default=-1.0)
    harvested: float = state(default=0.0)
    born: float = state(default=-100.0)
    fate: float = state(default=ALIVE)
    fertility: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    exploration: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    enterprise: float = state(default=0.5, domain=Domain.NONNEGATIVE)
    restlessness: float = state(default=0.1, domain=Domain.NONNEGATIVE)
    labour: float = parameter(default=5.0, domain=Domain.NONNEGATIVE)
    need: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)
    maturity: float = parameter(default=15.0, domain=Domain.NONNEGATIVE)
    lifespan: float = parameter(default=85.0, domain=Domain.NONNEGATIVE)
    spacing: float = parameter(default=2.0, domain=Domain.NONNEGATIVE)
    endowment: float = parameter(default=5.0, domain=Domain.NONNEGATIVE)
    mutation: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)
    sight: float = parameter(default=5.0, domain=Domain.NONNEGATIVE)
    reserve: float = parameter(default=5.0, domain=Domain.NONNEGATIVE)
    monetary: float = parameter(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rule of living in the phase of living.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.live, phase='living')]

    def _feed_child(self, agent: AgentView, model: ModelView) -> None:
        """Take food from the living parent and follow it."""
        parent = _people_by_id(model).get(int(agent.parent))
        if parent is None:
            return
        given = min(agent.need, max(parent.food - parent.need, 0.0))
        parent.food -= given
        agent.food += given
        agent.move_to(parent.location)

    def _breed(self, agent: AgentView, model: ModelView) -> None:
        """Have a child with mutated propensities at the probability of fertility."""
        fed = agent.food >= agent.endowment + agent.need
        spaced = model.time - agent.born >= agent.spacing
        if not (fed and spaced and model.rng.random() < agent.fertility):
            return
        traits = {
            name: float(np.clip(getattr(agent, name) + model.rng.normal(0.0, agent.mutation), 0.0, 1.0))
            for name in PROPENSITIES
        }
        agent.food -= agent.endowment
        agent.born = model.time
        agent.create(None, None, food=agent.endowment, age=0.0, parent=float(agent.id), born=model.time, **traits)

    def _work_or_found(self, agent: AgentView, model: ModelView) -> bool:
        """Quit a job at the restlessness, or found a firm on a free patch with food, and return whether employed."""
        if agent.employer >= 0:
            if model.rng.random() < agent.restlessness:
                agent.employer = -1.0
            return agent.employer >= 0
        if agent.agents_here('businesses') or _patch_here(agent).food <= 0:
            return False
        if model.rng.random() >= agent.enterprise:
            return False
        firm = agent.create('businesses', owner=float(agent.id))
        agent.employer = float(firm.id)
        return True

    def live(self, agent: AgentView, model: ModelView) -> None:
        """Age, eat, die, have children, and move or work.

        Args:
            agent:
                The person.

            model:
                The model.
        """
        agent.age += 1
        adult = agent.age >= agent.maturity
        if not adult:
            self._feed_child(agent, model)
        agent.food -= agent.need
        if agent.food < 0:
            agent.fate = STARVED
            return
        if agent.age > agent.lifespan and model.rng.random() < OLD_AGE_DEATH:
            agent.fate = AGED
            return
        if not adult:
            return
        self._breed(agent, model)
        employed = agent.monetary > 0 and self._work_or_found(agent, model)
        if not employed and agent.harvested < agent.need and agent.food < agent.reserve * agent.need:
            _wander(agent, model, agent.sight)
        agent.harvested = 0.0


class Business(Element):
    """Firm on a patch that harvests with its workers, pays wages with money it borrows, and sells food in its shop.

    Every step the workers pool their labour on the firm's patch, the firm pays each the wage, borrowing what it
    lacks from the bank, and pays the interest and a share of the principal of its loans. A firm that sold out
    raises its price, and one with a stock above `stock_limit` lowers it and lays a worker off. Otherwise a firm
    posts vacancies up to the workforce that harvests the whole capacity of its patch, raises the wage while a
    vacancy went unfilled, and lowers it while its sales fall short of its wage bill. Profits above twice the wage
    bill go to the owner, and a firm without a living owner passes to one of its workers.

    Args:
        label:
            The label of the firm or the group.

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
        >>> from skcomplex.models.economics.macroeconomics import Business
        >>> Business.group('businesses', 0, variable=True).size
        0
    """

    money: float = state(default=0.0, quantity='money')
    loans: float = state(default=0.0, quantity='credit')
    stock: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    wage: float = state(default=4.0, domain=Domain.NONNEGATIVE)
    price: float = state(default=1.0, domain=Domain.POSITIVE)
    owner: float = state(default=-1.0)
    workers: float = state(default=1.0, domain=Domain.NONNEGATIVE_INTEGER)
    vacancies: float = state(default=1.0, domain=Domain.NONNEGATIVE_INTEGER)
    revenue: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    interest: float = parameter(default=0.003, domain=Domain.NONNEGATIVE)
    repayment: float = parameter(default=0.004, domain=Domain.NONNEGATIVE)
    adjustment: float = parameter(default=0.05, domain=Domain.NONNEGATIVE)
    stock_limit: float = parameter(default=30.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the rule of production in the phase of work.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.produce, phase='work')]

    def _borrow(self, agent: AgentView, bank: AgentView, amount: float) -> None:
        """Borrow an amount the bank creates as money and records as a loan."""
        if amount > 0:
            transfer(bank, agent, amount, 'money')
            transfer(agent, bank, amount, 'loans')

    def _service(self, agent: AgentView, bank: AgentView) -> None:
        """Pay the interest and the repayment of the debt, borrowing anew what the firm lacks."""
        debt = -agent.loans
        interest, principal = agent.interest * debt, agent.repayment * debt
        self._borrow(agent, bank, interest + principal - agent.money)
        transfer(agent, bank, interest + principal, 'money')
        transfer(bank, agent, principal, 'loans')

    def _adjust(self, agent: AgentView, workers: list[AgentView], patch: AgentView, sold_out: bool) -> None:
        """Adjust the price, the wage, and the workforce to the stock, the labour, and the sales."""
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

    def produce(self, agent: AgentView, model: ModelView) -> None:
        """Harvest with the workers, pay wages and the bank, adjust, and pay a dividend to the owner.

        Args:
            agent:
                The firm.

            model:
                The model.
        """
        people = _people_by_id(model)
        workers = [person for person in people.values() if person.employer == agent.id]
        bank = model.agents('bank')[0]
        patch = _patch_here(agent)
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
        self._service(agent, bank)
        self._adjust(agent, workers, patch, sold_out)
        agent.workers, agent.revenue = float(len(workers)), 0.0
        owner = people.get(int(agent.owner))
        if owner is None and workers:
            owner = workers[0]
            agent.owner = float(owner.id)
        if owner is not None and agent.money > 2 * bill:
            transfer(agent, owner, agent.money - 2 * bill, 'money')


class CentralBank(Element):
    """Bank that creates money as loans and pays its interest income to a random person as a dividend.

    The money it issued, a negative holding, and the loans it holds sum to zero after every step, so every unit of
    money in the economy is matched by a loan.

    Args:
        label:
            The label of the bank or the group.

        money:
            The money of the bank, the negative of the money it issued plus its income of the step.

        loans:
            The loans the bank holds.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import CentralBank
        >>> CentralBank.group('bank', 1).size
        1
    """

    money: float = state(default=0.0, quantity='money')
    loans: float = state(default=0.0, quantity='credit')

    def laws(self, s: Symbols) -> list[Law]:
        """Return the dividend rule in the phase of banking and the invariant of the balance sheet.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        laws: list[Any] = [Rule(self.distribute, phase='banking'), Invariant('balance', _bank_balance, tolerance=1e-6)]
        return laws

    def distribute(self, agent: AgentView, model: ModelView) -> None:
        """Pay the income of the step to a random person, or to a random firm when nobody lives.

        Args:
            agent:
                The bank.

            model:
                The model.
        """
        income = agent.money + agent.loans
        holders = list(_people_by_id(model).values()) or model.agents('businesses')
        if income > 0 and holders:
            transfer(agent, holders[int(model.rng.integers(len(holders)))], income, 'money')


class Commons(Interaction):
    """Harvest of the hunter-gatherers, who pool the labour of the adults on a patch and share the food equally.

    The adults of a patch harvest `min(food, labour - resistance)`, and everyone on the patch, child or adult, gets an
    equal share.

    Args:
        label:
            The label of the interaction.

        people:
            The path of the group of people.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import Commons
        >>> Commons('commons', people='people').people
        'people'
    """

    people: str = role()

    def laws(self, s: Symbols) -> list[Law]:
        """Return the harvest rule in the phase of work.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.harvest, phase='work')]

    def harvest(self, model: Any) -> None:  # noqa: ANN401
        """Harvest every occupied patch and share the food among the people on it.

        Args:
            model:
                The model view of the interaction.
        """
        cells: dict[tuple[int, ...], list[AgentView]] = {}
        for person in model.agents('people'):
            cells.setdefault(tuple(int(item) for item in person.location), []).append(person)
        for group in cells.values():
            patch = _patch_here(group[0])
            labour = sum(person.labour for person in group if person.age >= person.maturity)
            harvest = min(patch.food, max(labour - patch.resistance, 0.0))
            patch.food -= harvest
            for person in group:
                person.food += harvest / len(group)
                person.harvested = harvest / len(group)


class LabourMarket(Interaction):
    """Labour market in which unemployed adults take the best-paying vacancy within their sight.

    Args:
        label:
            The label of the interaction.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import LabourMarket
        >>> LabourMarket('jobs').label
        'jobs'
    """

    def laws(self, s: Symbols) -> list[Law]:
        """Return the matching of vacancies with workers and the rule that hires along the pairs.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        laws: list[Any] = [
            Match('businesses', 'people', pairing=_offers, label='hires', phase='hiring'),
            Rule(self.hire, 'hiring'),
        ]
        return laws

    def hire(self, model: Any) -> None:  # noqa: ANN401
        """Employ every matched worker at its firm.

        Args:
            model:
                The model view of the interaction.
        """
        for firm, person in matches(model, 'hires'):
            person.employer = float(firm.id)
            person.move_to(firm.location)
            firm.vacancies = max(firm.vacancies - 1, 0.0)


class FoodMarket(Interaction):
    """Food market in which people short of food buy from the cheapest stocked shop within their sight.

    A buyer takes the food it lacks for `reserve` steps, as much as the stock and its money allow, and pays the price
    to the firm. The reporter `gdp` sums the value of the sales of a step.

    Args:
        label:
            The label of the interaction.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import FoodMarket
        >>> FoodMarket('shops').reports
        ('gdp',)
    """

    reports: ClassVar[tuple[str, ...]] = ('gdp',)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the matching of buyers with shops and the rule that trades along the pairs.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        laws: list[Any] = [
            Match('people', 'businesses', pairing=_shops, label='purchases', phase='market'),
            Rule(self.sell, 'market'),
        ]
        return laws

    def sell(self, model: Any) -> None:  # noqa: ANN401
        """Sell food along every pair of the step and report the value of the sales.

        Args:
            model:
                The model view of the interaction.
        """
        value = 0.0
        for person, firm in matches(model, 'purchases'):
            wanted = person.reserve * person.need - person.food
            quantity = max(min(wanted, firm.stock, person.money / firm.price), 0.0)
            transfer(person, firm, quantity * firm.price, 'money')
            firm.stock -= quantity
            firm.revenue += quantity * firm.price
            person.food += quantity
            value += quantity * firm.price
        model.report('gdp', value)


class Census(Interaction):
    """Census that counts births and deaths and settles the estate of the dead before removing them.

    The money and the loans of a dead person go to a random living person of its patch, or of anywhere. When nobody
    lives to inherit them, the dead person stays as an estate that holds them. The reporters `births`, `starvations`,
    and `old_age` count the events of a step.

    Args:
        label:
            The label of the interaction.

        people:
            The path of the group of people.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import Census
        >>> Census('census', people='people').reports
        ('births', 'starvations', 'old_age')
    """

    reports: ClassVar[tuple[str, ...]] = ('births', 'starvations', 'old_age')
    people: str = role()

    def _settle(self, dead: AgentView, living: list[AgentView], model: Any) -> None:  # noqa: ANN401
        """Leave the money and the loans of a dead person to an heir, and remove the person."""
        near = [person for person in living if person.location == dead.location] or living
        if near:
            heir = near[int(model.rng.integers(len(near)))]
            transfer(dead, heir, dead.money, 'money')
            transfer(dead, heir, dead.loans, 'loans')
        if dead.money == 0 and dead.loans == 0:
            dead.remove()

    def count(self, model: Any) -> None:  # noqa: ANN401
        """Count the births and deaths of the step and remove the dead.

        Args:
            model:
                The model view of the interaction.
        """
        people = model.agents('people')
        living = [person for person in people if person.fate == ALIVE]
        dead = [person for person in people if person.fate != ALIVE]
        model.report('births', float(sum(person.age == 0 for person in living)))
        model.report('starvations', float(sum(person.fate == STARVED for person in dead)))
        model.report('old_age', float(sum(person.fate == AGED for person in dead)))
        for person in dead:
            self._settle(person, living, model)

    def laws(self, s: Symbols) -> list[Law]:
        """Return the counting rule in the phase of the census.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.count, phase='census')]


@register_model(
    field='economics/macroeconomics',
    tags=('agents', 'mortality', 'evolution', 'money', 'hunter-gatherers', 'stock-flow consistent'),
    references=('Gooding 2019',),
    defaults={'until': 300},
    observables=lambda system: [
        Observable('population', group='people', variable='fate', reduce=_living),
        Observable('money_gini', group='people', variable='money', reduce=gini),
    ],
)
def metaphoria(
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
) -> System:
    """Return Gooding's Metaphoria, a mortal population of hunter-gatherers or of a monetary economy.

    Patches of a periodic lattice hold food that grows back to a random capacity and resist harvesting with a random
    resistance. People need food every step, age, die of starvation or old age, and have children who inherit their
    propensities to breed, explore, found firms, and quit jobs, with mutations, so the propensities evolve under the
    test of survival. Hunter-gatherers pool the labour of the adults on a patch and share the harvest. In the monetary
    economy firms harvest with paid workers, sell the food in shops, and borrow the wages from a bank that creates
    money as loans, so the money of all holders sums to zero at every step. Gooding found the hunter-gatherers stable,
    and the monetary population smaller and less healthy, with money concentrating to a Gini coefficient near 0.5. On
    land where the hunter-gatherers thrive the monetary population dwindles towards extinction, and with abundant food
    it lives on with persistent starvation.

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

    Returns:
        system:
            The model, with the group `patches`, one per cell, the variable group `people`, the reporters
            `census.births`, `census.starvations`, and `census.old_age`, and the observables `population` and
            `money_gini`. The monetary economy adds the variable group `businesses`, the group `bank`, and the reporter
            `shops.gdp`.

    Raises:
        ValueError:
            The mode is unknown.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import metaphoria
        >>> [group.label for group in metaphoria(10, mode='monetary', shape=(5, 5)).elements]
        ['patches', 'people', 'businesses', 'bank']
    """
    if mode not in MODES:
        error_msg = f'Unknown mode {mode!r}. Use one of {MODES}.'
        raise ValueError(error_msg)
    rng = np.random.default_rng(random_state)
    shape = tuple(int(item) for item in shape)
    space = LatticeSpace(shape)
    sites = list(np.ndindex(*shape))
    capacity = rng.uniform(0.0, max_food, len(sites))
    patches = Plot.group(
        'patches',
        len(sites),
        location=sites,
        food=capacity,
        capacity=capacity,
        resistance=rng.uniform(0.0, max_resistance, len(sites)),
        regrowth=regrowth,
    )
    cells = rng.integers(0, shape, size=(n_people, len(shape)))
    traits = {name: rng.uniform(0.0, bound, n_people) for name, bound in PROPENSITIES.items()}
    people = Settler.group(
        'people',
        n_people,
        variable=True,
        location=[tuple(int(item) for item in cell) for cell in cells],
        age=rng.integers(15, 40, n_people).astype(float),
        need=need,
        labour=labour,
        monetary=float(mode == 'monetary'),
        **traits,
    )
    firms = rng.integers(0, shape, size=(n_firms, len(shape)))
    businesses = Business.group(
        'businesses',
        n_firms,
        variable=True,
        location=[tuple(int(item) for item in cell) for cell in firms],
        interest=interest,
        repayment=repayment,
    )
    bank = CentralBank.group('bank', 1, location=(0,) * len(shape))
    economy = [businesses, bank] if mode == 'monetary' else []
    exchange: list[Interaction] = (
        [Commons('commons', people='people')]
        if mode == 'hunter_gatherer'
        else [LabourMarket('jobs'), FoodMarket('shops')]
    )
    return System(
        components=[patches, people, *economy],
        interactions=[*exchange, Census('census', people='people')],
        space=space,
        time=DiscreteTime(schedule=Phases([Phase(name) for name in PHASES])),
    )
