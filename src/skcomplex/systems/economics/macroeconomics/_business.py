"""Firm on a patch that harvests with its workers, pays wages with money it borrows, and sells food in its shop."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import math
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.laws.utils import transfer
from ....core.engine.solvers import SystemView
from ....core.specs import (
    ParameterSpec,
    StateSpec,
    VisualSpec,
)
from ._business_view import BusinessView
from ._central_bank_view import CentralBankView
from ._people_spec import _PeopleSpec
from ._plot_view import PlotView
from ._settler import SettlerElement
from ._settler_view import SettlerView


class BusinessElement(Element):
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

    def _service(self, agent: BusinessView, bank: CentralBankView) -> None:
        """Pay the interest and the repayment of the debt, borrowing anew what the firm lacks."""
        debt = -agent.loans
        interest, principal = agent.interest * debt, agent.repayment * debt
        self._borrow(agent, bank, interest + principal - agent.money)
        transfer(agent, bank, interest + principal, 'money')
        transfer(bank, agent, principal, 'loans')

    def _adjust(self, agent: BusinessView, workers: list[SettlerView], patch: PlotView, sold_out: bool) -> None:
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

    def produce(self, agent: BusinessView, system: SystemView) -> None:
        """Harvest with the workers, pay wages and the bank, adjust, and pay a dividend to the owner.

        Args:
            agent:
                The firm.

            system:
                The system.
        """
        people = BusinessElement._people_by_id(system)
        workers = [person for person in people.values() if person.employer == agent.id]
        bank = system.agents('bank', CentralBankView)[0]
        patch = SettlerElement._patch_here(agent)
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
        owner = agent.referenced('owner')
        if not isinstance(owner, SettlerView) and workers:
            owner = workers[0]
            agent.owner = float(owner.id)
        if isinstance(owner, SettlerView) and agent.money > 2 * bill:
            transfer(agent, owner, agent.money - 2 * bill, 'money')

    @staticmethod
    def _people_by_id(system: SystemView) -> dict[int, SettlerView]:
        """Return the living people by id, built once per step in the storage of the caller."""
        cached = system.storage.get('people')
        if not isinstance(cached, _PeopleSpec) or cached.time != system.time:
            people = {person.id: person for person in system.agents('people', SettlerView) if person.alive}
            cached = _PeopleSpec(system.time, people)
            system.storage['people'] = cached
        return dict(cached.people)
