"""Macroeconomics systems: the Mark-0 economy of firms, a household, and a bank.

It runs under a Covid shock and policies.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated, ClassVar

import numpy as np
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import FeedbackInteraction
from ....core.components.laws import BaseLaw
from ....core.components.spaces import BaseSpace
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import AgentView, SystemView
from ....core.errors import ValidationError
from ....core.specs import ParameterSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array


class Mark0FirmView(AgentView):
    """Firm as the economy sees it.

    It holds its workforce, demand, price, wage, cash, profit, and whether it is active.
    """

    workforce: float
    demand: float
    price: float
    wage: float
    cash: float
    profit: float
    active: float


class Mark0FirmElement(Element):
    """Firm of the Mark-0 economy.

    It holds its workforce, demand, price, wage, cash balance, and profit. The production is the productivity times the
    workforce, a negative cash balance is a debt, and the economy of the system updates every firm at once.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        workforce:
            The workers of the firm, as a share of the households.

        demand:
            The demand for the goods of the firm.

        price:
            The price of the goods of the firm.

        wage:
            The wage the firm pays.

        cash:
            The cash balance, negative for a debt.

        profit:
            The profit of the last step.

        active:
            1 for a firm in business, 0 for a bankrupt firm waiting to be revived.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.macroeconomics import Mark0FirmElement
        >>> GroupElement('firms', element=Mark0FirmElement, size=3).size
        3
    """

    law_classes = ()

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        workforce: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        demand: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        price: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        wage: Annotated[float, StateSpec(domain=RealSet.open(0, oo))] = 1.0,
        cash: Annotated[float, StateSpec()] = 0.0,
        profit: Annotated[float, StateSpec()] = 0.0,
        active: Annotated[float, StateSpec(domain=RealSet.closed(0, 1))] = 1.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.workforce = workforce
        self.demand = demand
        self.price = price
        self.wage = wage
        self.cash = cash
        self.profit = profit
        self.active = active

    @classmethod
    def _get_view(cls) -> type[Mark0FirmView]:
        """Return the view of the agents of the element."""
        return Mark0FirmView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since the economy updates every firm at once.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


CREDIT_POLICIES = ('none', 'naive', 'adaptive')


class Mark0EconomyInteraction(FeedbackInteraction[SystemView]):
    """Representative household and bank of the Mark-0 economy.

    They run the firms through one month every step. Every step the firms adjust their workforce, prices, and wages by
    the rules of thumb of Mark-0 (Eqs. 11 to 17 of Sharma et al. 2020) from the demand and the profit of the last month,
    pay their wages, and sell to the household, whose consumption budget `c (S + W + rho_d S)` (Eq. 5) splits among the
    firms by the logit demand of their prices relative to the mean price (Eq. 9). Firms pay interest and, when
    profitable with positive cash, a share `delta` of their cash as dividends (Eqs. 18 and 19). A firm whose debt to
    payroll `Phi = -E / (W N)` reaches the bankruptcy threshold `Theta` defaults, its debt falls on the bank, which
    moves the loan and deposit rates (Eqs. 21 and 22), and a bankrupt firm is revived with probability `phi` per month
    at the mean price and wage. Following the simplifications of the source, inflation expectations, the feedback of
    inflation on consumption, the effect of fragility on hiring, and the central bank are switched off, so the base rate
    is zero.

    During the shock, from `shock_time` for `shock_length` months, the productivity falls by the share `dzeta` and the
    consumption propensity by the share `dc`. The credit policy `'naive'` lifts the threshold during the shock and
    restores it at once after it, and `'adaptive'` restores it to `max(offset <Phi>, Theta)` with the
    production-weighted mean fragility `<Phi>` (Eq. 24). At the end of the shock the household's savings are multiplied
    by `kappa`, the helicopter money.

    Args:
        label:
            The label of the economy.

        rule:
            Unused, since the economy overrides `act`.

        phase:
            The phase the economy acts in, or `None` for the default phase.

        firms:
            The path of the group of firms.

        c0:
            The consumption propensity of the household.

        beta:
            The intensity of choice of the demand and of the workers.

        gamma:
            The relative size of the adjustments of prices and wages.

        eta_minus:
            The firing propensity `eta0-`.

        ratio:
            The hiring to firing ratio `R`, so the hiring propensity is `R eta0-`.

        delta:
            The share of their cash that profitable firms pay as dividends.

        theta:
            The bankruptcy threshold of the debt to payroll ratio.

        phi:
            The probability per month that a bankrupt firm is revived.

        zeta:
            The productivity of a worker.

        omega:
            The weight of the last inflation in its moving average.

        f:
            The share of the costs of defaults borne by the borrowers rather than the depositors.

        shock_time:
            The month the shock starts.

        shock_length:
            The months the shock lasts.

        dc:
            The relative fall of the consumption propensity during the shock.

        dzeta:
            The relative fall of the productivity during the shock.

        kappa:
            The factor of the household's savings at the end of the shock, 1 for no helicopter money.

        offset:
            The factor of the mean fragility of the adaptive credit policy.

        credit:
            The credit policy, `'none'`, `'naive'`, or `'adaptive'`.

    Raises:
        ValidationError:
            The credit policy is unknown, raised by `prepare`.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import Mark0EconomyInteraction
        >>> Mark0EconomyInteraction('economy', firms='firms').reports[:2]
        ('unemployment', 'inflation')
    """

    draws = True

    reports: ClassVar[tuple[str, ...]] = (
        'unemployment',
        'inflation',
        'output',
        'savings',
        'fragility',
        'bankruptcies',
        'threshold',
        'price',
        'wage',
    )

    def __init__(
        self,
        label: str,
        *,
        rule: None = None,
        phase: str | None = None,
        firms: Annotated[str, RoleSpec()],
        c0: Annotated[float, ParameterSpec(domain=RealSet.open(0, 1))] = 0.5,
        beta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        gamma: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, 1))] = 0.01,
        eta_minus: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.2,
        ratio: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
        delta: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.02,
        theta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 3.0,
        phi: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.1,
        zeta: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        omega: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.2,
        f: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.5,
        shock_time: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 100.0,
        shock_length: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        dc: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, 1))] = 0.0,
        dzeta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, 1))] = 0.0,
        kappa: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        offset: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.25,
        credit: str = 'none',
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.firms = firms
        self.c0 = c0
        self.beta = beta
        self.gamma = gamma
        self.eta_minus = eta_minus
        self.ratio = ratio
        self.delta = delta
        self.theta = theta
        self.phi = phi
        self.zeta = zeta
        self.omega = omega
        self.f = f
        self.shock_time = shock_time
        self.shock_length = shock_length
        self.dc = dc
        self.dzeta = dzeta
        self.kappa = kappa
        self.offset = offset
        self.credit = credit

    def _prepare(self, context: BaseSpace | None = None) -> None:
        """Raise an error for an unknown credit policy.

        Args:
            context:
                The space the interaction inherits from its system.

        Raises:
            ValidationError:
                The credit policy is unknown.
        """
        super()._prepare(context)
        if self.credit not in CREDIT_POLICIES:
            error_msg = f'The credit policy of {self.label!r} is {self.credit!r}. Use one of {CREDIT_POLICIES}.'
            raise ValidationError(error_msg)

    def _threshold(self, system: SystemView, month: int, fragility: float) -> float:
        """Return the bankruptcy threshold of the month under the credit policy."""
        start, length = system.parameter('shock_time'), system.parameter('shock_length')
        theta = system.parameter('theta')
        if self.credit == 'none' or length == 0 or month < start:
            return theta
        if month < start + length:
            return np.inf
        if self.credit == 'adaptive':
            return max(system.parameter('offset') * fragility, theta)
        return theta

    @staticmethod
    def _stored(system: SystemView, name: str, default: float) -> float:
        """Return a number the economy stored in an earlier step.

        Before it is stored, it is the default.
        """
        value = system.storage.get(name, default)
        return float(value) if isinstance(value, int | float | np.number) else default

    @staticmethod
    def _weighted(values: Array, weights: Array) -> float:
        """Return the mean of values weighted by nonnegative weights.

        It is the plain mean when they sum to zero.
        """
        total = weights.sum()
        return float(values @ weights / total) if total > 0 else float(values.mean())

    def _adjust(self, system: SystemView, state: dict[str, Array], zeta: float, unemployment: float) -> None:
        """Adjust the workforce, prices, and wages of the active firms.

        The adjustment follows last month's demand and profit.
        """
        rng, gamma = system.rng, system.parameter('gamma')
        active = state['active'] > 0
        workforce, demand, price, wage, profit = (state[name] for name in ('n', 'd', 'p', 'w', 'profit'))
        production = zeta * workforce
        mean_price = self._weighted(price[active], production[active])
        mean_wage = self._weighted(wage[active], production[active])
        attraction = np.where(active, np.exp(system.parameter('beta') * (wage / mean_wage - 1)), 0.0)
        available = unemployment * len(workforce) * attraction / attraction.sum()
        under, over = active & (production < demand), active & (production > demand)
        eta_minus = system.parameter('eta_minus')
        eta_plus = min(1.0, system.parameter('ratio') * eta_minus)
        hired = np.minimum(eta_plus * (demand - production), zeta * available) / zeta
        workforce = np.where(
            under,
            workforce + hired,
            np.where(over, workforce - eta_minus * (production - demand) / zeta, workforce),
        )
        xi = rng.random(len(price))
        price = np.where(under & (price < mean_price), price * (1 + gamma * xi), price)
        price = np.where(over & (price > mean_price), price * (1 - gamma * xi), price)
        xi = rng.random(len(wage))
        raised = under & (profit > 0)
        sales = state['sales']
        ceiling = np.divide(sales + state['interest'], workforce, out=np.full_like(wage, np.inf), where=workforce > 0)
        wage = np.where(
            raised,
            np.minimum(wage * (1 + gamma * (1 - unemployment) * xi), np.maximum(ceiling, wage)),
            wage,
        )
        wage = np.where(over & (profit < 0), wage * (1 - gamma * unemployment * xi), wage)
        state['n'], state['p'], state['w'] = np.maximum(workforce, 0.0), price, wage

    def act(self, system: SystemView) -> None:
        """Run one month of the economy.

        It adjusts, pays, sells, distributes, defaults, revives, and sets the rates.

        Args:
            system:
                The system view of the economy.
        """
        firms = sorted(system.agents('firms', Mark0FirmView), key=lambda firm: firm.id)
        names = ('workforce', 'demand', 'price', 'wage', 'cash', 'profit', 'active')
        keys = ('n', 'd', 'p', 'w', 'e', 'profit', 'active')
        state = {key: np.array([getattr(firm, name) for firm in firms]) for key, name in zip(keys, names, strict=True)}
        store = system.storage
        month = int(self._stored(system, 'month', 0.0))
        savings, inflation_ema = self._stored(system, 'savings', 0.0), self._stored(system, 'inflation_ema', 0.0)
        rho_l, rho_d = self._stored(system, 'rho_l', 0.0), self._stored(system, 'rho_d', 0.0)
        start, length = system.parameter('shock_time'), system.parameter('shock_length')
        shocked = start <= month < start + length
        zeta = system.parameter('zeta') * (1 - system.parameter('dzeta') * shocked)
        c = system.parameter('c0') * (1 - system.parameter('dc') * shocked)
        if length > 0 and month == start + length:
            savings *= system.parameter('kappa')
        size = len(firms)
        if month > 0:
            state['sales'] = np.asarray(store['sales'])
            state['interest'] = np.asarray(store['interest'])
            self._adjust(system, state, zeta, 1 - state['n'].sum() / size)
        active = state['active'] > 0
        production = zeta * state['n']
        bill = state['w'] * state['n']
        wages = float(bill.sum())
        state['e'] = state['e'] - bill
        mean_price = self._weighted(state['p'][active], production[active])
        mean_wage = self._weighted(state['w'][active], production[active])
        budget = c * (savings + wages + rho_d * savings)
        weights = np.where(active, np.exp(-system.parameter('beta') * state['p'] / mean_price), 0.0)
        state['d'] = budget * weights / (state['p'] * weights.sum())
        sales = state['p'] * np.minimum(production, state['d'])
        interest = rho_d * np.maximum(state['e'], 0.0) + rho_l * np.minimum(state['e'], 0.0)
        state['e'] = state['e'] + sales + interest
        state['profit'] = sales - bill + interest
        savings += wages + rho_d * savings - float(sales.sum())
        dividends = np.where((state['profit'] > 0) & (state['e'] > 0), system.parameter('delta') * state['e'], 0.0)
        state['e'] -= dividends
        savings += float(dividends.sum())
        fragility_of = np.divide(-state['e'], bill, out=np.zeros(size), where=bill > 0)
        fragility = self._weighted(fragility_of[active], production[active])
        threshold = self._threshold(system, month, fragility)
        defaulted = active & (fragility_of >= threshold)
        losses = float(-state['e'][defaulted].sum())
        for key in ('n', 'd', 'e', 'profit', 'active'):
            state[key] = np.where(defaulted, 0.0, state[key])
        unemployment = 1 - state['n'].sum() / size
        revived = (state['active'] == 0) & ~defaulted & (system.rng.random(size) < system.parameter('phi'))
        start_size = unemployment * system.rng.random(size)
        state['n'] = np.where(revived, start_size, state['n'])
        state['p'] = np.where(revived, mean_price, state['p'])
        state['w'] = np.where(revived, mean_wage, state['w'])
        state['active'] = np.where(revived, 1.0, state['active'])
        previous = self._stored(system, 'price', mean_price)
        inflation = (mean_price - previous) / previous
        inflation_ema = system.parameter('omega') * inflation + (1 - system.parameter('omega')) * inflation_ema
        debts, deposits = float(-np.minimum(state['e'], 0.0).sum()), float(np.maximum(state['e'], 0.0).sum())
        f = system.parameter('f')
        rho_l = f * losses / debts if debts > 0 else 0.0
        rho_d = -(1 - f) * losses / (savings + deposits) if savings + deposits > 0 else 0.0
        store.update(
            month=month + 1,
            savings=savings,
            inflation_ema=inflation_ema,
            rho_l=rho_l,
            rho_d=rho_d,
            price=mean_price,
            sales=np.where(defaulted, 0.0, sales),
            interest=np.where(defaulted, 0.0, interest),
        )
        for index, firm in enumerate(firms):
            for key, name in zip(keys, names, strict=True):
                setattr(firm, name, float(state[key][index]))
        reported = {
            'unemployment': 1 - state['n'].sum() / size,
            'inflation': inflation,
            'output': float((zeta * state['n']).sum()),
            'savings': savings,
            'fragility': fragility,
            'bankruptcies': float(defaulted.sum()),
            'threshold': min(threshold, 1e9),
            'price': mean_price,
            'wage': mean_wage,
        }
        for name, value in reported.items():
            system.report(name, value)


class Mark0System(BaseSystem):
    """The Mark-0 economy of Sharma, Bouchaud, and co-authors.

    It has a Covid shock to supply and demand. Firms produce perishable goods with a linear technology, sell them to a
    representative household at prices they adjust by rules of thumb, hire and fire by the gap between production and
    demand, and borrow from a representative bank, defaulting when their debt to payroll exceeds the threshold `Theta`
    (Sharma et al. 2020, Section 2). A month is a step, the money starts at zero, and the economy is stock-flow
    consistent apart from the written-off debts and the helicopter money. With the defaults of Table 1 the economy sits
    in the full employment phase. A lockdown lowers the productivity and the consumption propensity for `shock_length`
    months, after which a mild shock recovers in a V shape and a strong one tips the economy into an L-shaped recession
    of high unemployment, which easy credit and helicopter money can prevent. The source summarizes the model of earlier
    papers, so the initial firms (a workforce between 0.9 and 1, prices around 1, unit wages, no cash) and revived firms
    (the mean price and wage and a random share of the unemployed) follow Gualdi et al. (2015), and the demand reads
    prices relative to the mean price.

    The system has the group `firms` and the economy `economy`, whose reporters are `economy.unemployment`,
    `economy.inflation`, `economy.output`, `economy.savings`, `economy.fragility`, `economy.bankruptcies`,
    `economy.threshold`, `economy.price`, and `economy.wage`.

    Args:
        n_firms:
            The number of firms, equal to the number of households.

        c0:
            The consumption propensity of the household.

        beta:
            The intensity of choice of the demand and of the workers.

        gamma:
            The relative size of the adjustments of prices and wages.

        eta_minus:
            The firing propensity.

        ratio:
            The hiring to firing ratio.

        delta:
            The share of their cash that profitable firms pay as dividends.

        theta:
            The bankruptcy threshold of the debt to payroll ratio.

        phi:
            The probability per month that a bankrupt firm is revived.

        shock_time:
            The month the shock starts.

        shock_length:
            The months the shock lasts, 0 for no shock.

        dc:
            The relative fall of the consumption propensity during the shock.

        dzeta:
            The relative fall of the productivity during the shock.

        kappa:
            The factor of the household's savings at the end of the shock, 1 for no helicopter money.

        credit:
            The credit policy, `'none'`, `'naive'`, or `'adaptive'`.

        random_state:
            The seed or generator of the initial firms.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import Mark0System
        >>> Mark0System(10).prepare().system_.elements[0].prepare().size_
        10
    """

    field = 'economics/macroeconomics'
    topics = ('agents', 'macroeconomics', 'unemployment', 'crises', 'covid', 'policy', 'firms', 'stock-flow consistent')
    references = ('Sharma et al. 2020', 'Gualdi et al. 2015')
    until = 300

    def __init__(
        self,
        n_firms: int = 500,
        *,
        c0: float = 0.5,
        beta: float = 2.0,
        gamma: float = 0.01,
        eta_minus: float = 0.2,
        ratio: float = 2.0,
        delta: float = 0.02,
        theta: float = 3.0,
        phi: float = 0.1,
        shock_time: float = 100.0,
        shock_length: float = 0.0,
        dc: float = 0.0,
        dzeta: float = 0.0,
        kappa: float = 1.0,
        credit: str = 'none',
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.n_firms = n_firms
        self.c0 = c0
        self.beta = beta
        self.gamma = gamma
        self.eta_minus = eta_minus
        self.ratio = ratio
        self.delta = delta
        self.theta = theta
        self.phi = phi
        self.shock_time = shock_time
        self.shock_length = shock_length
        self.dc = dc
        self.dzeta = dzeta
        self.kappa = kappa
        self.credit = credit
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        rng = np.random.default_rng(self.random_state)
        firms = GroupElement(
            'firms',
            element=Mark0FirmElement,
            size=self.n_firms,
            values={
                'workforce': rng.uniform(0.9, 1.0, self.n_firms),
                'price': 1 + 0.1 * (rng.random(self.n_firms) - 0.5),
                'wage': 1.0,
            },
        )
        economy = Mark0EconomyInteraction(
            'economy',
            firms='firms',
            c0=self.c0,
            beta=self.beta,
            gamma=self.gamma,
            eta_minus=self.eta_minus,
            ratio=self.ratio,
            delta=self.delta,
            theta=self.theta,
            phi=self.phi,
            shock_time=self.shock_time,
            shock_length=self.shock_length,
            dc=self.dc,
            dzeta=self.dzeta,
            kappa=self.kappa,
            credit=self.credit,
        )
        return CompositeSystem(self.label, elements=[firms], interactions=[economy])
