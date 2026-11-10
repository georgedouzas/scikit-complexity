"""Finance systems: default cascades and the clearing of interbank payments.

The cascades run through interbank exposures.
"""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated, ClassVar

import numpy as np
from sage.sets.real_set import RealSet

from ....core.components.elements import GroupElement
from ....core.components.systems import BaseSystem, CompositeSystem
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, RoleSpec
from ....core.types import Array
from ..common.banking import BalanceSheetElement
from ..common.banking._balance_sheet import DEFAULTED, DEFAULTING, SOLVENT, BalanceSheetView
from ..common.policy import PolicyRuleInteraction


class DefaultCascadeInteraction(PolicyRuleInteraction[SystemView]):
    """Synchronous cascade of defaults.

    The creditors of a defaulting bank write off their claims on it. Every step the banks that defaulted in the last
    step pass their losses on: every creditor `j` of a defaulting bank `i` loses `(1 - recovery) L_ij` of its equity,
    and a solvent bank whose equity falls below the threshold defaults.

    Args:
        label:
            The label of the cascade.

        rule:
            Unused, since the cascade overrides `act`.

        phase:
            The phase the cascade acts in, or `None` for the default phase.

        banks:
            The path of the group of banks.

        liabilities:
            The interbank liabilities, `liabilities[i][j]` owed by bank `i` to bank `j`.

        threshold:
            The equity below which a bank defaults.

        recovery:
            The share of a claim on a defaulted bank that its creditor recovers.

    Examples:
        >>> from skcomplex.systems.economics.finance import DefaultCascadeInteraction
        >>> DefaultCascadeInteraction('cascade', banks='banks').reports
        ('defaults', 'losses')
    """

    reports: ClassVar[tuple[str, ...]] = ('defaults', 'losses')

    def __init__(
        self,
        label: str,
        *,
        rule: None = None,
        phase: str | None = None,
        banks: Annotated[str, RoleSpec()],
        liabilities: Sequence[Sequence[float]] = (),
        threshold: Annotated[float, ParameterSpec()] = 0.0,
        recovery: Annotated[float, ParameterSpec(domain=RealSet.closed(0, 1))] = 0.0,
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.banks = banks
        self.liabilities = liabilities
        self.threshold = threshold
        self.recovery = recovery

    def act(self, system: SystemView) -> None:
        """Pass on the losses of the defaulting banks.

        Then default the banks below the threshold.

        Args:
            system:
                The system view of the cascade.
        """
        banks = sorted(system.get_agents('banks', BalanceSheetView), key=lambda bank: bank.id)
        status = np.array([bank.status for bank in banks])
        equity = np.array([bank.equity for bank in banks])
        defaulting = status == DEFAULTING
        losses = (1 - system.get_parameter('recovery')) * np.asarray(self.liabilities, dtype=float).T @ defaulting
        equity = equity - losses
        hit = (status == SOLVENT) & (equity < system.get_parameter('threshold'))
        status = np.where(defaulting, DEFAULTED, np.where(hit, DEFAULTING, status))
        for bank, value, code in zip(banks, equity, status, strict=True):
            bank.equity, bank.status = float(value), float(code)
        system.report('defaults', float(np.sum(status != SOLVENT)))
        system.report('losses', float(losses.sum()))


class PaymentClearingInteraction(PolicyRuleInteraction[SystemView]):
    """Clearing of interbank payments of Eisenberg and Noe.

    It iterates `p <- min(p_bar, e + Pi^T p)`. Every step every bank pays the smaller of what it owes,
    `p_bar_i = sum_j L_ij`, and what it has, its external assets plus the payments it receives, which its debtors make
    in proportion to their liabilities, `Pi_ij = L_ij / p_bar_i`. From full payments the iteration falls to the greatest
    clearing vector, and a bank that pays less than it owes defaults.

    Args:
        label:
            The label of the clearing.

        rule:
            Unused, since the clearing overrides `act`.

        phase:
            The phase the clearing acts in, or `None` for the default phase.

        banks:
            The path of the group of banks.

        liabilities:
            The interbank liabilities, `liabilities[i][j]` owed by bank `i` to bank `j`.

    Examples:
        >>> from skcomplex.systems.economics.finance import PaymentClearingInteraction
        >>> PaymentClearingInteraction('clearing', banks='banks').reports
        ('shortfall', 'defaults')
    """

    reports: ClassVar[tuple[str, ...]] = ('shortfall', 'defaults')

    def __init__(
        self,
        label: str,
        *,
        rule: None = None,
        phase: str | None = None,
        banks: Annotated[str, RoleSpec()],
        liabilities: Sequence[Sequence[float]] = (),
    ) -> None:
        super().__init__(label, rule=rule, phase=phase)
        self.banks = banks
        self.liabilities = liabilities

    def act(self, system: SystemView) -> None:
        """Make one round of payments and record the shortfall and the defaults.

        Args:
            system:
                The system view of the clearing.
        """
        banks = sorted(system.get_agents('banks', BalanceSheetView), key=lambda bank: bank.id)
        matrix = np.asarray(self.liabilities, dtype=float)
        owed = matrix.sum(axis=1)
        shares = np.divide(matrix, owed[:, None], out=np.zeros_like(matrix), where=owed[:, None] > 0)
        payments = np.array([bank.payment for bank in banks])
        external = np.array([bank.external for bank in banks])
        payments = np.minimum(owed, external + shares.T @ payments)
        equity = external + shares.T @ payments - payments
        defaulted = payments < owed * (1 - 1e-12)
        for bank, paid, value, flag in zip(banks, payments, equity, defaulted, strict=True):
            bank.payment, bank.equity, bank.status = float(paid), float(value), float(DEFAULTED * flag)
        system.report('shortfall', float(np.sum(owed - payments)))
        system.report('defaults', float(defaulted.sum()))


class DefaultCascadeSystem(BaseSystem):
    """A cascade of defaults through interbank exposures.

    It starts from a shock to the external assets of some banks. Banks are linked by contracts, `liabilities[i][j]` the
    amount bank `i` owes bank `j`. A bank whose equity falls below the threshold `h` defaults, and its creditors write
    off their claims on it, losing `(1 - R) L_ij` of their equity at the next step, which may default them in turn. This
    is the cascade of Thurner et al. (2018, Section 5.5.3.2), and with `h = 0` and the recovery rate `R = 0` it is the
    contagion on default of Gai and Kapadia (2010) as Bardoscia et al. (2021, Section 3.1) review it. The shocked banks
    lose the share `shock` of their external assets at the start. Without a given network, every bank has assets of 100,
    a fifth of them interbank claims split evenly over a random number of debtors of mean `degree`, and capital of 4
    percent, the network of Gai and Kapadia, whose cascades are large in a window of intermediate degrees and rare for
    sparse or dense networks.

    The system has the group `banks` and the cascade `cascade`, whose reporters are the number of defaulted banks
    `cascade.defaults` and the losses written off in the step `cascade.losses`.

    Args:
        liabilities:
            The interbank liabilities, or `None` for the random network of Gai and Kapadia.

        equity:
            The equity of every bank before the shock, or `None` for the capital of the random network.

        shocked:
            The indices of the banks whose external assets the shock hits.

        shock:
            The share of their external assets the shocked banks lose.

        threshold:
            The equity below which a bank defaults.

        recovery:
            The share of a claim on a defaulted bank that its creditor recovers.

        n_banks:
            The number of banks of the random network.

        degree:
            The mean number of debtors of a bank of the random network.

        interbank_share:
            The share of the assets of a bank of the random network that are interbank claims.

        capital_ratio:
            The capital of a bank of the random network as a share of its assets.

        random_state:
            The seed or generator of the random network.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.finance import DefaultCascadeSystem
        >>> DefaultCascadeSystem([[0.0, 5.0], [0.0, 0.0]], [1.0, 2.0]).prepare().system_.elements[0].prepare().size_
        2
    """

    field = 'economics/finance'
    topics = ('systemic risk', 'network', 'contagion', 'default cascade', 'agents')
    references = ('Thurner et al. 2018', 'Bardoscia et al. 2021', 'Gai and Kapadia 2010')
    until = 30

    def __init__(
        self,
        liabilities: Sequence[Sequence[float]] | None = None,
        equity: Sequence[float] | None = None,
        *,
        shocked: Sequence[int] = (0,),
        shock: float = 1.0,
        threshold: float = 0.0,
        recovery: float = 0.0,
        n_banks: int = 250,
        degree: float = 4.0,
        interbank_share: float = 0.2,
        capital_ratio: float = 0.04,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.liabilities = liabilities
        self.equity = equity
        self.shocked = shocked
        self.shock = shock
        self.threshold = threshold
        self.recovery = recovery
        self.n_banks = n_banks
        self.degree = degree
        self.interbank_share = interbank_share
        self.capital_ratio = capital_ratio
        self.random_state = random_state

    @staticmethod
    def _get_gai_kapadia_network(
        n_banks: int,
        degree: float,
        interbank_share: float,
        capital_ratio: float,
        rng: np.random.Generator,
    ) -> tuple[Array, Array, Array]:
        """Return the liabilities, external assets, and capitals of a random network.

        The network is that of Gai and Kapadia. Every bank has assets of 100, of which the share `interbank_share` are
        claims split evenly over its debtors, links drawn with probability `degree / (n - 1)`, and capital of
        `capital_ratio` of its assets.
        """
        links = rng.random((n_banks, n_banks)) < degree / (n_banks - 1)
        np.fill_diagonal(links, val=False)
        debtors = links.sum(axis=0)
        claims = np.divide(100.0 * interbank_share, debtors, out=np.zeros(n_banks), where=debtors > 0)
        liabilities = np.asarray(links * claims[None, :], dtype=float)
        external = 100.0 - liabilities.sum(axis=0)
        capital = np.full(n_banks, 100.0 * capital_ratio)
        return liabilities, external, capital

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        if self.liabilities is None or self.equity is None:
            rng = np.random.default_rng(self.random_state)
            matrix, external, capital = self._get_gai_kapadia_network(
                self.n_banks,
                self.degree,
                self.interbank_share,
                self.capital_ratio,
                rng,
            )
        else:
            matrix, capital = np.asarray(self.liabilities, dtype=float), np.asarray(self.equity, dtype=float)
            external = np.maximum(capital, 0.0)
        size = len(matrix)
        hit = np.zeros(size, dtype=bool)
        hit[list(self.shocked)] = True
        losses = np.where(hit, self.shock * external, 0.0)
        after = capital - losses
        status = np.where(after < self.threshold, DEFAULTING, SOLVENT).astype(float)
        banks = GroupElement(
            'banks',
            element=BalanceSheetElement,
            size=size,
            values={'external': external - losses, 'equity': after, 'status': status},
        )
        cascade = DefaultCascadeInteraction(
            'cascade',
            banks='banks',
            liabilities=matrix.tolist(),
            threshold=self.threshold,
            recovery=self.recovery,
        )
        return CompositeSystem(self.label, elements=[banks], interactions=[cascade])


class EisenbergNoeClearingSystem(BaseSystem):
    """The clearing of interbank payments of Eisenberg and Noe.

    It is a fixed point of proportional payments. Bank `i` owes `L_ij` to bank `j`, `p_bar_i` in all, and holds external
    assets `e_i`. A clearing vector pays `p_i = min(p_bar_i, e_i + sum_j Pi_ji p_j)`, every bank paying in full or all
    it has, its debtors paying in proportion `Pi_ij = L_ij / p_bar_i`, the liquidity contagion of Bardoscia et al.
    (2021, Section 3.1). Every step is one round of the iteration from full payments, which falls monotonically to the
    greatest clearing vector, the solution of the linear programme `max sum_i p_i` with `p <= p_bar` and
    `p <= e + Pi^T p`. Without a given network the liabilities are random and the external assets are small for a few
    banks.

    The system has the group `banks` of the payments `banks.payment`, and the clearing `clearing`, whose reporters are
    the total unpaid liabilities `clearing.shortfall` and the number of defaulting banks `clearing.defaults`.

    Args:
        liabilities:
            The interbank liabilities, or `None` for a random network.

        external:
            The external assets of every bank, or `None` for random ones.

        n_banks:
            The number of banks of a random network.

        density:
            The probability of a liability between two banks of a random network.

        random_state:
            The seed or generator of a random network.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.finance import EisenbergNoeClearingSystem
        >>> system = EisenbergNoeClearingSystem([[0.0, 2.0], [1.0, 0.0]], [0.5, 1.0]).prepare().system_
        >>> system.elements[0].prepare().size_
        2
    """

    field = 'economics/finance'
    topics = ('systemic risk', 'network', 'clearing', 'payments', 'fixed point', 'agents')
    references = ('Bardoscia et al. 2021', 'Eisenberg and Noe 2001')
    until = 30

    def __init__(
        self,
        liabilities: Sequence[Sequence[float]] | None = None,
        external: Sequence[float] | None = None,
        *,
        n_banks: int = 50,
        density: float = 0.1,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.liabilities = liabilities
        self.external = external
        self.n_banks = n_banks
        self.density = density
        self.random_state = random_state

    def _build_system(self) -> CompositeSystem:
        """Return the parts of the system built from its model parameters."""
        if self.liabilities is None or self.external is None:
            rng = np.random.default_rng(self.random_state)
            links = rng.random((self.n_banks, self.n_banks)) < self.density
            np.fill_diagonal(links, val=False)
            matrix = links * rng.uniform(1.0, 10.0, (self.n_banks, self.n_banks))
            assets = rng.exponential(5.0, self.n_banks)
        else:
            matrix, assets = np.asarray(self.liabilities, dtype=float), np.asarray(self.external, dtype=float)
        banks = GroupElement(
            'banks',
            element=BalanceSheetElement,
            size=len(matrix),
            values={'external': assets, 'payment': matrix.sum(axis=1)},
        )
        clearing = PaymentClearingInteraction('clearing', banks='banks', liabilities=matrix.tolist())
        return CompositeSystem(self.label, elements=[banks], interactions=[clearing])
