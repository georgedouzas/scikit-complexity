"""Finance systems: default cascades through interbank exposures and the clearing of interbank payments."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from typing import Annotated, ClassVar

import numpy as np
from sage.sets.real_set import RealSet

from ....core.components.elements import GroupElement
from ....core.components.interactions import FeedbackInteraction
from ....core.components.system import System
from ....core.engine.solvers import SystemView
from ....core.specs import ParameterSpec, RoleSpec
from ....core.types import Array
from ..._catalog import register_system
from ._balance_sheet import DEFAULTED, DEFAULTING, SOLVENT, BalanceSheetElement, BalanceSheetView


class DefaultCascadeInteraction(FeedbackInteraction[SystemView]):
    """Synchronous cascade of defaults, in which the creditors of a defaulting bank write off their claims on it.

    Every step the banks that defaulted in the last step pass their losses on: every creditor `j` of a defaulting bank
    `i` loses `(1 - recovery) L_ij` of its equity, and a solvent bank whose equity falls below the threshold defaults.

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
        """Pass on the losses of the defaulting banks and default the banks below the threshold.

        Args:
            system:
                The system view of the cascade.
        """
        banks = sorted(system.agents('banks', BalanceSheetView), key=lambda bank: bank.id)
        status = np.array([bank.status for bank in banks])
        equity = np.array([bank.equity for bank in banks])
        defaulting = status == DEFAULTING
        losses = (1 - system.parameter('recovery')) * np.asarray(self.liabilities, dtype=float).T @ defaulting
        equity = equity - losses
        hit = (status == SOLVENT) & (equity < system.parameter('threshold'))
        status = np.where(defaulting, DEFAULTED, np.where(hit, DEFAULTING, status))
        for bank, value, code in zip(banks, equity, status, strict=True):
            bank.equity, bank.status = float(value), float(code)
        system.report('defaults', float(np.sum(status != SOLVENT)))
        system.report('losses', float(losses.sum()))


class PaymentClearingInteraction(FeedbackInteraction[SystemView]):
    """Clearing of interbank payments by the iteration `p <- min(p_bar, e + Pi^T p)` of Eisenberg and Noe.

    Every step every bank pays the smaller of what it owes, `p_bar_i = sum_j L_ij`, and what it has, its external assets
    plus the payments it receives, which its debtors make in proportion to their liabilities, `Pi_ij = L_ij / p_bar_i`.
    From full payments the iteration falls to the greatest clearing vector, and a bank that pays less than it owes
    defaults.

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
        banks = sorted(system.agents('banks', BalanceSheetView), key=lambda bank: bank.id)
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


def _gai_kapadia_network(
    n_banks: int,
    degree: float,
    interbank_share: float,
    capital_ratio: float,
    rng: np.random.Generator,
) -> tuple[Array, Array, Array]:
    """Return the liabilities, external assets, and capitals of the random network of Gai and Kapadia.

    Every bank has assets of 100, of which the share `interbank_share` are claims split evenly over its debtors, links
    drawn with probability `degree / (n - 1)`, and capital of `capital_ratio` of its assets.
    """
    links = rng.random((n_banks, n_banks)) < degree / (n_banks - 1)
    np.fill_diagonal(links, val=False)
    debtors = links.sum(axis=0)
    claims = np.divide(100.0 * interbank_share, debtors, out=np.zeros(n_banks), where=debtors > 0)
    liabilities = np.asarray(links * claims[None, :], dtype=float)
    external = 100.0 - liabilities.sum(axis=0)
    capital = np.full(n_banks, 100.0 * capital_ratio)
    return liabilities, external, capital


@register_system(
    field='economics/finance',
    tags=('systemic risk', 'network', 'contagion', 'default cascade', 'agents'),
    references=('Thurner et al. 2018', 'Bardoscia et al. 2021', 'Gai and Kapadia 2010'),
    defaults={'until': 30},
)
def default_cascade(
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
) -> System:
    """Return a cascade of defaults through interbank exposures, from a shock to the external assets of some banks.

    Banks are linked by contracts, `liabilities[i][j]` the amount bank `i` owes bank `j`. A bank whose equity falls
    below the threshold `h` defaults, and its creditors write off their claims on it, losing `(1 - R) L_ij` of their
    equity at the next step, which may default them in turn. This is the cascade of Thurner et al. (2018, Section
    5.5.3.2), and with `h = 0` and the recovery rate `R = 0` it is the contagion on default of Gai and Kapadia (2010) as
    Bardoscia et al. (2021, Section 3.1) review it. The shocked banks lose the share `shock` of their external assets at
    the start. Without a given network, every bank has assets of 100, a fifth of them interbank claims split evenly over
    a random number of debtors of mean `degree`, and capital of 4 percent, the network of Gai and Kapadia, whose
    cascades are large in a window of intermediate degrees and rare for sparse or dense networks.

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

    Returns:
        system:
            The system, with the group `banks` and the cascade `cascade`, whose reporters are the number of defaulted
            banks `cascade.defaults` and the losses written off in the step `cascade.losses`.

    Examples:
        >>> from skcomplex.systems.economics.finance import default_cascade
        >>> default_cascade([[0.0, 5.0], [0.0, 0.0]], [1.0, 2.0]).elements[0].prepare().size_
        2
    """
    if liabilities is None or equity is None:
        rng = np.random.default_rng(random_state)
        matrix, external, capital = _gai_kapadia_network(n_banks, degree, interbank_share, capital_ratio, rng)
    else:
        matrix, capital = np.asarray(liabilities, dtype=float), np.asarray(equity, dtype=float)
        external = np.maximum(capital, 0.0)
    size = len(matrix)
    hit = np.zeros(size, dtype=bool)
    hit[list(shocked)] = True
    losses = np.where(hit, shock * external, 0.0)
    after = capital - losses
    status = np.where(after < threshold, DEFAULTING, SOLVENT).astype(float)
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
        threshold=threshold,
        recovery=recovery,
    )
    return System(elements=[banks], interactions=[cascade])


@register_system(
    field='economics/finance',
    tags=('systemic risk', 'network', 'clearing', 'payments', 'fixed point', 'agents'),
    references=('Bardoscia et al. 2021', 'Eisenberg and Noe 2001'),
    defaults={'until': 30},
)
def eisenberg_noe_clearing(
    liabilities: Sequence[Sequence[float]] | None = None,
    external: Sequence[float] | None = None,
    *,
    n_banks: int = 50,
    density: float = 0.1,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the clearing of interbank payments of Eisenberg and Noe, a fixed point of proportional payments.

    Bank `i` owes `L_ij` to bank `j`, `p_bar_i` in all, and holds external assets `e_i`. A clearing vector pays
    `p_i = min(p_bar_i, e_i + sum_j Pi_ji p_j)`, every bank paying in full or all it has, its debtors paying in
    proportion `Pi_ij = L_ij / p_bar_i`, the liquidity contagion of Bardoscia et al. (2021, Section 3.1). Every step is
    one round of the iteration from full payments, which falls monotonically to the greatest clearing vector, the
    solution of the linear programme `max sum_i p_i` with `p <= p_bar` and `p <= e + Pi^T p`. Without a given network
    the liabilities are random and the external assets are small for a few banks.

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

    Returns:
        system:
            The system, with the group `banks` of the payments `banks.payment`, and the clearing `clearing`, whose
            reporters are the total unpaid liabilities `clearing.shortfall` and the number of defaulting banks
            `clearing.defaults`.

    Examples:
        >>> from skcomplex.systems.economics.finance import eisenberg_noe_clearing
        >>> eisenberg_noe_clearing([[0.0, 2.0], [1.0, 0.0]], [0.5, 1.0]).elements[0].prepare().size_
        2
    """
    if liabilities is None or external is None:
        rng = np.random.default_rng(random_state)
        links = rng.random((n_banks, n_banks)) < density
        np.fill_diagonal(links, val=False)
        matrix = links * rng.uniform(1.0, 10.0, (n_banks, n_banks))
        assets = rng.exponential(5.0, n_banks)
    else:
        matrix, assets = np.asarray(liabilities, dtype=float), np.asarray(external, dtype=float)
    banks = GroupElement(
        'banks',
        element=BalanceSheetElement,
        size=len(matrix),
        values={'external': assets, 'payment': matrix.sum(axis=1)},
    )
    clearing = PaymentClearingInteraction('clearing', banks='banks', liabilities=matrix.tolist())
    return System(elements=[banks], interactions=[clearing])
