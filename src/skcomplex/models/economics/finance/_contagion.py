"""Finance models: distress that spreads through a network of interbank loans."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from ....core.base import Domain, role, state
from ....core.model import Element, Interaction, Law, Observable, Rule, Symbols, System
from ....core.runtime import register_model
from ....core.solvers import AgentView, ModelView

type Array = npt.NDArray[np.float64]

UNDISTRESSED = 0
DISTRESSED = 1
INACTIVE = 2


def _random_exposures(n_banks: int, density: float, rng: np.random.Generator) -> tuple[Array, Array]:
    """Return random interbank liabilities and capitals of a given density of links."""
    links = rng.random((n_banks, n_banks)) < density
    np.fill_diagonal(links, val=False)
    liabilities = links * rng.uniform(1.0, 10.0, (n_banks, n_banks))
    capital = rng.uniform(5.0, 20.0, n_banks)
    return liabilities, capital


def _impact(liabilities: Array, capital: Array) -> Array:
    """Return the impact matrix `W_ij = min(1, M_ij / C_j)` of a default of `i` on `j`."""
    return np.minimum(1.0, np.asarray(liabilities, dtype=float) / np.asarray(capital, dtype=float)[None, :])


def _economic_values(liabilities: Array) -> Array:
    """Return the economic value of every bank, its share of all interbank lending."""
    lent = np.asarray(liabilities, dtype=float).sum(axis=0)
    return lent / lent.sum()


def _value_at_risk(agent: AgentView) -> float:
    """Return the distress of a bank weighted by its economic value."""
    return float(agent.h * agent.value)


class Bank(Element):
    """Bank of the DebtRank cascade, with its level of distress, its state, and its economic value.

    Args:
        label:
            The label of the agent or group.

        h:
            The level of distress, between 0 and 1, where 1 is default.

        status:
            0 for undistressed, 1 for distressed, and 2 for inactive.

        value:
            The economic value, the bank's share of all interbank lending.

    Examples:
        >>> from skcomplex.models.economics.finance import Bank
        >>> Bank.group('banks', 3, h=[1.0, 0.0, 0.0]).size
        3
    """

    h: float = state(default=0.0, domain=Domain.NONNEGATIVE)
    status: float = state(default=UNDISTRESSED, domain=Domain.NONNEGATIVE_INTEGER)
    value: float = state(default=0.0, domain=Domain.NONNEGATIVE)

    def laws(self, s: Symbols) -> list[Law]:
        """Return no laws, since the cascade updates every bank at once.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


class DebtRankCascade(Interaction):
    """Synchronous spread of distress, `h_i = min(1, h_i + sum_j W_ji h_j)` over the distressed banks `j`.

    A bank whose distress becomes positive turns distressed, and a distressed bank turns inactive after one step, so
    every bank passes its distress on once.

    Args:
        label:
            The label of the interaction.

        banks:
            The path of the group of banks.

        impact:
            The impact matrix, `impact[i][j]` the relative loss of `j` when `i` defaults.

    Examples:
        >>> from skcomplex.models.economics.finance import DebtRankCascade
        >>> DebtRankCascade('cascade', banks='banks', impact=[[0.0, 0.5], [0.0, 0.0]]).impact
        [[0.0, 0.5], [0.0, 0.0]]
    """

    banks: str = role()
    impact: Sequence[Sequence[float]] = ()

    def laws(self, s: Symbols) -> list[Law]:
        """Return the cascade rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [Rule(self.spread)]

    def spread(self, model: ModelView) -> None:
        """Advance the distress and the states of every bank by one step.

        Args:
            model:
                The model.
        """
        banks = sorted(model.agents('banks'), key=lambda bank: bank.id)
        h = np.array([bank.h for bank in banks])
        status = np.array([bank.status for bank in banks])
        distressed = status == DISTRESSED
        new_h = np.minimum(1.0, h + np.asarray(self.impact, dtype=float).T @ (h * distressed))
        hit = (status == UNDISTRESSED) & (new_h > 0)
        new_status = np.where(distressed, INACTIVE, np.where(hit, DISTRESSED, status))
        for bank, level, code in zip(banks, new_h, new_status, strict=True):
            bank.h, bank.status = float(level), float(code)


@register_model(
    field='economics/finance',
    tags=('systemic risk', 'network', 'contagion', 'agents'),
    references=('Thurner et al. 2018', 'Battiston et al. 2012'),
    defaults={'until': 10},
    observables=lambda system: [Observable('value_at_risk', _value_at_risk, group='banks', reduce='sum')],
)
def debtrank(
    liabilities: Sequence[Sequence[float]] | None = None,
    capital: Sequence[float] | None = None,
    *,
    distressed: Sequence[int] = (0,),
    psi: float = 1.0,
    n_banks: int = 10,
    density: float = 0.3,
    random_state: int | np.random.Generator | None = 0,
) -> System:
    """Return the DebtRank cascade of distress from a set of banks through their interbank loans.

    `liabilities[i][j]` is the loan from bank `j` to bank `i`, which `j` loses when `i` defaults, and the impact of `i`
    on `j` is `min(1, liabilities[i][j] / capital[j])`. The banks in `distressed` start with distress `psi`. The
    DebtRank is the final value at risk minus the initial one.

    Args:
        liabilities:
            The interbank liabilities, or `None` for a random network of `n_banks` banks.

        capital:
            The capital of every bank, or `None` for random capitals.

        distressed:
            The indices of the banks distressed at the start.

        psi:
            The initial distress of those banks, where 1 is default.

        n_banks:
            The number of banks of a random network.

        density:
            The probability of a loan between two banks of a random network.

        random_state:
            The seed or generator of a random network.

    Returns:
        system:
            The model, with the group `banks` and the distress weighted by economic value `value_at_risk`.

    Examples:
        >>> from skcomplex.models.economics.finance import debtrank
        >>> debtrank([[0.0, 5.0], [0.0, 0.0]], [10.0, 10.0]).elements[0].size
        2
    """
    if liabilities is None or capital is None:
        matrix, capitals = _random_exposures(n_banks, density, np.random.default_rng(random_state))
    else:
        matrix, capitals = np.asarray(liabilities, dtype=float), np.asarray(capital, dtype=float)
    size = len(matrix)
    h = np.zeros(size)
    h[list(distressed)] = psi
    status = np.where(h > 0, DISTRESSED, UNDISTRESSED).astype(float)
    banks = Bank.group('banks', size, h=h, status=status, value=_economic_values(matrix))
    return System(
        components=[banks],
        interactions=[DebtRankCascade('cascade', banks='banks', impact=_impact(matrix, capitals).tolist())],
    )
