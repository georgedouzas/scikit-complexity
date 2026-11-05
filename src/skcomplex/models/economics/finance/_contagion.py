"""Finance models: distress that spreads through a network of interbank loans."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence

import numpy as np

from ....core.elements import GroupElement
from ....core.system import System
from ....specs import ObservableSpec
from ....types import Array
from ..._catalog import register_model
from ._debt_rank_cascade import DISTRESSED, DebtRankCascadeInteraction
from ._interbank import UNDISTRESSED, InterbankElement
from ._interbank_view import InterbankView


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


def _value_at_risk(agent: InterbankView) -> float:
    """Return the distress of a bank weighted by its economic value."""
    return float(agent.h * agent.value)


@register_model(
    field='economics/finance',
    tags=('systemic risk', 'network', 'contagion', 'agents'),
    references=('Thurner et al. 2018', 'Battiston et al. 2012'),
    defaults={'until': 10},
    observables=lambda system: [ObservableSpec('value_at_risk', _value_at_risk, group='banks', reduce='sum')],
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
        >>> debtrank([[0.0, 5.0], [0.0, 0.0]], [10.0, 10.0]).elements[0].prepare().size_
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
    banks = GroupElement(
        'banks',
        element=InterbankElement,
        size=size,
        values={'h': h, 'status': status, 'value': _economic_values(matrix)},
    )
    return System(
        elements=[banks],
        interactions=[DebtRankCascadeInteraction('cascade', banks='banks', impact=_impact(matrix, capitals).tolist())],
    )
