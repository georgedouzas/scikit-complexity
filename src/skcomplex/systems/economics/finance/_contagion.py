"""Finance systems: distress that spreads through a network of interbank loans."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np
from sage.all__sagemath_symbolics import NN, oo
from sage.sets.real_set import RealSet

from ....core.components.elements import Element, GroupElement
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.components.system import BaseSystem, System
from ....core.engine.solvers import AgentView, SystemView
from ....core.specs import ObservableSpec, RoleSpec, StateSpec, VisualSpec
from ....core.types import Array


class InterbankView(AgentView):
    """Bank as the cascade sees it.

    It holds its distress `h`, its `status`, and its economic `value`.
    """

    h: float
    status: float
    value: float


UNDISTRESSED = 0


class InterbankElement(Element):
    """Bank of the DebtRank cascade.

    It holds its level of distress, its state, and its economic value.

    Args:
        label:
            The label of the agent or group.

        visual:
            How the element is drawn, or `None` for `default_visual_`.

        h:
            The level of distress, between 0 and 1, where 1 is default.

        status:
            0 for undistressed, 1 for distressed, and 2 for inactive.

        value:
            The economic value, the bank's share of all interbank lending.

    Examples:
        >>> from skcomplex.core.components.elements import GroupElement
        >>> from skcomplex.systems.economics.finance import InterbankElement
        >>> GroupElement('banks', element=InterbankElement, size=3, values={'h': [1.0, 0.0, 0.0]}).size
        3
    """

    law_classes = ()

    def __init__(
        self,
        label: str,
        visual: VisualSpec | None = None,
        *,
        h: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        status: Annotated[float, StateSpec(domain=NN)] = UNDISTRESSED,
        value: Annotated[float, StateSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
    ) -> None:
        super().__init__(label, visual=visual)
        self.h = h
        self.status = status
        self.value = value

    @classmethod
    def _get_view(cls) -> type[InterbankView]:
        """Return the view of the agents of the element."""
        return InterbankView

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return no laws, since the cascade updates every bank at once.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return []


DISTRESSED = 1


INACTIVE = 2


class DebtRankCascadeInteraction(Interaction):
    """Synchronous spread of distress over the distressed banks `j`.

    The distress follows `h_i = min(1, h_i + sum_j W_ji h_j)`. A bank whose distress becomes positive turns distressed,
    and a distressed bank turns inactive after one step, so every bank passes its distress on once.

    Args:
        label:
            The label of the interaction.

        banks:
            The path of the group of banks.

        impact:
            The impact matrix, `impact[i][j]` the relative loss of `j` when `i` defaults.

    Examples:
        >>> from skcomplex.systems.economics.finance import DebtRankCascadeInteraction
        >>> DebtRankCascadeInteraction('cascade', banks='banks', impact=[[0.0, 0.5], [0.0, 0.0]]).impact
        [[0.0, 0.5], [0.0, 0.0]]
    """

    law_classes = (RuleLaw,)

    def __init__(
        self,
        label: str,
        *,
        banks: Annotated[str, RoleSpec()],
        impact: Sequence[Sequence[float]] = (),
    ) -> None:
        super().__init__(label)
        self.banks = banks
        self.impact = impact

    def _get_laws(self, s: SimpleNamespace) -> list[BaseLaw]:
        """Return the cascade rule.

        Args:
            s:
                The namespace of the symbols the laws can use.

        Returns:
            laws:
                The laws of the component.
        """
        return [RuleLaw(self.spread)]

    def spread(self, system: SystemView) -> None:
        """Advance the distress and the states of every bank by one step.

        Args:
            system:
                The system.
        """
        banks = sorted(system.agents('banks', InterbankView), key=lambda bank: bank.id)
        h = np.array([bank.h for bank in banks])
        status = np.array([bank.status for bank in banks])
        distressed = status == DISTRESSED
        new_h = np.minimum(1.0, h + np.asarray(self.impact, dtype=float).T @ (h * distressed))
        hit = (status == UNDISTRESSED) & (new_h > 0)
        new_status = np.where(distressed, INACTIVE, np.where(hit, DISTRESSED, status))
        for bank, level, code in zip(banks, new_h, new_status, strict=True):
            bank.h, bank.status = float(level), float(code)


def _value_at_risk(agent: InterbankView) -> float:
    """Return the distress of a bank weighted by its economic value."""
    return float(agent.h * agent.value)


class DebtrankSystem(BaseSystem):
    """The DebtRank cascade of distress from a set of banks.

    The distress spreads through their interbank loans. `liabilities[i][j]` is the loan from bank `j` to bank `i`, which
    `j` loses when `i` defaults, and the impact of `i` on `j` is `min(1, liabilities[i][j] / capital[j])`. The banks in
    `distressed` start with distress `psi`. The DebtRank is the final value at risk minus the initial one. The system
    has the group `banks` and the distress weighted by economic value `value_at_risk`.

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

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.finance import DebtrankSystem
        >>> DebtrankSystem([[0.0, 5.0], [0.0, 0.0]], [10.0, 10.0]).prepare().system_.elements[0].prepare().size_
        2
    """

    field = 'economics/finance'
    topics = ('systemic risk', 'network', 'contagion', 'agents')
    references = ('Thurner et al. 2018', 'Battiston et al. 2012')
    until = 10
    observables = (ObservableSpec('value_at_risk', _value_at_risk, group='banks', reduce='sum'),)

    def __init__(
        self,
        liabilities: Sequence[Sequence[float]] | None = None,
        capital: Sequence[float] | None = None,
        *,
        distressed: Sequence[int] = (0,),
        psi: float = 1.0,
        n_banks: int = 10,
        density: float = 0.3,
        random_state: int | np.random.Generator | None = 0,
        label: str = 'system',
    ) -> None:
        super().__init__(label)
        self.liabilities = liabilities
        self.capital = capital
        self.distressed = distressed
        self.psi = psi
        self.n_banks = n_banks
        self.density = density
        self.random_state = random_state

    @staticmethod
    def _random_exposures(n_banks: int, density: float, rng: np.random.Generator) -> tuple[Array, Array]:
        """Return random interbank liabilities and capitals of a given density of links."""
        links = rng.random((n_banks, n_banks)) < density
        np.fill_diagonal(links, val=False)
        liabilities = links * rng.uniform(1.0, 10.0, (n_banks, n_banks))
        capital = rng.uniform(5.0, 20.0, n_banks)
        return liabilities, capital

    @staticmethod
    def _impact(liabilities: Array, capital: Array) -> Array:
        """Return the impact matrix `W_ij = min(1, M_ij / C_j)` of a default of `i` on `j`."""
        return np.minimum(1.0, np.asarray(liabilities, dtype=float) / np.asarray(capital, dtype=float)[None, :])

    @staticmethod
    def _economic_values(liabilities: Array) -> Array:
        """Return the economic value of every bank, its share of all interbank lending."""
        lent = np.asarray(liabilities, dtype=float).sum(axis=0)
        return lent / lent.sum()

    def _build_system(self) -> System:
        """Return the parts of the system built from its model parameters."""
        if self.liabilities is None or self.capital is None:
            matrix, capitals = self._random_exposures(
                self.n_banks,
                self.density,
                np.random.default_rng(self.random_state),
            )
        else:
            matrix, capitals = np.asarray(self.liabilities, dtype=float), np.asarray(self.capital, dtype=float)
        size = len(matrix)
        h = np.zeros(size)
        h[list(self.distressed)] = self.psi
        status = np.where(h > 0, DISTRESSED, UNDISTRESSED).astype(float)
        banks = GroupElement(
            'banks',
            element=InterbankElement,
            size=size,
            values={'h': h, 'status': status, 'value': self._economic_values(matrix)},
        )
        return System(
            self.label,
            elements=[banks],
            interactions=[
                DebtRankCascadeInteraction('cascade', banks='banks', impact=self._impact(matrix, capitals).tolist()),
            ],
        )
