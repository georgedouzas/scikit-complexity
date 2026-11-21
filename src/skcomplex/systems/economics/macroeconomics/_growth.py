"""Macroeconomics systems: growth and business cycles."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.components import Observable
from ....core.components.elements import Element
from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RateLaw
from ....core.engine.backends import BaseBackend
from ....core.engine.solvers import BaseSolver
from ....core.systems import BaseSystem, System
from ....core.types import Role
from ...general.dynamics import DynamicalSystem
from ...general.dynamics.types import Equations


class EconomyElement(Element):
    """Economy described by its capital per worker.

    Args:
        label:
            The label of the economy.

        k:
            The initial capital per worker.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import EconomyElement
        >>> EconomyElement('country', k=1.0).k
        1.0
    """

    def __init__(
        self,
        label: str,
        *,
        k: Annotated[float, RealSet.closed_open(0, oo)] = 1.0,
    ) -> None:
        super().__init__(label)
        self.k = k


class SolowGrowthInteraction(Interaction):
    """Solow growth of the capital per worker `k` of a target.

    It follows `dk/dt = s * k**alpha - delta * k`, the saving of a Cobb-Douglas output minus depreciation.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the economy.

        s:
            The saving rate.

        alpha:
            The output elasticity of capital.

        delta:
            The depreciation rate.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import SolowGrowthInteraction
        >>> SolowGrowthInteraction('solow', target='country', s=0.3).s
        0.3
    """

    law_classes = (RateLaw,)

    def __init__(
        self,
        label: str,
        *,
        target: Role,
        s: Annotated[float, RealSet.closed_open(0, oo)] = 0.2,
        alpha: Annotated[float, RealSet.open(0, oo)] = 0.3,
        delta: Annotated[float, RealSet.open(0, oo)] = 0.05,
    ) -> None:
        super().__init__(label)
        self.target = target
        self.s = s
        self.alpha = alpha
        self.delta = delta

    def _prepare(self, context: SimpleNamespace) -> None:
        """Set the rate `s * k**alpha - delta * k` of the capital per worker as the law.

        Args:
            context:
                The namespace of the elements a binding of the roles names.
        """
        super()._prepare(context)
        k = context.target.k
        self.laws_: list[BaseLaw] = [RateLaw(k, context.s * k**context.alpha - context.delta * k)]


class SolowSystem(BaseSystem):
    """An economy whose capital per worker approaches its steady state.

    It approaches through saving and depreciation. The system has the economy `economy`.

    Args:
        k:
            The initial capital per worker.

        s:
            The saving rate.

        alpha:
            The capital share of output.

        delta:
            The depreciation rate.

        label:
            The label of the system, which names it when it is nested in another one.

        solver:
            The solver of the problem, or `'auto'` for the one the system chooses.

        backend:
            The backend that evaluates the compiled functions, or `'auto'` for the one the system chooses.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import SolowSystem
        >>> SolowSystem(s=0.3).prepare().system_.interactions[0].s
        0.3
    """

    field = 'economics/macroeconomics'
    topics = ('growth', 'ode')
    references = ('Solow 1956',)
    until = 200.0
    dt = 1.0

    def __init__(
        self,
        k: float = 1.0,
        *,
        s: float = 0.2,
        alpha: float = 0.3,
        delta: float = 0.05,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label, solver=solver, backend=backend)
        self.k = k
        self.s = s
        self.alpha = alpha
        self.delta = delta

    def _prepare(self, context: Sequence[Observable] | None = None) -> None:
        """Set the system of the components built from the model parameters."""
        super()._prepare(context)
        self.system_ = System(
            self.label,
            elements=[EconomyElement('economy', k=self.k)],
            interactions=[
                SolowGrowthInteraction('growth', target='economy', s=self.s, alpha=self.alpha, delta=self.delta),
            ],
        )


class MultiplierAcceleratorSystem(DynamicalSystem):
    """Samuelson's multiplier-accelerator, `Y_{t+1} = c (1 + v) Y_t - c v Y_{t-1} + G`.

    Income oscillates around `G / (1 - c)`, with growing amplitude when `c v > 1` and complex characteristic roots. The
    system has the income `economy.Y` and its previous value `economy.Y_lag1`.

    Args:
        income:
            The income of the current period.

        previous:
            The income of the previous period.

        c:
            The marginal propensity to consume.

        v:
            The accelerator, the investment response to the change in consumption.

        G:
            The government spending.

        label:
            The label of the system, which names it when it is nested in another one.

    Examples:
        >>> from skcomplex.systems.economics.macroeconomics import MultiplierAcceleratorSystem
        >>> MultiplierAcceleratorSystem(v=0.5).prepare().system_.elements[0].values['v']
        0.5
    """

    field = 'economics/macroeconomics'
    topics = ('business cycle', 'discrete map', 'second order')
    references = ('Samuelson 1939',)
    until = 40
    dt = 1

    def __init__(
        self,
        income: float = 110.0,
        previous: float = 100.0,
        *,
        c: float = 0.8,
        v: float = 1.5,
        G: float = 20.0,
        label: str = 'system',
        solver: BaseSolver | str = 'auto',
        backend: BaseBackend | str = 'auto',
    ) -> None:
        super().__init__(label=label, solver=solver, backend=backend)
        self.income = income
        self.previous = previous
        self.c = c
        self.v = v
        self.G = G

    @staticmethod
    def _get_next_income(s: SimpleNamespace) -> Expression:
        """Return the income of the next period, `c (1 + v) Y_t - c v Y_{t-1} + G`."""
        economy = s.target
        income: Expression = (
            economy.c * (1 + economy.v) * economy.Y - economy.c * economy.v * economy.Y(t=s.t - 1) + economy.G
        )
        return income

    def _get_element_label(self) -> str:
        """Return the label of the element that holds the states."""
        return 'economy'

    def _get_time(self) -> str:
        """Return the discrete time of the map."""
        return 'discrete'

    def _get_states(self) -> dict[str, object]:
        """Return the initial value of every state by name."""
        return {
            'Y': self.income,
            'Y[-1]': 100.0,
        }

    def _get_parameters(self) -> dict[str, object]:
        """Return the value of every parameter by name."""
        return {
            'c': self.c,
            'v': self.v,
            'G': self.G,
        }

    def _get_equations(self) -> Equations:
        """Return the components of the system built from its model parameters."""
        return {
            'Y': self._get_next_income,
        }
