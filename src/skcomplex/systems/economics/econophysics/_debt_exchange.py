"""Random pairwise trades of Dragulescu and Yakovenko of a fixed amount, allowing debt down to a limit."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import TransferInteraction
from ....core.components.types import SystemViewProtocol
from ....core.specs import ParameterSpec
from ._saving_trader_view import SavingTraderView


class DebtExchangeInteraction(TransferInteraction):
    """Random pairwise trades of Dragulescu and Yakovenko of a fixed amount, allowing debt down to a limit.

    In every trade an agent pays a fixed amount to another agent drawn at random, unless its wealth would fall below
    minus the debt limit. Every step, every agent pays once, in random order. The stationary wealth distribution is
    the exponential `P(w) = exp(-(w + d) / T) / T` shifted to the debt limit `d`, with temperature `T = <w> + d`.

    Args:
        label:
            The label of the interaction.

        source:
            The path of the group of traders.

        target:
            The path of the group of traders, the source again.

        debt:
            The debt limit `d`, the largest debt an agent can have.

        amount:
            The amount paid in every trade.

        phase:
            The phase the trades run in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import DebtExchangeInteraction
        >>> DebtExchangeInteraction('trades', source='traders', target='traders', debt=5.0).debt
        5.0
    """

    def __init__(
        self,
        label: str,
        *,
        source: str,
        target: str,
        debt: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        amount: Annotated[float, ParameterSpec(domain=RealSet.open(0, oo))] = 1.0,
        phase: str | None = None,
    ) -> None:
        super().__init__(
            label,
            source=source,
            target=target,
            state='wealth',
            amount=amount,
            pairing='random',
            guard=self._is_solvent,
            form='agent',
            phase=phase,
        )
        self.debt = debt

    def _is_solvent(self, payer: SavingTraderView, payee: SavingTraderView, system: SystemViewProtocol) -> bool:
        """Return whether the payer stays above minus the debt limit after paying."""
        return bool(payer.wealth - system.parameter('amount') >= -system.parameter('debt'))
