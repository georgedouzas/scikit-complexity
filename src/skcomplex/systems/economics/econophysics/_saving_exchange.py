"""Random pairwise trades of Chakraborti and Chakrabarti, where every agent saves a fraction of its wealth."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Annotated

from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.components.interactions import TransferInteraction
from ....core.components.types import SystemViewProtocol
from ....core.specs import ParameterSpec
from ._saving_trader_view import SavingTraderView


class SavingExchangeInteraction(TransferInteraction):
    """Random pairwise trades of Chakraborti and Chakrabarti, where every agent saves a fraction of its wealth.

    A trade between agents `i` and `j` keeps total wealth and gives `w_i' = lambda w_i + eps (1 - lambda) (w_i + w_j)`
    and `w_j' = lambda w_j + (1 - eps) (1 - lambda) (w_i + w_j)` with `eps` uniform on `[0, 1]`, so `i` transfers
    `(1 - lambda) (w_i - eps (w_i + w_j))` to `j`, an amount of either sign. Every step, every agent trades once with
    another agent drawn at random, in random order. Without saving the stationary wealth distribution is exponential,
    and with saving propensity `lambda` it is close to a Gamma distribution of shape `n = 1 + 3 lambda / (1 - lambda)`.

    Args:
        label:
            The label of the interaction.

        source:
            The path of the group of traders.

        target:
            The path of the group of traders, the source again.

        saving:
            The saving propensity `lambda`, the fraction of its wealth that an agent keeps out of a trade.

        phase:
            The phase the trades run in, or `None` for the default phase.

    Examples:
        >>> from skcomplex.systems.economics.econophysics import SavingExchangeInteraction
        >>> SavingExchangeInteraction('trades', source='traders', target='traders').saving
        0.0
    """

    def __init__(
        self,
        label: str,
        *,
        source: str,
        target: str,
        saving: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.0,
        phase: str | None = None,
    ) -> None:
        super().__init__(
            label,
            source=source,
            target=target,
            state='wealth',
            amount=self._get_amount,
            pairing='random',
            form='agent',
            phase=phase,
        )
        self.saving = saving

    def _get_amount(self, first: SavingTraderView, second: SavingTraderView, system: SystemViewProtocol) -> float:
        """Return the wealth the first trader transfers to the second, of either sign."""
        saving = system.parameter('saving')
        share = float(system.rng.random())
        return (1 - saving) * (first.wealth - share * (first.wealth + second.wealth))
