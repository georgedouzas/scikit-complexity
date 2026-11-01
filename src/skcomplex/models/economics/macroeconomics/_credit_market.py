"""Credit market that also reports the spread of the rates firms pay over the cheapest bank's rate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar

import numpy as np

from ....core.interactions import MatchingInteraction
from ....engine.solvers import ModelView
from ....specs import (
    MatchSpec,
)
from ._firm_view import FirmView
from ._lender_view import LenderView


class CreditMarketInteraction(MatchingInteraction[ModelView]):
    """Credit market that also reports the spread of the rates firms pay over the cheapest bank's rate.

    The firms request loans and the banks provide them. The spread is the mean of `r_i / r_min - 1` over the borrowing
    firms, the dispersion of rates that a market with costly search leaves and that switching to cheaper lenders
    erodes.

    Args:
        label:
            The label of the market.

        requesters:
            The path of the group of firms.

        providers:
            The path of the group of banks.

        rule:
            The spec of the algorithm that matches firms with banks, such as `PartnerSwitchSpec()`.

        limit:
            The state with the rate of both sides, or a pair of the firms' and the banks' states.

        quantity:
            The state with the loan of both sides, or a pair of the firms' and the banks' states.

        budget:
            The state of a firm with what it can spend.

        transfers:
            The states every loan moves, each with the sign `-1` for a payment of the firm or `1` for a delivery to
            it.

        phase:
            The phase the market clears in, or `None` for the default phase.

        on_match:
            A function of every loan and the model view, run after the transfers, or `None`.

    Examples:
        >>> from skcomplex.models.economics.macroeconomics import CreditMarketInteraction
        >>> CreditMarketInteraction('credit', requesters='firms', providers='banks').reports[-1]
        'spread'
    """

    reports: ClassVar[tuple[str, ...]] = (*MatchingInteraction.reports, 'spread')

    def _report_matches(self, matches: list[MatchSpec], unmatched: float, model: ModelView) -> None:
        """Report the price, the volume, the unmatched quantity, and the spread of the rates over the cheapest rate.

        Args:
            matches:
                The loans of the clearing.

            unmatched:
                The quantity of both sides left after the clearing.

            model:
                The model view of the market.
        """
        super()._report_matches(matches, unmatched, model)
        cheapest = min((bank.ask for bank in model.agents('providers', LenderView) if bank.ask > 0), default=0.0)
        paid = [firm.rate for firm in model.agents('requesters', FirmView) if firm.loan > 0]
        spread = float(np.mean(paid) / cheapest - 1) if paid and cheapest > 0 else float('nan')
        model.report('spread', spread)
