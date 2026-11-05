"""Synchronous spread of distress, `h_i = min(1, h_i + sum_j W_ji h_j)` over the distressed banks `j`."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Annotated

import numpy as np

from ....core.components.interactions import Interaction
from ....core.components.laws import BaseLaw, RuleLaw
from ....core.engine.solvers import SystemView
from ....core.specs import RoleSpec
from ._interbank import UNDISTRESSED
from ._interbank_view import InterbankView

DISTRESSED = 1


INACTIVE = 2


class DebtRankCascadeInteraction(Interaction):
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
        >>> from skcomplex.systems.economics.finance import DebtRankCascadeInteraction
        >>> DebtRankCascadeInteraction('cascade', banks='banks', impact=[[0.0, 0.5], [0.0, 0.0]]).impact
        [[0.0, 0.5], [0.0, 0.0]]
    """

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
