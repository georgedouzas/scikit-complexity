"""Conservative encounters between wealth classes, with cooperation and competition."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.interactions import EncounterInteraction
from ....specs import ParameterSpec


def exchange_table(classes: int, cooperation: object, competition: object) -> npt.NDArray[np.object_]:
    """Return the transition table of encounters between wealth classes, `A[h, k, i]`.

    A candidate of class `h` that meets a richer field particle moves one class up with probability `cooperation`
    and one class down with probability `competition`. One that meets a poorer field particle moves one class up with
    probability `competition` and one class down with probability `cooperation`. So competition makes the rich richer
    and the poor poorer, and cooperation does the opposite. A move past the poorest or the richest class is not made.

    Args:
        classes:
            The number of wealth classes.

        cooperation:
            The probability of the move that narrows the gap between the two classes, a number or a parameter.

        competition:
            The probability of the move that widens it, a number or a parameter.

    Returns:
        table:
            The transition probabilities, which sum to one over the outcomes.

    Examples:
        >>> from skcomplex.models.economics.econophysics import exchange_table
        >>> exchange_table(3, 0.2, 0.3)[0, 2].astype(float).tolist()
        [0.8, 0.2, 0.0]
    """
    table = np.zeros((classes, classes, classes), dtype=object)
    for candidate in range(classes):
        for field in range(classes):
            gap = int(np.sign(field - candidate))
            moves = {candidate + gap: cooperation, candidate - gap: competition} if gap else {}
            for outcome, probability in moves.items():
                if 0 <= outcome < classes:
                    table[candidate, field, outcome] += probability
            table[candidate, field, candidate] += 1.0 - table[candidate, field].sum()
    return table


class WealthExchangeInteraction(EncounterInteraction):
    """Conservative encounters between wealth classes, with cooperation and competition.

    A candidate of class `h` meets a field particle of class `k` at the rate `eta` and moves by the transition table
    of `exchange_table`.

    Args:
        label:
            The label of the interaction.

        candidate:
            The path of the distribution over wealth classes whose particles move.

        field:
            The path of the distribution over wealth classes they meet, the candidate again.

        classes:
            The number of wealth classes.

        eta:
            The encounter rate.

        cooperation:
            The probability of the move that narrows the gap between the classes of an encounter.

        competition:
            The probability of the move that widens it.

    Examples:
        >>> from skcomplex.models.economics.econophysics import WealthExchangeInteraction
        >>> WealthExchangeInteraction('exchange', candidate='wealth', field='wealth').check().competition
        0.3
    """

    def __init__(
        self,
        label: str,
        *,
        candidate: str,
        field: str,
        classes: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 9,
        eta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        cooperation: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.2,
        competition: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
    ) -> None:
        super().__init__(label, candidate=candidate, field=field, rate=self._get_rate, kernel=self._get_kernel)
        self.classes = classes
        self.eta = eta
        self.cooperation = cooperation
        self.competition = competition

    def _get_rate(self, s: SimpleNamespace) -> Expression:
        """Return the encounter rate `eta`."""
        return s.eta

    def _get_kernel(self, s: SimpleNamespace) -> npt.NDArray[np.object_]:
        """Return the transition table of the encounters, with the cooperation and competition parameters."""
        return exchange_table(int(self.classes), s.cooperation, s.competition)
