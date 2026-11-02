"""Conservative encounters between citizens, in which crime attracts and honesty redeems."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.interactions import EncounterInteraction
from ....specs import ParameterSpec, RoleSpec


def contagion_table(classes: int, attraction: object, redemption: object) -> npt.NDArray[np.object_]:
    """Return the transition table of encounters between citizens, `A[h, k, i]`.

    A citizen of class `h` who meets a more criminal citizen moves one class up with probability `attraction`, and one
    who meets a more honest citizen moves one class down with probability `redemption`.

    Args:
        classes:
            The number of criminality classes.

        attraction:
            The probability that crime attracts, a number or a parameter.

        redemption:
            The probability that honesty redeems, a number or a parameter.

    Returns:
        table:
            The transition probabilities, which sum to one over the outcomes.

    Examples:
        >>> from skcomplex.models.social.crowds import contagion_table
        >>> contagion_table(3, 0.3, 0.1)[0, 2].astype(float).tolist()
        [0.7, 0.3, 0.0]
    """
    table = np.zeros((classes, classes, classes), dtype=object)
    for candidate in range(classes):
        for field in range(classes):
            if field > candidate:
                table[candidate, field, candidate + 1] += attraction
            elif field < candidate:
                table[candidate, field, candidate - 1] += redemption
            table[candidate, field, candidate] += 1 - table[candidate, field].sum()
    return table


class CriminalContagionInteraction(EncounterInteraction):
    """Conservative encounters between citizens, in which crime attracts and honesty redeems.

    Args:
        label:
            The label of the interaction.

        candidate:
            The path of the distribution of citizens whose members move.

        field:
            The path of the distribution of citizens they meet, usually the same distribution.

        classes:
            The number of criminality classes.

        eta:
            The encounter rate.

        attraction:
            The probability that a citizen who meets a more criminal one moves one class up.

        redemption:
            The probability that a citizen who meets a more honest one moves one class down.

    Examples:
        >>> from skcomplex.models.social.crowds import CriminalContagionInteraction
        >>> CriminalContagionInteraction('contagion', candidate='citizens', field='citizens').attraction
        0.3
    """

    def __init__(
        self,
        label: str,
        *,
        candidate: Annotated[str, RoleSpec()],
        field: Annotated[str, RoleSpec()],
        classes: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 5,
        eta: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 1.0,
        attraction: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.3,
        redemption: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, candidate=candidate, field=field, rate=self._get_rate, kernel=self._get_kernel)
        self.classes = classes
        self.eta = eta
        self.attraction = attraction
        self.redemption = redemption

    def _get_rate(self, s: SimpleNamespace) -> object:
        """Return the encounter rate."""
        return s.eta

    def _get_kernel(self, s: SimpleNamespace) -> object:
        """Return the table of the attraction of crime and the redemption of honesty."""
        return contagion_table(int(self.classes), s.attraction, s.redemption)
