"""Welfare policy that taxes the richest classes and helps the poorest, a micro-scale external action."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import oo
from sage.sets.real_set import RealSet

from ....core.interactions import EncounterInteraction
from ....specs import ParameterSpec


def policy_table(classes: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the rate pattern and the transition table of a welfare policy that taxes the rich and helps the poor.

    The policy moves a candidate of the richest third of the classes one class down, and a candidate of the poorest
    third one class up, and leaves the middle classes alone.

    Args:
        classes:
            The number of wealth classes.

    Returns:
        pattern:
            The rate pattern `eta[h, 0]`, 1 for the classes the policy acts on and 0 for the others.

        table:
            The transition table `A[h, 0, i]` of the policy.

    Examples:
        >>> from skcomplex.models.economics.econophysics import policy_table
        >>> policy_table(3)[0].ravel().tolist()
        [1.0, 0.0, 1.0]
    """
    pattern = np.zeros((classes, 1))
    table = np.zeros((classes, 1, classes))
    third = max(1, classes // 3)
    for candidate in range(classes):
        outcome = candidate + 1 if candidate < third else candidate - 1 if candidate >= classes - third else candidate
        pattern[candidate, 0] = float(outcome != candidate)
        table[candidate, 0, outcome] = 1.0
    return pattern, table


class WelfarePolicyInteraction(EncounterInteraction):
    """Welfare policy that taxes the richest classes and helps the poorest, a micro-scale external action.

    The policy is an encounter whose field role binds a group with a fixed density, so it acts on the transitions of
    the population at the rate `tax` times that density, by the pattern and the table of `policy_table`.

    Args:
        label:
            The label of the interaction.

        candidate:
            The path of the distribution over wealth classes.

        field:
            The path of the group of one member whose fixed density is the strength of the policy.

        classes:
            The number of wealth classes.

        tax:
            The rate at which the policy moves the richest classes down and the poorest up.

    Examples:
        >>> from skcomplex.models.economics.econophysics import WelfarePolicyInteraction
        >>> WelfarePolicyInteraction('welfare', candidate='wealth', field='state').check().tax
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        candidate: str,
        field: str,
        classes: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 9,
        tax: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, candidate=candidate, field=field, rate=self._get_rate, kernel=self._get_kernel)
        self.classes = classes
        self.tax = tax

    def _get_rate(self, s: SimpleNamespace) -> npt.NDArray[np.object_]:
        """Return the rate pattern of the policy times the tax."""
        pattern, _ = policy_table(int(self.classes))
        return pattern * s.tax

    def _get_kernel(self, s: SimpleNamespace) -> npt.NDArray[np.float64]:
        """Return the transition table of the policy."""
        _, table = policy_table(int(self.classes))
        return table
