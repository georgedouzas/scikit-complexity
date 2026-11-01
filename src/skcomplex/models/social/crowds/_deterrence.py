"""Encounters of citizens with security forces, which move criminal citizens toward honesty."""

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


def deterrence_table(classes: int, forces: int) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the rate pattern and the transition table of the encounters of citizens with security forces.

    A citizen of a criminal class `h > 0` who meets a security agent of efficacy `v_k` moves one class down at a rate
    proportional to `v_k`. Honest citizens are left alone.

    Args:
        classes:
            The number of criminality classes.

        forces:
            The number of efficacy classes of the security forces.

    Returns:
        pattern:
            The rate pattern `eta[h, k]`, the efficacy `v_k` for criminal classes and 0 for honest ones.

        table:
            The transition table `A[h, k, i]`.

    Examples:
        >>> from skcomplex.models.social.crowds import deterrence_table
        >>> deterrence_table(2, 2)[0].tolist()
        [[0.0, 0.0], [0.5, 1.0]]
    """
    efficacy = np.linspace(1.0 / forces, 1.0, forces)
    pattern = np.zeros((classes, forces))
    pattern[1:] = efficacy
    table = np.zeros((classes, forces, classes))
    for candidate in range(classes):
        table[candidate, :, max(candidate - 1, 0)] = 1.0
    return pattern, table


class DeterrenceInteraction(EncounterInteraction):
    """Encounters of citizens with security forces, which move criminal citizens toward honesty.

    Args:
        label:
            The label of the interaction.

        candidate:
            The path of the distribution of citizens.

        field:
            The path of the distribution of the security forces over efficacy.

        classes:
            The number of criminality classes.

        forces:
            The number of efficacy classes of the security forces.

        deterrence:
            The rate at which the most effective agents deter a criminal citizen.

    Examples:
        >>> from skcomplex.models.social.crowds import DeterrenceInteraction
        >>> DeterrenceInteraction('deterrence', candidate='citizens', field='security').deterrence
        2.0
    """

    def __init__(
        self,
        label: str,
        *,
        candidate: Annotated[str, RoleSpec()],
        field: Annotated[str, RoleSpec()],
        classes: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 5,
        forces: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 3,
        deterrence: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 2.0,
    ) -> None:
        super().__init__(label, candidate=candidate, field=field, rate=self._get_rate, kernel=self._get_kernel)
        self.classes = classes
        self.forces = forces
        self.deterrence = deterrence

    def _get_rate(self, s: SimpleNamespace) -> object:
        """Return the deterrence rate of every pair of a criminality class and an efficacy class."""
        return deterrence_table(int(self.classes), int(self.forces))[0] * s.deterrence

    def _get_kernel(self, s: SimpleNamespace) -> object:
        """Return the table that moves a deterred citizen one class down."""
        return deterrence_table(int(self.classes), int(self.forces))[1]
