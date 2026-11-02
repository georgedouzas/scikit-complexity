"""Recruitment of security agents in proportion to the mean criminality, a macro-scale external action."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from types import SimpleNamespace
from typing import Annotated

import numpy as np
import numpy.typing as npt
from sage.all__sagemath_symbolics import Expression, oo
from sage.sets.real_set import RealSet

from ....core.interactions import RateInteraction
from ....core.laws.utils import moment
from ....specs import ParameterSpec, RoleSpec


def criminality_grid(classes: int) -> npt.NDArray[np.float64]:
    """Return the criminality of every class of citizens, from 0 for honest to 1 for criminal.

    Args:
        classes:
            The number of criminality classes.

    Returns:
        criminality:
            The criminality of every class.

    Examples:
        >>> from skcomplex.models.social.crowds import criminality_grid
        >>> criminality_grid(3).tolist()
        [0.0, 0.5, 1.0]
    """
    return np.linspace(0.0, 1.0, classes)


class RecruitmentInteraction(RateInteraction):
    """Recruitment of security agents in proportion to the mean criminality, a macro-scale external action.

    It is a `RateInteraction` of the density `f` of the security forces that reads the citizens. The security forces
    grow at the rate `recruitment m share_k` in efficacy class `k`, where `m` is the mean criminality of the citizens,
    a moment of their distribution, and retire at the rate `retirement`.

    Args:
        label:
            The label of the interaction.

        target:
            The path of the distribution of the security forces over efficacy.

        citizens:
            The path of the distribution of citizens.

        classes:
            The number of criminality classes.

        recruitment:
            The recruitment rate per unit of mean criminality.

        retirement:
            The retirement rate.

    Examples:
        >>> from skcomplex.models.social.crowds import RecruitmentInteraction
        >>> RecruitmentInteraction('recruitment', target='security', citizens='citizens').retirement
        0.1
    """

    def __init__(
        self,
        label: str,
        *,
        target: Annotated[str, RoleSpec()],
        citizens: Annotated[str, RoleSpec()],
        classes: Annotated[int, ParameterSpec(domain=RealSet.open(0, oo))] = 5,
        recruitment: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.5,
        retirement: Annotated[float, ParameterSpec(domain=RealSet.closed_open(0, oo))] = 0.1,
    ) -> None:
        super().__init__(label, target=target, state='f', rate=self._get_rate)
        self.citizens = citizens
        self.classes = classes
        self.recruitment = recruitment
        self.retirement = retirement

    def _get_rate(self, s: SimpleNamespace) -> Expression:
        """Return the recruitment in proportion to the mean criminality of the citizens, less the retirement."""
        security, citizens = s.target, s.citizens
        criminality = moment(citizens.f, 1, criminality_grid(int(self.classes)), normalized=True)
        return s.recruitment * criminality * security.share - s.retirement * security.f
